compute_slurm skill — provisioning, launch, monitor when the target is `local-slurm` (this host is a Slurm login node).

## Provisioning checks (used by `provision_env`)

Local checks on the login node:

```bash
# slurm is functional
sinfo --version
squeue --noheader | head

# the partition exists
sinfo -p "$SLURM_PARTITION" --noheader | head

# the account is valid (best-effort; some sites don't expose this)
sacctmgr -P show account "$SLURM_ACCOUNT" 2>/dev/null | head -3
```

Provisioning checks for the *worker* environment (torch, verl, vllm) need to run *on a worker*, not on the login node — many login nodes have no GPU and no full verl install. Use a short interactive srun:

```bash
srun --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
     --gres=gpu:1 --time=0:10:00 --pty \
     bash -c 'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"; \
              python -c "import sys; sys.path.insert(0, \"'"$VERL_ROOT"'\"); import verl; print(\"verl OK\")"'
```

If the interactive srun blocks for > 5 minutes (queue), skip it and instead validate the slurm template syntactically:

```bash
sbatch --test-only workspace/job/job.slurm
```

The `--test-only` mode parses the script and returns "Job <id> to start at <time>" without actually queuing. Use this as the minimum acceptable check.

## Launch (used by `launch_training`)

Write `workspace/job/job.slurm` derived from `<verl_root>/examples/tutorial/slurm/ray_on_slurm.slurm` with the directives populated from `workspace/compute/compute_choice.md` and the launch command inserted into the body.

```bash
#!/bin/bash
#SBATCH --job-name=<run_id>
#SBATCH --nodes=<N>
#SBATCH --ntasks-per-node=1
#SBATCH --mem=<MEM_GB>G
#SBATCH --partition=<PARTITION>
#SBATCH --time=<HH:MM:SS>
#SBATCH --account=<ACCOUNT>
#SBATCH --gpus-per-node=<G>
#SBATCH --cpus-per-task=<C>
#SBATCH --output=<output_dir>/slurm-%j.out
#SBATCH --error=<output_dir>/slurm-%j.err

set -euo pipefail
source "<workspace>/env/launch_env.sh"

# Ray cluster bring-up — adapted from ray_on_slurm.slurm
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export ip_head="$head_node_ip:6379"

srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port=6379 \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
sleep 10

for ((i=1; i<${#nodes_array[@]}; i++)); do
  srun --nodes=1 --ntasks=1 -w "${nodes_array[$i]}" \
    ray start --address "$ip_head" \
              --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
  sleep 5
done

# The verl trainer entry-point — assembled by recipe state
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  <the full command from recipe.md, expanded with all env vars substituted>
```

If verl's published `ray_on_slurm.slurm` is significantly different on the user's checkout, *use the user's checkout's version as the structural source-of-truth* — patch its directives + body slots rather than the template above. The template above is a fallback for repos where the slurm example has drifted.

Submit:

```bash
JOBID=$(sbatch --parsable workspace/job/job.slurm)
echo "$JOBID" > workspace/job/slurm_jobid
```

Record `job_info.md` with `target: local-slurm`, `slurm_jobid: <id>`, `job_script: workspace/job/job.slurm`, etc.

## Monitor (used by `monitor_training`)

Polling cadence: **60 seconds**.

Each poll:

```bash
JOBID=$(cat workspace/job/slurm_jobid)
STATE=$(squeue -j "$JOBID" --noheader -o '%T')

# Empty means the job is no longer in queue → it ended.
if [ -z "$STATE" ]; then
    FINAL=$(sacct -j "$JOBID" --format=State --parsable2 --noheader | head -1)
    # FINAL ∈ {COMPLETED, FAILED, CANCELLED, TIMEOUT, PREEMPTED, NODE_FAIL, BOOT_FAIL, OUT_OF_MEMORY}
fi
```

Tail the slurm output / error files. They live at `<output_dir>/slurm-<JOBID>.out` and `.err`. Update `workspace/logs/job_log.md` with new bytes, and `progress.csv` with extracted step metrics.

For multi-node runs, slurm interleaves per-node output. Lines starting with `[node00X]` are common; preserve them as-is in the log file.

## Cancellation

```bash
scancel "$JOBID"
```

Then poll until `sacct` reports `CANCELLED`. Record the cancellation.

## Cost estimate

Slurm partitions have per-node hourly cost (or fair-share unit cost). The skill cannot know this without site-specific info; the cost gate should report node-hours (`nodes × time_limit`) and let the user translate it to dollars at their site.

For wall-clock estimate, defer to `compute_local`'s `steps_per_minute` heuristic — it doesn't change with target.

## Things you must not do

- Do not run training in a non-batch (`srun --pty`) session as a substitute for `sbatch`. The harness's monitor model assumes batch jobs.
- Do not silently retry sbatch on transient failures. A failed submission writes `launch_failed.md` and the harness exits via `finalize`.
- Do not over-poll. 60-second cadence is the floor; faster polling spams `squeue` and annoys cluster admins.
