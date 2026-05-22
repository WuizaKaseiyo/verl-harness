compute_slurm skill — provisioning, launch, monitor when the target is `local-slurm` (this host is a Slurm login node).

## Provisioning checks (used by `provision_env`)

### Env-collision pre-flight (REQUIRED before any verl launch)

Slurm clusters with heterogeneous-GPU-aware site configs commonly set `ROCR_VISIBLE_DEVICES` alongside `CUDA_VISIBLE_DEVICES` on every worker (via the GRES plugin, prolog scripts, or vendor module-files). verl's `_setup_env_cuda_visible_devices` (at `verl/single_controller/base/worker.py:256-267`) raises `ValueError("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.")` when both are set. This crash occurs inside the Ray worker actor's `__init__`, *past* every standard pre-flight (sbatch dry-run, login-node `import verl/torch/vllm`). It is therefore not catchable by import probes; we must probe the worker env directly.

The probe runs a **short, GPU-allocated srun** that just dumps env:

```bash
srun --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
     --gres=gpu:1 --time=0:02:00 --mem=4G --cpus-per-task=1 \
     bash -c 'env | grep -E "^(ROCR|HIP|CUDA)_VISIBLE_DEVICES" || echo "(none set)"'
```

If the worker reports both `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`, OR both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`, the harness **must inject** into the head of the generated `launch_env.sh`:

```bash
unset ROCR_VISIBLE_DEVICES   # cluster GRES plugin sets this for cross-vendor portability;
                              # collides with verl's worker.py:267 guard on NVIDIA partitions.
unset HIP_VISIBLE_DEVICES    # defensive — same family of variables.
```

Record the verbatim env-grep output and the injection in `env_state.md` under `## Env-collision pre-flight`.

If the probe srun is itself blocked by a long queue (> 5 min), fall back to:
1. document the gap explicitly in `env_state.md`;
2. inject the `unset` lines defensively (low-risk — `unset` is a no-op if the variable wasn't set);
3. proceed to `sbatch --test-only` for the directive syntax check.

### vLLM engine prerequisite (REQUIRED when recipe uses `rollout.name=vllm`)

verl's async rollout server (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:35`) imports `from vllm.v1.engine.async_llm import AsyncLLM` — **hardcoded V1**. With modern vllm (≥0.10.0) and certain config combinations (e.g., `enable_sleep_mode=True`, background-thread engine), vllm silently falls back to V0 unless `VLLM_USE_V1=1` is explicitly set. verl then refuses with `ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False.`

If `workspace/recipe/recipe.md` records `rollout.name=vllm`, the harness **must inject** into `launch_env.sh`:

```bash
export VLLM_USE_V1=1   # required by verl/workers/rollout/vllm_rollout/vllm_async_server.py (V1-only).
```

Record the injection (with the recipe-side trigger) in `env_state.md` under `## vLLM engine prerequisite`.

### Container wrapping (optional — depends on the user's verl template)

The verl checkout's official slurm template (`<verl_root>/examples/tutorial/slurm/ray_on_slurm.slurm`) wraps every srun in an Apptainer/Singularity container (`apptainer run --nv --bind $verl_workdir $apptainer_image_path`). Detect with:

```bash
grep -E '^\s*(apptainer|singularity|docker)\s+run' "$VERL_ROOT/examples/tutorial/slurm/ray_on_slurm.slurm"
```

- If a container wrap is present **and** the user's `compute_choice.md` records `container_image: <path>` (or a non-container Conda/venv env hasn't been declared), preserve the wrap.
- If the user has a Conda env that already contains verl+torch+vllm and `compute_choice.md` records `container: none`, drop the wrap and replace it with `source <conda_root>/etc/profile.d/conda.sh && conda activate <env_name>` inside `launch_env.sh`.

### Standard slurm checks (run after the pre-flights above)

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
