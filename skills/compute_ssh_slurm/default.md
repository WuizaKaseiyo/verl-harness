compute_ssh_slurm skill — provisioning, launch, monitor when the target is `ssh-slurm` (remote Slurm cluster reached via ssh).

Identical to `compute_slurm` in *what* gets run; the difference is *where* — every slurm command is wrapped in `ssh <alias>`. The agent's local shell never enters the cluster.

## Pre-requisites

- An ssh alias (in `~/.ssh/config`) for the remote login node, with key-based auth — no interactive prompts. Default alias source: `training_intent.md` `ssh.alias`; fallback: `$VERL_HARNESS_REMOTE`.
- The remote `$HOME` has a `verl-harness-runs/` working directory writable by the agent. The harness creates it on first use.
- A remote verl checkout — record its path as `REMOTE_VERL_ROOT`. The agent asks the user for it during `provision_env` (or it's recorded in `training_intent.md` as `ssh.remote_verl_root`).

## Provisioning checks (used by `provision_env`)

### Env-collision pre-flight (REQUIRED — same root cause as `compute_slurm`)

verl's `_setup_env_cuda_visible_devices` (at `verl/single_controller/base/worker.py:256-267`) raises `ValueError` when both `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set. Probe the remote worker env:

```bash
ssh <alias> "srun --partition=<remote_partition> --account=<remote_account> \
                  --gres=gpu:1 --time=0:02:00 --mem=4G --cpus-per-task=1 \
                  bash -c 'env | grep -E \"^(ROCR|HIP|CUDA)_VISIBLE_DEVICES\" || echo \"(none set)\"'"
```

If both `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` (or both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`) appear, inject into `launch_env.sh`:

```bash
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
```

Record the verbatim probe output + injection in `env_state.md` under `## Env-collision pre-flight`.

### vLLM engine prerequisite (REQUIRED when recipe uses `rollout.name=vllm`)

verl's async rollout server hardcodes the V1 AsyncLLM constructor (`vllm_async_server.py:35`). vllm ≥0.10.0 may silently fall back to V0 unless `VLLM_USE_V1=1` is set, causing verl to refuse with `ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False.`

If `workspace/recipe/recipe.md` records `rollout.name=vllm`, inject into `launch_env.sh`:

```bash
export VLLM_USE_V1=1
```

Record the injection in `env_state.md` under `## vLLM engine prerequisite`.

### Container wrapping (same detection as `compute_slurm`)

```bash
ssh <alias> "grep -E '^\s*(apptainer|singularity|docker)\s+run' $REMOTE_VERL_ROOT/examples/tutorial/slurm/ray_on_slurm.slurm"
```

Preserve the container wrap if `compute_choice.md` did not record `container: none`. If the user has a Conda env on the remote, switch to the conda activation pattern (see `compute_slurm` skill).

### Standard remote slurm checks

```bash
ssh <alias> "sinfo --version; squeue -V"
ssh <alias> "test -d $REMOTE_VERL_ROOT && echo OK || echo MISSING"
ssh <alias> "ls $REMOTE_VERL_ROOT/examples/tutorial/slurm/ray_on_slurm.slurm 2>&1 || echo MISSING"
```

For the worker-env check, attempt a short remote interactive srun (same shape as `compute_slurm`'s, prefixed with `ssh <alias>`). If that's not feasible, fall back to `ssh <alias> "sbatch --test-only ..."` after the slurm script is written.

## Launch (used by `launch_training`)

The slurm script is **assembled locally** (so the agent has it on disk for inspection / re-submission), then rsync'd to the remote host before submission.

```bash
# Local: write workspace/job/job.slurm (same template as compute_slurm)
# Local: write workspace/env/launch_env.sh (same as compute_local; paths inside reference REMOTE_VERL_ROOT, not VERL_ROOT)

# Remote staging directory
REMOTE_RUN_DIR="\$HOME/verl-harness-runs/<run_id>"
ssh <alias> "mkdir -p $REMOTE_RUN_DIR"

# Push
rsync -avz workspace/job/job.slurm workspace/env/launch_env.sh <alias>:$REMOTE_RUN_DIR/

# Submit
JOBID=$(ssh <alias> "cd $REMOTE_RUN_DIR && sbatch --parsable job.slurm")
echo "$JOBID" > workspace/job/slurm_jobid
echo "<alias>" > workspace/job/remote_alias
echo "$REMOTE_RUN_DIR" > workspace/job/remote_path
```

Record `job_info.md` with `target: ssh-slurm`, `remote_alias`, `remote_path`, `slurm_jobid`.

## Monitor (used by `monitor_training`)

Polling cadence: **90 seconds** (ssh overhead is real; over-polling racks up handshake cost).

```bash
JOBID=$(cat workspace/job/slurm_jobid)
ALIAS=$(cat workspace/job/remote_alias)
STATE=$(ssh "$ALIAS" "squeue -j $JOBID --noheader -o '%T'")
# … same logic as compute_slurm
```

Log tailing requires a small protocol:

```bash
ssh "$ALIAS" "tail -c +$LAST_OFFSET $REMOTE_OUTPUT_DIR/slurm-$JOBID.out" > workspace/logs/_chunk.txt
NEW_BYTES=$(stat -c%s workspace/logs/_chunk.txt)
cat workspace/logs/_chunk.txt >> workspace/logs/job_log.md
echo $((LAST_OFFSET + NEW_BYTES)) > workspace/logs/_tail_offset.txt
```

(Adjust for macOS/BSD `stat` if the agent's local host is macOS — use `stat -f%z` there.)

## Cancellation

```bash
ssh <alias> "scancel $JOBID"
```

## Cost estimate

Same heuristic as `compute_slurm`. The user is presumed to know their remote cluster's pricing model.

## Things you must not do

- Do not use ssh with password auth — the harness must not block on a password prompt. If key-based auth isn't set up, `select_compute`'s probe fails and the target is rejected up-front.
- Do not exfiltrate workspace artefacts to the remote that the user did not authorise. The rsync targets are precisely the slurm script and the launch_env.sh — nothing else.
- Do not run interactive ssh sessions. Every ssh invocation is single-command and short-lived.
