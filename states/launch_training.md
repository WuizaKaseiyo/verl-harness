# launch_training

## Description

Fire the training job. The exact mechanics depend on the compute target chosen at `select_compute`. After this state, the job is *running* (or queued); the next state (`monitor_training`) watches it to a terminal status.

Apply whichever of `compute_local`, `compute_slurm`, `compute_ssh_slurm` matches the chosen target.

Concretely:

1. **Read** `workspace/intake/training_intent.md`, `workspace/recipe/recipe.md`, `workspace/dataset/dataset.md`, `workspace/compute/compute_choice.md`, `workspace/env/env_state.md`, `workspace/env/launch_env.sh`.
2. **Assemble the launch command.** Start from the recipe's launch path:
   - **Shell-script recipe** (`recipe.md` has `script_path`): the script reads env-vars (`MODEL_PATH`, `NNODES`, `NDEVICES_PER_NODE`, `TRAIN_BATCH_SIZE`, …). Set them all from training_intent + recipe defaults, patch `data.train_files` / `data.val_files` to the dataset paths, source `launch_env.sh`, then invoke the script.
   - **Direct python module** (`recipe.md` has `module: verl.trainer.main_<algo>`): assemble the full `python -m verl.trainer.main_<algo> arg1=v1 arg2=v2 …` command, patch `data.train_files` / `data.val_files`, `actor_rollout_ref.model.path`, `trainer.default_local_dir`, etc.
3. **Compose the launch command by target:**
   - **local-direct.** Write `workspace/job/launch.sh` that sources `launch_env.sh` and runs the assembled command, redirecting stdout/stderr to `workspace/job/stdout.log` / `stderr.log`. Run it in the background; capture the PID; record `workspace/job/job_info.md` with `pid`, `start_time`, `target: local-direct`, `cmd: <full command>`.
   - **local-slurm.** Write `workspace/job/job.slurm` based on `<VERL_ROOT>/examples/tutorial/slurm/ray_on_slurm.slurm` (or a clean equivalent — see `compute_slurm` skill) with the sbatch directives from `compute_choice.md` and the assembled command inserted into the appropriate slot. Run `sbatch workspace/job/job.slurm` locally; capture the returned `JOBID`; record it in `job_info.md` with `target: local-slurm`, `slurm_jobid`, `cmd: sbatch …`, `job_script: workspace/job/job.slurm`.
   - **ssh-slurm.** Same as local-slurm except: rsync (or scp) the slurm script and `launch_env.sh` to the remote login node first (default destination `~/verl-harness-runs/<run_id>/`), then `ssh <login> sbatch …`. Capture the remote JOBID; record `target: ssh-slurm`, `remote_alias`, `remote_path`, `slurm_jobid`.
4. **HITL checkpoint (cost gate).** Before actually running the launch command, present:
   - The full final command (env vars + invocation)
   - The compute target
   - The expected resource cost: N nodes × G GPUs × estimated wall-clock from the recipe's `TOTAL_EPOCHS` and the dataset's row count. (compute_local / compute_slurm skills have rough heuristics.)
   - The output_dir where checkpoints will land
   Ask the user to confirm. Skipped with `--no-hitl`.
5. **Execute the launch.** After confirmation, run the command. Record exit code (for local-direct: the *launch* exit code, since the process is detached) or sbatch's stdout (JOBID). If the launch command itself fails (sbatch returns nonzero, ssh fails, the background process can't start), do not proceed to monitor_training — transition to `finalize` with a `launch_failed` status.
6. **Write the canonical `workspace/job/job_info.md`** the monitor state will read:
   ```
   target: local-direct | local-slurm | ssh-slurm
   started_at: <ISO8601>
   pid: <local-direct only>
   slurm_jobid: <slurm targets only>
   remote_alias: <ssh-slurm only>
   output_dir: <absolute path>
   stdout_log: <path the monitor will tail>
   stderr_log: <path the monitor will tail>
   cmd: <the full command that was run>
   ```

## Skills

- skills/compute_local
- skills/compute_slurm
- skills/compute_ssh_slurm
- skills/global

> Of the three `compute_*` skills, **read only the one matching the chosen target** in `workspace/compute/compute_choice.md`. The other two are listed for validator coverage and are not consulted on this run.

## Human Checkpoints

- **Cost gate.** Step 4. Skipped with `--no-hitl`.

## Next States

### monitor_training

**Condition:** The launch command succeeded (background PID acquired for local-direct, sbatch returned a JOBID for slurm targets). `workspace/job/job_info.md` is written.

**Deliverables:**

- job_info: Canonical job_info.md (target, started_at, pid or slurm_jobid, output_dir, stdout/stderr log paths, the literal launch command).

### finalize

**Condition:** The launch command itself failed (sbatch returned nonzero, ssh connection refused, the local process failed to start). Training never began.

**Deliverables:**

- launch_failed: A `workspace/job/launch_failed.md` recording the command that was attempted, the exit code, stderr, and a one-line diagnosis if possible.
