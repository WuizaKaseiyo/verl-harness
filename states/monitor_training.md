# monitor_training

## Description

Watch the running training job until it reaches a terminal status: success (training finished, final checkpoint exists), crash (process exited nonzero, slurm job FAILED), preempted (slurm preemption / timeout — possibly resumable), or user-cancelled. **This state loops internally** — it does not transition per poll cycle. Each iteration polls job state, tails new log lines, evaluates terminal conditions, and either sleeps (job still running) or transitions out (terminal status reached).

Apply the `training_monitor` skill (`skills/training_monitor`) for the polling rules, log-anomaly detection, and terminal-condition definitions, plus whichever of `compute_local` / `compute_slurm` / `compute_ssh_slurm` matches the chosen target (for the *mechanism* of polling).

Concretely, each iteration:

1. **Read** `workspace/job/job_info.md`.
2. **Poll job status:**
   - **local-direct:** `kill -0 <pid>`. If alive → still running. If dead → check exit code from a side-channel (the launch script wrote `workspace/job/exit_code` on completion).
   - **local-slurm:** `squeue -j <jobid> --noheader -o '%T'` returns one of `PENDING`, `RUNNING`, `COMPLETING`. If empty, the job exited; `sacct -j <jobid> --format=State --parsable2 --noheader` gives the final state (`COMPLETED`, `FAILED`, `TIMEOUT`, `PREEMPTED`, `CANCELLED`).
   - **ssh-slurm:** same as local-slurm but via `ssh <alias> "squeue …"` / `ssh <alias> "sacct …"`.
3. **Tail new log lines.** Read incremental content from stdout / stderr log paths since the last poll. Append all new lines to `workspace/logs/job_log.md`. Use simple file-offset tracking (record the last byte-offset polled in `workspace/logs/_tail_state.json`).
4. **Anomaly scan.** Apply training_monitor's pattern set on the new log lines:
   - OOM (`torch.cuda.OutOfMemoryError`, `CUDA out of memory`, `killed`-with-OOM-context)
   - NaN / Inf loss (`loss is nan`, `loss=inf`, `gradient is nan`)
   - vLLM rollout crash (`vllm.engine` errors)
   - NCCL hang (`NCCL WARN`, `Timeout waiting for ack`)
   - Slurm preemption (`slurmstepd: error: *** STEP … CANCELLED AT …`)
   Record any matches in `workspace/logs/anomalies.md` with timestamp + matched line.
5. **Progress sample.** Every N poll cycles, extract: current step, current epoch, recent train loss, recent reward, recent throughput. Append a row to `workspace/logs/progress.csv`. The verl trainer's own log lines have a predictable `step=… epoch=… loss=…` format; the training_monitor skill has the regex.
6. **Evaluate terminal conditions:**
   - **Success.** Job exit code 0 (local-direct) or slurm state `COMPLETED` AND `<output_dir>/checkpoints/` has at least one numbered checkpoint dir AND the training log's last line contains a "training finished" marker (the regex is in training_monitor).
   - **Crash.** Job exit code nonzero (local-direct) or slurm state in `{FAILED, NODE_FAIL, BOOT_FAIL, OUT_OF_MEMORY}`. Record the last 200 lines of stderr to `workspace/logs/crash_tail.md`.
   - **Preempted / Timeout.** Slurm state `TIMEOUT` or `PREEMPTED`. Record the last checkpoint step in `workspace/logs/preempted.md` — if there is one, training is resumable.
   - **User cancelled.** A `workspace/job/cancel_requested` file exists (the user creates it externally, or invokes the harness with `--cancel`). Issue `scancel <jobid>` (slurm) or `kill <pid>` (local-direct), set status `cancelled`.
7. **Sleep before next poll.** Local-direct: 30s. Slurm: 60s (be polite to the scheduler). ssh-slurm: 90s.

When a terminal condition is reached, **write `workspace/job/job_status.md`** with `status: success | crashed | preempted | cancelled`, the final step / epoch / checkpoint path, the last loss/reward values from `progress.csv`, and pointers to the anomaly + crash log files.

## Skills

- skills/training_monitor
- skills/compute_local
- skills/compute_slurm
- skills/compute_ssh_slurm
- skills/global

> Of the three `compute_*` skills, **read only the one matching the chosen target** in `workspace/compute/compute_choice.md` for the polling mechanism. The other two are listed for validator coverage.

## Human Checkpoints

- **Cancellation request.** This state honours a user-initiated cancel during the loop (via the `workspace/job/cancel_requested` file). No proactive pause — the loop runs continuously. Skipped with `--no-hitl` is irrelevant here; the loop is automatic.

## Next States

### summarize

**Condition:** A terminal status (success | crashed | preempted | cancelled) has been reached and `workspace/job/job_status.md` is written.

**Deliverables:**

- job_status: The terminal status, the final step / epoch / checkpoint path (when applicable), the last training metrics (loss, reward) from `progress.csv`, and pointers to `anomalies.md`, `crash_tail.md`, or `preempted.md` depending on status.
- training_logs: The accumulated `workspace/logs/job_log.md`, `progress.csv`, `anomalies.md` (and crash/preempted siblings as relevant) — the raw evidence summarize consumes.
