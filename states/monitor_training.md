# monitor_training

## Description

Watch the running training job until it reaches a terminal status: success (training finished, final checkpoint exists), crash (process exited nonzero, slurm job FAILED), preempted (slurm preemption / timeout — possibly resumable), or user-cancelled — then write `workspace/job/job_status.md` and transition to `summarize`.

Apply the `training_monitor` skill (`skills/training_monitor`) for the polling rules, anomaly + divergence detection, and terminal-condition definitions, plus whichever of `compute_local` / `compute_slurm` / `compute_ssh_slurm` matches the chosen target (for the *mechanism* of polling).

**Arm-and-detach — do not sit in a busy poll loop.** Training runs are routinely *hours to days*. An LLM agent must **not** spin a 30/60/90 s wake loop across that span: it would re-read context hundreds of times, get auto-compacted repeatedly, and burn enormous cost on a state that barely changes between polls. Instead this state arms a cheap, detached watcher and steps back:

1. **Author + launch the detached poller.** Write `workspace/job/watch.sh` from the template `skills/training_monitor/templates/watch_poller.py` and start it detached (`nohup … &` for local; a tiny background process otherwise). It is a **non-LLM** process that owns the mechanical loop and survives independently of any agent session — it polls job state at the target cadence (local-direct 30 s / slurm 60 s / ssh-slurm 90 s; those cadences are fine for a *script*, just not for the agent), tails logs by byte-offset into `workspace/logs/job_log.md`, runs the `training_monitor` anomaly + divergence scan, appends `workspace/logs/progress.csv`, and **on terminal status writes `workspace/job/job_status.md` and touches the sentinel `workspace/job/terminal`**.
2. **(slurm — preferred for the terminal event) submit a dependency job.** `sbatch --dependency=afterany:<jobid>` a one-liner that records the final `sacct` state and touches `workspace/job/terminal`. The scheduler then *tells* the harness the instant training ends — zero polling for the terminal event; the poller's role narrows to mid-run health.
3. **Re-engage on events, not on a fixed clock.** After arming, the agent returns to this state only when one fires: the `terminal` sentinel appears; the poller escalates an anomaly into `workspace/logs/escalation.md` (OOM / NaN / NCCL hang / **divergence** / stalled-stdout hang); or a coarse **heartbeat** elapses (default 15–30 min, derived as a few × observed seconds-per-step) at which it confirms the poller is still alive and stdout is still advancing (a frozen byte-offset while the job is still `RUNNING` = hang). On a hosted runner that keeps the session, "re-engage" can be a long sleep between checks rather than a process exit; either way the cadence is event-driven and coarse, never the poller's 30/60/90 s.

**Workspace is the source of truth.** The watch can span days and the runner's context may be auto-compressed (or the session may end and be re-invoked into `goal=resume_monitor`) between engagements. Re-read everything from the workspace each time — never rely on agent memory of upstream states:

- `workspace/intake/training_intent.md` (intent), `workspace/recipe/recipe.md` (launch args), `workspace/compute/compute_choice.md` (target + directives)
- `workspace/job/job_info.md` (`slurm_jobid`, `pid`, log paths), `workspace/job/job_status.md` (written by the poller on terminal), `workspace/logs/{progress.csv, anomalies.md, escalation.md}`

The mechanical loop the **poller** runs each cycle (the agent does not do these by hand):

1. **Read** `workspace/job/job_info.md`.
2. **Poll job status:**
   - **local-direct:** `kill -0 <pid>`. Alive → running. Dead → read `workspace/job/exit_code` (the launch script writes it on completion).
   - **local-slurm:** `squeue -j <jobid> --noheader -o '%T'` (`PENDING`/`RUNNING`/`COMPLETING`); if empty, `sacct -j <jobid> --format=State --parsable2 --noheader` gives the final state.
   - **ssh-slurm:** same via `ssh <alias> "squeue …"` / `"sacct …"`.
3. **Tail new log lines** by byte-offset into `workspace/logs/job_log.md` (offset tracked in `workspace/logs/_tail_state.json`).
4. **Anomaly + divergence scan.** Apply training_monitor's pattern set (OOM / NaN-Inf / vLLM crash / NCCL hang / slurm preemption) **and** the RL-stability + divergence rules (KL drift, entropy collapse *and* entropy explosion, validation regression, the joint divergence signal). Record matches in `workspace/logs/anomalies.md`; promote fatal/divergence findings into `workspace/logs/escalation.md` to wake the agent.
5. **Progress sample.** Parse the namespaced metric dict / tqdm line (regexes in training_monitor) and append a row to `workspace/logs/progress.csv` (dynamic columns).
6. **Evaluate terminal conditions:**
   - **Success.** Exit code 0 (local-direct) or slurm `COMPLETED` AND (`<output_dir>/global_step_<N>/` exists, OR — when `save_freq=-1` — stdout shows verl's terminal marker `Final validation metrics:`). See `training_monitor`; there is no "training finished" string in verl.
   - **Crash.** Exit code nonzero or slurm state in `{FAILED, NODE_FAIL, BOOT_FAIL, OUT_OF_MEMORY}`. Capture the last 200 lines of stderr to `workspace/logs/crash_tail.md`.
   - **Preempted / Timeout.** Slurm `TIMEOUT`/`PREEMPTED`. Record the last checkpoint step in `workspace/logs/preempted.md` — if there is one, training is resumable.
   - **User cancelled.** `workspace/job/cancel_requested` exists. Issue `scancel <jobid>` (slurm) or `kill <pid>` (local-direct), set status `cancelled`.
7. **Sleep before next poll:** local-direct 30 s / slurm 60 s / ssh-slurm 90 s.

On any terminal condition the poller **writes `workspace/job/job_status.md`** (`status: success | crashed | preempted | cancelled`, final step/epoch/checkpoint path, last loss/reward from `progress.csv`, pointers to anomaly + crash logs) and touches `workspace/job/terminal`. The agent, on its next engagement, reads `job_status.md` and transitions to `summarize`.

## Skills

- skills/training_monitor
- skills/compute_local
- skills/compute_slurm
- skills/compute_ssh_slurm
- skills/builtin-tools
- skills/global

> Of the three `compute_*` skills, **read only the one matching the chosen target** in `workspace/compute/compute_choice.md` for the polling mechanism. The other two are listed for validator coverage.

## Hand-off Points

- **Cancellation request.** This state honours a user-initiated cancel (the `workspace/job/cancel_requested` file — created externally or via `--cancel`). The detached poller observes it each cycle and issues the `scancel`/`kill`; no proactive agent pause. `--no-hitl` is irrelevant here; cancellation is automatic whenever the file appears.

## Next States

### summarize

**Condition:** The `workspace/job/terminal` sentinel exists and `workspace/job/job_status.md` records a terminal status (success | crashed | preempted | cancelled). The agent reaches this on the engagement that observes the sentinel (woken by it, by an escalation, or by a heartbeat check).

**Deliverables:**

- job_status: The terminal status, the final step / epoch / checkpoint path (when applicable), the last training metrics (loss, reward) from `progress.csv`, and pointers to `anomalies.md`, `crash_tail.md`, or `preempted.md` depending on status.
- training_logs: The accumulated `workspace/logs/job_log.md`, `progress.csv`, `anomalies.md` (and crash/preempted siblings as relevant) — the raw evidence summarize consumes.
