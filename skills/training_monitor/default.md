training_monitor skill — what to watch for in a running verl training job and how to call its status.

## Polling cadence (by target)

| Target        | Cadence | Rationale                                                 |
|---------------|---------|-----------------------------------------------------------|
| local-direct  | 30 s    | Local kernel + filesystem — cheap to poll                 |
| local-slurm   | 60 s    | `squeue` hits the slurm controller — don't over-poll      |
| ssh-slurm     | 90 s    | ssh handshake + `squeue` — coarsest cadence               |

## Terminal conditions

The monitor loop exits when one of these is true. Each maps to a status the `summarize` and `finalize` states branch on.

### Success

All three must hold:

1. Job exit signal — `kill -0 <pid>` returns nonzero (local-direct) or `sacct` reports `COMPLETED` (slurm).
2. `<output_dir>/checkpoints/` contains at least one numbered checkpoint directory (verl convention: `global_step_<N>/`).
3. The trainer's stdout contains a "training finished" / "training complete" marker, OR the final logged step equals the recipe's `total_training_steps` (i.e., the trainer ran the whole curriculum).

### Crash

- Exit code nonzero (local-direct) OR slurm state ∈ `{FAILED, NODE_FAIL, BOOT_FAIL, OUT_OF_MEMORY}`.

When crash is detected, capture the last 200 lines of stderr to `workspace/logs/crash_tail.md`.

### Preempted / Timeout

- Slurm state ∈ `{TIMEOUT, PREEMPTED}`. The job ran but the scheduler reclaimed it. Often the run has a usable last checkpoint and can resume.

When preempted, record the highest-numbered checkpoint to `workspace/logs/preempted.md`.

### User cancelled

- A file `workspace/job/cancel_requested` exists (the user touched it or passed `--cancel`).

When cancellation is detected: issue `scancel <jobid>` (slurm) or `kill -INT <pid>` (local-direct); then poll the job to actual termination before declaring `cancelled`.

## Anomaly patterns

Scan each new log chunk for these. Record matches in `workspace/logs/anomalies.md` (timestamp + matched line). The patterns are case-sensitive unless noted.

### OOM family

- `torch.cuda.OutOfMemoryError`
- `CUDA out of memory`
- `cuda runtime error \(2\): out of memory`     (case-insensitive)
- kernel log: `Out of memory: Killed process <pid>` (from `dmesg`)
- vllm: `Engine.*?CUDA out of memory`

OOM is usually fatal; the harness writes a crash anomaly *and* records the suggested fix (reduce `train_batch_size` or `ppo_max_token_len_per_gpu`, reduce `rollout.gpu_memory_utilization`, or shrink the model).

### NaN / Inf

- `loss is nan`
- `loss=inf`
- `gradient is nan`
- `RuntimeError: Function .* returned nan values`

Fix suggestion: lower learning rate, enable gradient clipping (already 1.0 in verl defaults), check the dataset for poisoned rows.

### NCCL hang

- `NCCL WARN`
- `Timeout waiting for ack`
- `Watchdog caught collective operation timeout`
- The process becomes silent for > 10 minutes despite the slurm job still running.

Fix suggestion: check inter-node network (`ibstat`, `ucx_info`), confirm NCCL env vars (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`).

### vLLM rollout crash

- `vllm.engine.*Error`
- `vllm.engine.*Aborted`
- `actor_rollout_ref.rollout` config-related errors

Fix suggestion: lower `rollout.gpu_memory_utilization`, reduce `tensor_model_parallel_size`, switch to sglang.

### Slurm preemption / step error

- `slurmstepd: error: \*\*\* STEP \d+ CANCELLED AT`
- `slurmstepd: error: \*\*\* JOB \d+ ON \S+ CANCELLED AT`

Treat as a normal `preempted` terminal status if `sacct` confirms; otherwise it's a partial-node failure that may still finish.

## Progress extraction

verl's trainer logs follow a fairly stable format. The skill parses lines like:

```
step:50, epoch:0, train_loss:0.234, reward:0.812, response_length:148.7, kl:0.0034, lr:1.0e-06
```

with the regex:

```
^step:(?P<step>\d+),\s*epoch:(?P<epoch>\d+)(?:.*?train_loss:(?P<loss>[-+]?\d*\.?\d+(?:e[-+]?\d+)?))?(?:.*?reward:(?P<reward>[-+]?\d*\.?\d+))?(?:.*?response_length:(?P<resp_len>[\d.]+))?(?:.*?kl:(?P<kl>[-+]?\d*\.?\d+))?
```

(Adjust if the user's verl version logs different keys — peek at `verl/trainer/main_*.py` for the actual log strings.)

Append each parsed row to `workspace/logs/progress.csv` with columns `wallclock,step,epoch,loss,reward,response_length,kl,lr`.

## Throughput estimate

Once `progress.csv` has ≥ 10 rows, compute:

- mean seconds-per-step (from consecutive `wallclock` diffs)
- mean tokens-per-second (from `step × train_batch_size × max_response_length / wallclock_delta`)

Use these to refine the cost estimate the cost-gate originally presented. Record the refinement in `workspace/logs/throughput.md`.

## Status report — `workspace/job/job_status.md`

```markdown
# Job status

## Status
success | crashed | preempted | cancelled

## Job
- target: local-direct | local-slurm | ssh-slurm
- pid / slurm_jobid: …
- output_dir: …

## Terminal facts
- final_step: 1500
- final_epoch: 5
- last_checkpoint: <output_dir>/checkpoints/global_step_1500/
- final_loss: 0.214
- final_reward: 0.872

## Anomalies observed
- 2026-05-21T14:33:12 — "NCCL WARN: ring 0 timed out" (recovered)

## Crash tail (only if crashed)
- pointer to workspace/logs/crash_tail.md
- inferred cause: <OOM|NaN|NCCL|vLLM|other>
- suggested fix: <one-line remediation>
```

## Things you must not do

- Do not over-poll. The cadences above are *minimums*; do not poll faster than them.
- Do not invent log lines. The parser reads what the trainer actually wrote. If a metric column is missing on a row, it stays empty in `progress.csv`, not guessed.
- Do not "interpret" crash tails — quote them verbatim, then add the one-line inferred cause. The user will read the tail and judge for themselves.
