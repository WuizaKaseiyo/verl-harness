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

1. **Job exit signal** — `kill -0 <pid>` returns nonzero (local-direct) or `sacct` reports `COMPLETED` (slurm).
2. **Checkpoint OR last-step parity** — at least one of:
   - `<output_dir>/checkpoints/global_step_<N>/` exists (verl saves checkpoints at this path; the convention is hardcoded in `verl/trainer/ppo/ray_trainer.py:940` as `f"global_step_{self.global_steps}"`). Note: if the recipe sets `trainer.save_freq=-1` (common for smoke / no-save runs), there will be **no** checkpoint directory even on a successful run — fall through to the next bullet.
   - The trainer's stdout shows the final-step marker (next bullet) — sufficient on its own when `save_freq=-1`.
3. **Terminal marker in stdout** — verl prints exactly one line at the end of `fit()`:
   ```
   'Final validation metrics: {...}'
   ```
   (the source is `verl/trainer/ppo/ray_trainer.py:1756`: `pprint(f"Final validation metrics: {last_val_metrics}")`). Match with:
   ```python
   TERMINAL_SUCCESS = re.compile(r"Final validation metrics:")
   ```
   No "training finished" / "training complete" string exists in verl source — do not look for those.

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

**Caveat:** the specific phrasing of NaN messages depends on whether they originate from PyTorch's `_check_finite_grad` paths, vLLM, or the recipe's own asserts. The patterns below mix verl-typical and torch-generic shapes; verify on the user's checkout that at least one matches:

- `loss is nan`                                   (typical custom logger string)
- `loss=inf`
- `gradient is nan`
- `RuntimeError: Function .* returned nan values` (torch-side)
- `assert.*not torch.isnan`                       (verl asserts)
- A metric-dict line where `'actor/pg_loss': nan` (also `'reward/mean': nan`, `'critic/value_loss': nan`)

If none of the literal-string patterns match on a real verl run, fall back to scanning the namespaced metric stream (from the progress parser above) for any value parsing as `nan` / `inf` — that catches the failure regardless of how the trainer chose to surface it.

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

verl's trainer does NOT emit a single-line `step:N, epoch:N, train_loss:X, ...` log format. Progress flows through the `Tracking` logger at `verl/trainer/ppo/ray_trainer.py:1722` (`logger.log(data=metrics, step=self.global_steps)`) and, when the user configures `trainer.logger=["console"]`, lands on stdout as a **namespaced metric dict** with keys like:

```
training/global_step
training/epoch
actor/grad_norm
actor/lr
actor/pg_loss
actor/entropy
critic/value_loss          (PPO only)
response/length/mean
response/length/std
response/length/max
reward/mean
reward/std
training/timing/step       (seconds)
throughput/tokens_per_sec
```

The exact stdout shape varies by console backend (`pprint`-style multi-line, or one-line repr). Parse defensively — accept both. The recommended two-pass parser:

```python
# Pass 1: detect that a metrics block is starting (the trainer pprints the dict)
METRIC_DICT_START = re.compile(r"'training/global_step'\s*:\s*(?P<step>\d+)")

# Pass 2: harvest individual key:value pairs within the block.
#   Matches both `'key': 0.123`, `'key': 1.0e-6`, and `'key': 50` shapes.
METRIC_KV = re.compile(
    r"'(?P<key>[\w/]+)'\s*:\s*(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
```

For each block where `METRIC_DICT_START` matched, scan all `METRIC_KV` hits and append one row to `workspace/logs/progress.csv`. The CSV **columns are dynamic** (the union of keys observed so far) rather than the legacy fixed `step,epoch,loss,reward,...` schema. Suggested baseline column ordering: `wallclock, training/global_step, training/epoch, actor/lr, actor/pg_loss, actor/entropy, actor/grad_norm, critic/value_loss, reward/mean, reward/std, response/length/mean, training/timing/step` — but add new columns as new keys appear.

If the user's `trainer.logger` includes a wandb / tensorboard backend rather than `console`, the console may emit a smaller / different subset; do not assume any key is always present.

### Fallback: tqdm progress bar

verl uses `tqdm` (`ray_trainer.py:1364`: `progress_bar = tqdm(total=self.total_training_steps, ...)`) as the visible step counter. Each `progress_bar.update(1)` writes a line like:
```
Training Progress:  50%|##########          | 50/100 [01:00<01:00,  1.20it/s]
```
to stderr (sometimes interleaved into stdout under slurm). Match with:
```python
TQDM_LINE = re.compile(r"Training Progress:\s*\d+%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)")
```
This is a useful liveness signal even when the metric dict isn't yet on stdout (e.g., the first few rollout-bound steps before the first `logger.log` call).

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
