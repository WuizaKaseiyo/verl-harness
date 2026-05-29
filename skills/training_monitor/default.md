training_monitor skill — what to watch for in a running verl training job and how to call its status.

## Two cadences — the poller's, and the agent's

`monitor_training` runs **arm-and-detach**: a cheap non-LLM **poller** owns the tight loop; the **agent** re-engages only on events. They run on different clocks — keep them separate.

### Poller cadence (the detached script — fine to be frequent)

| Target        | Cadence | Rationale                                                 |
|---------------|---------|-----------------------------------------------------------|
| local-direct  | 30 s    | Local kernel + filesystem — cheap to poll                 |
| local-slurm   | 60 s    | `squeue` hits the slurm controller — don't over-poll      |
| ssh-slurm     | 90 s    | ssh handshake + `squeue` — coarsest cadence               |

These are minimums *for a script*; `kill -0` / `squeue` / a byte-offset tail cost ~nothing and annoy no one at this rate.

### Agent engagement cadence (event-driven — never the 30/60/90 s above)

The LLM agent must not wake on the poller's clock. It re-engages only when:

- the `workspace/job/terminal` sentinel appears (poller or slurm dependency job wrote it), **or**
- the poller promotes a finding into `workspace/logs/escalation.md` (OOM / NaN / NCCL hang / divergence / stalled-stdout hang), **or**
- a **heartbeat** elapses — default **15–30 min**, derived as a few × observed seconds-per-step — at which the agent confirms the poller is alive and the log byte-offset is still advancing.

Rationale: a days-long run polled by the agent every 90 s = hundreds of context-reloads and compactions for a state that barely changes. Past the 5-minute prompt-cache window a longer heartbeat is strictly cheaper (one cache miss buys a long wait); so favour 15–30 min, not a round 5 min.

## Terminal conditions

The monitor loop exits when one of these is true. Each maps to a status the `summarize` and `finalize` states branch on.

### Success

All three must hold:

1. **Job exit signal** — `kill -0 <pid>` returns nonzero (local-direct) or `sacct` reports `COMPLETED` (slurm).
2. **Checkpoint OR last-step parity** — at least one of:
   - `<output_dir>/global_step_<N>/` exists (verl saves checkpoints directly under `trainer.default_local_dir` — i.e. the run's `output_dir` — with **no** intervening `checkpoints/` subdir; verified at `verl/trainer/ppo/ray_trainer.py:939-940`, `local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")`). Note: if the recipe sets `trainer.save_freq=-1` (common for smoke / no-save runs), there will be **no** checkpoint directory even on a successful run — fall through to the next bullet.
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

**Escalate only on genuine collective failures / hangs:**

- `Watchdog caught collective operation timeout`
- `Timeout waiting for ack`
- `NCCL.*unhandled (system|cuda) error`, `NCCL.*remote process exited`
- The process becomes silent for > 10 minutes despite the slurm job still running (stalled-stdout hang).

**Do NOT escalate on bare `NCCL WARN`.** During Ray/vLLM bring-up verl routinely logs
`NCCL WARN Call to bind failed: Address already in use` (a RAS-daemon port retry) — this is
**benign**, appears on healthy runs, and must not trip a divergence/crash alarm. Record such
`NCCL WARN` lines in `anomalies.md` for visibility, but classify them informational, not fatal.
(Verified: jobs 8579 and 8633 both logged this bind warning during startup and ran fine.)

Fix suggestion (for the real hangs): check inter-node network (`ibstat`, `ucx_info`), confirm NCCL env vars (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`).

### vLLM rollout crash

- `vllm.engine.*Error`
- `vllm.engine.*Aborted`
- `actor_rollout_ref.rollout` config-related errors

Fix suggestion: lower `rollout.gpu_memory_utilization`, reduce `tensor_model_parallel_size`, switch to sglang.

### Per-algorithm progress signals

Beyond the universal stability thresholds below, each algorithm family has progress signals that *only* make sense for it. The monitor emits anomalies / liveness signals from this set in addition to the universal ones, picking the right set from `workspace/algorithm/algorithm_config.md`.

| Algorithm | Signal | Healthy band | Anomaly when |
|---|---|---|---|
| PPO | `critic/value_loss` | Declining trend over first 200 steps | Flat or growing after step 100 — critic not learning |
| PPO | `critic/explained_variance` | `> 0.0`, ideally `> 0.5` | `≤ 0.0` for 50 consecutive steps |
| PPO + GRPO family | `actor/pg_loss` | Negative-and-trending-toward-zero | Stays positive — policy is anti-learning |
| GRPO (any) | Group-reward std (in `actor/<...>/group_std` if logged; else inferred) | `> 0.05 × mean reward magnitude` | `< 0.01 × mean` for 20 steps — group variance collapse, advantages are noise |
| GRPO-passk | `val/pass_at_k` | Monotone non-decreasing | Drop > 5% from running max |
| GDPO | per-component `gdpo/<key>/mean` | Each balanced w.r.t. its weight | One component dominates the sum by > 5× |
| SFT | `val/loss` | Monotone decreasing for first epoch | Increasing after step 100 — over-fit or LR too high |
| Distillation (online) | `algorithm.distill_loss` | Decreasing | Saturates at 0 immediately → student already mimics teacher |
| Distillation (offline) | `train/distill_loss` ≈ `train/ce_loss` (when `ce_loss_weight > 0`) | Both decreasing | Distill loss decreasing while CE stays flat — student copies teacher's mistakes |

### RL-stability thresholds (PPO-family — universal)

These are not crashes — they are *signals* that the policy is drifting. Each fires as an anomaly when the parsed metric crosses the threshold for 2+ consecutive logged steps:

| Metric (canonical key from progress.csv) | Threshold | Inferred problem |
|---|---|---|
| `actor/kl` (or `training/kl`, depending on logger) | `> 0.05` | KL drift; policy diverging from reference. Suggest lowering `actor.kl_loss_coef` floor or raising `kl_ctrl.target_kl`. |
| `actor/entropy` (lower bound) | `< 0.5` | Policy collapsing toward greedy. Suggest raising `actor.entropy_coeff` or lowering `rollout.temperature`. |
| `actor/entropy` (upper bound) | rises above `2 × median(first 20 logged steps)` **and** still trending up over the last 50 steps | **Entropy explosion** — policy drifting toward random/uniform, the opposite failure from collapse. Suggest lowering `actor.entropy_coeff` (or setting it to 0), lowering `rollout.temperature`, or raising the KL anchor (`actor.kl_loss_coef`). The absolute value is vocab-dependent, so the signal is the *trend ratio*, not a fixed number. |
| `response/length/mean` | `≥ 0.95 × data.max_response_length` | Truncation; the trainer can't see the end of responses, structural signal lost. Suggest raising `data.max_response_length` (and `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` to match). |
| `response/length/std` | `< 5` (after step 20) | Mode collapse — all responses the same length. Often co-occurs with low entropy. |
| Canonical **val** metric (`val/reward/mean` for RL; `val/loss` for SFT) | drops `> 20%` below its running best (for higher-is-better) — or rises `> 20%` above its running best (for `val/loss`) — across `2+` consecutive validations | **Validation regression** — the policy is getting *worse* on held-out data even though training "runs". The best checkpoint is an earlier one, not the latest. Suggest stopping early and either resuming from the best step or revisiting `lr` / `entropy_coeff` / KL anchor. |

**Divergence (joint signal — highest confidence).** When `actor/entropy` is climbing (upper-bound rule above) **and** the canonical val metric is regressing **at the same time**, treat it as policy divergence, not a transient wobble — record a single `divergence` anomaly that cites both series. This is exactly the failure observed in the C2 1000-step run (entropy ~0.4 → ~10 while `val/reward/mean` fell 0.10 → 0.05) and the "policy collapse" the recipe comments warn about. A diverging run will not recover on its own; surface it prominently in `anomalies.md` and `job_status.md` so the human can decide to stop. (The monitor only *surfaces* — it does not auto-`scancel`; auto-intervention is a separate, default-off policy.)

These thresholds are conservative defaults. The user can override per intent (`monitor.kl_threshold`, `monitor.entropy_explosion_ratio`, `monitor.val_regression_pct`, etc.); document any override in `workspace/intake/training_intent.md`.

Frontend mirroring (dashboard): when the progress chart panel exists, draw a faint horizontal threshold line at each value so the user can see the safety zone visually. (Front-end task — separate change.)

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

## Best checkpoint selection

`summarize` and `finalize` need the **best** checkpoint, not just the last. The "best" is defined by the canonical validation metric for the trainer:

| Trainer family | Canonical val metric (lower is better unless noted) | Source-of-truth in this verl |
|---|---|---|
| SFT | `val/loss` | `sft_trainer.py` |
| PPO (`adv_estimator=gae`) | `val/reward/mean` (higher is better) — fall back to `val/score/mean` | `main_ppo` + `algorithm.adv_estimator=gae` |
| GRPO family (`grpo`, `grpo_passk`, `grpo_vectorized`) | `val/reward/mean` (higher is better); for `grpo_passk` prefer `val/pass_at_k` if logged | `main_ppo` + `adv_estimator=grpo*` |
| Other PPO-family RL (rloo, remax, gpg, reinforce_plus_plus, gdpo, gmpo, dppo, cispo, sapo, mtp, otb, opo) | `val/reward/mean` (higher is better) | `main_ppo` + `adv_estimator=<name>` |
| GDPO (Group reward-Decoupled) | `val/reward/mean` for the summed reward; ALSO surface per-component `val/gdpo/<key>/mean` if logged | `main_ppo` + `adv_estimator=gdpo` + `gdpo_reward_keys` |
| DPO (classic) | n/a — not first-class in this verl; if external trainer used, expects `val/reward_gap` (higher is better) | external (not in this verl) |
| RM training | n/a — not first-class in this verl; if external trainer used, expects `val/accuracy` (higher is better) | external |
| On-policy distillation | `val/distill_loss` (lower is better); for mixed distill+CE also surface `val/ce_loss` | `verl/trainer/distillation/` |

A working helper lives at **`skills/training_monitor/templates/pick_best_checkpoint.py`** — `summarize` imports and calls it:

```
from pick_best_checkpoint import pick_best_checkpoint
result = pick_best_checkpoint(
    progress_csv="workspace/logs/progress.csv",
    output_dir="<output_dir from job_info.md>",
    trainer_family="grpo",      # from algorithm_config.md
)
# result is (best_step, best_value, best_path) or None.
```

The template records the `CANONICAL_VAL_METRIC` table for every trainer family covered by `algo_*` skills. Extend the table when adding new families.

`job_status.md` records the result under `## Best checkpoint`. `summarize` echoes it to the run report.

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
- last_checkpoint: <output_dir>/global_step_1500/
- final_loss: 0.214
- final_reward: 0.872

## Best checkpoint
- best_step: 1200
- best_metric: val/reward/mean = 0.891    # name + value of the trainer-family's canonical val metric
- best_path: <output_dir>/global_step_1200/
- (or `best_checkpoint: none — no val metric logged or no save matched the best step` when applicable)

## Anomalies observed
- 2026-05-21T14:33:12 — "NCCL WARN: ring 0 timed out" (recovered)
- 2026-05-21T14:45:00 — `actor/kl = 0.073 > 0.05 threshold` for steps 200-202 (KL drift)

## Crash tail (only if crashed)
- pointer to workspace/logs/crash_tail.md
- inferred cause: <OOM|NaN|NCCL|vLLM|other>
- suggested fix: <one-line remediation>
```

## Things you must not do

- Do not over-poll. The poller cadences are *minimums*; do not poll faster than them.
- **Do not let the agent busy-loop on the poller cadence.** The agent arms the detached poller and re-engages on events / a 15–30 min heartbeat. Sitting in a 30/60/90 s agent loop across a multi-hour/day run is the specific anti-pattern this state was rewritten to remove.
- Do not invent log lines. The parser reads what the trainer actually wrote. If a metric column is missing on a row, it stays empty in `progress.csv`, not guessed.
- Do not "interpret" crash tails — quote them verbatim, then add the one-line inferred cause. The user will read the tail and judge for themselves.
