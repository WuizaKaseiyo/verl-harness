# VerL-Harness Experiment Progress

_Last updated: 2026-06-17_

Companion to `experiment_design.md`. Records what has actually been executed,
what is in flight, and what remains.

---

## Executive summary

| Block | Status | Notes |
|---|---|---|
| RQ2 — Failure recovery | ✅ **Complete** | 5 controlled injections all PASS + 2 free real-world incidents |
| RQ3 — Monitoring cost | ✅ **Complete** | Event Runtime 32.7× cheaper than 30 s polling, no missed events |
| Exp 5 — Two-axis dispatch | ✅ **Complete** | 21 / 21 cells PASS + 3 / 3 negative controls correctly rejected |
| RQ1 — End-to-end workflows | 🟡 **2 / 3 datapoints done** | N11 ✓, N9 ✓, N3 in queue |
| RQ4 — Human effort | 🔴 **Not started** | Needs human subjects |

**3 out of 5 paper blocks can be drafted today**; only RQ1 (waiting on N3) and
RQ4 (waiting on participants) are not yet complete.

---

## ✅ Completed experiments

### Exp 1 — End-to-end workflow completion (2 / 3)

#### N11 — GSM8K + GRPO on local-slurm (Qwen2.5-3B-Instruct, 4 × H100)

| | |
|---|---|
| jobid | 15583 (15531 first attempt crashed; see RQ2 bonus row) |
| State | COMPLETED, exit 0:0 |
| Wallclock | 4 h 14 min 38 s |
| Training | 210 / 300 steps (hit recipe `TOTAL_EPOCHS=15` cap before `total_training_steps`) |
| Checkpoints | `global_step_{50, 100, 150, 200}` under `output_dir/` (canonical layout) |
| Final reward / mean | **0.961** (GSM8K essentially solved) |
| Workspace | `runs/n11-gsm8k-3b-grpo-2026-06-09/workspace/` |
| Poller verdict | success, last_checkpoint = global_step_200 |
| Anomalies | none |

#### N9 — BoolQ + SFT with auto-preprocess (Qwen2.5-3B-Instruct, 4 × H100)

| | |
|---|---|
| jobid | 15669 (silently backfilled and finished while harness believed it was still PENDING) |
| State | COMPLETED, exit 0:0 |
| Wallclock | 5 min 9 s |
| Training | 2 / 2 epochs completed (`Epoch 2/2: 97% [35/36]` then exit) |
| Final checkpoint | `global_step_72/` (36 GB FSDP world_size=4 shards) |
| Workspace | `runs/n9-boolq-sft-2026-06-09/workspace/` |
| Side note | 1-GPU retry (jobid 15700) was launched on the assumption 15669 was still queued; it failed at verl's `resume_mode=auto` because the on-disk ckpt is world_size=4 and the retry was world_size=1. Archived at `archive_1gpu_retry_attempted/` with a diagnosis.md — not a training failure, just a stale-ckpt artifact. Documented for future spec coverage. |

**`generate_preprocess` branch was fully exercised end-to-end**: BoolQ not in
verl's data_preprocess registry → harness authored `preprocess.py` from the
gsm8k.py template → `py_compile` + 100-row smoke PASS → HITL approval (required;
not skippable by `--no-hitl`) → ran preprocess → 9 427 + 3 270 parquet rows on
disk. This is the unknown-HF-dataset branch the experiment design specifically
targeted.

#### N3 — GSM8K + GRPO (pivoted from srun → 2-GPU sbatch)

🟡 **In flight.** jobid 15746 PENDING (Resources). Original plan was
local-direct via `srun --gres=gpu:4` to validate the harness's local-direct
compute branch with real training. The user's account caps interactive srun at
1 GPU, which is too tight for GRPO 3B with vLLM rollout, so the harness
pivoted to 2-GPU sbatch on local-slurm. The local-direct compute branch is now
covered by probe-only evidence (same status as ssh-slurm); this is recorded
honestly in `runs/n3-.../workspace/compute/compute_choice.md`.

Expected wallclock ~8 h once allocated.

---

### Exp 2 — Failure recovery and safe halt (5 / 5 PASS)

All five planned injections completed; two more failure-recovery datapoints
were captured for free from real-world incidents during the campaign.

| ID | Injection | Verdict | Evidence |
|---|---|---|---|
| F5 | 72 B GRPO at cap = 4 H100, util = 0.6 | **PASS** — estimator halted with multi-node / TP / LoRA / quant advise; 0 sbatch attempted; 3 B same-recipe positive control passed | `runs/expF5/result.md` |
| F8 | sbatch with `--partition=does-not-exist-deadbeef` | **PASS** — sbatch exit 1, no jobid, no allocation; harness wrote `launch_failed.md` + remediation (list of valid partitions); 0 GPU·h | `runs/expF8/result.md` |
| F9 | Legacy `output_dir/checkpoints/global_step_200/` alongside canonical `output_dir/global_step_300/` | **PASS** — harness correctly resolved canonical path; legacy ignored | `runs/expF9_1781535759/final_report.md` |
| F10 | Synthetic NCCL `WARN Call to bind failed: Address already in use` (Ray / vLLM startup chatter) | **PASS** — classified `NCCL-warn`, recorded in `anomalies.md`, did NOT escalate | `runs/expF10F11/confusion_matrix.md` |
| F11 | Synthetic `Watchdog caught collective operation timeout` | **PASS** — classified `NCCL-hang`, escalation written | `runs/expF10F11/confusion_matrix.md` |

**Free bonus failure-recovery datapoints**:

| | What happened | What the harness did |
|---|---|---|
| N11 #1 (jobid 15531) | First N11 attempt crashed in 30 s because `python -c 'import verl'` returned `ModuleNotFoundError` — the chess conda env had its verl install removed between 5/28 and 6/15 | watch_poller picked up FAILED sacct state + exit 1 within the 30 s window, wrote `job_status.md=crashed`, recorded `last_checkpoint: none` honestly, no false success. Evidence archived at `runs/n11-.../workspace/crash_log/crash_1/` |
| N9 retry (jobid 15700) | verl's `resume_mode=auto` tried to load `model_world_size_1_rank_0.pt` from a world_size=4 ckpt left over from 15669 | Detected as crashed by the poller; on inspection it was not a training failure but a real verl semantic surface around cross-world-size resume. Documented as a finding the harness specs should call out |

**Confusion matrix (F10 + F11)**

| truth ↓ / predicted → | hang | none | warn |
|---|---|---|---|
| hang | 2 | 0 | 0 |
| warn | 0 | 0 | 2 |

- hang detection rate: 100 % (2 / 2)
- warn detection rate: 100 % (2 / 2)
- false-escalation rate on warn: 0 / 2

---

### Exp 3 — Event-driven monitoring cost (R1 – R6 × 4 runtimes)

Replayed the 6 scenarios through 4 monitoring runtimes. Real corpora wherever
possible:

- R1 (normal progress): N11 step 50 – 75 slice, 63 lines real
- R2 (checkpoint produced): N11 around save_freq = 50, 91 lines real
- R3 (launch failure): N11 #1 ModuleNotFoundError crash, 128 lines real
- R4 (NCCL benign warn): synthetic F10 corpus
- R5 (true NCCL hang): synthetic F11 corpus
- R6 (training completed): N11 terminal window, 30 lines real

Cost model: Claude Sonnet 4.6 rates ($3 / Mtok in, $15 / Mtok out), 10 K
prompt + 150 completion per LLM wakeup ⇒ $0.0323 / call.

**Aggregate across 6 scenarios:**

| Runtime | Total Calls | Cost (USD) | Detected / Actionable | False Wakeups | Cost ratio vs Event |
|---|---|---|---|---|---|
| LLM Polling 30 s | **131** | **$4.22** | 4 / 4 | 127 | 32.7× |
| LLM Polling 60 s | 65 | $2.10 | **3 / 4** (misses R3) | 62 | 16.2× |
| LLM Polling 90 s | 42 | $1.36 | **3 / 4** (misses R3) | 39 | 10.5× |
| **Event Runtime** | **4** | **$0.13** | **4 / 4** | **0** | 1.0× |

Headline: **Event Runtime is 32.7× cheaper than 30 s polling, with the same
detection completeness and zero false wakeups.** Polling at ≥ 60 s would have
missed N11 #1's 30-second crash entirely.

Artifacts: `runs/expExp3/results.md`, `per_cell.csv`, `slices/`.

---

### Exp 5 — Two-axis algorithm dispatch coverage (3 × 7)

| adv_estimator ↓ / loss_mode → | vanilla | gspo | cispo | geo_mean | sapo | dppo_tv | dppo_kl |
|---|---|---|---|---|---|---|---|
| **grpo** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **grpo_passk** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **grpo_vectorized** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- **21 / 21 cells PASS** (each cell: enum membership ✓, registry membership ✓,
  Hydra compose ✓, round-trip ✓)
- `sapo`'s extra knobs verified at `actor_rollout_ref.actor.tau_{pos,neg}`
  (top-level `actor.*`, not under `policy_loss.*`)
- **3 / 3 negative controls** (`nonsense_xyz` × `vanilla`, `grpo` × `junk_loss_mode`,
  both invalid) correctly rejected at the static-check level

Artifacts: `runs/expExp5/coverage.md`, `matrix.csv`, `negatives.csv`,
`sweep_matrix.py`.

---

## 🟡 In flight

| Experiment | jobid / state | Notes |
|---|---|---|
| **N3** GSM8K + GRPO 2 GPU sbatch | 15746 PENDING (Resources) | Pivoted from local-direct/srun. watch_poller PID 2257049 alive. Expected wallclock ~8 h once allocated. Last datapoint needed to close RQ1 (with a caveat about the local-direct branch). |

## ⬜ Not started

| Experiment | Blocker |
|---|---|
| **H1** Human effort: GRPO on GSM8K | Needs subjects for manual / naive-LLM baselines (harness side is already done — reuse N11 run) |
| **H4** Human effort: BoolQ SFT with generated preprocessing | Same — harness side is N9, baselines need humans |

---

## Per-RQ paper readiness

| Block | Drafting status |
|---|---|
| **RQ2 — Failure recovery** | ✅ All evidence on disk. Can write the section today. |
| **RQ3 — Monitoring cost** | ✅ Headline number + per-cell table ready. Can write today. |
| **Exp 5 — Two-axis dispatch coverage** | ✅ 3 × 7 table + per-cell + negative controls ready. Can write today. |
| **RQ1 — End-to-end workflows** | 🟡 Awaiting N3. Two datapoints (N11 + N9) already strong; N3 adds one more datapoint at a different parallel scale and a probe-only nod to local-direct. |
| **RQ4 — Human effort** | 🔴 Needs subjects. Harness side already covered by N11 / N9 — manual / naive-LLM baselines need to be collected. |

---

## Artifact index

```
runs/
├── expF5/                            — RQ2 — GPU budget halt-and-advise
├── expF8/                            — RQ2 — launch-failure short-circuit
├── expF9_1781535759/                 — RQ2 — ckpt path correction
├── expF10F11/                        — RQ2 — NCCL warn vs hang
├── expExp3/                          — RQ3 — monitoring cost simulation
├── expExp5/                          — Exp 5 — two-axis dispatch matrix
├── n11-gsm8k-3b-grpo-2026-06-09/     — RQ1 — N11 (success) + N11 #1 (free RQ2)
├── n9-boolq-sft-2026-06-09/          — RQ1 — N9 (success, full generate_preprocess)
└── n3-gsm8k-3b-grpo-local-2026-06-09/  — RQ1 — N3 (in flight)
```
