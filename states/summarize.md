# summarize

## Description

Turn the raw monitoring artefacts into a one-page human-readable run report. Branch on the terminal status from `monitor_training`. Whatever the outcome, the summary is honest — a crashed run gets a crashed-run report, not a softened story.

Apply the `training_monitor` skill (`skills/training_monitor`) for log-parsing conventions and the `global` skill for the honesty principle.

Concretely:

1. **Read** `workspace/intake/training_intent.md`, `workspace/recipe/recipe.md`, `workspace/dataset/dataset.md`, `workspace/compute/compute_choice.md`, `workspace/job/job_info.md`, `workspace/job/job_status.md`, and the contents of `workspace/logs/`.
2. **Branch on `job_status.status`:**
   - **success:** assemble the success report (next section).
   - **crashed:** the crash report (after).
   - **preempted:** the preemption / resume-instructions report.
   - **cancelled:** the cancellation report.
3. **Compute the training-curve summary.** Read `workspace/logs/progress.csv`. Compute first-step / last-step / mid-training values for loss, reward, throughput. If wandb is configured and the run url was logged, record it.
4. **Resolve the final and best checkpoints.**
   - **Final** — list the highest-step checkpoint directory directly under `<output_dir>/` (verl writes `global_step_<N>/` there, no `checkpoints/` subdir) and its disk size. For preempted: the last completed checkpoint (the basis for a resume). For crashed: the last checkpoint, if any, even if it's an early one.
   - **Best** — apply `training_monitor`'s `pick_best_checkpoint(...)` helper using the trainer-family's canonical val metric (see the skill's "Best checkpoint selection" table). Record `best_step`, `best_metric` (key + value), and `best_path`. When the trainer never logged the canonical val metric (e.g., `test_freq=0` or recipe didn't define val files), record `best_checkpoint: none — no val metric logged`. When the best step's ckpt isn't on disk (often because `save_freq` missed it), record `best_checkpoint: none — best step <N> not saved`.
5. **Write `workspace/summary/summary.md`** following the per-status template (below).

### Success report

- Topline: "Training completed: \<algorithm\> on \<dataset\> with \<model\>. Final checkpoint at \<path\>. Best checkpoint at \<best_path\> (\<best_metric\>)."
- Training-curve numbers: initial / mid / final train loss; initial / mid / final reward; mean step throughput.
- **Final checkpoint:** path + size.
- **Best checkpoint:** path + size + `(<best_metric_name>=<value>)` — the path the user should hand to `main_eval` / `main_generation_server`.
- wandb run url (if configured).
- Compute used: target, nodes, GPUs, wall-clock.
- Anomalies recorded mid-run (if any), with the note that the run completed despite them.

### Crash report

- Topline: "Training crashed at step \<N\>, epoch \<E\>. Last checkpoint: \<path or none\>."
- The matched anomaly pattern (OOM / NaN / NCCL / vLLM / generic).
- The last 50 lines of stderr (excerpt of `crash_tail.md`).
- Training-curve numbers up to the crash step.
- Compute used so far.
- **A specific remediation suggestion**: OOM → reduce `train_batch_size` / `ppo_max_token_len_per_gpu`; NaN → lower LR or clip more aggressively; NCCL hang → check inter-node network; vLLM crash → check `rollout.gpu_memory_utilization`. Skill `training_monitor` has the full mapping.

### Preempted / timeout report

- Topline: "Training was \<preempted | timed out\> at step \<N\>. Last checkpoint: \<path\>. Resumable."
- The resume command — verl supports `trainer.resume_mode=auto` / `trainer.resume_from_path=<ckpt>`. Construct the exact resume command using the existing recipe with the resume args appended.
- Training-curve numbers up to preemption.

### Cancelled report

- Topline: "Training cancelled by user at step \<N\>. Last checkpoint: \<path or none\>."
- The user-supplied reason if recorded; otherwise "no reason recorded".
- Compute used.

## Skills

- skills/training_monitor
- skills/builtin-tools
- skills/global

## Hand-off Points

- None. Summarization is a deterministic report over recorded run artefacts and introduces no new decision.

## Next States

### reflect

**Condition:** `workspace/summary/summary.md` is written AND `workspace/intake/training_intent.md` contains a `refine` block (the user opted into closed-loop refinement at intake). Fires regardless of terminal status — a crashed iteration is diagnosable too; `reflect` decides whether the loop continues or stops.

**Deliverables:**

- summary: `workspace/summary/summary.md` — the status-branched run report for the finished iteration, the evidence `reflect` diagnoses against.

### finalize

**Condition:** `workspace/summary/summary.md` is written and no `refine` block exists in `workspace/intake/training_intent.md` (the default, single-iteration path).

**Deliverables:**

- summary: `workspace/summary/summary.md` — the status-branched run report with topline, training curve numbers, checkpoint path, compute cost, anomalies, and (for non-success) a specific remediation or resume command.
