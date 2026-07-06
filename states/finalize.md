# finalize

## Description

**Terminal state.** Produce exactly one `workspace/final_report.md` from the terminal deliverable that caused this state to be entered. This state does not infer success from the furthest completed stage: it reads the incoming artefact, preserves its status, and reports whether training or post-training work actually ran.

Concretely:

1. **Identify the terminal input.** Use the deliverable on the transition into this state and verify that its canonical file from `## Terminal Inputs` exists. Do not select an older artefact merely because it is also present in a resumed workspace. If the transition context is unavailable, choose the newest canonical terminal-input file by modification time and record that fallback explicitly.
2. **Read shared context when present.** Read `workspace/intake/training_intent.md` plus any recipe, dataset, compute, environment, job, log, or checkpoint records relevant to the selected terminal input. Missing upstream files are expected for early halts and must not be invented.
3. **Classify the outcome:**
   - `completed` — `summary` with job status `success`, `generate_report`, `eval_report`, or `reflect_report` with stop reason `target_met`.
   - `incomplete` — `summary` with job status `crashed`, `preempted`, or `cancelled`, or `reflect_report` with any other stop reason.
   - `failed` — any `*_failed`, `algorithm_unsupported`, or `gpu_budget_exceeded` input.
4. **Write `workspace/final_report.md`** with:
   - outcome and terminal stage;
   - one-line headline copied from or strictly supported by the terminal input;
   - normalized intent fields that actually exist;
   - what ran and what did not run;
   - verbatim failure evidence or measured results;
   - remediation/resume command when the input provides one;
   - pointers to the terminal input and every existing relevant artefact.
5. **Update run metadata.** Set `runs/<run_id>/meta.json.status` to the same outcome vocabulary (`completed`, `incomplete`, or `failed`) and record `terminal_input`, `terminal_stage`, and `finished_at`. Preserve unrelated metadata fields.
6. **Tell the user** the headline, outcome, and path to `workspace/final_report.md`.

### Report rules by terminal family

- **Training summary (`summary`).** Preserve the exact job status. For success, report final and best checkpoints only when they exist on disk. For crash/preemption/cancellation, hoist the remediation or resume command and state that training did not complete.
- **Early training halt (`algorithm_unsupported`, `gpu_budget_exceeded`, `env_failed`, `sanity_failed`, `launch_failed`).** State plainly that training never started, except when `sanity_failed` spent a bounded sanity probe; in that case say that full training never started. Include the exact failed gate and unblock action.
- **Generation (`generate_report` / `generate_failed`).** Report output parquet path and row count only when verified. Do not imply that training occurred in this run.
- **Evaluation (`eval_report` / `eval_failed`).** Report scores verbatim by `data_source` on success. On failure, report no aggregate score unless the terminal input explicitly labels it partial.
- **Refinement loop (`reflect_report`).** Report the per-iteration history verbatim (delta, diagnosis, metric), the stop reason, and the best iteration's checkpoint path. Never present `budget_exhausted` or `no_further_delta` as success.

## Terminal Inputs

Every transition into `finalize` must deliver exactly one of these names at the canonical path. This section is machine-checked by `tools/validate_harness.py`.

- `summary` — `workspace/summary/summary.md` — terminal training report after monitoring.
- `algorithm_unsupported` — `workspace/algorithm/algorithm_unsupported.md` — requested algorithm has no accepted trainer binding.
- `gpu_budget_exceeded` — `workspace/compute/gpu_budget_exceeded.md` — minimum viable allocation exceeds the GPU cap.
- `env_failed` — `workspace/env/env_failed.md` — environment provisioning could not complete.
- `sanity_failed` — `workspace/sanity/sanity_failed.md` — model/data/reward sanity probe failed.
- `launch_failed` — `workspace/job/launch_failed.md` — the full job never started.
- `generate_report` — `workspace/generate/generate_report.md` — generation completed and no chained eval was requested.
- `generate_failed` — `workspace/generate/generate_failed.md` — generation crashed or timed out.
- `eval_report` — `workspace/eval/eval_report.md` — standalone or chained evaluation completed.
- `eval_failed` — `workspace/eval/eval_failed.md` — standalone or chained evaluation failed.
- `reflect_report` — `workspace/reflect/reflect_report.md` — closed-loop refinement ended; per-iteration history and stop reason.

## Skills

- skills/builtin-tools
- skills/global

## Hand-off Points

- None. `finalize` reports the already-determined terminal outcome and does not introduce a new approval gate.

<!-- Terminal state: no `## Next States`. -->
