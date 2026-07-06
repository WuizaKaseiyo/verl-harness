# reflect

## Description

Closed-loop refinement. Entered from `summarize` when the user opted into
refinement at intake. Diagnose the finished iteration against the recorded
intent, then either propose one bounded configuration delta and loop back to
`configure_algorithm`, or stop and hand the loop's outcome to `finalize`.
This state refines the training configuration only — it never edits states,
skills, or tools (the harness itself), and it never changes the model,
dataset, algorithm, or compute target.

Apply the `training_monitor` skill (`skills/training_monitor`) for metric
parsing, anomaly patterns, and the best-checkpoint helper, and the `global`
skill for the honesty principle: every diagnosis claim must cite recorded
evidence verbatim, and a loop that failed to improve is reported as such.

Concretely:

1. **Load loop state.** Read `workspace/reflect/loop_state.json`; if absent,
   initialize `{"iteration": 1, "max_iterations": <bound>, "history": []}`.
   The bound is `refine.max_iterations` from
   `workspace/intake/training_intent.md`, capped by the `**Loop:**` bound on
   the `configure_algorithm` transition below — the edge bound is the hard
   cap regardless of what the intent file requests.
2. **Read the evidence.** `workspace/intake/training_intent.md` (the target:
   `refine.target_metric`, `refine.target_value`),
   `workspace/summary/summary.md`, `workspace/job/job_status.md`,
   `workspace/logs/progress.csv`, `workspace/logs/anomalies.md`, and
   `workspace/algorithm/algorithm_config.md` (the knob table of the
   just-finished iteration).
3. **Diagnose.** Classify the finished iteration, citing file + verbatim
   numbers for every claim:
   - `target_met` — the target metric reached the requested value.
   - `improving` — the metric moved toward the target but did not reach it.
   - `plateaued` — no meaningful movement over the last third of training.
   - `degraded_or_unstable` — NaN, entropy collapse, or reward regression.
   - `crashed` — `job_status.status` is crashed; reuse the remediation
     mapping from `skills/training_monitor` (the same table `summarize`
     uses for crash reports).
4. **Decide.** Stop (transition to `finalize`) when the diagnosis is
   `target_met`, the iteration bound is exhausted, or no plausible bounded
   delta exists. Otherwise propose exactly one delta for the next iteration.
5. **Propose the delta (loop branch only).** Write
   `workspace/reflect/refinement_plan.md`: the diagnosis with its evidence
   quotes, the proposed change as a table (knob, old value, new value, why),
   and the expected effect on the target metric. Deltas are bounded: only
   knobs already surfaced in `algorithm_config.md`'s knob table or
   `recipe.md`'s `## Key hyperparameters` may change, at most 3 knobs per
   iteration.
6. **Snapshot the finished iteration.** Copy `summary.md`,
   `algorithm_config.md`, and `job_status.md` to
   `workspace/reflect/iter_<N>/` (small report files only — never
   checkpoints or raw logs; checkpoint paths are recorded, not copied), and
   append the iteration record (iteration, diagnosis, delta or `none`,
   metric value) to `loop_state.json.history`. Increment `iteration` when
   looping.
7. **Hand off for approval (loop branch).** Present the diagnosis and the
   delta table; on approval, transition to `configure_algorithm`, which
   treats the approved plan as user-provided knob overrides for the next
   iteration. On decline, stop.
8. **Write the loop report (stop branch).**
   `workspace/reflect/reflect_report.md`: one line per iteration (delta,
   diagnosis, metric value), the stop reason (`target_met` /
   `budget_exhausted` / `no_further_delta` / `user_stopped`), and the best
   iteration's checkpoint path taken from its snapshot.

## Skills

- skills/training_monitor          # metric parsing, anomaly patterns, best-checkpoint helper
- skills/builtin-tools
- skills/global

## Hand-off Points

- **Approve refinement delta.** Step 7, before looping back. Skipped with
  `--no-hitl`; the always-on gates downstream (sanity approval, cost gate)
  still pause each iteration's relaunch.

## Next States

### configure_algorithm

**Condition:** The diagnosis proposes a bounded delta,
`workspace/reflect/loop_state.json` records `iteration < max_iterations`,
and the user approved the plan (or `--no-hitl`). `configure_algorithm`
treats `workspace/reflect/refinement_plan.md` as user-provided knob
overrides for the next iteration.

**Loop:** max_iterations: 3

**Deliverables:**

- refinement_plan: `workspace/reflect/refinement_plan.md` — the evidence-linked diagnosis and the bounded knob delta (at most 3 knobs, existing knob table only) for the next iteration.

### finalize

**Condition:** The loop stops: the target was met, the iteration bound is
exhausted, no plausible bounded delta remains, or the user declined another
iteration. The report states which, honestly.

**Deliverables:**

- reflect_report: `workspace/reflect/reflect_report.md` — the per-iteration history (delta, diagnosis, metric), the stop reason, and the best iteration's checkpoint path.
