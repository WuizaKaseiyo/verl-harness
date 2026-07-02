Cross-cutting principles for the whole verl-harness run. These bind every state.

## Honesty is the cardinal rule

The output of this harness is a *training run*: real compute spent, real checkpoints written, real logs accumulated. Lying about what happened wastes more compute next time and erodes the user's trust in the harness's reports. Therefore:

- **Never report a checkpoint that does not exist.** If `<output_dir>/global_step_500/` is not on disk, the run did not produce that checkpoint. The summary may not name it. (verl writes `global_step_<N>/` directly under `trainer.default_local_dir`; there is no `checkpoints/` subdir.)
- **Never report a metric the trainer did not log.** Every number in the summary and the final report traces to a row in `workspace/logs/progress.csv`, which itself was parsed from the trainer's stdout. If the trainer never printed a reward line, the summary says "reward not logged", not a guess.
- **A crashed run is reported as crashed.** Soften nothing. The crash report is itself a complete deliverable.
- **An empty / unhelpful slurm queue state is reported as that.** If `squeue` returns empty before the job appears, that is a real-world condition, not a bug to hide.
- **Costs are stated up-front.** Before `launch_training` actually fires the job, the user sees the expected node-hours and confirms. If the harness is wrong about the cost estimate, that's a known limitation, not a reason to skip the disclosure.

## Scope discipline

- All run artefacts live under `runs/<run_id>/workspace/`. Do not write outside it. The training job's own outputs (checkpoints, the trainer's logs, slurm `.out` files) live wherever the user pointed `output_dir`; the harness records *the path*, not a copy.
- The harness is the *runtime*, not the trainer. It does not invent algorithms, write training code, or modify the verl repo's source tree. The verl checkout is read-only from the harness's perspective.
- "Cheap before expensive" — every state runs cheap checks before triggering expensive actions. `provision_env` dry-runs sbatch before submission; `launch_training` shows the cost gate before spending GPU hours; `monitor_training` checks job status frequently enough that crashes are caught early.

## Defaults

Unless the user overrides them in `intake`:

- `compute_pref`: `auto` (let `select_compute` decide).
- `output_dir`: `<VERL_ROOT>/outputs/<run_id>/`.
- `seed`: 1.
- `wandb`: disabled.
- `reward_kind`: `rule` (the safe default; known datasets ship a rule-based reward).
- `cost_gate_threshold_node_hours`: `50`. Below this estimate, `launch_training` may pass silently under `--no-hitl`. At or above, the cost gate fires regardless of HITL mode (see "Always-on hand-off points" below).
- HITL: every documented hand-off point pauses. `--no-hitl` skips most of them and records the escape in the run log, with two exceptions (see below).
- Polling intervals during `monitor_training`: 30 s (local-direct), 60 s (local-slurm), 90 s (ssh-slurm).

### Always-on hand-off points (cannot be skipped by `--no-hitl`)

Four pauses are unconditional because skipping them silently has historically destroyed whole training runs:

1. **`generate_preprocess` script approval.** A bad auto-generated `preprocess.py` will produce a structurally-valid parquet whose semantic content is wrong, and the trainer will spend its full epoch budget learning from garbage. The user must approve the generated script and the HF-field → verl-column rationale before `prepare_data` runs it.
2. **`configure_reward` approval when `reward_kind ∈ {custom, shaped}`.** A wrong custom reward function fires zero or near-zero on every response, training is all-noise, and the model collapses or stays unchanged for the full epoch budget. The user must approve the generated `compute_score.py` (or the user-supplied path) before `sanity_rollout` runs it.
3. **`sanity_rollout` report approval.** The 10-row reward distribution and the row-0 trace are the last cheap chance to catch tokenizer / chat-template / reward-fn mismatches before sbatch. Even a `verdict=green` sanity report must be eyeballed; the heuristic catches obvious failures but not subtle ones (e.g., the model emits the right answer in a slightly off format that the reward fn just barely accepts).
4. **`launch_training` cost gate when estimated node-hours ≥ `cost_gate_threshold_node_hours`** (default 50). Small smoke runs pass silently; expensive jobs require explicit user sign-off even in autonomous mode.

All other hand-off points (intake confirmation, recipe selection, prepared-data confirmation, compute target, provisioning result, key hyperparams override, configure_algorithm confirmation when not halting, configure_reward for `rule`/`model`) are skipped by `--no-hitl`.

## Mode semantics — what `--no-hitl` actually means

`--no-hitl` is **semi-autonomous, not "fully autonomous"**. Six-or-so hand-off points pass silently; **four always-on points still pause** (listed above). A user expecting "kick it off, walk away, check tomorrow" needs to know that the four always-on points will block the run until acknowledged.

### What to do if you genuinely need lights-out autonomous operation

The supported pattern is **pre-flight everything that triggers an always-on pause**, so the harness encounters no decision it has to escalate. Concretely:

1. **Skip the `generate_preprocess` always-on:** use a dataset that's in the registry (or that you've already preprocessed yourself and supply as a local parquet path). When `prepare_data` takes branches (a) or (b), `generate_preprocess` never fires.
2. **Skip the `configure_reward` always-on:** set `reward_kind: rule` in intake. The custom/shaped authoring branch never fires.
3. **Skip the `sanity_rollout` always-on:** run `skills/reward_rule/templates/sanity_probe.py` yourself ahead of time, place the resulting JSON at `workspace/sanity/sanity_report.md` with `verdict: green`, and the state will detect the existing report and pass it through. *(This is a manual side-channel; document the override in the run log.)*
4. **Skip the cost gate always-on:** keep estimated node-hours below `cost_gate_threshold_node_hours` (default 50). For larger runs, raise the threshold in intake (`cost_gate_threshold_node_hours: 200`) — accepting the silent autonomous spend that implies.

If those four conditions all hold, `--no-hitl` becomes fully autonomous for the duration of the run. If any one of them doesn't hold, the harness will pause and stay paused.

This is intentional. Burning many GPU-hours on a misconfigured run is more expensive than the human time to confirm those four points.

## State logging contract

The web dashboard's live-state highlighting and the `re-attach to a running job` flow both depend on two artefacts every run must produce. These are part of the inter-state contract; without them, the dashboard renders a cosmetic graph rather than a live one.

### `runs/<run_id>/meta.json`

Created by the first state (`intake`) on entry, finalised by `finalize`:

```json
{
  "run_id": "<run_id>",
  "started_at": "<ISO8601 UTC>",
  "status": "running",
  "hitl": true,
  "driver": "<runner identifier — e.g., claude-opus-4-7, custom>",
  "harness_commit": "<git rev-parse --short HEAD of this harness folder>",
  "purpose": "<one-line free text from intake>"
}
```

`finalize` updates this file with:

```json
{
  "status": "completed | incomplete | failed",
  "finished_at": "<ISO8601 UTC>",
  "terminal_state": "finalize",
  "terminal_input": "<deliverable name declared in states/finalize.md>",
  "terminal_stage": "<state that transitioned to finalize>",
  "final_report": "workspace/final_report.md"
}
```

`completed` means a successful training summary, generation report, or evaluation report. `incomplete` means training started but ended crashed, preempted, or cancelled. `failed` means an early gate, launch, generation, or evaluation failed. The detailed trainer/job status remains in the terminal input and `final_report.md`.

If HITL is escaped mid-run (the user flipped to `--no-hitl` after entry, or the run started in `--no-hitl`), append `hitl_switched_at` and `hitl_switched_reason` so the audit trail is unambiguous.

### `runs/<run_id>/workspace/logs/state_log.md`

Created by `intake` on entry, appended to by **every** subsequent state on entry. The format is one line per FSM step:

```
- [<ISO8601 UTC>] #<step> entered <state_name>, from <previous_state_name>
```

Examples:

```
- [2026-05-22T00:32:43Z] #1 entered intake, from start
- [2026-05-22T00:34:10Z] #2 entered locate_recipe, from intake
- [2026-05-22T00:35:00Z] #3 entered prepare_data, from locate_recipe
```

A terminal transition (`finalize` entry) gets the same shape; no separate "exit" line. Off-pipeline events (HITL switches, cancel requests, preemption) get a free-form annotation line bracketed by `--`:

```
- [2026-05-22T00:05:00Z] -- HITL switched off (user request) — remaining cost-gate / summarize / finalize pauses skipped --
```

The dashboard parser (`web/src/verl_harness_web/parser.py` regex `_STATE_LOG_RE`) reads this file directly; do not deviate from the format.

### Responsibility

Every state is responsible for appending its own entry as its first action on entry, before any other side-effect. If a state's `## Description` doesn't say so, the agent still does it — this is a global rule, not a per-state instruction.

## Tone

- American English.
- Quote tool output verbatim (squeue lines, srun output, trainer log lines). Never paraphrase a numerical result.
- Be precise about *who* did what: the harness reports its own actions; the trainer reports its own; do not blur the two.
