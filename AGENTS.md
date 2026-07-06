# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo is (and isn't)

This repo is primarily a set of Markdown specs that drive an agent through a [verl](https://github.com/volcengine/verl) training run end-to-end. Every `.md` under `states/` and `skills/` is runtime source read by the agent. The agent is the FSM runtime; there is no in-tree training runner. Supporting Python exists under `tools/` (fallback tools and contract validation) and `web/` (the dashboard). `runs/` is git-ignored scratch space for per-execution workspaces.

When asked to "fix a bug" or "add a feature," the change is almost always an edit to a markdown spec — re-wording an instruction, tightening a condition, adding a deliverable, adjusting a HITL checkpoint. Treat these files as **agent-facing source code**: precision, scope discipline, and consistency across states matter more than prose quality.

## Architecture: an FSM expressed in markdown

The harness is a finite state machine. **`states/*.md` is the single source of truth for control flow.** `task-overview.md`, README diagrams, and the dashboard are derived views and must be synchronized after state changes. The main training path is:

```
intake → locate_recipe → configure_algorithm → prepare_data ⇄ generate_preprocess
       → configure_reward → select_compute → provision_env → sanity_rollout
       → launch_training → monitor_training → summarize → [reflect] → finalize
```

`intake` also dispatches resume, generation, and evaluation goals. All normal and failure branches converge on `finalize` through the terminal-input contract declared in `states/finalize.md`. The opt-in `reflect` loop (`summarize → reflect → configure_algorithm`, bounded by the `**Loop:** max_iterations` declaration on the back-edge) and the dataset bounce are the only cycles; the validator rejects any undeclared cycle.

Two kinds of files:

- **`states/<name>.md`** — one per FSM node. Each follows a strict schema the agent depends on: `## Description` (what to do, which skills to apply) → `## Skills` (which skill dirs to read) → `## Hand-off Points` (HITL pause points; the dashboard parser still accepts the older `## Human Checkpoints` for back-compat, but new states should use `## Hand-off Points`) → `## Next States` (each transition has a `**Condition:**` and `**Deliverables:**` block). Breaking this schema breaks the FSM.
- **`skills/<area>/default.md`** — domain knowledge the states reference. Skills are *consulted*, not transitioned to. The `global/` skill (`scientific_principles.md`) is the only one that binds every state; the others are area-specific.

The state files are the FSM; the skills are the library the FSM calls into. A state delegates non-trivial logic to a skill — e.g., `locate_recipe.md` explicitly defers candidate scoring to `skills/verl_recipes/`. When editing a state, check whether the detail belongs in the state (control flow, transitions, deliverables) or in the skill (rules, regexes, templates, tables).

## The three branching axes

These are baked into the design — most edits touch one of them:

1. **Algorithm binding** (`locate_recipe` + `skills/verl_recipes`). The user names a trainer (`ppo`/`grpo`/`sft`/…); the harness searches `<VERL_ROOT>/examples/<algo>_trainer/` and `<VERL_ROOT>/recipe/<algo>_trainer/` for `run_*.sh`, scores by model-slug + backend + scale, and either picks one or falls back to `python -m verl.trainer.main_<algo>` with Hydra-style CLI overrides. **There is no curated trainer registry** — whatever the user names is what the harness goes looking for, and it halts honestly if neither a script nor a trainer module exists.
2. **Dataset binding** (`prepare_data` ⇄ `generate_preprocess`). Three branches: (a) known verl-preprocessable name → run `<VERL_ROOT>/examples/data_preprocess/<name>.py` per `skills/dataset_registry`; (b) user-supplied parquet path → use as-is; (c) HF dataset id not in the registry → bounce to `generate_preprocess`, which authors a `preprocess.py` per `skills/dataset_autogen` and returns. The registry's table of ~14 known datasets is a hint — `dataset_registry` instructs the agent to verify on-disk under `examples/data_preprocess/` before binding.
3. **Compute target** (`select_compute` + `compute_local`/`compute_slurm`/`compute_ssh_slurm`). Three targets — `local-direct`, `local-slurm`, `ssh-slurm` — are gated by capability probes. The authoritative `auto` ranking and tie-break rules live in `skills/compute_select/default.md`; do not duplicate them in guidance files. `provision_env`, `launch_training`, and `monitor_training` each consult only the compute skill matching the selected target.

## Workspace as inter-state contract

Every state writes a canonical file under `runs/<run_id>/workspace/<area>/` that the next state reads. This is the only communication channel between states — no shared memory, no globals. The canonical files are:

```
workspace/intake/training_intent.md      ← single authoritative record of user intent
workspace/recipe/recipe.md               ← launch path + full args + verl commit
workspace/algorithm/algorithm_config.md  ← resolved estimator/loss mode + algorithm knobs
workspace/dataset/dataset.md             ← train_files / val_files / row counts
workspace/reward/reward_config.md        ← reward implementation + CLI injection
workspace/compute/compute_choice.md      ← target + probe results + (for slurm) sbatch directives
workspace/env/env_state.md, launch_env.sh
workspace/sanity/sanity_report.md         ← bounded model/data/reward probe
workspace/job/job_info.md                ← target, pid|slurm_jobid, cmd, log paths
workspace/job/job_status.md              ← success | crashed | preempted | cancelled
workspace/logs/{job_log.md, progress.csv, anomalies.md, crash_tail.md}
workspace/summary/summary.md
workspace/reflect/{refinement_plan.md,reflect_report.md,loop_state.json}
workspace/generate/{generate_report.md,generate_failed.md}
workspace/eval/{eval_report.md,eval_failed.md}
workspace/final_report.md
```

The training job's own outputs (checkpoints, slurm `.out`/`.err`, the trainer's logs) live under the user-specified `output_dir`. The workspace **records the path**; it does not copy the artefacts. When editing states, preserve this: never propose copying large training outputs into `workspace/`.

## Cardinal rules (`skills/global/scientific_principles.md`)

These bind every state and override any specific instruction that conflicts:

- **Honesty.** Never report a checkpoint that isn't on disk, a metric the trainer never logged, or a "success" verdict for a crashed run. Quote tool output verbatim — `squeue` lines, log lines, regex matches. If the trainer didn't print a reward, the summary says "reward not logged", not a guess.
- **Read-only over verl.** The harness never modifies the verl source tree. Recipe behaviour is configured via env-var overrides or Hydra CLI overrides at launch time, never by patching `run_*.sh`.
- **Cheap before expensive.** `provision_env` uses `sbatch --test-only` before submission; `launch_training` enforces a cost gate (estimated node-hours, presented to the user) before spending GPU time.
- **HITL on by default.** Each state's `## Hand-off Points` is authoritative. `--no-hitl` skips ordinary pauses, but the always-on gates defined in `skills/global/scientific_principles.md` still apply: generated preprocess approval, custom/shaped reward approval, sanity-rollout approval, and the cost gate above its configured threshold.
- **American English in all written artefacts.**

## Conventions when editing the specs

- **Preserve the state-file schema.** Required H2 sections: `## Description`, `## Skills`, `## Hand-off Points`, `## Next States`. (The dashboard parser at `web/src/verl_harness_web/parser.py` accepts the older `## Human Checkpoints` for back-compat, but new state files should use `## Hand-off Points`.) Each `### <next-state>` under `## Next States` must have a `**Condition:**` and a `**Deliverables:**` block. A transition that closes a cycle must additionally carry a `**Loop:** max_iterations: <n>` line between its `**Condition:**` and `**Deliverables:**` blocks. Skip a section only when truly inapplicable (e.g., `finalize.md` has no `## Next States` — the comment in that file explains why).
- **Workspace paths are part of the contract.** `workspace/intake/training_intent.md` is referenced by name from multiple downstream states; renaming it requires updating every reader. Same for `recipe.md`, `dataset.md`, `compute_choice.md`, `job_info.md`, `job_status.md`.
- **State vs skill placement.** Concrete rules, regex sets, tables of options, command templates → skill. Control flow, transition conditions, deliverables, HITL points → state. If a state file starts accumulating regexes or detail tables, that's the signal to move them into a skill.
- **Polling cadences are minimums, not targets.** 30 s / 60 s / 90 s for local-direct / local-slurm / ssh-slurm. Don't propose faster polling — it spams `squeue` and annoys cluster admins.
- **No new trainer registry.** Resist any change that turns the algorithm field into an enum or a hand-curated list. The harness's commitment is to try whatever the user names and halt honestly on miss.
- **Validate every FSM edit.** Run `python tools/validate_harness.py .`. It checks the state schema, transition targets and deliverables, reachability, terminal convergence on the loop-free graph, declared-loop bounds (undeclared cycles are rejected; declared loop edges must close a real cycle), skill references, and `finalize` terminal-input coverage.

## Invocation

The harness is invoked by running a compatible runner against this directory. The user passes (or `$VERL_HOME` provides) the absolute path to a verl checkout; from there the FSM drives itself. See `README.md` for the invocation surface.
