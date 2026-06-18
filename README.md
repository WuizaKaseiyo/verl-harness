# verl-harness

A markdown-driven agent harness that walks an LLM through a full
[verl](https://github.com/volcengine/verl) training run — from "I want
GRPO on gsm8k with Qwen3-4B" to a trained checkpoint and a structured
report.

Every state under `states/` and every skill under `skills/` is an
instruction file the agent reads and executes. **There is no executable
code in this repo; the agent is the runtime.**

## FSM

```
intake → locate_recipe → configure_algorithm → prepare_data ⇄ generate_preprocess
       → configure_reward → select_compute → provision_env → sanity_rollout
       → launch_training → monitor_training → summarize → finalize
```

Three branching axes:

| axis | what the user picks | what the harness does |
|---|---|---|
| **algorithm** | `ppo` / `grpo` / `gspo` / `sft` / … | searches `examples/<algo>_trainer/` and `recipe/<algo>_trainer/` in the verl checkout; falls back to `python -m verl.trainer.main_<algo>`. No curated trainer registry — halts honestly when verl has no match. |
| **dataset** | a name (gsm8k, math, …), an HF id, or a parquet path | known names bind to verl's preprocess scripts; unknown HF ids route through `generate_preprocess`, which authors a preprocess script from one of verl's templates. |
| **compute** | `auto` (default), `local-direct`, `local-slurm`, `ssh-slurm` | capability probes (`gpu.access` / `slurm.access` / `ssh.exec`) pick a target. |

See `task-overview.md` for the full diagram (resume / generate / eval
goals included) and `CLAUDE.md` for editing conventions.

## Drive it

Point an agent runner at this directory. Minimal prompt:

```
You are driving the verl-harness FSM at /path/to/verl-harness.
Read CLAUDE.md, task-overview.md, and states/intake.md, then apply intake.
Walk transitions per each state's `## Next States` block.
Honor every `## Hand-off Points` block.

Intent: <one sentence>.
verl checkout: <absolute path or $VERL_HOME>.
HITL: on   (or --no-hitl).
```

`--no-hitl` is semi-autonomous: four hand-off points stay always-on —
`generate_preprocess` script approval, custom-reward approval,
`sanity_rollout` approval, and the cost gate when estimated node-hours
≥ 50. See `skills/global/scientific_principles.md`.

The harness ships the spec; the agent owns the transition logic. There
is no runner in this repo. Tested with [Claude Code](https://claude.com/claude-code).

## Dashboard

```bash
uv run --project web verl-harness-web .
```

Opens `http://127.0.0.1:8766`. Observe-only — renders the FSM, the
progress chart, anomalies, job card, and the log tail from each run's
workspace. See `web/README.md`.

## What it does NOT do

- Modify the verl source tree (the verl repo is read-only from the
  harness's perspective).
- Invent metrics, checkpoints, or success verdicts. A crashed run is
  reported as crashed, with a specific remediation.

## License

Apache 2.0 (matches verl).
