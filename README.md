# verl-harness

A markdown-driven agent harness that drives an LLM agent through the full lifecycle of a [verl](https://github.com/volcengine/verl) training run: parse the user's intent, find the right recipe in the verl checkout, prepare the dataset (verl's preprocess scripts for known datasets; auto-generated for HuggingFace datasets that aren't in the registry), pick a compute target (local GPU, local Slurm login node, or remote Slurm via ssh), provision the env, launch the job, monitor it, and produce a run report.

The harness is a workflow that produces a concrete output (a trained checkpoint + a structured run report). It does not optimise verl itself; it drives it.

## What it does

The harness is a finite state machine specified entirely in markdown — every state under `states/` and every skill under `skills/` is a runtime instruction file the agent reads and executes. There is no executable code in this repo; the agent is the runtime.

- **One agent session** walks the FSM from `intake` to `finalize`.
- **No registry of trainers.** The user names the algorithm (`ppo`, `grpo`, `gspo`, `sft`, …); the harness goes looking for a matching script under the verl checkout's `examples/<algo>_trainer/` or `recipe/<algo>_trainer/`, or falls back to a direct `python -m verl.trainer.main_<algo>` command. If the user names a trainer verl doesn't have, the harness halts honestly.
- **Two dataset paths.** Known datasets (gsm8k, math, hellaswag, full_hh_rlhf, aime, …) bind to verl's existing preprocess scripts. Unknown HuggingFace datasets route through `generate_preprocess`, which writes a verl-compatible preprocess script from one of verl's preprocess templates.
- **Three compute paths.** `local-direct` (run on this host's GPUs), `local-slurm` (sbatch from this Slurm login node), `ssh-slurm` (ssh to a remote login node and sbatch).
- **Monitor + report.** The `monitor_training` state polls until terminal (success / crashed / preempted / cancelled), tails logs, scans for OOM / NaN / NCCL / vLLM crashes, and parses progress lines into a CSV. `summarize` and `finalize` produce branch-aware reports — a crashed run is reported as crashed, with a specific remediation suggestion. The harness never fabricates a metric.

## FSM diagram

```
                       intake dispatches on `goal`
                       ─────────────────────────────────────────────────────────────
goal=train (default)  →  locate_recipe → configure_algorithm → prepare_data ⇄ generate_preprocess
                                                                              ↓
                                  configure_reward → select_compute → provision_env → sanity_rollout → launch_training
                                                                                                ↓                ↓
                                                                                       sanity_failed     launch_failed
                                                                                                ↓                ↓                ▼
                                                                                                                  monitor_training → summarize → finalize

goal=resume_monitor   →  monitor_training (re-attach to in-flight run; reads existing workspace) → summarize → finalize
goal=resume_train     →  launch_training (+trainer.resume_mode=auto patched in) → monitor_training → summarize → finalize
goal=generate         →  (select_compute → provision_env →) run_generate → (chain_eval → run_eval →) finalize
goal=eval             →  run_eval (no GPU; CPU-only) → finalize
```

`configure_algorithm` also halts to `finalize` when the algorithm has no first-class trainer in this verl (dpo / rm). `provision_env` failure → `finalize`. Sanity verdict `fail` → `finalize`. Honesty over heroics.

See `task-overview.md` for the full diagram and the parsing conventions.

## Layout

```
verl-harness/
├── task-overview.md
├── CLAUDE.md               — repo guidance for Claude Code (and other agents)
├── states/
│   ├── intake.md                    — dispatches on `goal`: train / resume_monitor / resume_train / generate / eval
│   ├── locate_recipe.md
│   ├── configure_algorithm.md       — Phase 2: apply algo_<name> skill, surface algo knobs
│   ├── prepare_data.md
│   ├── generate_preprocess.md
│   ├── configure_reward.md          — Phase 1: pick reward_kind (rule/model/custom/shaped)
│   ├── sanity_rollout.md            — Phase 1: load model, run 1 prompt, run reward fn
│   ├── select_compute.md
│   ├── run_generate.md              — Phase 3: batch generation (main_generation_server)
│   ├── run_eval.md                  — Phase 3: offline scoring (main_eval; CPU-only)
│   ├── provision_env.md
│   ├── launch_training.md
│   ├── monitor_training.md
│   ├── summarize.md
│   └── finalize.md
├── skills/
│   ├── intake/             — canonical training-intent fields, how to elicit them
│   ├── verl_recipes/       — recipe scoring, direct-module fallback, recipe.md format
│   ├── dataset_registry/   — the ~14 known verl-preprocessable datasets + column conventions
│   ├── dataset_autogen/    — author a verl preprocess script from an HF dataset schema
│   ├── compute_select/     — capability probes (gpu/slurm/ssh) and target selection
│   ├── compute_local/      — local-direct provisioning, launch, monitoring
│   ├── compute_slurm/      — local-slurm provisioning, launch, monitoring
│   ├── compute_ssh_slurm/  — ssh-slurm provisioning, launch, monitoring
│   ├── training_monitor/   — polling cadences, terminal conditions, anomaly patterns, progress parsing
│   ├── reward_rule/        — built-in deterministic rewards (Phase 1)
│   ├── reward_model/       — pre-trained reward-model scoring (Phase 1)
│   ├── reward_custom/      — author a custom_reward_function.path file (Phase 1)
│   ├── reward_shaping/     — composing format + correctness + length rewards (Phase 1)
│   ├── algo_ppo/           — PPO-only knobs (critic, value_loss, kl_ctrl) (Phase 2)
│   ├── algo_grpo/          — GRPO group-rollout knobs (n, norm_adv_by_std) (Phase 2)
│   ├── algo_dpo/           — DPO handling (note: not first-class in verl) (Phase 2)
│   ├── algo_sft/           — SFT knobs (packing, chat template, dynamic batch) (Phase 2)
│   ├── algo_rm/            — RM training (note: not first-class in verl) (Phase 2)
│   ├── algo_distill/       — on-policy distillation (teacher + distill loss) (Phase 2)
│   ├── run_generate/       — main_generation_server CLI + pitfalls (Phase 3)
│   ├── run_eval/           — main_eval CLI + reward fn integration (Phase 3)
│   ├── builtin-tools/      — FastHarness filesystem/shell/web tools (used by every state)
│   └── global/             — honesty principle, scope discipline, defaults, state-log contract
├── runs/                   — per-execution workspace dirs (gitignored)
└── web/                    — sibling Python package: `verl-harness-web` live dashboard
```

## How to invoke

Point an agent runner at this directory and have it start at `states/intake.md`. The agent walks the FSM as specified — at each state it reads the state file, applies the listed skills, executes the described work, writes the named deliverables under `runs/<run_id>/workspace/`, and transitions per the `## Next States` rules.

The harness asks for `verl_root` (or reads `$VERL_HOME`), the algorithm, the model, the dataset, the reward kind, and the compute preference. From there it pauses at hand-off points for confirmation — recipe selection (when multiple match), prepared-data confirmation (with row-0 sample), compute-target confirmation, provisioning result, and the cost gate before launching the actual training job.

**HITL mode semantics:**
- **HITL on (default):** every documented hand-off point pauses.
- **`--no-hitl`:** **semi-autonomous, not fully autonomous.** Most pauses pass silently, but **four always-on hand-off points still block** — `generate_preprocess` script approval, `configure_reward` approval for custom/shaped reward, `sanity_rollout` report approval, and the cost gate when estimated node-hours ≥ `cost_gate_threshold_node_hours` (default 50). See `skills/global/scientific_principles.md` → "Mode semantics" for the four conditions to pre-flight if you want lights-out operation.

### Running with Claude Code

The harness is designed to be driven by [Claude Code](https://claude.com/claude-code) (or any compatible agent runner). A minimal starting prompt:

```
You are driving the verl-harness FSM at /path/to/verl-harness.

1. Read CLAUDE.md, task-overview.md, and skills/global/scientific_principles.md.
2. Read states/intake.md and apply it.
3. After each state's work is complete, read the named state file for its
   `## Next States` block and evaluate which transition condition holds.
4. Append one line to runs/<run_id>/workspace/logs/state_log.md on every
   state entry (format: `- [<ISO8601 UTC>] #<step> entered <state>, from <prev>`).
5. Honor every hand-off point per the state's `## Hand-off Points` block.

My intent: <one sentence — e.g., "GRPO on gsm8k with Qwen3-4B on agent-dev partition">.

verl checkout: <absolute path or $VERL_HOME>.
HITL: on (or `--no-hitl` for autonomous mode).
```

The agent owns the FSM transition logic. The harness does not ship a runner; it ships the spec the runner follows.

### Re-attaching to a running job

If your Claude Code session disconnects mid-`monitor_training` (SSH drop, container restart, conversation auto-compression), start a new session and prompt:

```
The harness at /path/to/verl-harness has an in-flight run.
1. Read runs/<latest_run_id>/meta.json to confirm status: "running".
2. Read runs/<latest_run_id>/workspace/logs/state_log.md to recover the last
   state entered. If it's `monitor_training`, re-enter it.
3. Read workspace/job/job_info.md for the slurm_jobid / pid and resume polling.
```

The monitor state explicitly re-reads all upstream decisions from `workspace/` on every poll (see `states/monitor_training.md`), so it is safe to re-attach. Do **not** rely on agent memory from the previous session.

## Running on a cloud Slurm cluster

Cloud-hosted Slurm clusters (cluster login node on a remote host, GPUs behind a partition) introduce a few operational concerns the local-laptop case does not:

- **Wrap your Claude Code session in `tmux` / `screen`.** If your SSH connection drops, the session keeps polling. Re-attach with `tmux a -t verl-harness`.
- **Point `$HF_HOME` and `$OUTPUT_DIR` at scratch, not $HOME.** Most cloud clusters have a small `$HOME` quota and a large `$SCRATCH` (or shared filesystem) tier. Set in your shell profile:
  ```bash
  export SCRATCH=${SCRATCH:-/mnt/scratch/$USER}            # cluster-specific
  export HF_HOME=$SCRATCH/.cache/huggingface
  ```
  The harness's `provision_env` will warn if either resolves under `$HOME`.
- **Container detection.** If the cluster's verl install uses Apptainer/Singularity (common on shared HPC), the harness's `compute_slurm` skill grep's `examples/tutorial/slurm/ray_on_slurm.slurm` for `apptainer run` and preserves the container wrap. If you have a Conda env that already has verl + torch + vllm, declare `container: none` in `compute_choice.md` to drop the wrap.
- **Where slurm fields come from.** `slurm.partition`, `slurm.account`, `slurm.time_limit` are *not* in the recipe — they are intake fields you supply. `sinfo --format='%P %a %l'` lists the partitions you can pick from.
- **Cost gate.** Even under `--no-hitl`, the launch_training cost gate fires when estimated node-hours ≥ 50 (configurable via `cost_gate_threshold_node_hours` in `training_intent.md`). This is a safety rail against silent autonomous spend.

## Dashboard

A sibling Python package at `web/` ships a live dashboard, `verl-harness-web`, that watches a harness folder and renders the FSM graph, the active state, the selected recipe / prepared dataset / compute target, and (once training is running) the progress chart, anomalies list, job card, and incremental log tail.

```bash
uv run --project web verl-harness-web .
```

Defaults to `http://127.0.0.1:8766`. The dashboard is **observe-only** — it does not execute the harness; it watches the workspace directory and renders what's on disk. The only writes it permits are to `task-overview.md`, `states/*.md`, and `skills/**/*.md`. See `web/README.md` for endpoints, modes (`live` / `--static`), and design notes.

## Required capabilities

Declared in `task-overview.md`. At least one of `gpu.access`, `slurm.access`, `ssh.exec` must be present (otherwise no training can run). Other capabilities (`filesystem.read`/`write`, `shell.exec`, `code.execute`, `web.search`, `web.fetch`) are mandatory.

## What it does not do

- Does not modify the verl source tree (the verl repo is read-only from this harness's perspective).
- Does not curate "supported trainers" — whatever the user names, the harness tries to bind to a script or trainer module verl has.
- Does not run interactive ssh sessions or interactive Slurm srun shells.
- Does not invent metrics, checkpoints, or success verdicts.

## License

This harness is licensed under the Apache 2.0 License, matching the verl repo's license.
