# verl-harness — Drive an agent through a full verl training run

## Overview

This harness drives a FastHarness-compatible agent through the full lifecycle of a [verl](https://github.com/volcengine/verl) training run: it reads the user's training intent (which algorithm, which dataset, which model, what compute), locates the right recipe inside the verl checkout, prepares the dataset (using verl's built-in preprocessing or auto-generating one for an unknown HuggingFace dataset), picks a compute target (local GPU, local Slurm login node, or remote Slurm via ssh), provisions the environment, launches the training job, monitors it to terminal status, and writes a summary report.

This is a **Category B** harness — a workflow demo that produces a concrete output (a trained checkpoint + a structured run report). It does not optimise verl itself; it drives it. It is structured like `web2bigtable/`, `ai-scientist-v2/`, `eduplanner/`: a workflow harness whose deliverable is a concrete artefact a user can hand off.

### What this reproduces

The behaviour each item references has been **verified against `/Users/steven/verl`** (verl source tree); line and path references are the contract.

- **The PPO-family algorithm dispatch.** All RL algorithms route through `verl.trainer.main_ppo` with `algorithm.adv_estimator=<name>`. The legal set of estimator names is **`AdvantageEstimator`** at `verl/trainer/ppo/core_algos.py` (an `Enum`) plus the `@register_adv_est(...)` decorator registry in the same file — the harness reads that registry instead of hardcoding a list. Verified estimators in the current verl checkout include `gae`, `grpo`, `grpo_vectorized`, `gdpo`, `grpo_passk`, `rloo`, `rloo_vectorized`, `opo`, `reinforce_plus_plus`, `remax`, `gpg`, `optimal_token_baseline`, `tir_optimal_token_baseline`.
- **The SFT dispatch.** SFT does **not** go through `main_ppo`; the harness selects `verl.trainer.sft_trainer` (FSDP, single-node) or `verl.trainer.sft_trainer_ray` (Ray, multi-node) by user scale.
- **The `examples/<algo>_trainer/run_*.sh` recipe-discovery flow.** The harness enumerates every shell script under `examples/<algo>_trainer/` (e.g., `examples/grpo_trainer/run_qwen3_4b_fsdp.sh`, `examples/ppo_trainer/run_qwen3_8b_fsdp.sh`, `examples/sft/...`), scores them by model-slug, backend (`fsdp` / `megatron` / `veomni`), and scale fit, and binds the user's intent to the best match. If no script matches, it falls back to a direct `python -m verl.trainer.main_ppo ...` invocation with Hydra-style overrides drawn from `verl/trainer/config/ppo_trainer.yaml`.
- **The dataset preprocessing flow.** Known datasets bind to `examples/data_preprocess/<name>.py` (14 scripts verified on disk: `gsm8k`, `math_dataset`, `aime2024_multiturn_w_tool`, `dapo_multiturn_w_tool`, `full_hh_rlhf`, `geo3k`, `geo3k_multiturn_w_tool`, `gsm8k_multiturn_sft`, `gsm8k_multiturn_w_tool`, `gsm8k_tool_agent_loop`, `hellaswag`, `multiturn`, `pokemon`, `preprocess_search_r1_dataset`). Unknown HuggingFace datasets route through `generate_preprocess`, which writes a verl-compatible preprocess script from one of the existing templates, then `prepare_data` re-enters and runs it.
- **The Slurm + Ray bring-up.** The harness adapts `examples/tutorial/slurm/ray_on_slurm.slurm` — a real upstream template — populating sbatch directives and splicing in the assembled launch command.
- **Three compute targets.** `local-direct` (bash the script in the agent's shell), `local-slurm` (sbatch from this host as a login node), `ssh-slurm` (rsync the slurm script + `ssh <alias> sbatch`).
- **Real footguns inside the agent shell.** Two preflight checks the import-only provisioning checks miss:
  - **ROCR/CUDA env-collision** — `verl/single_controller/base/worker.py` raises inside the Ray worker actor's `__init__` if both `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set. The harness greps the env up-front and injects `unset` lines into `launch_env.sh`.
  - **vLLM V1 requirement** — `verl/workers/rollout/vllm_rollout/vllm_async_server.py` imports the V1 `AsyncLLM` unconditionally; modern vllm silently falls back to V0 under some configs. When the recipe uses `rollout.name=vllm`, the harness injects `export VLLM_USE_V1=1`.
- **Monitor + report.** `monitor_training` loops on the chosen target's polling mechanism (`kill -0 <pid>` for local-direct; `squeue` / `sacct` for slurm; same via `ssh` for ssh-slurm), tails logs incrementally, scans for OOM / NaN / NCCL / vLLM / preemption patterns, and exits on success / crashed / preempted / cancelled. `summarize` and `finalize` produce branch-aware reports — a crashed run becomes a crash report with a specific remediation suggestion.
- **HITL gates.** Recipe selection (when multiple match), prepared-data confirmation, compute-target confirmation, and the **cost gate** before launch.

### What this does NOT reproduce

- **The verl source code itself.** The harness reads and drives verl as-is. It never edits `verl/` or any trainer code. If a bug fix needs source changes, that belongs in the verl repo, not here.
- **`verl.trainer.main_generation_server`** (generation-server runs) and **`verl.trainer.main_eval`** (eval-only runs). Both modules exist in verl but the harness focuses on training. Adding them would mean new states (`launch_server` / `launch_eval`), not refactoring the existing ones.
- **`verl/trainer/distillation/`** — knowledge-distillation training flows. The harness can bind to `examples/on_policy_distillation_trainer/` scripts via recipe search, but it does not yet model the distillation-specific intent fields (teacher model / temperature / distillation loss weights).
- **NPU / Ascend specifics.** `examples/ascend_extras/`, `requirements-npu.txt`, `docker/ascend/` exist; the harness auto-detects `DEVICE` via `python3 -c 'import torch_npu'` and otherwise treats NPU as a pass-through (the underlying recipe handles it). Site-specific NPU provisioning (HCCL env vars, `npu-smi` health checks beyond the basic probe) is not modelled.
- **Hyperparameter sweeps.** `examples/tuning/` exists in verl; this harness drives **one** set of hyperparameters per run. Multi-config sweeps are the user's job (or a future meta-harness over this one).
- **Inference-time generation pipelines.** `examples/generation/` is for generation experiments, not training. Out of scope here.
- **Docker / container builds.** `docker/Dockerfile.stable.vllm`, `docker/Dockerfile.stable.sglang`, etc. exist; the harness assumes the user has a working `python -c 'import verl'` and does not build or push containers.
- **Megatron parallelism design decisions.** TP / PP / CP / DP splits come from the chosen recipe's defaults; the harness passes user-supplied `--nodes` / `--gpus-per-node` through but does not advise on the parallelism mix.
- **Reward-model training itself.** The harness does not yet train a reward model from scratch (roadmap Phase 2). For *using* a pre-trained reward model in PPO/GRPO etc., see the supported `reward_kind: model` flow below.

### What this DOES reproduce (reward engineering — Phase 1+)

verl exposes four ways to score model outputs; the harness surfaces all of them as first-class concerns:

| `reward_kind` | Mechanism | Configured via |
|---|---|---|
| `rule` | Built-in deterministic reward fn (e.g., gsm8k correctness, math final-answer extraction) | `reward_model.style=rule` + `reward_model.ground_truth` in the dataset row; nothing else needed |
| `model` | Pre-trained reward model that scores assistant responses | `reward_model.path=<HF id or local path>` + `reward_model.input_tokenizer` |
| `custom` | User-supplied Python function called per response | `reward.custom_reward_function.path=<py file>` + `reward.custom_reward_function.name=<callable>`; see `skills/reward_custom/` |
| `shaped` | Composition of multiple rewards (format + correctness + length) with weights | `skills/reward_shaping/` describes the composition pattern; usually realised as a custom function |

The `configure_reward` state (between `prepare_data` and `select_compute`) is where the user picks one of these and the harness writes `workspace/reward/reward_config.md` for `launch_training` to patch into the CLI.
- **`recipe/` subtree contents.** verl's `recipe/` directory exists but is empty in some checkouts (including the current `/Users/steven/verl`). The harness enumerates it as a fallback search location; if a future checkout populates it, recipes there will be picked up automatically.
- **Fabricated metrics, checkpoints, or success verdicts.** If training crashes, the run report says so plainly. The harness never invents a loss curve or a checkpoint path.

### When to use this harness

When you have a verl checkout, a dataset (named or referenceable by HuggingFace id), a model (HF id or local path), an algorithm, and either a GPU machine or a Slurm cluster, and you want an agent to drive the whole "go from spec to running job to trained checkpoint" pipeline without you sitting through it.

## Workflow Diagram

```
┌────────────┐
│   intake   │  parse intent + dispatch on `goal`:
│            │    train (default) → locate_recipe
│            │    resume_monitor  → monitor_training (re-attach to in-flight run)
│            │    resume_train    → launch_training with +trainer.resume_mode=auto
│            │    generate        → run_generate (skip locate_recipe / configure_*)
│            │    eval            → run_eval (no GPU; skip select_compute / provision_env)
└─────┬──────┘
      ▼ (goal=train)
┌──────────────────┐
│  locate_recipe   │  find the verl example/recipe script that matches the trainer + model,
│                  │  or fall back to constructing a launch command from verl.trainer.main_*
└─────┬────────────┘
      ▼
┌──────────────────────┐
│ configure_algorithm  │  apply algo_<name> skill (ppo/grpo/dpo/sft/rm/distill);
│                      │  surface algorithm-specific knobs + dataset-column requirements;
│                      │  write workspace/algorithm/algorithm_config.md.
│                      │  halts if algorithm has no first-class trainer (dpo/rm).
└─────┬────────────────┘
      ▼
┌──────────────────┐    unknown dataset    ┌────────────────────┐
│  prepare_data    │──────────────────────▶│ generate_preprocess │
│                  │◀──────────────────────│ (write a verl-style │
│  (verl preprocess script for known        │ preprocess script   │
│   datasets; produces parquet; row-0       │ for an HF dataset)  │
│   sample displayed in HITL; column         └────────────────────┘
│   shape validated against algorithm)
└─────┬────────────┘
      ▼
┌──────────────────┐
│ configure_reward │  pick reward_kind ∈ {rule, model, custom, shaped};
│                  │  author compute_score.py if custom/shaped;
│                  │  write workspace/reward/reward_config.md
└─────┬────────────┘
      ▼
┌──────────────────┐
│  select_compute  │  decide: local-direct | local-slurm | ssh-slurm
└─────┬────────────┘
      ▼
┌──────────────────┐
│  provision_env   │  ROCR/CUDA + VLLM_USE_V1 pre-flight, then torch/cuda/verl import,
│                  │  resolve model weights, mkdir output_dir, write launch_env.sh
└─────┬────────────┘
      ▼
┌──────────────────┐
│  sanity_rollout  │  re-uses provision_env's launch_env.sh; loads model + samples
│                  │  1 response + invokes reward fn + 10-row distribution.
│                  │  fail → short-circuit to finalize (no real training spent)
└─────┬────────────┘
      ▼
┌──────────────────┐
│ launch_training  │  three sub-paths:
│                  │   • local-direct: nohup bash launch.sh & (capture pid)
│                  │   • local-slurm: sbatch job.slurm (capture jobid)
│                  │   • ssh-slurm: rsync + ssh <alias> sbatch (capture remote jobid)
└─────┬────────────┘
      ▼
┌──────────────────┐
│ monitor_training │  loop: poll status / tail logs / scan OOM/NaN/NCCL/vLLM/preempt.
│                  │  exits to summarize on success / crashed / preempted / cancelled.
└─────┬────────────┘
      ▼
┌──────────────────┐
│    summarize     │  status-branched report: success / crashed (+remediation) /
│                  │  preempted (+resume cmd) / cancelled
└─────┬────────────┘
      ▼
┌──────────────────┐
│    finalize      │  terminal — final_report.md + artefact pointers
└──────────────────┘

(goal=generate / goal=eval — post-train lifecycle tracks, also reachable standalone from intake)

┌──────────────────┐
│  run_generate    │  batch generation via verl.trainer.main_generation_server.
│                  │  reads prompts.parquet + checkpoint, writes generations.parquet.
│                  │  re-uses provision_env's launch_env.sh.
└─────┬────────────┘
      ▼ (chain_eval=true) or standalone goal=eval
┌──────────────────┐
│  run_eval        │  CPU-only scorer via verl.trainer.main_eval.
│                  │  reads generations.parquet + reward fn,
│                  │  emits per-data_source test_score.
└─────┬────────────┘
      ▼
┌──────────────────┐
│    finalize      │  terminal — eval_report.md / generate_report.md
└──────────────────┘
```

## Starting State
states/intake.md

## Hand-off Points
allowed

Several states pause for user confirmation by default — choosing a recipe when multiple match (`locate_recipe`), confirming dataset destination and quota (`prepare_data`), confirming the compute target picked (`select_compute`), and confirming the final launch command before it starts spending GPU time (`launch_training`). These checkpoints can be skipped by passing `--no-hitl` in the invocation; the harness records that escape in the run log.

## Required Capabilities

- filesystem.read
- filesystem.write
- shell.exec
- code.execute            # for running verl's Python preprocess scripts (e.g., gsm8k.py)
- web.search              # finding HF dataset / model docs
- web.fetch               # fetching HF dataset cards, model cards, dataset schemas
- slurm.access            # CUSTOM token — host can run `sbatch` / `squeue` / `scancel` locally (i.e., this machine is a Slurm login node). If not present, the harness silently disables the `local-slurm` branch of `select_compute`.
- ssh.exec                # CUSTOM token — host has an ssh client and a configured remote login node (the agent runs `ssh <login> sbatch …`). If not present, the harness silently disables the `ssh-slurm` branch.
- gpu.access              # CUSTOM token — at least one CUDA-visible GPU on the host. If not present, the harness silently disables the `local-direct` branch.

At least one of `slurm.access` / `ssh.exec` / `gpu.access` must be present, or no training can run and `select_compute` halts with an error.

**Built-in tools fallback.** The five standard FastHarness capabilities (`filesystem.read`, `filesystem.write`, `shell.exec`, `web.search`, `web.fetch`) are all backed in-tree at `tools/registry.py` via the **builtin-tools** skill (`skills/builtin-tools/`). A host that lacks native equivalents can run `python tools/registry.py <tool_name> '<json_args>' --workspace "<WORKSPACE>"` for any of the nine bundled tools (`list_dir`, `read_file`, `grep`, `file_create`, `append_file`, `mkdir`, `shell_exec`, `search_web`, `fetch_webpage`). `search_web` requires `SERPER_API_KEY` in the env; the rest have no external requirements.

## Notes

- **Paths.** The harness assumes a verl checkout exists somewhere on disk and refers to it as `VERL_ROOT`. The user passes it in via the invocation (e.g., "run the harness with verl at /opt/verl") or by setting the `VERL_HOME` env var. If neither is set, `intake` asks. All references to verl's example scripts, slurm templates, and `verl.trainer.main_*` modules are anchored to `VERL_ROOT`. The reference verl checkout used to validate this harness's claims is `/Users/steven/verl`.
- **Workspace layout.** All run artefacts live under `runs/<run_id>/workspace/`. Per-state deliverables go in well-known subdirectories: `workspace/intake/`, `workspace/recipe/`, `workspace/dataset/`, `workspace/compute/`, `workspace/env/`, `workspace/job/`, `workspace/logs/`, `workspace/summary/`. The training job's *own* output (checkpoints, the trainer's log files, the slurm `.out` / `.err` files) lives wherever the user pointed it (an `OUTPUT_DIR`) — we record the path under `workspace/job/output_dir.txt` rather than copying the entire checkpoint into the workspace.
- **Honesty over impressiveness.** If the training job OOM-crashes, the summary says so plainly. If only one step ran before a NaN, the summary reports the one step. The harness must never fabricate a metric, a checkpoint path, or a "success" verdict that didn't happen.
- **Cost awareness.** Training jobs can cost hundreds of GPU-hours. Before `launch_training` actually fires the job (if HITL is allowed), the agent must present the final launch command and the expected node-hours and ask the user to confirm.
- **Cancellation.** During `monitor_training`, if the user sends a stop signal (or `--cancel` is set externally), the agent issues `scancel` (Slurm) or kills the local process and writes a `cancelled` summary.
- **Reproducibility.** The harness records the exact training command, env-vars, seeds, and `git log -1` of the verl repo at launch time, so a second run can be re-launched verbatim.
- **American English in all written artefacts.**
