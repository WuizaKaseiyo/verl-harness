# verl-harness — Drive an agent through a full verl training run

## Overview

This harness drives a FastHarness-compatible agent through the full lifecycle of a [verl](https://github.com/volcengine/verl) training run: it reads the user's training intent (which algorithm, which dataset, which model, what compute), locates the right recipe inside the verl checkout, prepares the dataset (using verl's built-in preprocessing or auto-generating one for an unknown HuggingFace dataset), picks a compute target (local GPU, local Slurm login node, or remote Slurm via ssh), provisions the environment, launches the training job, monitors it to terminal status, and writes a summary report.

**Authority rule:** `states/*.md` is the single source of truth for FSM control flow. This overview is a human-readable projection. After changing a state or transition, run `python tools/validate_harness.py .` and update this projection when the visible flow changed.

This is a **Category B** harness — a workflow demo that produces a concrete output (a trained checkpoint + a structured run report). It does not optimise verl itself; it drives it. It is structured like `web2bigtable/`, `ai-scientist-v2/`, `eduplanner/`: a workflow harness whose deliverable is a concrete artefact a user can hand off.

### What this reproduces

The behaviour each item references has been **verified against `/Users/steven/verl`** (verl source tree); line and path references are the contract.

- **The PPO-family algorithm dispatch — two orthogonal axes.** All RL algorithms route through `verl.trainer.main_ppo`, configured along two independent registries in `verl/trainer/ppo/core_algos.py`, both read off the checkout (never hardcoded):
  - **Advantage estimator** (`algorithm.adv_estimator=<name>`) — the `@register_adv_est(...)` registry. Verified in the current checkout: `gae`, `grpo`, `grpo_passk`, `grpo_vectorized`, `gdpo`, `rloo`, `rloo_vectorized`, `opo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `remax`, `gpg`, `optimal_token_baseline`, `tir_optimal_token_baseline`.
  - **Policy loss mode** (`actor_rollout_ref.actor.policy_loss.loss_mode=<name>`) — the `@register_policy_loss(...)` registry: `vanilla` (default), `gspo`, `cispo`, `geo_mean`, `sapo`, `gpg`, `dppo_tv`, `dppo_kl`, `clip_cov`, `kl_cov`, `bypass_mode`.

  A named algorithm is often a *pair*: GSPO/CISPO/GMPO/SAPO/DPPO all run `adv_estimator=grpo` with a non-`vanilla` `loss_mode` (`examples/<name>_trainer/run_*.sh` carry both inline). `gdpo` is a real adv_estimator (Group reward-Decoupled), not classic DPO. `mtp` (Multi-Token Prediction) is a grpo run plus a model/training feature. `configure_algorithm` resolves the pair from the matched recipe, validates against both registries, and halts honestly on a name in neither.
- **The SFT dispatch.** SFT does **not** go through `main_ppo`; the harness selects `verl.trainer.sft_trainer` (FSDP, single-node) or `verl.trainer.sft_trainer_ray` (Ray, multi-node) by user scale.
- **The `examples/<algo>_trainer/run_*.sh` recipe-discovery flow.** The harness enumerates every shell script under `examples/<algo>_trainer/` (e.g., `examples/grpo_trainer/run_qwen3_4b_fsdp.sh`, `examples/ppo_trainer/run_qwen3_8b_fsdp.sh`, `examples/sft/...`), scores them by model-slug, backend (`fsdp` / `megatron` / `veomni`), and scale fit, and binds the user's intent to the best match. If no script matches, it falls back to a direct `python -m verl.trainer.main_ppo ...` invocation with Hydra-style overrides drawn from `verl/trainer/config/ppo_trainer.yaml`.
- **The dataset preprocessing flow.** Known datasets bind to scripts verified at runtime under `examples/data_preprocess/`; `skills/dataset_registry/default.md` contains a non-authoritative snapshot. Unknown HuggingFace datasets route through `generate_preprocess`, which writes a verl-compatible preprocess script from one of the existing templates, then `prepare_data` re-enters and runs it.
- **The Slurm + Ray bring-up.** The harness adapts `examples/tutorial/slurm/ray_on_slurm.slurm` — a real upstream template — populating sbatch directives and splicing in the assembled launch command.
- **Three compute targets.** `local-direct` (bash the script in the agent's shell), `local-slurm` (sbatch from this host as a login node), `ssh-slurm` (rsync the slurm script + `ssh <alias> sbatch`).
- **Real footguns inside the agent shell.** Two preflight checks the import-only provisioning checks miss:
  - **ROCR/CUDA env-collision** — `verl/single_controller/base/worker.py` raises inside the Ray worker actor's `__init__` if both `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set. The harness greps the env up-front and injects `unset` lines into `launch_env.sh`.
  - **vLLM V1 requirement** — `verl/workers/rollout/vllm_rollout/vllm_async_server.py` imports the V1 `AsyncLLM` unconditionally; modern vllm silently falls back to V0 under some configs. When the recipe uses `rollout.name=vllm`, the harness injects `export VLLM_USE_V1=1`.
- **Monitor + report (arm-and-detach).** Because runs span hours to days, `monitor_training` does not busy-loop an LLM. It launches a cheap **detached poller** (`skills/training_monitor/templates/watch_poller.py`) that owns the tight loop — polls the target's mechanism (`kill -0` / `squeue`+`sacct` / `ssh`), tails logs by byte-offset, parses verl's metric dict into `progress.csv`, scans OOM / NaN / NCCL / vLLM / preemption **plus** RL-divergence (entropy explosion + validation regression), and on terminal status writes `job_status.md` + touches a `terminal` sentinel. The agent re-engages only on events (sentinel / escalation / a 15–30 min heartbeat), not on the poller's 30/60/90 s. On slurm the terminal event can instead be delivered by an `sbatch --dependency=afterany` job. `summarize` and `finalize` produce branch-aware reports — a crashed run becomes a crash report with a specific remediation suggestion.
- **HITL gates.** Each state's `## Hand-off Points` declares ordinary pauses. The four always-on gates are defined centrally in `skills/global/scientific_principles.md`: generated preprocessing, custom/shaped rewards, sanity rollout, and expensive-job cost approval.
- **Post-training tracks.** `run_generate` drives `verl.trainer.main_generation_server`; `run_eval` drives `verl.trainer.main_eval`. They can run standalone from `intake`, and generation can chain into evaluation.

### What this does NOT reproduce

- **The verl source code itself.** The harness reads and drives verl as-is. It never edits `verl/` or any trainer code. If a bug fix needs source changes, that belongs in the verl repo, not here.
- **`verl/trainer/distillation/`** — knowledge-distillation training flows. The harness can bind to `examples/on_policy_distillation_trainer/` scripts via recipe search, but it does not yet model the distillation-specific intent fields (teacher model / temperature / distillation loss weights).
- **NPU / Ascend specifics.** `examples/ascend_extras/`, `requirements-npu.txt`, `docker/ascend/` exist; the harness auto-detects `DEVICE` via `python3 -c 'import torch_npu'` and otherwise treats NPU as a pass-through (the underlying recipe handles it). Site-specific NPU provisioning (HCCL env vars, `npu-smi` health checks beyond the basic probe) is not modelled.
- **Hyperparameter sweeps.** `examples/tuning/` exists in verl; this harness drives **one** set of hyperparameters per run. Multi-config sweeps are the user's job (or a future meta-harness over this one).
- **Long-running online inference services.** `run_generate` supports bounded batch generation; lifecycle management for a persistent public serving endpoint remains out of scope.
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
│  select_compute  │  decide target (local-direct|local-slurm|ssh-slurm) +
│                  │  estimate GPU budget from model size+algo (skills/gpu_budget)
│                  │  → right-size gpus-per-node (cap-aware); halt if over budget.
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
│ monitor_training │  arm a detached poller (status / tail / metrics→csv /
│                  │  scan OOM·NaN·NCCL·vLLM·preempt·divergence); agent re-engages
│                  │  on terminal sentinel / escalation / heartbeat → summarize.
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

Every state's `## Hand-off Points` is authoritative. `--no-hitl` skips ordinary pauses and records that escape in the run log. It does not skip the four always-on gates in `skills/global/scientific_principles.md`: generated preprocess approval, custom/shaped reward approval, sanity-rollout approval, and the cost gate when the estimated node-hours meet the configured threshold.

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
- **Workspace layout.** All run artefacts live under `runs/<run_id>/workspace/`. Canonical state areas are `intake/`, `recipe/`, `algorithm/`, `dataset/`, `reward/`, `compute/`, `env/`, `sanity/`, `job/`, `logs/`, `summary/`, `generate/`, and `eval/`; `states/finalize.md` declares every accepted terminal artefact. The training job's own outputs remain under the user-selected `output_dir`; the workspace records paths rather than copying checkpoints or Slurm logs.
- **Honesty over impressiveness.** If the training job OOM-crashes, the summary says so plainly. If only one step ran before a NaN, the summary reports the one step. The harness must never fabricate a metric, a checkpoint path, or a "success" verdict that didn't happen.
- **Cost awareness.** Training jobs can cost hundreds of GPU-hours. Before `launch_training` actually fires the job (if HITL is allowed), the agent must present the final launch command and the expected node-hours and ask the user to confirm.
- **Cancellation.** During `monitor_training`, if the user sends a stop signal (or `--cancel` is set externally), the agent issues `scancel` (Slurm) or kills the local process and writes a `cancelled` summary.
- **Reproducibility.** The harness records the exact training command, env-vars, seeds, and `git log -1` of the verl repo at launch time, so a second run can be re-launched verbatim.
- **American English in all written artefacts.**
