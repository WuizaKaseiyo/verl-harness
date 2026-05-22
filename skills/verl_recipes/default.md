verl_recipes skill — how to bind the user's intent to an actual verl launch path.

## What verl ships

The verl checkout has multiple locations for runnable training scripts. **Always verify the layout on disk before assuming** — verl evolves, and the on-disk truth wins. Common patterns observed:

1. **`<verl_root>/examples/<algorithm>_trainer/run_*.sh`** — the long-standing example recipes. Naming convention: `run_<model_slug>_<backend>.sh` (e.g., `run_qwen3_8b_fsdp.sh`, `run_qwen2_5_32b_fsdp.sh`, `run_deepseek_v3_671b_megatron.sh`). Each script accepts env-var overrides for model path, scale, batch size, learning rate, rollout config, total epochs.
2. **`<verl_root>/examples/sft/<dataset>/run_*.sh`** — SFT lives in a separate tree shaped by *dataset*, not algorithm. SFT does not go through `main_ppo`; it has its own trainer module (see "Module landscape" below).
3. **`<verl_root>/recipe/<name>/`** — newer / more curated recipes. **Layout varies between checkouts**: some versions nest `recipe/<algorithm>_trainer/`, others put recipes in a flat `recipe/<name>/` tree (e.g., one folder per dataset or per published paper). `ls <verl_root>/recipe/` before assuming.

Some algorithms only have entries in one location; some have both. Always enumerate.

### Module landscape (verified against `verl/trainer/`)

The harness must NOT assume a `main_<algorithm>.py` exists for every algorithm. Concretely, on a typical recent verl checkout (e.g., 0.8.0.dev), the only `verl.trainer.main_*` modules that exist are:

```
verl/trainer/main_ppo.py            ← PPO + the whole PPO-family RL algorithm set
verl/trainer/main_ppo_sync.py       ← synchronous-rollout variant of main_ppo
verl/trainer/main_eval.py           ← evaluation only
verl/trainer/main_generation_server.py
verl/trainer/sft_trainer.py         ← SFT (NOT main_sft; module name differs)
verl/trainer/sft_trainer_ray.py     ← SFT with Ray
```

So the rule for algorithm binding is:

- **PPO-family RL algorithms (grpo, rloo, remax, gae, reinforce_plus_plus, dapo, dppo, gpg, gspo, gdpo, gmpo, cispo, sapo, mtp, opo, otb, …)** → route through `verl.trainer.main_ppo` with `algorithm.adv_estimator=<algo>`. The authoritative list of accepted estimator names is the **`AdvantageEstimator` enum** at `verl/trainer/ppo/core_algos.py` (class definition; do not hardcode the list in the harness — read the enum). Custom advantage estimators registered via `@register_adv_est("<name>")` are also valid.
- **SFT** → use `verl.trainer.sft_trainer` (FSDP) or `verl.trainer.sft_trainer_ray` (Ray); choose by the user's scale (Ray for multi-node, FSDP for single-node).
- **An algorithm name the user types that is neither in the enum, nor maps to an SFT/eval/generation module, nor has a script under `examples/` or `recipe/`** → halt with a clear "no such trainer in this verl checkout" message.

If a shell script under `examples/<algorithm>_trainer/` or `recipe/<name>/` matches the request, **prefer it over the direct-module path** — recipes encode the recommended hyperparameters and rollout config for the (algo, model, backend) combo.

If no shell script matches but the algorithm is binding-able per the rules above, construct the launch command from the appropriate module entry point. verl uses Hydra-style overrides — every config field is exposed as `dotted.path=value`. Common ones:

- `data.train_files=<path>` / `data.val_files=<path>`
- `data.train_batch_size=<N>`
- `data.max_prompt_length=<N>` / `data.max_response_length=<N>`
- `actor_rollout_ref.model.path=<HF id or local path>`
- `critic.model.path=<...>` (PPO only)
- `algorithm.adv_estimator=<gae|grpo|...>` (PPO; defaults vary by trainer)
- `trainer.nnodes=<N>` / `trainer.n_gpus_per_node=<G>`
- `trainer.total_epochs=<E>`
- `trainer.default_local_dir=<output_dir>`
- `trainer.project_name=<wandb_project>` / `trainer.experiment_name=<run_name>`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=<TP>`
- `actor_rollout_ref.rollout.gpu_memory_utilization=<float>`

The exact set of fields a trainer accepts is in **`<verl_root>/verl/trainer/config/`** — load it before assembling the command. The layout is Hydra-hierarchical:

- `ppo_trainer.yaml` — the PPO-family entry config (used by all `algorithm.adv_estimator` values).
- `ppo_megatron_trainer.yaml` — Megatron backend variant.
- `sft_trainer_engine.yaml` — SFT engine config.
- `_generated_*.yaml` — composed snapshots (read-only; for reference only).
- Subdirectories `actor/`, `algorithm/`, `critic/`, `data/`, `engine/`, `model/`, `optim/`, `ref/`, `reward/`, `rollout/`, `profiler/`, … — Hydra config groups; each contains the validated field set for that subsection.

There is **no per-algorithm YAML** (no `grpo_trainer.yaml`, no `sft_trainer.yaml`); the algorithm is selected by the `algorithm.adv_estimator` override at launch time, not by a different config file. (Older or differently-laid-out verl checkouts may differ — verify with `ls <verl_root>/verl/trainer/config/`.)

## Scoring candidates

Given the user's `(algorithm, model, optional backend, optional scale)`, score each candidate script in both locations:

- **Algorithm match** is binary — only scripts whose filename or parent dir name contains `<algorithm>` count.
- **Model slug match.** Lowercase the user's `model` HF-id (`Qwen/Qwen3-4B` → `qwen3_4b`); lowercase the script name (`run_qwen3_4b_fsdp.sh` → `qwen3_4b_fsdp`). Match score = longest-common substring + bonuses for exact size match (`4b == 4b`, not `4b ~ 8b`). The user almost always wants the same model size; size mismatches are strong negatives.
- **Backend match.** If the user specified `fsdp` / `megatron` / `veomni`, require the script's name to contain that token. If unspecified, prefer `fsdp` (it is the most-common verl baseline).
- **Scale fit.** If the user gave `nodes` / `gpus_per_node`, the script's hard-coded defaults (NNODES, NDEVICES_PER_NODE) should be ≤ user's. A script defaulting to 8 nodes that the user wants to run on 1 node is workable (env-var override) but the closer match wins.

If exactly one candidate scores above a confidence threshold → pick it. If multiple → pause for the user. If none → fall back to direct-module invocation.

## Direct-module fallback

When no shell-script recipe matches, construct the launch command from the appropriate module entry per the "Module landscape" rules above. Steps:

1. **Decide the module.** Run:
   ```bash
   python -c "import importlib; importlib.import_module('verl.trainer.main_<algorithm>')"
   ```
   If it succeeds, use it (rare — usually only `main_ppo`/`main_ppo_sync`/`main_eval` succeed).

   Otherwise, check whether `<algorithm>` is a valid PPO-family `adv_estimator`:
   ```bash
   python -c "from verl.trainer.ppo.core_algos import AdvantageEstimator, ADV_ESTIMATOR_REGISTRY; \
              print(<algorithm> in {e.value for e in AdvantageEstimator} | set(ADV_ESTIMATOR_REGISTRY))"
   ```
   If `True`, the module is `verl.trainer.main_ppo` and the CLI must include `algorithm.adv_estimator=<algorithm>`.

   Otherwise, if `<algorithm> == "sft"`, the module is `verl.trainer.sft_trainer` (or `sft_trainer_ray` for multi-node). Note that SFT does not accept the PPO-family overrides below; its config surface is `sft_trainer_engine.yaml` + the `data`/`model`/`optim` Hydra groups.

   If none of the above: halt with "no such trainer in this verl checkout: <algorithm>".

2. Load `<verl_root>/verl/trainer/config/ppo_trainer.yaml` (or `sft_trainer_engine.yaml` for SFT) to discover the trainer's full set of accepted fields.
3. Construct the CLI:
   ```
   python -m verl.trainer.main_ppo \
     algorithm.adv_estimator=<algorithm> \
     data.train_files=<from prepare_data> \
     data.val_files=<from prepare_data> \
     data.train_batch_size=<user or default> \
     data.max_prompt_length=<user or default> \
     data.max_response_length=<user or default> \
     actor_rollout_ref.model.path=<user> \
     actor_rollout_ref.rollout.name=<vllm|sglang> \
     trainer.nnodes=<user or default> \
     trainer.n_gpus_per_node=<user or default> \
     trainer.total_epochs=<user or default> \
     trainer.default_local_dir=<output_dir> \
     trainer.project_name=<wandb_project, if any> \
     trainer.experiment_name=<wandb_run_name, if any>
   ```
   Add critic-related overrides for PPO; add reward-model overrides if relevant.
4. Present the constructed command to the user (HITL checkpoint in `locate_recipe`) before recording it.

## Output: `workspace/recipe/recipe.md`

```markdown
# Recipe

## Launch path
- type: script | module
- script_path: /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh   # if type=script
- module: verl.trainer.main_ppo                                       # if type=module (all PPO-family RL algorithms)

## Key hyperparameters

These are surfaced in the locate_recipe HITL display. The user may override any of them before recipe.md is finalised.

| Field | Recipe default | User override |
|---|---|---|
| actor_rollout_ref.actor.optim.lr | 1e-6 | — |
| critic.optim.lr | 1e-5 | — (n/a unless PPO) |
| actor_rollout_ref.actor.kl_loss_coef | 0.001 | — |
| actor_rollout_ref.actor.ppo_mini_batch_size | 16 | — |
| actor_rollout_ref.rollout.gpu_memory_utilization | 0.6 | — |
| actor_rollout_ref.rollout.tensor_model_parallel_size | 1 | — |
| actor_rollout_ref.rollout.n | 8 | — |
| data.max_response_length | 1024 | — |
| trainer.total_epochs | 1 | — |
| trainer.total_training_steps | (unset — runs to epochs end) | — |
| trainer.test_freq | 2 | — |
| trainer.save_freq | 10 | — |

## Resolved arguments

### Env-var overrides (script path)
- MODEL_PATH=Qwen/Qwen3-4B
- NNODES=1
- NDEVICES_PER_NODE=8
- TRAIN_BATCH_SIZE=1024
- TOTAL_EPOCHS=5

### CLI overrides (module path)
- algorithm.adv_estimator=grpo
- data.train_files=workspace/dataset/gsm8k/train.parquet
- data.val_files=workspace/dataset/gsm8k/test.parquet
- actor_rollout_ref.model.path=Qwen/Qwen3-4B
- trainer.nnodes=1
- trainer.n_gpus_per_node=8
- trainer.total_epochs=5
- trainer.default_local_dir=/opt/verl/outputs/<run_id>/

## Recipe defaults inherited (the user did not override these)
- ROLLOUT_TP=2
- ROLLOUT_GPU_MEM_UTIL=0.6
- ACTOR_LR=1e-6

## verl version
- commit: 1234abc Add foo bar
- branch: main
```

## Things you must not do

- Do not modify the verl recipe script. The harness reads it; it does not edit it. The recipe's behaviour comes from env-var overrides at launch time.
- Do not invent a trainer module name. If `verl.trainer.main_<algorithm>` does not import, halt — the user's algorithm name is wrong.
- Do not "pick a model that's close enough" silently. A mismatched model size will OOM or underperform; if no candidate scores above threshold, pause for the user.
