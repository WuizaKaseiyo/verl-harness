verl_recipes skill — how to bind the user's intent to an actual verl launch path.

## What verl ships

The verl checkout has two parallel locations for runnable training scripts:

1. **`<verl_root>/examples/<algorithm>_trainer/run_*.sh`** — the long-standing example recipes. Naming convention: `run_<model_slug>_<backend>.sh` (e.g., `run_qwen3_8b_fsdp.sh`, `run_qwen2_5_32b_fsdp.sh`, `run_deepseek_v3_671b_megatron.sh`). Each script accepts env-var overrides for model path, scale, batch size, learning rate, rollout config, total epochs.
2. **`<verl_root>/recipe/<algorithm>_trainer/run_*.sh`** — newer / more curated recipes. Same naming convention. Tends to lag the examples but is more curated.

Some algorithms only have entries in one location; some have both. Always search both.

If neither has a script for the requested `(algorithm, model)`, fall back to invoking the trainer module directly: `python -m verl.trainer.main_<algorithm>` with explicit CLI overrides. verl uses Hydra-style overrides — every config field is exposed as `dotted.path=value`. Common ones:

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

The exact set of fields a trainer accepts is in `<verl_root>/verl/trainer/config/<algorithm>_*.yaml` — load that to validate overrides before assembling the command.

## Scoring candidates

Given the user's `(algorithm, model, optional backend, optional scale)`, score each candidate script in both locations:

- **Algorithm match** is binary — only scripts whose filename or parent dir name contains `<algorithm>` count.
- **Model slug match.** Lowercase the user's `model` HF-id (`Qwen/Qwen3-4B` → `qwen3_4b`); lowercase the script name (`run_qwen3_4b_fsdp.sh` → `qwen3_4b_fsdp`). Match score = longest-common substring + bonuses for exact size match (`4b == 4b`, not `4b ~ 8b`). The user almost always wants the same model size; size mismatches are strong negatives.
- **Backend match.** If the user specified `fsdp` / `megatron` / `veomni`, require the script's name to contain that token. If unspecified, prefer `fsdp` (it is the most-common verl baseline).
- **Scale fit.** If the user gave `nodes` / `gpus_per_node`, the script's hard-coded defaults (NNODES, NDEVICES_PER_NODE) should be ≤ user's. A script defaulting to 8 nodes that the user wants to run on 1 node is workable (env-var override) but the closer match wins.

If exactly one candidate scores above a confidence threshold → pick it. If multiple → pause for the user. If none → fall back to direct-module invocation.

## Direct-module fallback

When no shell-script recipe matches, construct the launch command directly. Steps:

1. Confirm `verl.trainer.main_<algorithm>` exists as a Python module: `python -c "import importlib; importlib.import_module('verl.trainer.main_<algorithm>')"`. If it doesn't, the user's algorithm name is wrong — halt with "no such trainer".
2. Load `<verl_root>/verl/trainer/config/<algorithm>_megatron_trainer.yaml` (or the matching FSDP config) to discover the trainer's full set of accepted fields.
3. Construct the CLI:
   ```
   python -m verl.trainer.main_<algorithm> \
     algorithm.adv_estimator=<inferred default> \
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
- module: verl.trainer.main_grpo                                      # if type=module

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
