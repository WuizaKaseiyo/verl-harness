# intake

## Description

Parse the user's training intent into a structured, complete training spec that the rest of the harness can consume. This is the only state where the harness solicits free-form input from the user; from here onward, every state reads `workspace/intake/training_intent.md` rather than asking the user again.

Apply the `intake` skill (`skills/intake`).

Concretely:

1. **Collect the intent.** The user typically describes their goal in one or two sentences: *"GRPO on gsm8k with Qwen3-4B on my local 8-GPU box"*, or *"PPO on a custom HF dataset (myorg/mydata) with Qwen2.5-7B on our slurm cluster, 2 nodes × 8 GPU each"*. The intake skill explains the canonical fields the harness needs.
2. **Resolve `VERL_ROOT`.** The absolute path to the verl checkout. From: (a) the invocation, (b) `VERL_HOME` env var, (c) ask the user. Confirm the path exists and looks like a verl checkout (has `verl/` source, `examples/`, `requirements.txt`).
3. **Normalise the fields.** Translate the user's prose into a structured training intent:
   - `algorithm` — the trainer name the user wants. Examples: `ppo`, `grpo`, `gspo`, `dpo`, `dapo`, `rloo`, `remax`, `sft`. The harness does not curate a registry; whatever the user names is what `locate_recipe` will go look for.
   - `model` — either a HuggingFace model id (e.g. `Qwen/Qwen3-4B`) or an absolute local path. Record which.
   - `dataset` — either a known verl-preprocessable name (`gsm8k`, `math_dataset`, `hellaswag`, `full_hh_rlhf`, `aime2024_multiturn_w_tool`, etc.) or an HF dataset id (`myorg/mydata`) or an absolute path to a parquet directory. The dataset_registry skill (used by `prepare_data`) decides the branch.
   - `compute_pref` — `auto` (let `select_compute` decide), `local-direct`, `local-slurm`, or `ssh-slurm`. If `auto`, leave it; `select_compute` will probe.
   - `scale` — number of nodes, GPUs per node, train batch size, mini-batch size, max prompt/response length. If the user did not specify, leave them blank and let `locate_recipe` adopt the recipe's defaults.
   - `output_dir` — absolute path where training will write its checkpoints / logs. If not provided, default to `<VERL_ROOT>/outputs/<run_id>/` and record that.
   - `wandb` — optional: `wandb.project`, `wandb.entity`, `wandb.run_name`. If unset, training launches with wandb disabled.
   - `hf_token` — required if pulling a gated model or dataset. Pull from `$HF_TOKEN` env var by default; ask if missing and the user's spec needs it.
4. **Confirm with the user** (HITL checkpoint). Show the normalised intent and ask: *"Is this what you want? Edit any field or confirm."*
5. **Write `workspace/intake/training_intent.md`** with the confirmed, structured fields. Every subsequent state reads this file.

## Skills

- skills/intake
- skills/builtin-tools
- skills/global

## Hand-off Points

- **Confirm normalised intent.** After step 4, present the fully-normalised training spec and pause for user confirmation. Skipped with `--no-hitl`.

## Next States

### locate_recipe

**Condition:** `workspace/intake/training_intent.md` exists with all required fields populated (algorithm, model, dataset, compute_pref, output_dir).

**Deliverables:**

- training_intent: The structured training spec — algorithm, model, dataset, compute_pref, scale knobs, output_dir, wandb config, hf_token-source — written as a key/value markdown document.
- verl_root: The confirmed absolute path to the verl checkout, recorded both as a field in training_intent.md and on its own at `workspace/intake/verl_root.txt` for downstream states that just need the path.
