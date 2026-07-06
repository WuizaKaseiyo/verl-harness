# intake

## Description

Parse the user's training intent into a structured, complete training spec that the rest of the harness can consume. This is the only state where the harness solicits free-form input from the user; from here onward, every state reads `workspace/intake/training_intent.md` rather than asking the user again.

Apply the `intake` skill (`skills/intake`).

Concretely:

1. **Detect re-attach / resume opportunity** *(before any user input)*. If `runs/` contains a workspace whose `meta.json.status` is `"running"`, OR the user explicitly named an existing run as the target, branch out of the train track:
   - **`meta.json.status == "running"` + `state_log.md` last entry is `monitor_training`** → the previous session got disconnected mid-poll. Set `goal: resume_monitor` and transition straight to `monitor_training` (it re-reads everything from workspace on each iteration; safe to re-enter cold).
   - **`meta.json.status ∈ {"crashed", "preempted"}`** AND the user asks to resume → set `goal: resume_train`, read the existing workspace's `recipe.md` + `compute_choice.md` + `launch_env.sh`, transition straight to `launch_training` with the resume-mode CLI patch (see step 3's `resume_mode`).
   - **Otherwise** → continue with the fresh-train flow below.

   If the user is starting a fresh run rather than re-attaching, also pick the lifecycle goal:
   - `goal: train` (default — the full train track described in this state's downstream)
   - `goal: generate` — produce a generations parquet from an existing checkpoint (no training; jumps to `run_generate`)
   - `goal: eval` — score an existing generations parquet (no training, no GPU; jumps to `run_eval`)

2. **Collect the intent.** The user typically describes their goal in one or two sentences: *"GRPO on gsm8k with Qwen3-4B on my local 8-GPU box"*, or *"PPO on a custom HF dataset (myorg/mydata) with Qwen2.5-7B on our slurm cluster, 2 nodes × 8 GPU each"*. For `goal: generate` or `goal: eval` the intent is shaped differently (checkpoint path + prompts path; or generations path + reward fn). The intake skill explains the canonical fields per goal.
3. **Resolve `VERL_ROOT`.** The absolute path to the verl checkout. From: (a) the invocation, (b) `VERL_HOME` env var, (c) ask the user. Confirm the path exists and looks like a verl checkout (has `verl/` source, `examples/`, `requirements.txt`).
4. **Normalise the fields.** Translate the user's prose into a structured training intent:
   - `goal` — `train` (default) | `resume_monitor` | `resume_train` | `generate` | `eval`. Decides the FSM dispatch in the `## Next States` block.
   - `resume_mode` — only meaningful when `goal: resume_train`. Values: `auto` (default; verl scans `<output_dir>` for the latest `global_step_<N>/` and resumes there) or `resume_path` with an explicit `resume_from_path: <ckpt path>`. Mapped to the verl CLI fields `+trainer.resume_mode=...` and `+trainer.resume_from_path=...`.
   - `algorithm` — the trainer name the user wants. Examples: `ppo`, `grpo`, `gspo`, `dpo`, `dapo`, `rloo`, `remax`, `sft`. The harness does not curate a registry; whatever the user names is what `locate_recipe` will go look for. (Not used for `goal: generate` or `goal: eval` — those are post-train tracks.)
   - `model` — either a HuggingFace model id (e.g. `Qwen/Qwen3-4B`) or an absolute local path. Record which.
   - `dataset` — either a known verl-preprocessable name (`gsm8k`, `math_dataset`, `hellaswag`, `full_hh_rlhf`, `aime2024_multiturn_w_tool`, etc.) or an HF dataset id (`myorg/mydata`) or an absolute path to a parquet directory. The dataset_registry skill (used by `prepare_data`) decides the branch.
   - `compute_pref` — `auto` (let `select_compute` decide), `local-direct`, `local-slurm`, or `ssh-slurm`. If `auto`, leave it; `select_compute` will probe.
   - `scale` — number of nodes, GPUs per node, train batch size, mini-batch size, max prompt/response length, and `gpu_cap` (the max GPUs the account may request in one allocation — clusters commonly cap this, e.g. 4). If the user did not specify nodes / gpus_per_node, leave them blank: `select_compute` will **estimate** the GPU count from model size + algorithm via `skills/gpu_budget` (capped by `gpu_cap`), rather than inheriting a guess. Other scale knobs fall back to the recipe defaults in `locate_recipe`.
   - `output_dir` — absolute path where training will write its checkpoints / logs. If not provided, default to `<VERL_ROOT>/outputs/<run_id>/` and record that.
   - `wandb` — optional: `wandb.project`, `wandb.entity`, `wandb.run_name`. If unset, training launches with wandb disabled.
   - `refine` — optional block; its presence opts the run into the bounded closed-loop refinement stage (`reflect`, entered after `summarize`). Fields: `refine.target_metric` (a `workspace/logs/progress.csv` column or a val metric the trainer logs, e.g. `val-core/openai/gsm8k/acc/mean@1`), `refine.target_value` (the number to reach), `refine.max_iterations` (≥ 1; values above the FSM bound declared on the `reflect → configure_algorithm` transition are clamped to that bound, and the clamp is recorded in `loop_state.json`). If the user asks for refinement but any field is unset, ask — never guess a target. Omit the block entirely when the user did not ask for refinement: its absence is what routes `summarize → finalize` directly.
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

The transition fires per `goal`. Exactly one of the five branches below applies.

### locate_recipe

**Condition:** `goal: train` (default). `workspace/intake/training_intent.md` exists with all required fields populated (algorithm, model, dataset, compute_pref, output_dir).

**Deliverables:**

- training_intent: The structured training spec — goal, algorithm, model, dataset, compute_pref, scale knobs, output_dir, wandb config, optional refine block, hf_token-source — written as a key/value markdown document.
- verl_root: The confirmed absolute path to the verl checkout, recorded both as a field in training_intent.md and on its own at `workspace/intake/verl_root.txt` for downstream states that just need the path.

### monitor_training

**Condition:** `goal: resume_monitor` — detected in step 1 (the previous session disconnected mid-poll; `meta.json.status == "running"` and last `state_log.md` entry is `monitor_training`). The existing workspace is intact; no FSM re-entry needed beyond appending the `state_log.md` line for the re-attach.

**Deliverables:**

- resume_marker: A line in `state_log.md` annotating the re-attach: `- [<ISO8601>] -- re-attached to in-flight monitor_training, jobid=<jobid> --`.

### launch_training

**Condition:** `goal: resume_train` — user explicitly asks to resume a preempted/crashed run from an existing workspace. The harness reads the existing `recipe.md`, `compute_choice.md`, `env_state.md`, `launch_env.sh` AS-IS (no re-decision), and `launch_training` re-assembles the command with `+trainer.resume_mode=<auto|resume_path>` (and `+trainer.resume_from_path=<ckpt>` when applicable) appended.

**Deliverables:**

- resume_intent: A `workspace/intake/resume_intent.md` recording: source run_id, resume_mode (auto / resume_path), resume_from_path (if explicit), and which artefacts from the source workspace will be re-used verbatim.

### run_generate

**Condition:** `goal: generate` — produce a generations parquet from a checkpoint + prompts parquet. No training, no recipe selection. `training_intent.md` carries: `model_path` (the checkpoint to generate from), `prompts_path` (the parquet of prompts), `sampling_params` (n / temperature / top_p / top_k / max_tokens), `output_path` (where to write the generations parquet).

**Deliverables:**

- generate_intent: `training_intent.md` containing only the generate-track fields (`model_path`, `prompts_path`, `sampling_params`, `output_path`, optional `chain_eval: true` to also transition to `run_eval` after).

### run_eval

**Condition:** `goal: eval` — score an existing generations parquet using a reward function. CPU-only; no GPU provisioning. `training_intent.md` carries: `generations_path` (parquet with prompts + responses + ground truth), `reward_fn_path` (Python file with `compute_score`), `reward_fn_name` (default `compute_score`).

**Deliverables:**

- eval_intent: `training_intent.md` containing only the eval-track fields (`generations_path`, `reward_fn_path`, `reward_fn_name`).
