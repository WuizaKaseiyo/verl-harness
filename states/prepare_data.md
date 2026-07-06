# prepare_data

## Description

Produce the train / validation parquet files that the recipe's `data.train_files` and `data.val_files` arguments will point at. Two paths: known datasets (verl ships a preprocess script for them) and unknown datasets (the harness writes one). Both end with parquet files on disk and a record of where they are.

Apply the `dataset_registry` skill (`skills/dataset_registry`).

Concretely:

1. **Read `workspace/intake/training_intent.md`** for the `dataset` field, AND **`workspace/algorithm/algorithm_config.md`** for the algorithm's dataset-column requirements (PPO/GRPO need `reward_model.ground_truth` if `reward_kind=rule`; SFT needs `response`; gdpo needs the per-component reward keys; distillation may need `teacher_response`).
2. **Branch on dataset type:**
   - **(a) Known verl-preprocessable name.** The dataset_registry skill maps names like `gsm8k`, `math_dataset`, `hellaswag`, `full_hh_rlhf`, `geo3k`, `aime2024_multiturn_w_tool`, `gsm8k_multiturn_sft`, etc. to their preprocess script under `<VERL_ROOT>/examples/data_preprocess/<dataset>.py`. Verify the script exists; run it via `python <script> --local_dir <workspace/dataset/<name>/>`.
   - **(b) Direct local parquet path.** The user already has parquet files. Validate they exist and contain the expected columns; record paths; skip preprocessing.
   - **(c) HuggingFace dataset id, but not in the registry.** Transition to `generate_preprocess` to author a custom preprocess script. After `generate_preprocess` returns, the script will be at `workspace/dataset/<name>/preprocess.py` and we run it the same way the registry path does.
3. **Verify outputs + algorithm-column validation.** After preprocessing runs, confirm that `workspace/dataset/<name>/train.parquet` and (if applicable) `workspace/dataset/<name>/test.parquet` (or `val.parquet`) exist and are non-empty. Use `python -c 'import pyarrow.parquet as pq; print(pq.read_table(...).num_rows)'` (or `parquet-tools` if installed) to record row counts.

   Then validate against `algorithm_config.md`'s column requirements:
   - PPO/GRPO with `reward_kind=rule` → row must have `reward_model.ground_truth` non-empty on ≥ 95% of rows.
   - SFT → row must have `response` (single-turn) or `messages_key` content with an assistant turn (multi-turn).
   - GDPO → if `gdpo_reward_keys` are set, the custom reward fn (from `configure_reward`) must return those keys; the dataset itself doesn't need them.
   - DPO (external trainer) / RM (external trainer) → `chosen` / `rejected` columns required.
   - Distillation (offline) → `teacher_response` column required.

   On a column mismatch, halt with a clear "dataset shape does not match algorithm requirements" error. Do not silently accept the run.
4. **HITL hand-off point — confirm prepared data.** Present:
   - dataset name + branch taken (a / b / c)
   - output paths and row counts
   - disk size of the prepared data
   - HF cache used (so the user can see what was downloaded into `$HF_HOME`)
   - **Row-0 sample, verbatim** (mandatory, not optional). Display:
     - `prompt` — the full chat-message list as it sits in the parquet, formatted readably
     - `reward_model.ground_truth` — verbatim
     - `response` — only for SFT data; verbatim
     - `extra_info` — verbatim
     - Column list, with any missing canonical column flagged red
   Ask the user to confirm before proceeding; if unsatisfied (e.g., wrong split was downloaded, prompt template malformed, ground_truth empty on most rows), they can ask to re-run with different args.
5. **Write `workspace/dataset/dataset.md`** recording the chosen branch, the script invoked, the resolved `train_files` / `val_files` paths (verl's recipe arguments will be patched to point here), row counts, and the dataset's git source if applicable.

## Skills

- skills/dataset_registry
- skills/dataset_autogen     # only consulted on the unknown-dataset path
- skills/builtin-tools
- skills/global

## Hand-off Points

- **Confirm prepared data.** After step 3, before transitioning. Skipped with `--no-hitl`.

## Next States

### generate_preprocess

**Condition:** The dataset is an HF dataset id (or otherwise unknown) and no preprocess script exists for it in the verl registry. (Branch (c) above.) After `generate_preprocess` returns, the dataset counts as known, so this branch cannot re-fire.

**Loop:** max_iterations: 1

**Deliverables:**

- dataset_intent: A note at `workspace/dataset/intent.md` describing the HF dataset id, the schema fields, the conversation format (single-turn / multi-turn / tool-calling), and the target reward signal — enough for `generate_preprocess` to author a verl-compatible preprocess script.

### configure_reward

**Condition:** Train and (when applicable) validation parquet files exist on disk; `workspace/dataset/dataset.md` records their paths, row counts, and the row-0 sample. (Branches (a), (b), or post-generate_preprocess path.) `prepare_data` always transitions here next — the four reward kinds (`rule`/`model`/`custom`/`shaped`) all flow through `configure_reward`, which handles each branch internally.

**Deliverables:**

- dataset: Path to `workspace/dataset/<name>/train.parquet` (and `test.parquet` or `val.parquet`), row counts, disk size, the source of the data (verl preprocess script, user-supplied parquet, or generated-from-HF-schema), and the **row-0 sample** (`prompt`, `reward_model.ground_truth`, `response` if SFT, `extra_info`). The recipe's `data.train_files` / `data.val_files` arguments are patched to these paths in `workspace/recipe/recipe.md` (or noted as patch-to-apply at launch time).
