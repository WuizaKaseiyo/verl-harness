# generate_preprocess

## Description

Author a verl-compatible preprocess script for an HF dataset that doesn't have one in the verl registry. The output is a Python file at `workspace/dataset/<name>/preprocess.py` whose contract matches verl's other preprocess scripts (writes `train.parquet` and `test.parquet` to a `--local_dir`, with the columns the verl trainer expects). After this state writes the script, control returns to `prepare_data` to run it.

Apply the `dataset_autogen` skill (`skills/dataset_autogen`). That skill has the full schema rules for what columns a verl parquet must contain (depending on whether this is SFT, RM, or PPO/GRPO-style RL data) and the templates the preprocess script should follow.

Concretely:

1. **Read** `workspace/intake/training_intent.md` (for algorithm) and `workspace/dataset/intent.md` (the unknown-dataset note from `prepare_data`).
2. **Inspect the HF dataset schema.** Use `web.fetch` to retrieve the HF dataset card, and prefer `python -c "import datasets; d=datasets.load_dataset(<id>, split='train', streaming=True); print(next(iter(d)))"` to see one real example. Note: this is the moment to detect whether the dataset is conversational (chat turns), single-turn (prompt → response), tool-calling, multi-choice, math problem + final-answer, etc.
3. **Pick a template.** Per the dataset_autogen skill, the right verl preprocess template depends on (a) the trainer type from `training_intent.md` (`sft` vs `ppo`/`grpo`/etc.) and (b) the dataset shape (single-turn / multi-turn / tool / RM-pairs). Reference an existing verl preprocess script in the same category as the model template:
   - SFT, single-turn: model after `examples/data_preprocess/gsm8k.py`
   - SFT, multi-turn: model after `examples/data_preprocess/multiturn.py`
   - SFT, multi-turn with tools: model after `examples/data_preprocess/gsm8k_multiturn_sft.py`
   - PPO/GRPO, single-turn math: model after `examples/data_preprocess/math_dataset.py`
   - PPO/GRPO, multi-turn with tools: model after `examples/data_preprocess/gsm8k_multiturn_w_tool.py`
   - Reward model / preference pairs: model after `examples/data_preprocess/full_hh_rlhf.py`
4. **Write the preprocess script** to `workspace/dataset/<name>/preprocess.py`. The script must:
   - take `--local_dir` and `--hdfs_dir` arguments (matching verl's other preprocess scripts);
   - load the HF dataset by the user's id;
   - transform each example into the canonical verl row shape (`prompt`, `data_source`, `ability`, `reward_model.style`, `reward_model.ground_truth`, `extra_info.*` — exact fields depend on the trainer);
   - write `train.parquet` and `test.parquet` (or just `train.parquet` if no test split).
5. **Self-check** the script:
   - It parses with `python -m py_compile`.
   - The output columns it would produce match what verl's `data.<trainer>` config expects.
   - It does not invent any field — every field comes from the HF row or from a deterministic formula on HF row fields.
6. **HITL checkpoint** — show the generated script and the reasoning (which template, which template's columns map to which HF fields). Ask the user to approve or edit. Skipped with `--no-hitl` (default: proceed with the generated script).
7. **Set the return-to flag.** Note that `prepare_data` should now treat this dataset as if it were a known dataset, with `preprocess.py` as the script. Transition back to `prepare_data`.

## Skills

- skills/dataset_autogen
- skills/dataset_registry    # for the canonical column rules
- skills/builtin-tools
- skills/global

## Hand-off Points

- **Approve generated preprocess script.** Step 6. Skipped with `--no-hitl`.

## Next States

### prepare_data

**Condition:** `workspace/dataset/<name>/preprocess.py` exists, passes `python -m py_compile`, and `dataset.md` (or a sibling generate-note) records which template was chosen and the HF-field → verl-column mapping.

**Deliverables:**

- generated_preprocess: The authored `preprocess.py` plus a one-page rationale at `workspace/dataset/<name>/preprocess_rationale.md` explaining the template chosen, the field mapping, and any decisions made about reward signal / ground truth.
