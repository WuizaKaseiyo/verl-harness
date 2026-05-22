dataset_registry skill — known verl-preprocessable datasets + the canonical column rules.

## Registry (datasets verl ships a preprocess script for)

Located under `<verl_root>/examples/data_preprocess/`. The harness binds the user-supplied dataset name to the script as follows:

| Dataset name (user input)         | Preprocess script                                              | Output split(s)         | Trainer-side use      |
|-----------------------------------|----------------------------------------------------------------|-------------------------|-----------------------|
| `gsm8k`                           | `examples/data_preprocess/gsm8k.py`                            | train / test            | PPO/GRPO single-turn  |
| `math_dataset`                    | `examples/data_preprocess/math_dataset.py`                     | train / test            | PPO/GRPO single-turn  |
| `aime2024_multiturn_w_tool`       | `examples/data_preprocess/aime2024_multiturn_w_tool.py`        | train / test            | PPO/GRPO multi-turn + tool |
| `dapo_multiturn_w_tool`           | `examples/data_preprocess/dapo_multiturn_w_tool.py`            | train / test            | DAPO multi-turn + tool|
| `full_hh_rlhf`                    | `examples/data_preprocess/full_hh_rlhf.py`                     | train / test            | Reward modelling / preference pairs |
| `geo3k`                           | `examples/data_preprocess/geo3k.py`                            | train / test            | PPO/GRPO single-turn  |
| `geo3k_multiturn_w_tool`          | `examples/data_preprocess/geo3k_multiturn_w_tool.py`           | train / test            | PPO/GRPO multi-turn + tool |
| `gsm8k_multiturn_sft`             | `examples/data_preprocess/gsm8k_multiturn_sft.py`              | train / test            | SFT multi-turn        |
| `gsm8k_multiturn_w_tool`          | `examples/data_preprocess/gsm8k_multiturn_w_tool.py`           | train / test            | PPO/GRPO multi-turn + tool |
| `gsm8k_tool_agent_loop`           | `examples/data_preprocess/gsm8k_tool_agent_loop.py`            | train / test            | Tool-agent loop       |
| `hellaswag`                       | `examples/data_preprocess/hellaswag.py`                        | train / val             | PPO/GRPO single-turn  |
| `multiturn`                       | `examples/data_preprocess/multiturn.py`                        | train / test            | Multi-turn SFT        |
| `pokemon`                         | `examples/data_preprocess/pokemon.py`                          | train / test            | Tutorial / sanity     |
| `preprocess_search_r1_dataset`    | `examples/data_preprocess/preprocess_search_r1_dataset.py`     | train / test            | Search-R1 style       |
| `chess_fen_cycle`                 | `examples/data_preprocess/chess_fen_cycle.py`                  | train / test            | Chess-DuPO FEN cycle (rule reward; ground_truth = original FEN) |

The agent should **re-confirm the list against the actual files on disk** at `<verl_root>/examples/data_preprocess/` before binding — verl evolves, and the on-disk truth wins. The table above is the current snapshot; treat it as a hint, not as source-of-truth.

## Invocation pattern

Every verl preprocess script accepts at least:

- `--local_dir <output dir>` — where to write the resulting parquet files.
- `--hdfs_dir <hdfs path>` — optional; skip if not using HDFS.

Some accept `--local_dataset_path` (path to an already-downloaded raw dataset) so HF doesn't re-download. Use it when the user's HF cache already has the raw data.

Canonical run:

```bash
python <verl_root>/examples/data_preprocess/<name>.py \
  --local_dir <workspace>/dataset/<name>/
```

After completion, the directory contains `train.parquet` and (usually) `test.parquet`. Validate with:

```python
import pyarrow.parquet as pq
t = pq.read_table('<workspace>/dataset/<name>/train.parquet')
print('rows:', t.num_rows, 'cols:', t.column_names)
```

## Column conventions (what verl trainers expect)

All verl trainers expect parquet rows with these columns (names verbatim):

| Column          | Type   | Used by | Notes |
|-----------------|--------|---------|-------|
| `prompt`        | list[dict] | all | OpenAI-style chat messages: `[{"role":"user","content":"..."}]` |
| `data_source`   | string | logging | e.g., `"openai/gsm8k"` |
| `ability`       | string | logging | a tag like `"math"` or `"qa"` |
| `reward_model`  | dict   | PPO/GRPO etc. | `{"style": "rule" or "model", "ground_truth": "..."}` |
| `extra_info`    | dict   | any | optional bag of extras |

SFT-style data uses additional columns:

| Column      | Type | Notes |
|-------------|------|-------|
| `response`  | string (or list[dict] for multi-turn) | the target completion |

Reward-model / preference data uses:

| Column        | Type | Notes |
|---------------|------|-------|
| `chosen`      | list[dict] | preferred completion |
| `rejected`    | list[dict] | dispreferred completion |

These are the columns the user-supplied preprocess script (whether registry or auto-generated) must produce. The `dataset_autogen` skill enforces them for new datasets.

## Recording dataset state

After preprocessing succeeds, write `workspace/dataset/dataset.md`:

```markdown
# Dataset

## Source
- name: gsm8k                                  # user-supplied
- branch: registry                             # registry | local | autogen
- registry_script: <verl_root>/examples/data_preprocess/gsm8k.py
- hf_id: openai/gsm8k                          # discovered, if applicable

## Outputs
- train_files: workspace/dataset/gsm8k/train.parquet (rows: 7473, size: 4.2 MB)
- val_files: workspace/dataset/gsm8k/test.parquet (rows: 1319, size: 0.7 MB)

## Schema
- columns: [prompt, data_source, ability, reward_model, extra_info]
- prompt sample (row 0): [{"role":"user","content":"..."}]

## HF cache used
- $HF_HOME = /home/.../.cache/huggingface
```

## Things you must not do

- Do not invent registry entries. If the user names a dataset that isn't on disk under `examples/data_preprocess/`, treat it as unknown and route via `generate_preprocess`.
- Do not silently rename columns. If a preprocess script's output is missing the column verl expects, the dataset is wrong — report it.
- Do not write outside `workspace/dataset/`. The HF cache lives in `$HF_HOME` (the user's home, by convention); the preprocessed parquet lives in the workspace. Never touch the registry script.
