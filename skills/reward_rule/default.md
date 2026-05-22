reward_rule skill — `reward_kind: rule`. Deterministic, dataset-row-derived rewards. The simplest and lowest-risk reward kind.

## When to use

- The dataset has a `reward_model.style="rule"` column on every row (this is the convention `dataset_registry` documents and `dataset_autogen` produces).
- A correct response is checkable by a deterministic function from the ground truth: math final answer, multiple-choice letter, exact string match, regex, JSON-schema validation, etc.
- You do NOT need a learned reward model.

Most known datasets in `dataset_registry` (`gsm8k`, `math_dataset`, `geo3k`, `aime2024_multiturn_w_tool`, `chess_fen_cycle`, …) ship rule rewards.

## How verl realises it

When `reward_model.style="rule"` on a row, verl invokes the **`reward.compute_score`** entry — a callable resolved from one of:

1. **Built-in default** — verl bundles `compute_score` implementations under `verl/utils/reward_score/<dataset_name>.py`. The function's signature is `compute_score(data_source, solution_str, ground_truth, extra_info) -> float | dict`. If `data_source` matches one of the bundled rewards, verl picks it automatically. Examples on a typical checkout:
   - `verl/utils/reward_score/gsm8k.py`
   - `verl/utils/reward_score/math.py`
   - `verl/utils/reward_score/hellaswag.py`
   - `verl/utils/reward_score/full_hh_rlhf.py`
2. **Custom override** — if the user wants to override (e.g., stricter regex), they switch `reward_kind` to `custom` and pass `reward.custom_reward_function.path` (see `reward_custom` skill).

For `reward_kind: rule`, the harness:
- Verifies that `<verl_root>/verl/utils/reward_score/<data_source>.py` exists OR that the recipe's `reward.custom_reward_function.path` already points to one (in which case the recipe baked in a project-specific rule reward — common on research forks like the chess checkout).
- Writes nothing extra into the CLI; the rule reward is already wired via the dataset's `reward_model.style` column.

## Pitfalls

- **Regex specificity.** Built-in math reward fns use boxed-answer regexes (`\\boxed{...}`). If the model's response format diverges (e.g., bare numbers, fractions vs decimals), the reward returns 0 even on correct answers. Mitigation: run `sanity_rollout` (which runs the actual reward fn on a real model output) before committing 50 GPU·hours.
- **Ground-truth quality.** `reward_model.ground_truth` was extracted by the preprocess script; verify it on row-0 (see `prepare_data` step 4). A dataset with empty or malformed ground_truth fields silently produces all-zero rewards.
- **Multi-turn rule rewards.** For multi-turn datasets, the reward typically applies only to the *final* assistant turn. Verify the preprocess script captures the final answer in the canonical position; see `gsm8k_multiturn_w_tool.py` for the reference shape.

## Configuration in `workspace/reward/reward_config.md`

```markdown
# Reward config

## Kind
rule

## Function source
- built_in_path: <verl_root>/verl/utils/reward_score/<data_source>.py    # ONE of these is set
- recipe_baked_path: <verl_root>/verl/utils/reward_score/chess_fen_cycle.py  # if the recipe wired it via reward.custom_reward_function.path
- callable_name: compute_score                                            # default for built-in; recipes may override

## Ground-truth check
- column: reward_model.ground_truth
- row-0 sample: "72"                                                      # verbatim from prepare_data step 4
- non-empty rate: 100% (sampled 100 rows)                                 # mandatory; fail this state if < 95%

## CLI injection
(none — verl reads reward_model.style from the parquet row directly)
```

## Things you must not do

- Do not silently substitute a `model` reward when the user asked for `rule`. If `reward_model.style` is rule but the recipe injects `reward_model.path=<HF id>`, that is a recipe bug — surface it.
- Do not edit the built-in `compute_score` functions in `verl/utils/reward_score/`. The verl source is read-only from the harness's perspective. If a built-in is wrong for the task, escalate to `reward_custom`.
