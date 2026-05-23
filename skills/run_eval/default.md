run_eval skill — offline scoring of an existing generations parquet via `verl.trainer.main_eval`. Grounded in the actual source at `verl/trainer/main_eval.py` and config at `verl/trainer/config/evaluation.yaml`.

## What this is (and what it isn't)

`main_eval` is a **CPU-only reward-fn invoker**. It reads a parquet that already contains generated responses, calls a reward function on each row (in parallel via Ray), and prints a `metric_dict` of `test_score/<data_source>: <mean_score>`. It does **not** run inference. It does **not** need a GPU. It does **not** load the model.

If the user wants to "evaluate my checkpoint", that's two steps: `run_generate` produces the parquet, `run_eval` scores it. This skill covers only the second step.

## Real CLI surface (from `verl/trainer/config/evaluation.yaml`)

```bash
python3 -m verl.trainer.main_eval \
    data.path=<generations_parquet> \
    data.prompt_key=prompt \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    custom_reward_function.path=<reward_fn.py> \
    custom_reward_function.name=compute_score \
    ray_kwargs.ray_init.num_cpus=<N>
```

All `data.*_key` fields default to the verl canonical column names; only override when scoring a non-canonical parquet.

## Input parquet shape

The parquet must have **all four** of:
- `prompt` (list of `{role, content}`) — used by some reward fns for context
- `responses` (list of strings; one per `rollout.n`) — the generated outputs
- `data_source` (string) — what `test_score/<key>` is keyed on
- `reward_model` (dict with `ground_truth`) — what `compute_score` reads as `ground_truth`

These are the columns `run_generate` writes (it preserves the input's `data_source` / `reward_model` and adds `responses`). If the user supplies a different parquet, validate the four columns are present before launching.

## Reward function interface

The skill at `skills/reward_custom/` documents the `compute_score(data_source, solution_str, ground_truth, extra_info)` signature in detail. For `run_eval`, three sources of the reward fn:

1. **From a previous train run's `workspace/reward/compute_score.py`** — most common when evaluating a model you trained with this harness. Path: `runs/<train_run_id>/workspace/reward/compute_score.py`.
2. **From verl's built-in `verl/utils/reward_score/<data_source>.py`** — when the dataset is a registered one (gsm8k, math, etc.) and you want the default scorer.
3. **User-supplied path** — if the user has a separate reward fn (e.g., the same one used in training but vetted independently).

Set both `custom_reward_function.path` and `custom_reward_function.name` (default `compute_score`).

## Output

`main_eval` prints a single `metric_dict` like:
```python
{'test_score/openai/gsm8k': 0.847, 'test_score/another_source': 0.612}
```
The harness captures stdout and parses this dict; per-source scores go into `workspace/eval/eval_report.md`.

There is **no parquet output** from `main_eval`. The scoring is one-shot; the dict is the deliverable.

## Compute footprint

CPU-only. The skill recommends:
- `ray_kwargs.ray_init.num_cpus=null` (use all CPUs) for free-standing eval on a workstation.
- `ray_kwargs.ray_init.num_cpus=<N>` for slurm — must match `--cpus-per-task` to avoid Ray over-subscribing.
- Wall-clock is dominated by the reward fn's complexity. A pure regex match is < 1 sec for 10k rows; a code-execution reward (run-the-generated-Python-and-check-stdout) is minutes-to-hours.

A slurm job for eval typically requests `--time=00:30:00 --cpus-per-task=16 --mem=32G --partition=cpu`. No `--gres=gpu`.

## Failure modes

- **Missing column.** If `responses` is absent (e.g., the user pointed at a prompts-only parquet by mistake), Ray crashes with KeyError. Validate before launch.
- **Reward fn raises.** A reward fn that doesn't handle an empty `responses[i]` (when `n>1` and one generation hit EOS at zero tokens) raises and kills the worker. Pre-test the fn at the equivalent of `sanity_rollout` step before submitting the full batch.
- **Multi-`data_source` confusion.** `test_score/<data_source>` is keyed verbatim; if the generations parquet was built from a mixed dataset (multiple `data_source` values), the report has one score per source. The user must know this — surface in the HITL pre-launch summary.
- **Ray init hang on Slurm.** If `num_cpus=null` is left on a slurm node, Ray sees the whole machine's CPUs (often > allocation), oversubscribes, and hangs. Always set `num_cpus` explicitly on slurm.

## Output: `workspace/eval/eval_report.md`

```markdown
# Eval report

## Source
- generations_parquet: <path>
- reward_fn: <path>::<callable_name>

## Scores (verbatim from main_eval stdout)
- test_score/openai/gsm8k: 0.847
- test_score/chess_fen_cycle: 0.612

## Distribution per data_source
- openai/gsm8k:        n=1319, mean=0.847, p50=0.91, p10=0.0, p90=1.0
- chess_fen_cycle:     n=2000, mean=0.612, p50=0.74, p10=0.0, p90=1.0

## Compute
- target: local-direct (cpu) | local-slurm (cpu partition)
- cpus: 16
- wallclock: 8 min 32 sec

## Sample (rows of interest)
- best row (rewarded 1.0): row_id=42, prompt=..., response=..., ground_truth=...
- median row: row_id=731, prompt=..., response=..., reward=0.5
- failing row (rewarded 0.0): row_id=88, prompt=..., response=..., ground_truth=...
```

## Things you must not do

- Do not request GPUs for eval — they're unused and waste cluster allocation.
- Do not score an SFT generations parquet using a PPO/GRPO rule reward without verifying the reward fn applies — SFT often doesn't have `reward_model.ground_truth` (just `response`), and the rule scorer will return 0 on every row, painting a false picture.
- Do not aggregate `test_score` across data sources by simple mean — the source distribution may be skewed. Report per-source and let the user weight as they want.
