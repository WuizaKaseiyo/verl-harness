# run_eval

## Description

Score an existing generations parquet using a reward function. Calls `python -m verl.trainer.main_eval`. CPU-only, no GPU provisioning needed.

Apply the `run_eval` skill (`skills/run_eval`) for the CLI surface + pitfalls. Compute provisioning is minimal — CPU-only slurm job (`--partition=cpu` if available) or just run locally — so `compute_select` / `provision_env` are largely skipped on the standalone entry.

This state is reachable from three entry paths:
- **Standalone**: `intake` with `goal: eval` — user has a generations parquet and a reward fn, wants the scores.
- **Chained from generate**: `run_generate` with `chain_eval: true` — generate then immediately score.
- **Chained from train** (future Phase 4): a train track followed by an automatic eval on the val set.

Concretely:

1. **Read** `workspace/intake/training_intent.md` for `generations_path`, `reward_fn_path`, `reward_fn_name`. If chained from `run_generate`, instead read `workspace/generate/generate_report.md` to discover the generations parquet path.

2. **Validate the parquet** has the four required columns (`prompt`, `responses`, `data_source`, `reward_model`). Halt if any is missing.

3. **Pre-test the reward fn** on row 0 of the generations parquet:
   ```python
   import importlib.util, pyarrow.parquet as pq
   spec = importlib.util.spec_from_file_location("reward_mod", reward_fn_path)
   mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
   row = pq.read_table(generations_path).slice(0, 1).to_pylist()[0]
   score = getattr(mod, reward_fn_name)(
       row["data_source"], row["responses"][0],
       row["reward_model"]["ground_truth"], row.get("extra_info")
   )
   print(f"row-0 score: {score}")
   ```
   If this raises, halt — the reward fn isn't compatible with the parquet shape.

4. **Hand-off point — confirm eval plan.** Present:
   - Generations parquet path + row count + `data_source` value counts
   - Reward fn path + callable name + row-0 sanity score
   - Compute target (local CPU process or slurm CPU partition)
   - Expected wall-clock (heuristic: rows × 0.01 sec for regex rewards, more for code-execution rewards)

   Skipped with `--no-hitl`.

5. **Assemble the launch command** (matching `verl/trainer/config/evaluation.yaml`):
   ```
   python -m verl.trainer.main_eval \
       data.path=<generations_path> \
       data.prompt_key=prompt \
       data.response_key=responses \
       data.data_source_key=data_source \
       data.reward_model_key=reward_model \
       custom_reward_function.path=<reward_fn_path> \
       custom_reward_function.name=<reward_fn_name> \
       ray_kwargs.ray_init.num_cpus=<num_cpus>
   ```
   On `local-direct` (or a CPU-only host like a slurm login node) just bash it. On slurm, sbatch a CPU job with `--cpus-per-task=<N> --mem=32G --time=00:30:00` and the command above.

6. **Capture stdout** — `main_eval` prints `print(metric_dict)` at the end. Parse with:
   ```python
   import re, ast
   m = re.search(r"\{['\"]test_score/.*?\}", stdout, re.DOTALL)
   metric_dict = ast.literal_eval(m.group(0)) if m else {}
   ```
   (`ast.literal_eval` is safe — the output is a Python dict literal.)

7. **Write `workspace/eval/eval_report.md`** per the `run_eval` skill template: source paths, scores verbatim, per-source distribution, compute used, sample rows (best / median / worst by reward).

## Skills

- skills/run_eval
- skills/reward_rule
- skills/reward_model
- skills/reward_custom
- skills/reward_shaping
- skills/training_monitor          # for the polling cadences + log-anomaly patterns
- skills/builtin-tools
- skills/global

> Read the matching `reward_*` skill if the reward fn was authored / configured by an earlier `configure_reward` state in the same workspace. Otherwise the four are reference material for understanding the function shape.

## Hand-off Points

- **Confirm eval plan.** Step 4. Skipped with `--no-hitl`.

## Next States

### finalize

**Condition:** `workspace/eval/eval_report.md` is written with parsed `test_score/<data_source>` values. Either standalone (eval succeeded) or chained-from-generate (both succeeded).

**Deliverables:**

- eval_report: Per-data_source scores, distribution stats, compute used, sample rows.

### finalize

**Condition:** Eval crashed (reward fn raised on a real row, parquet was malformed mid-iteration, Ray init failed). The harness short-circuits honestly rather than reporting a partial score.

**Deliverables:**

- eval_failed: A `workspace/eval/eval_failed.md` recording the row that broke + the reward fn's verbatim traceback.
