reward_custom skill — `reward_kind: custom`. A user-supplied Python function called per response.

## When to use

- The task's correctness check is more complex than a regex / boxed-number extractor but still deterministic (run a unit test, validate JSON against schema, parse FEN and recompute board legality, execute a script and check stdout).
- Or: the user wants a shaped reward but prefers writing one Python file over composing verl's built-in shaping primitives.
- The function returns a scalar (or a dict of scalars; verl sums them).

## verl's interface contract

verl reads `reward.custom_reward_function.path=<py file>` + `reward.custom_reward_function.name=<callable name>`. The callable is imported and invoked once per rollout response with this signature:

```python
def compute_score(
    data_source: str,        # the parquet row's data_source column
    solution_str: str,       # the assistant's response (decoded text)
    ground_truth: str,       # the parquet row's reward_model.ground_truth
    extra_info: dict | None, # the parquet row's extra_info column
) -> float | dict:
    """Return a scalar reward, or a dict like {"score": 0.7, "format_bonus": 0.1}.
    verl sums dict values when the trainer expects a scalar.
    """
```

Return-type rules:
- `float` → used directly as the per-response reward.
- `dict[str, float]` → verl sums values for the per-response reward, AND logs each key separately under `reward/<key>/*` in the metric stream (visible in `progress.csv` and the dashboard).

The function is imported **on every rollout-worker process**. Keep imports cheap and side-effect-free at module load time. Do not open files / load models at module level; do it lazily inside the function with a module-level cache if needed.

## Authoring template

A working template lives at **`skills/reward_custom/templates/compute_score.py`** — copy it to `workspace/reward/compute_score.py` and adapt the four helper functions (`_match_final_answer`, `_canonical`, `_check_format`, `_length_penalty`) to the task. The template's structure is intentionally minimal: a single `compute_score(...)` entry that returns a dict, with task-specific logic factored into private helpers.

What the template demonstrates (and what your custom function must preserve):

- Signature: `compute_score(data_source, solution_str, ground_truth, extra_info)` returning `float` or `dict[str, float]`.
- Lazy imports — no `vllm`, `torch`, or heavyweight ML imports at module top level (they'd load on every rollout-worker spawn).
- Defensive parsing — empty `ground_truth`, empty `response`, and parse-failures all degrade gracefully to `0.0` rather than raising.
- A small `_canonical` normaliser for numeric equality (`"1/2"` == `"0.5"` after canonicalisation) — extend per task.
- A length-penalty helper that's a no-op for short responses and only kicks in over a soft budget.

Authoring loop:
1. Copy template to `workspace/reward/compute_score.py`.
2. Edit `_match_final_answer` for the task's correctness check (regex / parser / external call).
3. Edit `_check_format` for the task's expected output shape (or remove if the task has no format constraint).
4. Edit `_length_penalty` (or remove if length is not a concern).
5. Run sanity_rollout (next state) to verify on 10 real rows.

## CLI injection

```
reward.custom_reward_function.path=<workspace>/reward/compute_score.py
reward.custom_reward_function.name=compute_score
```

The harness places the file at `<workspace>/reward/compute_score.py` so it is co-located with the run artefacts and survives across resumes (no path rot when the user moves the verl tree).

## Self-check before committing

Mandatory `sanity_rollout` pass before `select_compute`:

1. Module imports without error (`python -m py_compile`).
2. The function is callable with the documented signature.
3. Run on row 0 of `train.parquet`:
   - tokenize prompt
   - generate one response with the actual model + sampler (vllm, `n=1`, `temperature=0.8`)
   - call `compute_score(data_source, response, ground_truth, extra_info)`
   - print the return value, with each component named
4. Run on rows 0..99; record min / median / max / non-zero-rate.
5. Verify `non_zero_rate >= 0.05` — if essentially all responses score 0, the function is too strict (or wrong); the training signal is noise. Surface the histogram and ask the user before launching.

## Pitfalls

- **Returning `None` or raising.** verl treats a missing return as 0; an exception kills the rollout worker. The function MUST be total — handle empty `response`, missing `ground_truth`, malformed parsing.
- **State across calls.** Module-level mutables (counters, caches) are per-worker, not shared. Don't use them for ground-truth-like data — read from `extra_info` or `ground_truth` instead.
- **External calls.** `subprocess.run` for a code-execution reward is allowed but expensive. Budget for it in compute estimates; consider a timeout (`timeout=5`).
- **Numerical scale.** Mixing a `correctness` of `{0, 1}` with a `length_penalty` of `[-20, 0]` swamps the correctness signal. Either scale components to a comparable range or return them as a dict and let `reward_shaping` (see that skill) compose.

## Things you must not do

- Do not write a custom function the user did not request. If `reward_kind=rule` and a built-in exists, use the built-in.
- Do not silently swallow exceptions inside the function. Return 0 explicitly with a comment; never `try/except Exception: pass`.
- Do not import verl from inside the function. The function runs in the rollout worker; it has its own import path and additional imports add cold-start latency.
