# sanity_rollout

## Description

Run the cheapest possible end-to-end test of the full **model + dataset + reward fn** stack before spending any real GPU hours on training. Load the model (via vllm or transformers), sample one response on row 0 of `train.parquet`, invoke the reward function on that response, display the result. Failure short-circuits to `finalize` with a `sanity_failed` deliverable — the harness will not proceed to a doomed training run.

This state is the cheapest catch-net the harness has for the most common "model + reward + data shape" mismatches that the FSM's earlier import / dry-run checks cannot detect: tokenizer template mismatch, ground-truth field absent or malformed, reward fn returns 0 for valid responses, RM tokenizer disagreement, model loads but the response shape isn't what the recipe expects.

Apply the matching `reward_*` skill (from `workspace/reward/reward_config.md`) for the reward-fn-invocation mechanics, and the matching `compute_*` skill (from `workspace/compute/compute_choice.md`) for *where* to run the load (locally on `local-direct`; via a short srun under the same slurm directives as the real training job on slurm targets).

**FSM ordering rationale:** sanity_rollout sits **after** `provision_env` (env is validated, `launch_env.sh` exists) and **before** `launch_training` (catches problems before sbatch'ing the real training job). It re-uses the same launch env, same worker resource shape, same model path — anything that breaks in the real launch except OOM-at-scale will surface here in ~1 minute instead of after a 30-minute slurm queue.

Concretely:

1. **Read** `workspace/intake/training_intent.md`, `workspace/recipe/recipe.md`, `workspace/dataset/dataset.md`, `workspace/reward/reward_config.md`, `workspace/compute/compute_choice.md`, `workspace/env/env_state.md`, `workspace/env/launch_env.sh`.

2. **Plan the load.** Decide what minimal artefacts to materialise:
   - The actor model (HF safetensors or local shards) loaded into vllm with `n=1`, `temperature=0.8`, `max_new_tokens=min(256, data.max_response_length)`.
   - The tokenizer.
   - For `reward_kind=model`: the RM tokenizer + the RM (loadable into memory; can use 8-bit if available).
   - For `reward_kind=custom|shaped`: the `compute_score.py` module — `importlib.import_module` it, confirm the callable is present.

3. **Probe environment.** Same probe pattern as `provision_env` step 1a (ROCR/CUDA collision check) — sanity_rollout uses the same worker that `launch_training` will, so reuse `workspace/env/launch_env.sh` to set up the env. `source $WS/env/launch_env.sh` before any python.

4. **Run the sanity case + multi-row distribution.** Use the worker-side probe template at `skills/reward_rule/templates/sanity_probe.py` (it's generic across reward kinds — it branches on `reward_kind` from `reward_config.md`). Invoke it with `WORKSPACE` env var pointing at the run's workspace dir:

   ```bash
   WORKSPACE=<workspace> python skills/reward_rule/templates/sanity_probe.py > workspace/sanity/probe_output.json
   ```

   The template:
   - reads paths from `workspace/intake/training_intent.md`, `dataset/dataset.md`, `reward/reward_config.md`
   - loads the actor model into vLLM with `n=1, temperature=0.8, max_tokens=max_response_length`
   - samples one response on each of rows 0..9 of `train.parquet`
   - invokes the reward fn (rule / custom / shaped); RM is caller-handled (this state must load the RM separately for `reward_kind=model`)
   - returns a JSON report with row-level traces + the distribution summary (min/median/max/non_zero_rate) + load time + a `verdict` ∈ `{green, fail}`

   **Fail condition baked into the template:** `non_zero_rate < 0.05` over 10 rows → verdict=`fail`. Reward fn is too strict (or wrong); training signal is noise.

5. **Verify** the JSON report and extract the values to copy into `sanity_report.md`. Record:
   - Reward histogram (min / p50 / max).
   - Non-zero rate. **Fail this state if the non-zero rate < 5%** — the reward fn is too strict (or wrong); training will be all-noise.
   - For `shaped`: per-component means.
   - Median response length; flag if ≥ 0.95 × `data.max_response_length` (truncation; the rollout never sees the end of responses).

6. **Always-on hand-off point — present the sanity report.** Display:
   - Row-0 prompt (first 200 chars), the model's response (full text), ground truth, the reward value (decomposed for shaped/custom).
   - The 10-row distribution.
   - GPU memory peak (so the user can audit whether the recipe's `gpu_memory_utilization` is realistic on this hardware).
   - Time taken (so the user sees what one rollout step actually costs).

   **Always-on**: this pause is not skipped by `--no-hitl`. Rationale: this is the only place before the cost-gate where a wrong reward fn can be caught for a few seconds of compute instead of hundreds of GPU-hours. Same family of always-on hand-offs as `generate_preprocess` step 6 and `configure_reward` for custom/shaped (see `skills/global/scientific_principles.md`).

7. **Write `workspace/sanity/sanity_report.md`** with the row-0 trace, the 10-row distribution, GPU-memory peak, and a verdict line (`green` | `warn` | `fail`). On `fail` (non-zero rate < 5%, OOM, or import error), short-circuit to `finalize` with `sanity_failed`.

## Skills

- skills/reward_rule
- skills/reward_model
- skills/reward_custom
- skills/reward_shaping
- skills/compute_local
- skills/compute_slurm
- skills/compute_ssh_slurm
- skills/builtin-tools
- skills/global

> Of the four `reward_*` skills, **read only the one matching `reward_kind`**. Of the three `compute_*` skills, **read only the one matching the compute target**. The others are listed for validator coverage.

## Hand-off Points

- **Sanity report approval.** Step 6. **Always-on**: cannot be skipped by `--no-hitl` (see `skills/global/scientific_principles.md`).

## Next States

### launch_training

**Condition:** `workspace/sanity/sanity_report.md` records verdict `green` or `warn` (with user acceptance of the warnings). The reward fn returned non-zero on at least 5% of the 10-row sample. No OOM at model load.

**Deliverables:**

- sanity_report: Row-0 trace (prompt + response + reward), 10-row distribution, GPU-memory peak, verdict.

### finalize

**Condition:** Sanity report verdict is `fail` — reward fn essentially never fires, model fails to load, OOM at model load, or the reward fn raises an exception on a valid response. Training would be doomed; the harness short-circuits.

**Deliverables:**

- sanity_failed: A `workspace/sanity/sanity_failed.md` recording what was tried and what broke. Common modes: tokenizer chat-template mismatch, ground_truth column empty on most rows, reward fn raises on empty response, RM tokenizer mismatch.
