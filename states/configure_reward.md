# configure_reward

## Description

Resolve the reward function the trainer will use, given the user's `reward_kind` field. Inserted between `prepare_data` and `select_compute`. Four branches — one per reward kind — each driven by its own skill:

| `reward_kind` | Skill | Output |
|---|---|---|
| `rule` | `skills/reward_rule` | Confirm a built-in or recipe-baked `compute_score` exists; no extra files |
| `model` | `skills/reward_model` | Resolve the RM path / tokenizer / micro-batch size; verify RM is loadable |
| `custom` | `skills/reward_custom` | Author or accept a user-supplied `compute_score.py` |
| `shaped` | `skills/reward_shaping` (delegates to `reward_custom` for mechanics) | Author a dict-returning compose function with weights |

In all branches the state writes `workspace/reward/reward_config.md` so that `launch_training` can patch the Hydra CLI without re-deciding anything.

Apply the appropriate reward skill (the one matching `reward_kind` in `workspace/intake/training_intent.md`). Other reward skills are listed in `## Skills` for validator coverage and are not consulted on this run.

Concretely:

1. **Read** `workspace/intake/training_intent.md` (for `algorithm`, `reward_kind`, `model`), `workspace/dataset/dataset.md` (for `data_source` and the row-0 schema), and `workspace/recipe/recipe.md` (which may have already baked a custom reward path — common on research forks).

2. **Branch by `reward_kind`:**
   - **`rule`** — verify either:
     - `<verl_root>/verl/utils/reward_score/<data_source>.py` exists (verl built-in), OR
     - `recipe.md` already records `reward.custom_reward_function.path=<...>` pointing at a rule-based fn in the verl checkout (project-specific built-in).
     In either case, record the resolved fn path + callable name. No CLI injection beyond what the recipe already does.
   - **`model`** — apply `reward_model` skill. Resolve `reward_model.path` (HF id or local path; pre-download if HF and not cached). Confirm `reward_model.input_tokenizer` is the RM's tokenizer, not the actor's. Plan `micro_batch_size_per_gpu` against available GPU memory (recipe defaults rarely fit when RM is co-located with actor on a single GPU).
   - **`custom`** — apply `reward_custom` skill. If the user supplied an existing path (`reward.custom_reward_function.path` in their intent), accept it after a `python -m py_compile` check. Otherwise, author a new `compute_score.py` at `workspace/reward/compute_score.py` per the skill's template, using the trainer family + dataset shape to pick the function's body.
   - **`shaped`** — apply `reward_shaping` skill (which itself uses the `reward_custom` mechanism). Pick components and weights with the user, then author the dict-returning compose function at `workspace/reward/compute_score.py`.

3. **Always-on hand-off point — approve reward configuration.** Present:
   - The resolved branch and the reward fn / RM identity
   - For `custom` / `shaped`: the full generated `compute_score.py` plus the per-component weight table
   - For `model`: the RM path, tokenizer, micro-batch size, and expected GPU-memory footprint
   - For `rule`: the resolved built-in path and one row-0 trace (prompt → ground_truth → expected scoring path)
   - The CLI overrides that `launch_training` will splice in

   The HITL pause here is **always-on** when `reward_kind ∈ {custom, shaped}` because a wrong reward fn silently destroys whole runs (same rationale as `generate_preprocess`'s always-on rule). For `rule` and `model`, the pause is skipped by `--no-hitl`.

4. **Write `workspace/reward/reward_config.md`** per the templates in the four reward skills. The file must include a `## CLI injection` block that `launch_training` will splice into the assembled command.

## Skills

- skills/reward_rule
- skills/reward_model
- skills/reward_custom
- skills/reward_shaping
- skills/dataset_registry        # for the row schema the reward fn must read from
- skills/builtin-tools
- skills/global

> Of the four `reward_*` skills, **read only the one matching `reward_kind`**. The others are listed for validator coverage.

## Hand-off Points

- **Approve reward configuration.** Step 3. **Always-on for `reward_kind ∈ {custom, shaped}`** (cannot be skipped by `--no-hitl` — see `skills/global/scientific_principles.md`). Skipped with `--no-hitl` for `rule` and `model`.

## Next States

### select_compute

**Condition:** `workspace/reward/reward_config.md` is written with a resolved reward kind and (for custom/shaped) the `compute_score.py` exists + passes `py_compile`.

**Deliverables:**

- reward_config: The chosen `reward_kind`, the resolved fn path / RM path, the CLI overrides for `launch_training`, and (for custom/shaped) the authored `compute_score.py` plus its rationale.
