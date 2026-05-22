# configure_algorithm

## Description

Resolve algorithm-specific configuration the trainer will use. Inserted between `locate_recipe` and `prepare_data`. This is where the harness applies an algorithm's specific knowledge — what knobs matter, what failure modes look like, what's a valid range — *before* the data is prepped, so the data shape can be validated against the algorithm's requirements.

Apply the matching `algo_*` skill from `workspace/intake/training_intent.md`'s `algorithm` field. The mapping:

| `algorithm` (user input) | Applied skill | Notes |
|---|---|---|
| `ppo` | `skills/algo_ppo` | Only adv_estimator that uses a critic |
| `grpo`, `grpo_passk`, `grpo_vectorized` | `skills/algo_grpo` | Group-based; no critic |
| `gdpo` | `skills/algo_dpo` (the gdpo branch within that skill) | Per-component reward decoupling — verl realises this as a PPO-family adv_estimator |
| `dpo` | `skills/algo_dpo` | **Halt** (classic pair-likelihood DPO not first-class in this verl) |
| `sft` | `skills/algo_sft` | Different trainer module entirely (`verl.trainer.sft_trainer`) |
| `rm`, `reward_model` | `skills/algo_rm` | **Halt** (RM training not first-class in this verl) |
| `distill`, `distillation`, `on_policy_distillation` | `skills/algo_distill` | `main_ppo` + `distillation.*` namespace |
| `rloo`, `remax`, `gpg`, `opo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `gmpo`, `dppo`, `cispo`, `sapo`, `mtp`, `otb`, `optimal_token_baseline`, `tir_optimal_token_baseline`, any other `adv_estimator` registered via `@register_adv_est` | **no dedicated skill yet** — apply `skills/algo_ppo` for the shared loss/KL knobs (clip, entropy, kl_loss_coef) plus the chosen recipe's defaults verbatim; do **not** apply `algo_grpo` because group-statistics knobs (`rollout.n`, `norm_adv_by_std_in_grpo`) are not relevant to single-rollout REINFORCE-family algorithms | Document this explicitly in `algorithm_config.md`: "no algo-specific skill — using recipe defaults + algo_ppo for shared knobs". Future work: add `algo_reinforce_family` and `algo_rloo` as Phase 2.1 follow-ups when a real user case demands them. |

Other algorithm names the user types are not blocked here — they pass through to `locate_recipe`'s recipe search; if a script matches by name (e.g., a research fork's custom algorithm with its own `examples/<name>_trainer/`), this state surfaces whichever knobs the recipe exposes via env vars + lets the user override them, without a dedicated skill.

**Honesty over fake coverage.** When an algorithm has no dedicated skill (the last row of the dispatch table), the state explicitly records this in `algorithm_config.md`:

```markdown
## Algo-specific skill
- skill: (none — no dedicated skill for `<algorithm>` yet)
- fallback: skills/algo_ppo (for shared loss/KL knobs only)
- recipe-side defaults: surfaced verbatim from recipe.md `## Key hyperparameters`
- known limitation: group-statistics / SFT-specific / distillation-specific knobs are NOT being curated for this algorithm; the user is responsible for any algo-specific tuning beyond what the recipe sets.
```

This prevents the agent from silently pretending a non-matching skill applies — the past version of this state mapped `rloo`/`remax`/`gpg` into `algo_grpo` which has group-style content irrelevant to those algorithms.

Concretely:

1. **Read** `workspace/intake/training_intent.md` (for `algorithm`, `algorithm`-specific intake fields if any), `workspace/recipe/recipe.md` (for the resolved launch path + the `## Key hyperparameters` block written by `locate_recipe`).

2. **Dispatch.** Pick the algo skill per the mapping table above. If the algorithm is in the "no first-class trainer" set (`dpo`, `rm`), call the matching skill, which itself decides whether to halt or to redirect (gdpo / external trainer / pre-trained RM use).

3. **Surface knobs in HITL.** Display the skill's "Knobs to surface in `configure_algorithm`" table with the recipe's current default values. The user may override any value. Algorithm-specific knobs add to (not replace) the `## Key hyperparameters` block in `recipe.md` — the union is what `launch_training` will splice into the CLI.

4. **Validate against recipe constraints.** If the user picked `algorithm=ppo` but the chosen recipe doesn't bind a critic (e.g., the recipe was tagged grpo and has no `critic.*` directives), halt — recipe and algorithm disagree. Same for SFT picked with a PPO recipe, etc.

5. **Write `workspace/algorithm/algorithm_config.md`** with:
   - The applied skill (`skills/algo_<name>`)
   - The full knob table with recipe-default and user-override columns
   - Algorithm-specific dataset column requirements (e.g., DPO needs `chosen`/`rejected`; SFT needs `response`; PPO/GRPO needs `reward_model.ground_truth` for rule rewards; distillation may need `teacher_response`)
   - The CLI overrides to splice in (matching the skill's "CLI injection" block)
   - A `## Halt condition` block if the skill decided to halt (dpo / rm without a first-class trainer)

6. **Hand-off point — confirm algorithm configuration.** Present the knob table + dataset-shape requirements + CLI overrides. Skipped with `--no-hitl` for the supported algorithms; **always-on** when the skill emits a halt condition (the user must explicitly accept the halt or redirect).

## Skills

- skills/algo_ppo
- skills/algo_grpo
- skills/algo_dpo
- skills/algo_sft
- skills/algo_rm
- skills/algo_distill
- skills/verl_recipes              # for the recipe-side knob defaults
- skills/builtin-tools
- skills/global

> Of the six `algo_*` skills, **read only the one matching `algorithm`**. The others are listed for validator coverage and are not consulted on this run.

## Hand-off Points

- **Confirm algorithm configuration.** Step 6. Skipped with `--no-hitl` for supported algorithms (ppo/grpo-family/sft/distill). **Always-on** when the matching skill emits a halt condition (dpo/rm without first-class trainer) — the user must accept the halt or pick a redirect.

## Next States

### prepare_data

**Condition:** `workspace/algorithm/algorithm_config.md` is written, the chosen algorithm has a working trainer binding in this verl, and the user has not requested a halt. `prepare_data` reads the algorithm config to validate the dataset's columns match the algorithm's requirements (e.g., DPO/gdpo needs pair data; SFT needs `response`; etc.).

**Deliverables:**

- algorithm_config: The applied algo skill, the knob table, dataset-column requirements, CLI overrides for `launch_training`.

### finalize

**Condition:** The chosen algorithm has no first-class trainer in this verl (dpo / rm) AND the user did not accept a redirect (gdpo / external trainer / pre-trained RM use). The harness short-circuits honestly rather than driving a missing trainer.

**Deliverables:**

- algorithm_unsupported: A `workspace/algorithm/algorithm_unsupported.md` recording which algorithm was requested, what's missing in this verl, and the redirects the user declined.
