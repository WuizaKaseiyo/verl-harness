# configure_algorithm

## Description

Resolve algorithm-specific configuration the trainer will use. Inserted between `locate_recipe` and `prepare_data`. This is where the harness applies an algorithm's specific knowledge — what knobs matter, what failure modes look like, what's a valid range — *before* the data is prepped, so the data shape can be validated against the algorithm's requirements.

### verl has two orthogonal algorithm axes — resolve both

Every RL algorithm runs through `verl.trainer.main_ppo`, configured along **two independent axes**. A named "algorithm" is often a *pair*, not a single estimator:

- **Axis 1 — advantage estimator** (`algorithm.adv_estimator=<name>`). Legal set = the `@register_adv_est` registry in `verl/trainer/ppo/core_algos.py` (currently `gae, grpo, grpo_passk, grpo_vectorized, gdpo, rloo, rloo_vectorized, opo, reinforce_plus_plus, reinforce_plus_plus_baseline, remax, gpg, optimal_token_baseline, tir_optimal_token_baseline`).
- **Axis 2 — policy loss mode** (`actor_rollout_ref.actor.policy_loss.loss_mode=<name>`). Legal set = the `@register_policy_loss` registry in the same file (currently `vanilla` (default), `gspo, cispo, geo_mean, sapo, gpg, dppo_tv, dppo_kl, clip_cov, kl_cov, bypass_mode`).

**Do not hardcode either list** — read both registries off the verl checkout at run time (the same "no curated trainer registry" rule the harness applies everywhere), and **prefer reading the actual `adv_estimator=` and `policy_loss.loss_mode=` values straight out of the recipe** `locate_recipe` already matched (the recipe script is the source of truth; the registries are for validation + fallback). E.g. `examples/gspo_trainer/run_*.sh` sets `adv_estimator=grpo policy_loss.loss_mode=gspo`.

The mapping (resolve the user's `algorithm` to the axis pair + skill):

| `algorithm` (user input) | adv_estimator | policy_loss.loss_mode | Applied skill | Notes |
|---|---|---|---|---|
| `ppo` | `gae` | `vanilla` | `skills/algo_ppo` | Only estimator that uses a critic |
| `grpo`, `grpo_passk`, `grpo_vectorized` | `grpo*` | `vanilla` | `skills/algo_grpo` | Group-based; no critic |
| `gspo` | `grpo` | `gspo` | `skills/algo_grpo` + loss_mode knob | Group Sequence PO (Qwen) |
| `cispo` | `grpo` | `cispo` | `skills/algo_grpo` + loss_mode knob | MiniMax CISPO |
| `gmpo` | `grpo` | `geo_mean` | `skills/algo_grpo` + loss_mode knob | Geometric-Mean PO |
| `sapo` | `grpo` | `sapo` | `skills/algo_grpo` + loss_mode knob | + `policy_loss.tau_pos`/`tau_neg` knobs (read from recipe) |
| `dppo` | `grpo` | `dppo_tv` or `dppo_kl` | `skills/algo_grpo` + loss_mode knob | Decoupled PPO; read *which* variant from the recipe |
| `gdpo` | `gdpo` | `vanilla` | `skills/algo_dpo` (the gdpo branch) | Group reward-Decoupled PO — **not** classic DPO; a real adv_estimator |
| `rloo`, `rloo_vectorized`, `remax`, `gpg`, `opo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `otb`/`optimal_token_baseline`, `tir_optimal_token_baseline` | (same-named estimator) | `vanilla` unless the recipe sets one | **no dedicated skill** — apply `skills/algo_ppo` for shared loss/KL knobs (clip, entropy, kl_loss_coef) + recipe defaults verbatim | Estimator-axis variants. Document the no-skill fallback in `algorithm_config.md`. **OTB family (`optimal_token_baseline`, `tir_optimal_token_baseline`) additionally requires `actor_rollout_ref.actor.calculate_sum_pi_squared=True` — without it the trainer asserts `Step-dependent optimal baseline requires sum_pi_squared from actor` at the first PPO step. The harness must auto-inject this when the estimator is in the OTB family.** |
| `mtp` | `grpo` | (from recipe) | `skills/algo_grpo` group knobs + recipe defaults | Multi-Token Prediction is a model/training feature layered on a grpo run (fully-async megatron) |
| `dpo` | — | — | `skills/algo_dpo` | **Halt** (classic pair-likelihood DPO not first-class in this verl) |
| `sft` | — | — | `skills/algo_sft` | Different trainer module entirely (`verl.trainer.sft_trainer` / `sft_trainer_ray`) |
| `rm`, `reward_model` | — | — | `skills/algo_rm` | **Halt** (RM training not first-class in this verl) |
| `distill`, `distillation`, `on_policy_distillation` | (from recipe) | — | `skills/algo_distill` | `main_ppo` + `distillation.*` namespace |

Note the correction from older versions: the loss-mode variants (`gspo`/`cispo`/`gmpo`/`sapo`/`dppo`) **do** use `algo_grpo` — they run `adv_estimator=grpo`, so the group-statistics knobs (`rollout.n`, `norm_adv_by_std_in_grpo`) genuinely apply — *plus* their loss-mode-specific knob(s), which no dedicated skill curates yet (surface them from the recipe + the registry signature). They are not in the "no skill / algo_ppo fallback" bucket.

Other algorithm names the user types are not blocked here — they pass through to `locate_recipe`'s recipe search; if a script matches by name (e.g., a research fork's custom algorithm with its own `examples/<name>_trainer/`), this state surfaces whichever knobs the recipe exposes via env vars + lets the user override them, without a dedicated skill.

**Honesty over fake coverage.** When an algorithm has no dedicated skill (the estimator-variant row), the state explicitly records this in `algorithm_config.md`:

```markdown
## Algo-specific skill
- skill: (none — no dedicated skill for `<algorithm>` yet)
- fallback: skills/algo_ppo (for shared loss/KL knobs only)
- recipe-side defaults: surfaced verbatim from recipe.md `## Key hyperparameters`
- known limitation: group-statistics / SFT-specific / distillation-specific knobs are NOT being curated for this algorithm; the user is responsible for any algo-specific tuning beyond what the recipe sets.
```

This prevents the agent from silently pretending a non-matching skill applies — the past version of this state both mapped `rloo`/`remax`/`gpg` into `algo_grpo` (wrong — those are estimator variants, not group-loss variants) **and** miscategorised `gmpo`/`dppo`/`cispo`/`sapo` as estimator variants (wrong — those are `loss_mode` variants of grpo).

Concretely:

1. **Read** `workspace/intake/training_intent.md` (for `algorithm`, `algorithm`-specific intake fields if any), `workspace/recipe/recipe.md` (for the resolved launch path + the `## Key hyperparameters` block written by `locate_recipe`).

2. **Resolve the axis pair, then dispatch.** Determine `(adv_estimator, policy_loss.loss_mode)` for the user's `algorithm`:
   - First read them straight out of `recipe.md` / the matched recipe script (the recipe is the source of truth — e.g. a `gspo` recipe already carries `adv_estimator=grpo policy_loss.loss_mode=gspo`).
   - Validate each against the live verl registries (`grep '@register_adv_est' / '@register_policy_loss' verl/trainer/ppo/core_algos.py`). If the user named an algorithm but no recipe was matched (direct-module fallback), set the pair from the mapping table and confirm both names exist in the registries; if a name is in neither registry, **halt honestly** (unknown algorithm — not a curated-list miss, a genuinely absent one).
   - Then pick the algo skill per the mapping table. If the algorithm is in the "no first-class trainer" set (`dpo`, `rm`), call the matching skill, which itself decides whether to halt or to redirect (gdpo / external trainer / pre-trained RM use).

3. **Surface knobs in HITL.** Display the skill's "Knobs to surface in `configure_algorithm`" table with the recipe's current default values. The user may override any value. Algorithm-specific knobs add to (not replace) the `## Key hyperparameters` block in `recipe.md` — the union is what `launch_training` will splice into the CLI.

4. **Validate against recipe constraints.** If the user picked `algorithm=ppo` but the chosen recipe doesn't bind a critic (e.g., the recipe was tagged grpo and has no `critic.*` directives), halt — recipe and algorithm disagree. Same for SFT picked with a PPO recipe, etc.

5. **Write `workspace/algorithm/algorithm_config.md`** with:
   - The resolved **axis pair** — `adv_estimator: <name>` and `policy_loss.loss_mode: <name>` (record `vanilla` explicitly when the recipe leaves it default) — plus any loss-mode-specific knobs (e.g. `policy_loss.tau_pos`/`tau_neg` for `sapo`). `launch_training` splices `actor_rollout_ref.actor.policy_loss.loss_mode=<name>` (and the loss-mode knobs) into the CLI only when `loss_mode != vanilla`.
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
