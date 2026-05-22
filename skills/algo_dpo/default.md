algo_dpo skill — DPO (Direct Preference Optimization) handling for this verl.

## Important: DPO is not a first-class trainer in this verl checkout

Probing this verl shows:

- **No `verl.trainer.main_dpo` module.** The PPO entry (`main_ppo`) is the only RL entry.
- **No `examples/dpo_trainer/` recipe directory.**
- **`algorithm.loss_type` values are `"ppo_clip"` / `"reinforce"`** — there is no `"dpo_sigmoid"` / `"dpo_hinge"` / `"ipo"`. The DPO-family loss types are not implemented at the trainer level.
- **There IS a related variant: `gdpo` (Group reward-Decoupled Normalisation Policy Optimization)**, exposed as `algorithm.adv_estimator=gdpo` and a `gdpo` reward-manager at `verl/experimental/reward_loop/reward_manager/gdpo.py`. GDPO is a *PPO-family* algorithm — it uses rollouts, group statistics, and policy-gradient updates, not pair-likelihood maximisation. It is **not** classic DPO.

## What the harness does when the user says `algorithm: dpo`

Three honest options, decided in `configure_algorithm`:

1. **Suggest `gdpo` instead.** If the user wanted "preference-style training with multiple reward signals", GDPO is the supported analogue: `adv_estimator=gdpo` + `algorithm.gdpo_reward_keys=[...]` + `algorithm.gdpo_reward_weights=[...]` to compose per-component rewards. Surface this to the user with the trade-off (rollout-based, not pair-likelihood).

2. **Halt with a clear "no such trainer" message.** Per `skills/global/scientific_principles.md`'s honesty principle and CLAUDE.md's "halt honestly on miss". The agent does not invent a DPO trainer.

3. **External-trainer pointer (optional).** Suggest the user use an out-of-tree DPO trainer (`trl.DPOTrainer`, `OpenRLHF`, etc.); the harness can still drive `prepare_data` for pair datasets (using `full_hh_rlhf` template under `dataset_autogen`) but `launch_training` is out of scope — the user runs the external trainer themselves and re-enters the harness at `monitor_training` (Phase 3 resume track).

## If the user opts for GDPO (the supported path)

### Trainer binding
- Module: `verl.trainer.main_ppo`
- adv_estimator: `gdpo`

### Knobs specific to GDPO
| Field | What it controls | Typical default |
|---|---|---|
| `algorithm.adv_estimator` | `gdpo` | — |
| `algorithm.gdpo_reward_keys` | Which keys in the per-response reward dict participate in advantage normalisation (e.g., `["correctness", "format_bonus"]`) | None — must be set if `compute_score` returns a dict |
| `algorithm.gdpo_reward_weights` | Per-key weights for aggregation | equal weights |
| `actor_rollout_ref.rollout.n` | Group size (same role as GRPO) | `8`+ |

The per-component reward keys are produced by the **custom reward function** (`reward_kind=custom` or `shaped` — see `skills/reward_custom/`, `skills/reward_shaping/`). GDPO without a dict-returning compute_score has nothing to decouple and degenerates to GRPO.

### CLI injection
```
algorithm.adv_estimator=gdpo
algorithm.gdpo_reward_keys=["correctness","format_bonus","length_penalty"]
algorithm.gdpo_reward_weights=[1.0, 0.2, -0.005]
actor_rollout_ref.rollout.n=8
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

### Canonical val metric
`val/reward/mean` for the summed reward. Also surface per-component val metrics if logged: `val/gdpo/correctness/mean`, `val/gdpo/format_bonus/mean`, etc. (the trainer logs each `gdpo_reward_keys` member separately during training; recipe must enable val-time logging the same way).

## Failure modes

- **User says `algorithm: dpo` and expects pair-likelihood loss.** The agent cannot deliver it here — halt and explain. Do not silently route to GDPO without consent.
- **GDPO with a non-dict `compute_score`.** Degenerates to GRPO; the `gdpo_reward_keys`/`gdpo_reward_weights` overrides are ignored. Verify the reward function returns a dict before committing.
- **Pair datasets used with GDPO.** GDPO is rollout-based; chosen/rejected pair data doesn't fit. If the dataset has `chosen`/`rejected` columns, route to option (3) above.

## Things you must not do

- Do not author a `loss_type="dpo_sigmoid"` override — verl will reject it with an unknown-loss-type error.
- Do not lie to the user that "DPO is supported, the harness will figure it out". State the situation plainly and let the user pick between GDPO, halt, or external trainer.
- Do not fabricate a `verl.trainer.main_dpo` import; the module does not exist in this checkout.
