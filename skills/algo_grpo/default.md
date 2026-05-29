algo_grpo skill — knobs and pitfalls specific to GRPO (`algorithm.adv_estimator=grpo`).

GRPO eliminates the critic by computing advantage from **group statistics**: sample `n` responses per prompt, normalise the per-response reward by the group mean (and optionally the group std). The variance signal is structural, not learned.

## Trainer binding

Module: `verl.trainer.main_ppo` (same entry as PPO, branch by `adv_estimator`).
Config root: same as PPO (`ppo_trainer.yaml`), but the `critic.*` block is unused.
Variants in this verl: `grpo`, `grpo_passk`, `grpo_vectorized` (see `verl/trainer/ppo/core_algos.py` `AdvantageEstimator` enum).

## Knobs to surface in `configure_algorithm`

### Group-rollout shape (the defining GRPO knob)
| Field | What it controls | Typical default |
|---|---|---|
| `actor_rollout_ref.rollout.n` | Responses per prompt (= group size) | `8` (small recipes), `16`+ (paper-scale) |
| `algorithm.adv_estimator` | `grpo` / `grpo_passk` / `grpo_vectorized` | `grpo` |
| `algorithm.norm_adv_by_std_in_grpo` | Divide group advantage by std (original GRPO); `False` reproduces Dr.GRPO | `True` |

Effective per-prompt cost ≈ `n × max_response_length × tokens-per-second`. Doubling `n` doubles rollout cost; tune it against the rollout-side budget, not vibes.

### Loss / KL (shared with PPO)
| Field | What it controls | Typical default |
|---|---|---|
| `actor_rollout_ref.actor.clip_ratio` | Policy clip ε | `0.2` |
| `actor_rollout_ref.actor.entropy_coeff` | Entropy bonus | `0.0`–`0.005` |
| `actor_rollout_ref.actor.use_kl_loss` | KL as actor loss term | `True` |
| `actor_rollout_ref.actor.kl_loss_coef` | KL coefficient | `0.001` |
| `actor_rollout_ref.actor.kl_loss_type` | `"low_var_kl"` (recommended for GRPO) | `"low_var_kl"` |

GRPO has the same KL-in-reward vs KL-loss choice as PPO. Same rule: pick one.

### GRPO-passk variant
`adv_estimator=grpo_passk` — advantage = `pass@k` indicator across the group. Useful for code / math tasks where partial correctness signals are weak and you want a binary "did any of the n samples solve it" signal back-propagated.

### Policy-loss-mode variants (GSPO / CISPO / GMPO / SAPO / DPPO)

These keep `adv_estimator=grpo` (so **all the group knobs above still apply**) and only swap the **policy loss objective** via `actor_rollout_ref.actor.policy_loss.loss_mode` (the second algorithm axis — `@register_policy_loss` in `verl/trainer/ppo/core_algos.py`). They are *not* separate advantage estimators. Read the exact `loss_mode` (and its extra knobs) from the matched `examples/<name>_trainer/run_*.sh`; the registry is the validity check.

| User name | `policy_loss.loss_mode` | Extra knobs | One-line intent |
|---|---|---|---|
| `gspo` | `gspo` | — | Group Sequence PO — sequence-level (not token-level) importance ratio; steadier for long responses / MoE (Qwen). |
| `cispo` | `cispo` | — | Clipped-IS PO — clips the importance-sampling weight rather than the ratio (MiniMax M1). |
| `gmpo` | `geo_mean` | — | Geometric-Mean PO — geometric- instead of arithmetic-mean over tokens; robust to outlier tokens. |
| `sapo` | `sapo` | `actor.tau_pos`, `actor.tau_neg` (NOT under `policy_loss.*`) | Asymmetric clipping with separate positive/negative temperatures. |
| `dppo` | `dppo_tv` *or* `dppo_kl` | — | Decoupled PPO — TV- or KL-regularised; read which from the recipe. |

Other registered modes (`clip_cov`, `kl_cov`, `bypass_mode`) are research/diagnostic and pass through the same way. `vanilla` is the GRPO default and needs no `loss_mode` on the CLI.

## Failure modes

- **Group-reward variance collapses.** All `n` samples in a group score the same → group std = 0 → if `norm_adv_by_std_in_grpo=True` and the divide-by-zero is unguarded, you get NaN advantages. verl guards this with an epsilon, but the signal is also gone. Either lower temperature (more deterministic samples for harder prompts → more useful variance), or **switch `norm_adv_by_std_in_grpo=False`** (Dr.GRPO style) which avoids the division entirely.
- **Mode-flat group.** Group mean creeps toward maximum reward while group entropy stays high — model has memorised the dataset, advantage signal becomes noise. Train a smaller-step horizon or add more diverse data.
- **Long-response collapse (chess-DuPO-like).** Specific to multi-stage / verifier-recompute setups: if `max_response_length` truncates the structural part of responses, the verifier sees garbage and the advantage stops correlating with correctness. Watch `response/length/mean` against `data.max_response_length` (see `training_monitor` anomaly thresholds).

## Canonical val metric

`val/reward/mean` — higher is better. For binary-reward tasks (math correctness), also surface `val/pass_rate` if logged. For GRPO-passk specifically, prefer `val/pass_at_k`.

## CLI injection from `configure_algorithm`

```
algorithm.adv_estimator=grpo                                # or grpo_passk, grpo_vectorized
algorithm.use_kl_in_reward=False
algorithm.norm_adv_by_std_in_grpo=True                      # False = Dr.GRPO style
actor_rollout_ref.rollout.n=8                                # group size
actor_rollout_ref.actor.clip_ratio=0.2
actor_rollout_ref.actor.entropy_coeff=0.0
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
# loss-mode variants only (omit for vanilla GRPO):
# actor_rollout_ref.actor.policy_loss.loss_mode=gspo|cispo|geo_mean|sapo|dppo_tv|dppo_kl
# actor_rollout_ref.actor.tau_pos=...   actor_rollout_ref.actor.tau_neg=...   # sapo only
# WARNING: verl's own examples/sapo_trainer/*.sh write `+actor.policy_loss.tau_pos=...`
# That path is WRONG on this verl: `tau_pos`/`tau_neg` are fields of ActorConfig, NOT
# PolicyLossConfig (verified at verl/workers/config/actor.py:158-159 for ActorConfig vs.
# 78-100 for PolicyLossConfig — PolicyLossConfig has no tau_* fields, and Hydra reports
# `TypeError: PolicyLossConfig.__init__() got an unexpected keyword argument 'tau_pos'`).
# Use `actor_rollout_ref.actor.tau_pos=` (no `policy_loss.` infix) regardless of what the
# example script shows.
```

## Things you must not do

- Do not surface `critic.*` knobs for GRPO — there is no critic in this algorithm; verl ignores those overrides.
- Do not assume `rollout.n=1` is valid for GRPO. The group requires ≥ 2 samples per prompt to compute variance; with `n=1` GRPO degenerates to REINFORCE.
- Do not silently switch to `grpo_passk` when the user said `grpo`; they are different algorithms with different optimal hyperparameters.
