algo_ppo skill — knobs and pitfalls specific to PPO (`algorithm.adv_estimator=gae`).

PPO is the only PPO-family algorithm that uses a critic. The other family members (grpo / rloo / remax / gpg / gdpo / …) compute advantage without a value baseline and skip the critic entirely.

## Trainer binding

Module: `verl.trainer.main_ppo` with `algorithm.adv_estimator=gae`.
Config root: `<verl_root>/verl/trainer/config/ppo_trainer.yaml` + `critic/` Hydra subdir (`critic.yaml`, `dp_critic.yaml`, `megatron_critic.yaml`, etc.).

## Knobs to surface in `configure_algorithm`

These are the PPO-specific fields the user is most likely to want to tune in the HITL display. Recipe defaults are inherited; the user may override any of them.

### Critic (PPO's value baseline — none of the other adv_estimators use this)
| Field | What it controls | Typical default |
|---|---|---|
| `critic.model.path` | Critic backbone (often same as actor; can be smaller) | inherited from `actor_rollout_ref.model.path` |
| `critic.optim.lr` | Critic learning rate | `1e-5` (usually 10× the actor's `1e-6`) |
| `critic.ppo_micro_batch_size_per_gpu` | Critic forward micro-batch | matches actor `ppo_mini_batch_size` |
| `critic.cliprange_value` | Value-loss clip; large = unstable critic | `0.5` |

### PPO loss
| Field | What it controls | Typical default |
|---|---|---|
| `algorithm.loss_type` | `"ppo_clip"` (clipped surrogate) — the canonical | `"ppo_clip"` |
| `actor_rollout_ref.actor.clip_ratio` | PPO policy-clip ε | `0.2` |
| `actor_rollout_ref.actor.entropy_coeff` | Entropy bonus | `0.0` |
| `algorithm.adv_estimator` | Must be `gae` for PPO | `gae` |
| `algorithm.gamma` | Discount factor (per-token rewards rare → usually 1.0) | `1.0` |
| `algorithm.lam` | GAE bias/variance lambda | `0.95` |

### KL control
| Field | What it controls | Typical default |
|---|---|---|
| `algorithm.use_kl_in_reward` | Apply KL penalty *in the reward* (subtracts KL from per-token reward) | `False` |
| `algorithm.kl_ctrl.kl_coef` | Coefficient of the in-reward KL penalty | `0.001` |
| `algorithm.kl_ctrl.target_kl` | Adaptive KL target | `0.1` |
| `actor_rollout_ref.actor.use_kl_loss` | Add KL as a separate loss term on the actor | `True` |
| `actor_rollout_ref.actor.kl_loss_coef` | KL loss coefficient | `0.001` |
| `actor_rollout_ref.actor.kl_loss_type` | `"kl"`, `"low_var_kl"`, `"k1"` … | `"low_var_kl"` |

Pick **one** KL control regime (in-reward OR loss-term), not both. Using both double-counts the KL signal and crushes exploration.

## Failure modes

- **Critic divergence** — `actor/grad_norm` stable while `critic/value_loss` explodes. Lower `critic.optim.lr` or tighten `critic.cliprange_value`.
- **Actor collapse to greedy** — `actor/entropy` falls below 0.5 within ~50 steps. Either KL is too weak (raise `kl_loss_coef`) or `clip_ratio` too loose (lower to 0.15).
- **Reward-KL fight** — when `use_kl_in_reward=True` AND `use_kl_loss=True`, the policy fights itself. Pick one.
- **Value head not learning** — if `critic.model.path` is a fresh init rather than a pre-warmed critic, the first ~100 steps have noisy GAE estimates; tolerate higher variance early.

## Canonical val metric (for `pick_best_checkpoint` in `training_monitor`)

`val/reward/mean` — higher is better. Falls back to `val/score/mean` if the recipe doesn't compute reward at val time.

## CLI injection from `configure_algorithm`

```
algorithm.adv_estimator=gae
algorithm.use_kl_in_reward=False                          # or True if user opts in
algorithm.gamma=1.0
algorithm.lam=0.95
algorithm.kl_ctrl.kl_coef=0.001                            # only if use_kl_in_reward=True
actor_rollout_ref.actor.clip_ratio=0.2
actor_rollout_ref.actor.entropy_coeff=0.0
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
critic.optim.lr=1e-5
critic.cliprange_value=0.5
```

## Things you must not do

- Do not surface `algorithm.norm_adv_by_std_in_grpo` for PPO — that's GRPO-only and ignored under GAE.
- Do not advise turning off the critic for PPO — that's not PPO, that's a different algorithm (REINFORCE / RLOO / GRPO).
- Do not silently combine `use_kl_in_reward=True` and `use_kl_loss=True` without flagging the double-count to the user.
