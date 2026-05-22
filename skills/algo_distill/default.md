algo_distill skill — on-policy distillation. Grounded in this verl's actual `distillation.*` config namespace (`verl/trainer/config/distillation/distillation.yaml`).

## Trainer binding

**There is no separate distillation main module.** Distillation runs through **`verl.trainer.main_ppo`** with `algorithm.adv_estimator=grpo` + `distillation.enabled=True` + the `distillation.*` config tree below. Loss code lives in `verl/trainer/distillation/{losses.py, fsdp/, megatron/}`; teacher-side workers live in `verl/experimental/teacher_loop/`.

Use `examples/on_policy_distillation_trainer/run_*.sh` as the authoritative reference for any particular (student, teacher, backend) combo. Recipes available on this checkout: `run_qwen3_8b_fsdp.sh`, `run_qwen3_8b_megatron.sh`, `run_qwen3_vl_8b_fsdp.sh`.

## Knobs to surface in `configure_algorithm`

### Teacher (resource pool + per-teacher inference)
| Field | What it controls | Default in distillation.yaml |
|---|---|---|
| `distillation.enabled` | Master switch | `false` (must override to `True`) |
| `distillation.n_gpus_per_node` | GPUs per node in teacher pool | `8` |
| `distillation.nnodes` | Nodes in teacher pool | `0` (must set) |
| `distillation.teacher_models.<name>.model_path` | Teacher HF id or local path | `null` |
| `distillation.teacher_models.<name>.inference.tensor_model_parallel_size` | Teacher TP | `2` |
| `distillation.teacher_models.<name>.inference.gpu_memory_utilization` | Teacher vLLM mem cap | `0.5` |
| `distillation.teacher_models.<name>.inference.max_model_len` | Teacher max token count (= `max_prompt + max_response + 1`) | `null` (compute) |
| `distillation.teacher_models.<name>.inference.name` | `vllm` / `sglang` (resolves from `actor_rollout_ref.rollout.name`) | inherited |
| `distillation.teacher_key` | Field in the data row that routes prompts to a teacher (for multi-teacher) | `data_source` |

**Multi-teacher gotcha (from distillation.yaml):** the single-teacher slot is named `teacher_model`. Adding `teacher_model2`, `teacher_model3` works **only** if you also rename the first one (`teacher_model1`). Using the default `teacher_model` together with `teacher_model2` silently drops `teacher_model`. Surface this to the user when they configure multiple teachers.

### Distillation loss (`distillation.distillation_loss.*`)
| Field | What it controls | Default in distillation.yaml | Common override |
|---|---|---|---|
| `loss_mode` | Distillation loss formulation — see `verl/trainer/distillation/losses.py` for registered modes (e.g., `k1`, `k3`); `k3` is the verl default, examples use `k1` | `k3` | `k1` (qwen3-8b recipe) |
| `topk` | Top-k logits used for sparse distillation (lower k = cheaper, less faithful) | `32` | `64` |
| `use_task_rewards` | If `True`, also accumulate task-reward signal alongside distill loss (combined objective) | `true` | `False` for pure distill |
| `distillation_loss_coef` | Weight on distill loss when `use_task_rewards=True` | `1.0` | per-recipe |
| `loss_max_clamp` | Cap the distill loss magnitude per token (None = unclamped) | `null` | `10.0` (8b recipe) for stability |
| `log_prob_min_clamp` | Floor log-probs to avoid `log(0)` in KL | `null` | `-10.0` (8b recipe) |
| `use_policy_gradient` | If `True`, wrap distill loss into a PPO-style policy-gradient update (clipped surrogate) | `false` | `True` (qwen3-8b recipe; lets `clip_ratio*` apply) |
| `policy_loss_mode` | When `use_policy_gradient=True`, the surrogate form (`"vanilla"` or other) | `"vanilla"` | — |
| `clip_ratio` / `clip_ratio_low` / `clip_ratio_high` | PPO clip ε when `use_policy_gradient=True` | `0.2` each | — |

### Student / actor / rollout knobs
Distillation reuses the **`actor_rollout_ref.*`** namespace for the student (it IS the actor). Same shape as GRPO recipes — see `algo_grpo` for `rollout.n`, `rollout.gpu_memory_utilization`, `actor.optim.lr`, `actor.ppo_max_token_len_per_gpu`, `actor.fsdp_config.param_offload/optimizer_offload`, etc.

### Trainer
Same `trainer.*` fields as PPO/GRPO (`nnodes`, `n_gpus_per_node`, `total_epochs`, `save_freq`, `test_freq`, etc.).

## Failure modes

- **Teacher pool sizing wrong.** `distillation.n_gpus_per_node` × `distillation.nnodes` must allocate enough capacity for the teacher(s) at their `tensor_model_parallel_size`. The `qwen3_8b_fsdp` recipe uses `TEACHER_WORLD_SIZE=4` for a 32B teacher at TP=2 — 4 GPUs serving the teacher pool, separate from the student's GPUs. Co-locating teacher and student on the same GPUs typically OOMs.
- **`distillation.teacher_key` mismatch.** If you list multiple teachers but the dataset's `data_source` (or whatever `teacher_key` points at) doesn't have values matching the teacher names' `key` fields, prompts get routed to the wrong teacher (or none) silently. Verify at `sanity_rollout` by sampling 10 rows and printing which teacher each routes to.
- **`use_task_rewards=True` with `use_policy_gradient=False` + a noisy reward fn.** Distill loss decreases smoothly while task-reward adds high-variance perturbations; net effect is unstable distillation. Either set `use_task_rewards=False` (pure distill) or set `use_policy_gradient=True` (clips the combined update).
- **Top-k too small.** `topk=4` or `topk=8` produces unstable distillation on tasks where the teacher's distribution has mass spread across many tokens. Default `topk=32` is conservative; raise to `64` if loss curves are noisy.
- **`loss_max_clamp=None` + bad teacher.** A misaligned teacher (different tokenizer family, different chat template) emits gigantic per-token KL contributions early in training. The 8B recipe clamps at `10.0` for a reason; unclamping is asking for blow-ups.

## Canonical val metric (for `pick_best_checkpoint`)

`val/reward/mean` when `use_task_rewards=True` (the combined objective's reward part is logged). For pure distill (`use_task_rewards=False`), there is **no canonical val metric** — the distill loss itself is the only signal, and lower-is-better. Look for `val/distillation_loss` if logged; otherwise fall back to the training-side `actor/distillation_loss` mid-training samples.

## CLI injection from `configure_algorithm`

Exact shape (matches `examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh`'s EXTRA block):

```
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
distillation.enabled=True
distillation.n_gpus_per_node=<TEACHER_WORLD_SIZE>
distillation.nnodes=<NNODES>
distillation.teacher_models.teacher_model.model_path=<TEACHER_HF_ID>
distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=<teacher_tp>
distillation.teacher_models.teacher_model.inference.name=vllm
distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=<float>
distillation.teacher_models.teacher_model.inference.max_model_len=<int>
distillation.distillation_loss.loss_mode=k1
distillation.distillation_loss.topk=64
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.use_policy_gradient=True
distillation.distillation_loss.loss_max_clamp=10.0
distillation.distillation_loss.log_prob_min_clamp=-10.0
```

(Student-side overrides — actor/rollout/data/trainer — are layered on top per `algo_grpo` shape, since distillation borrows GRPO's actor loop.)

## Things you must not do

- Do not assume a `verl.trainer.main_distill` module. It does not exist; distillation is `main_ppo` with `distillation.enabled=True`.
- Do not invent fields like `algorithm.distill_loss_type=kl` or `teacher.model.path=...` (older drafts of this skill had these — they are NOT real verl fields). The authoritative namespace is `distillation.*`.
- Do not silently co-locate teacher and student on the same GPU set unless the recipe explicitly does so (most don't — teacher pool is separate).
- Do not enable `use_policy_gradient=True` without also setting `clip_ratio*` — the unclipped surrogate is numerically unstable on most data.
