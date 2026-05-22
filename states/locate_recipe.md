# locate_recipe

## Description

Find the verl script that will actually launch training, given the user's `algorithm`, `model`, and (when set) `scale` fields. verl ships dozens of example shell scripts under `<VERL_ROOT>/examples/<algorithm>_trainer/run_<model_slug>_<backend>.sh` and additional in-repo recipes under `<VERL_ROOT>/recipe/<algorithm>_trainer/`. This state binds the user's intent to the most appropriate one — or, when no script matches, constructs a direct `python -m verl.trainer.main_<algorithm>` invocation from the trainer's documented surface.

Apply the `verl_recipes` skill (`skills/verl_recipes`).

Concretely:

1. **Read `workspace/intake/training_intent.md`** for algorithm, model, optional backend (`fsdp` / `megatron` / `veomni`), and scale.
2. **Search the verl checkout** for candidate scripts:
   - First: `<VERL_ROOT>/examples/<algorithm>_trainer/run_*.sh`
   - Then: `<VERL_ROOT>/recipe/<algorithm>_trainer/run_*.sh` (newer recipes live here)
   - Then: any `<VERL_ROOT>/examples/**/run_*<algorithm>*.sh`
3. **Score candidates** by model-name match, backend match (if the user specified one), and scale fit. Score by the `verl_recipes` skill's rules.
4. **If exactly one strong match exists:** pick it. Record the path.
5. **If multiple strong matches exist:** (HITL checkpoint) present the top 3 to the user with one-line summaries; ask which to use.
6. **If no script matches:** build a launch command directly from `python -m verl.trainer.main_<algorithm>`, populating the standard verl CLI overrides (`data.train_files`, `data.val_files`, `actor_rollout_ref.model.path`, `trainer.nnodes`, `trainer.n_gpus_per_node`, …) from the training intent. The `verl_recipes` skill documents which trainer modules exist and which fields each accepts. If the trainer module also doesn't exist (typo'd algorithm), halt with a clear "no such trainer" message.
7. **Surface key hyperparameters in the HITL display** (and in `recipe.md`). Regardless of whether step 5 or step 6 fired, the agent must promote the following fields from "inherited default" to top-level visible before transitioning. The user sees them with the recipe's default value and may override any of them in-band:

   | Field | Why surface it |
   |---|---|
   | `actor_rollout_ref.actor.optim.lr` | Most-tuned hyperparameter; recipe defaults vary 10× |
   | `critic.optim.lr` (PPO only) | Same as above; only relevant for `adv_estimator=gae` |
   | `algorithm.kl_ctrl.kl_coef` / `actor_rollout_ref.actor.kl_loss_coef` | KL control; misuse kills training |
   | `actor_rollout_ref.actor.ppo_mini_batch_size` | Memory + stability trade-off |
   | `actor_rollout_ref.rollout.gpu_memory_utilization` | OOM risk at launch; depends on model + GPU |
   | `actor_rollout_ref.rollout.tensor_model_parallel_size` | Wrong value causes init crash |
   | `actor_rollout_ref.rollout.n` (PPO-family) | Rollout sample count; affects group-reward variance for GRPO |
   | `data.max_response_length` | Truncation; mode-collapse signal |
   | `trainer.total_epochs` / `trainer.total_training_steps` | Wall-clock + cost driver |
   | `trainer.test_freq` | In-training eval cadence; 0 disables |
   | `trainer.save_freq` | Checkpoint cadence; -1 means no save |

   Additional algorithm-specific knobs come from `skills/algo_<name>/` (when present — see roadmap Phase 2); until those exist, fall back to the recipe's own header comments.

8. **Write `workspace/recipe/recipe.md`** recording:
   - The chosen launch path: either `script_path: …` (path to a verl shell script) or `module: verl.trainer.main_ppo` (direct python module invocation; SFT uses `verl.trainer.sft_trainer`).
   - The `## Key hyperparameters` section listing every field from step 7 with the recipe's value (or the user's override).
   - The complete argument set the launcher will be invoked with — script env vars (`MODEL_PATH`, `NNODES`, `NDEVICES_PER_NODE`, …) and / or CLI overrides for the python module.
   - The verl commit (`git -C <VERL_ROOT> log -1 --format='%H %s'`) for reproducibility.
   - All other default values inherited from the recipe that the user did not override.

## Skills

- skills/verl_recipes
- skills/builtin-tools
- skills/global

## Hand-off Points

- **Recipe selection.** When step 5 fires (multiple matches), pause for the user to pick. Skipped with `--no-hitl` (default: pick the highest-scoring candidate).
- **Direct-module fallback.** When step 6 fires (no shell script matched), pause to show the constructed `python -m verl.trainer.main_<algorithm>` command and ask the user to confirm or edit. Skipped with `--no-hitl` (default: proceed with constructed command, recording it in `recipe.md`).
- **Key hyperparameters override.** After step 7, present the key-hyperparameters table and allow the user to override any value before `recipe.md` is finalised. Skipped with `--no-hitl` (default: keep all recipe values).

## Next States

### configure_algorithm

**Condition:** `workspace/recipe/recipe.md` is written with a resolved launch path and a complete argument set; the verl commit hash is recorded; the `## Key hyperparameters` block is present (per step 7) so that `configure_algorithm` can extend it with algorithm-specific knobs.

**Deliverables:**

- recipe: The chosen verl launch path (shell script or direct python module), the full argument set, the inherited defaults, the `## Key hyperparameters` block, and the verl commit hash. Sufficient that `configure_algorithm` can layer algorithm-specific knobs on top, then `launch_training` can run the training without re-deciding anything.
