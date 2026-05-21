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
7. **Write `workspace/recipe/recipe.md`** recording:
   - The chosen launch path: either `script_path: …` (path to a verl shell script) or `module: verl.trainer.main_<algorithm>` (direct python module invocation).
   - The complete argument set the launcher will be invoked with — script env vars (`MODEL_PATH`, `NNODES`, `NDEVICES_PER_NODE`, …) and / or CLI overrides for the python module.
   - The verl commit (`git -C <VERL_ROOT> log -1 --format='%H %s'`) for reproducibility.
   - Any default values inherited from the recipe that the user did not override.

## Skills

- skills/verl_recipes
- skills/global

## Human Checkpoints

- **Recipe selection.** When step 5 fires (multiple matches), pause for the user to pick. Skipped with `--no-hitl` (default: pick the highest-scoring candidate).
- **Direct-module fallback.** When step 6 fires (no shell script matched), pause to show the constructed `python -m verl.trainer.main_<algorithm>` command and ask the user to confirm or edit. Skipped with `--no-hitl` (default: proceed with constructed command, recording it in `recipe.md`).

## Next States

### prepare_data

**Condition:** `workspace/recipe/recipe.md` is written with a resolved launch path and a complete argument set; the verl commit hash is recorded.

**Deliverables:**

- recipe: The chosen verl launch path (shell script or direct python module), the full argument set, the inherited defaults, and the verl commit hash. Sufficient that `launch_training` can run the training without re-deciding anything.
