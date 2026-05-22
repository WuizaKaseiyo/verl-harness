# select_compute

## Description

Pick the compute target the training job will run on: `local-direct` (run training directly on this machine with its CUDA GPUs), `local-slurm` (this machine is a Slurm login node, sbatch from here), or `ssh-slurm` (this machine has ssh credentials for a remote Slurm cluster's login node). The choice constrains how `launch_training` will be invoked.

Apply the `compute_select` skill (`skills/compute_select`).

Concretely:

1. **Read** `workspace/intake/training_intent.md` for `compute_pref`.
2. **Probe host capabilities.** The compute_select skill defines the probes; in short:
   - `gpu.access`: does `nvidia-smi` succeed and report ≥ 1 GPU? Record device count, names, free memory, driver version. (If `nvidia-smi` is absent, the host has no GPUs and `local-direct` is unavailable.)
   - `slurm.access`: do `sinfo` and `squeue` succeed locally? Record partition list and one user-friendly partition recommendation. (If they fail, `local-slurm` is unavailable.)
   - `ssh.exec`: is there a configured remote slurm cluster? The user normally records its alias in `$VERL_HARNESS_REMOTE` or passes it in. Probe with `ssh <alias> sinfo -V`. (If missing or unconfigured, `ssh-slurm` is unavailable.)
3. **Resolve the choice:**
   - If `compute_pref` is explicit (`local-direct` / `local-slurm` / `ssh-slurm`) and the corresponding capability is present → use it.
   - If `compute_pref` is explicit but the capability is missing → halt with a clear error.
   - If `compute_pref` is `auto`, the compute_select skill applies its ranking (typically `local-slurm` > `local-direct` > `ssh-slurm`, but the skill documents the full rule).
4. **For Slurm targets**, capture the scheduling parameters the user will need to fill into the sbatch directives:
   - partition, account, time limit, nodes (= `scale.nodes`), gpus-per-node (= `scale.gpus_per_node`), mem-per-node, cpus-per-task. If `intake` left scale fields blank, inherit the recipe's defaults from `workspace/recipe/recipe.md`. If the partition / account aren't set in the intent, ask.
5. **HITL checkpoint** — present:
   - Chosen target (local-direct / local-slurm / ssh-slurm)
   - The probe results that motivated the choice (e.g., "host has 8× A100 80GB, driver 535, slurm not available → local-direct")
   - For slurm: the scheduling parameters that will go into sbatch
6. **Write `workspace/compute/compute_choice.md`** with:
   - target: one of the three labels
   - host probe results (GPU list, slurm partitions, ssh alias)
   - For slurm targets, the populated sbatch directives ready to splice into `<VERL_ROOT>/examples/tutorial/slurm/ray_on_slurm.slurm` (or a fresh slurm template the harness writes — see `compute_slurm` skill).

## Skills

- skills/compute_select
- skills/builtin-tools
- skills/global

## Hand-off Points

- **Confirm compute target + slurm parameters.** Step 5. Skipped with `--no-hitl`.

## Next States

### provision_env

**Condition:** `workspace/compute/compute_choice.md` is written and names a target that has at least one available capability (gpu.access, slurm.access, or ssh.exec). For Slurm targets, the sbatch directives are fully populated.

**Deliverables:**

- compute_choice: The chosen compute target (`local-direct` | `local-slurm` | `ssh-slurm`), the probe results that supported the choice, and (for slurm targets) the complete sbatch directive set.
