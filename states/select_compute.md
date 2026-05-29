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
4. **Estimate the GPU budget from model size + algorithm, then set the GPU count.** Before deciding `gpus_per_node`, apply `skills/gpu_budget` — do not just inherit a number or blindly max out the account. Run `skills/gpu_budget/templates/estimate_gpu_budget.py` with the model (path/id from `recipe.md` / `training_intent.md`), the algorithm (from `workspace/algorithm/algorithm_config.md`), the per-GPU VRAM (from the `gpu.access` probe, or the partition's GPU type for slurm), the recipe's `rollout.gpu_memory_utilization`, `train_batch_size`, max prompt+response length, and the account GPU **cap** (`scale.gpu_cap` in `training_intent.md`; if unset, ask — clusters commonly cap a single allocation, e.g. 4 GPUs).
   - The estimator returns **minimum-to-fit** `N_min`, a **recommended** `N_rec` (= cap, since data-parallel throughput scales ~linearly once it fits), and a TP factor if the model is too big for one GPU.
   - **If `N_min > cap`** → the run cannot fit within the budget: **transition to `finalize`** with a `gpu_budget_exceeded` deliverable stating the per-GPU estimate and the options (TP + multi-node, LoRA, quantized/8-bit optimizer, lower `gpu_mem_util`, or a smaller model). Do not submit a doomed job.
   - Otherwise set `gpus_per_node = N_rec` (and `tensor_model_parallel_size = TP` when TP > 1). This estimate also feeds `launch_training`'s cost gate (node-hours).
5. **For Slurm targets**, capture the remaining scheduling parameters for the sbatch directives:
   - partition, account, time limit, nodes (= `scale.nodes`), mem-per-node, cpus-per-task. `gpus-per-node` comes from the step-4 budget estimate (not blindly from `scale.gpus_per_node`; an explicit user value overrides but warn if it is below `N_min`). If `intake` left scale fields blank, inherit the recipe's defaults from `workspace/recipe/recipe.md`. If the partition / account aren't set in the intent, ask.
6. **HITL checkpoint** — present:
   - Chosen target (local-direct / local-slurm / ssh-slurm)
   - The probe results that motivated the choice (e.g., "host has 8× A100 80GB, driver 535, slurm not available → local-direct")
   - **The GPU-budget estimate**: per-GPU VRAM estimate, `N_min` (minimum-to-fit), `N_rec` (chosen, vs cap), and TP — with the one-line reasoning ("1.7B GRPO at util=0.6 fits at 2 GPUs; using cap=4 for ~2× throughput")
   - For slurm: the scheduling parameters that will go into sbatch
7. **Write `workspace/compute/compute_choice.md`** with:
   - target: one of the three labels
   - host probe results (GPU list, slurm partitions, ssh alias)
   - **GPU budget**: model size, algorithm, per-GPU VRAM, `gpu_mem_util`, `N_min`, `N_rec` (= chosen `gpus_per_node`), `tensor_model_parallel_size`, and the per-GPU estimate — so `launch_training` can reuse it for the cost gate and the user can audit the sizing.
   - For slurm targets, the populated sbatch directives ready to splice into `<VERL_ROOT>/examples/tutorial/slurm/ray_on_slurm.slurm` (or a fresh slurm template the harness writes — see `compute_slurm` skill).

## Skills

- skills/compute_select
- skills/gpu_budget
- skills/builtin-tools
- skills/global

## Hand-off Points

- **Confirm compute target + GPU budget + slurm parameters.** Step 6. Skipped with `--no-hitl` (the GPU-budget estimate is still computed and recorded; only the pause is skipped).

## Next States

### provision_env

**Condition:** `workspace/compute/compute_choice.md` is written and names a target that has at least one available capability (gpu.access, slurm.access, or ssh.exec). For Slurm targets, the sbatch directives are fully populated.

**Deliverables:**

- compute_choice: The chosen compute target (`local-direct` | `local-slurm` | `ssh-slurm`), the probe results that supported the choice, the GPU-budget estimate (`N_min` / `N_rec` / TP / per-GPU estimate), and (for slurm targets) the complete sbatch directive set with `--gpus-per-node` = `N_rec`.

### finalize

**Condition:** The GPU budget estimate (step 4) reports `N_min > gpu_cap` — the model + algorithm cannot fit within the account's GPU allocation cap, and no in-budget configuration (TP, lower `gpu_mem_util`) closes the gap.

**Deliverables:**

- gpu_budget_exceeded: A `workspace/compute/gpu_budget_exceeded.md` recording the model size, algorithm, per-GPU estimate, `N_min` vs `cap`, and the concrete options (TP + multi-node, LoRA, quantized/8-bit optimizer, lower `gpu_mem_util`, smaller model). The run exits through `finalize` with a `failed` status rather than submitting a job that will OOM.
