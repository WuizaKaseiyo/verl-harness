# provision_env

## Description

Ensure the environment that will run training (locally, or on the slurm worker nodes) is ready: Python and torch versions are correct, verl is importable, the model weights are reachable, the HF / wandb tokens are configured, and the output directory exists. The state catches "environment broken" failures *before* a slurm job sits in queue for 30 minutes only to crash on `import verl`.

Apply the appropriate compute-target skill (`skills/compute_local`, `skills/compute_slurm`, or `skills/compute_ssh_slurm`) — only the one matching the chosen target. The compute_target skill defines *where* to run each check (locally on this host, on a slurm worker via `srun --pty`, or via `ssh <login> "command"`).

Concretely:

1. **Read** `workspace/intake/training_intent.md`, `workspace/recipe/recipe.md`, `workspace/dataset/dataset.md`, and `workspace/compute/compute_choice.md`.

   Two **pre-flights** are mandatory (see compute_target skill, "Env-collision pre-flight" and "vLLM engine prerequisite" sections):

   1a. **Worker-side env-collision probe.** Run the appropriate probe per the chosen target (in-place for `local-direct`; short srun for `local-slurm` / `ssh-slurm`). If both `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` (or both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`) are set on the worker, the agent **must** inject `unset ROCR_VISIBLE_DEVICES` and `unset HIP_VISIBLE_DEVICES` into `launch_env.sh`. This protects against verl's worker guard at `verl/single_controller/base/worker.py:256-267`, which fires inside the Ray worker actor's `__init__` and is therefore not catchable by import-only checks.

   1b. **vLLM V1 prerequisite.** If `recipe.md` records `rollout.name=vllm`, the agent **must** inject `export VLLM_USE_V1=1` into `launch_env.sh`. This protects against verl's async rollout server (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:35`), which hardcodes the V1 AsyncLLM constructor while modern vllm may silently fall back to V0.

   Both injections are no-ops when the underlying conditions don't apply; defensive emission is safer than gating on a possibly-stale probe.

2. **Python + torch + verl import check.** Run:
   ```
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
   python -c "import sys; sys.path.insert(0, '<VERL_ROOT>'); import verl; print('verl OK')"
   ```
   on the target. If verl is not importable, attempt the standard install (`cd <VERL_ROOT> && pip install -e .`) and re-check. Record torch+cuda+nccl versions.
3. **Inference backend check.** If the recipe uses vllm (default) or sglang, verify it imports: `python -c "import vllm; print(vllm.__version__)"`. If not present, install it.
4. **Resolve model weights.** Two cases:
   - `model` is an HF id (e.g., `Qwen/Qwen3-4B`) and not yet cached: pre-download via `huggingface-cli download <id> --local-dir <hf_cache>/...` so the training launch isn't waiting on HF. Use `$HF_TOKEN` for gated models.
   - `model` is a local path: verify it exists, list `config.json` + safetensor shards, sanity-check the shard count vs `config.json`'s `num_hidden_layers`.
   Record the resolved local path; the recipe's `actor_rollout_ref.model.path` (or equivalent) will be patched to it.
5. **Output dir.** `mkdir -p <output_dir>`. Confirm writable. Compute available disk: `df -h <output_dir>` and warn if < 100 GB free.
6. **Wandb.** If wandb is configured in the intent, run `wandb login --verify` (or echo `$WANDB_API_KEY` is set). If wandb is unset, ensure `WANDB_DISABLED=true` is in the launch env.
7. **For slurm targets only:** dry-run the slurm template — `sbatch --test-only <template>` — to confirm the directives are syntactically valid and the partition exists. Don't actually queue the job; this is a syntactic check.
8. **Write `workspace/env/env_state.md`** with:
   - **`## Env-collision pre-flight`** — verbatim `env | grep -E '^(ROCR|HIP|CUDA)_VISIBLE_DEVICES'` output from the worker, plus whether `unset ROCR/HIP` was injected.
   - **`## vLLM engine prerequisite`** — whether `recipe.md` triggered the `export VLLM_USE_V1=1` injection (and the rollout backend the trigger keyed on).
   - torch/cuda/verl/vllm versions
   - model path resolved
   - output dir + free disk
   - wandb status
   - slurm dry-run result (if applicable)

   Also write a `workspace/env/launch_env.sh` that:
   - leads with the `unset ROCR_VISIBLE_DEVICES; unset HIP_VISIBLE_DEVICES` (defensive — no-op when not needed);
   - exports `VLLM_USE_V1=1` if vllm rollout is in the recipe;
   - sets `PYTHONPATH=<VERL_ROOT>` (in case verl isn't `pip install -e .`'d into the active env);
   - exports `HF_HOME`, `HF_TOKEN`, `WANDB_*`, and any other run-specific env-vars.

   `launch_training` will `source` this file before any python launch.

## Skills

- skills/compute_local
- skills/compute_slurm
- skills/compute_ssh_slurm
- skills/builtin-tools
- skills/global

> Of the three `compute_*` skills, **read only the one matching the chosen target** in `workspace/compute/compute_choice.md`. The other two are listed so the harness validator sees them as registered, but they are not consulted on this run.

## Hand-off Points

- **Confirm provisioning result.** After step 8 — show the env_state.md summary, the resolved model path, the output dir size, and the launch_env.sh. Skipped with `--no-hitl`.

## Next States

### launch_training

**Condition:** `workspace/env/env_state.md` confirms verl + torch + inference backend importable on the target; model weights are resolved to a local path; output directory exists and is writable; slurm dry-run passed (if slurm target).

**Deliverables:**

- env_state: The provisioning report — torch/cuda/verl/vllm versions, resolved model path, output dir + free disk, wandb status, slurm syntactic-validation result.
- launch_env: A `launch_env.sh` script (`workspace/env/launch_env.sh`) that exports every env-var the launch needs. `launch_training` sources this so the launched job has a deterministic env.

### finalize

**Condition:** Provisioning failed in a way the harness cannot fix (verl install failed, model cannot be downloaded, output dir not writable, slurm partition does not exist).

**Deliverables:**

- env_failed: A `workspace/env/env_failed.md` explaining what broke, what was tried, and what the user must do to unblock. (The run will exit through `finalize` with a `failed` status rather than launching a doomed training job.)
