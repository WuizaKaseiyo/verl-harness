# run_generate

## Description

Run batch generation off a checkpoint to produce a parquet of generated responses. Calls `python -m verl.trainer.main_generation_server`. Despite the module name this is a finite job, not a long-running server — it spins vllm replicas, submits the prompts, collects responses, writes the parquet, exits.

Apply the `run_generate` skill (`skills/run_generate`) for the CLI surface + pitfalls, and whichever of `compute_*` matches the chosen target for the launch mechanism.

This state is reachable from two entry paths:
- **Standalone**: `intake` with `goal: generate` — user is generating from a previously-trained checkpoint.
- **Chained**: from a finished `train` track when the user requested `chain_eval: true` at intake (then `run_generate` transitions to `run_eval` instead of `finalize`).

Concretely:

1. **Read** `workspace/intake/training_intent.md` for `model_path`, `prompts_path`, `sampling_params`, `output_path`, optional `chain_eval`. Also read `workspace/compute/compute_choice.md`, `workspace/env/env_state.md`, `workspace/env/launch_env.sh` — generation re-uses the same launch env shape as training. (If entering standalone, the earlier compute-pick / provision_env states still ran first; they only skipped the recipe/dataset/reward train-track-specific steps.)

2. **Validate the prompts parquet.** Per the `run_generate` skill: the parquet must have a `prompt` column with list-of-`{role, content}` rows. Also confirm `data_source` and `reward_model.ground_truth` columns are present if the user plans to chain into `run_eval` (since main_eval needs them).

3. **Assemble the launch command** (matching `examples/generation/run_deepseek_llm_7b.sh`):
   ```
   python -m verl.trainer.main_generation_server \
       trainer.nnodes=<from compute_choice> \
       trainer.n_gpus_per_node=<from compute_choice> \
       data.train_files=<prompts_path> \
       data.prompt_key=prompt \
       +data.output_path=<output_path> \
       actor_rollout_ref.model.path=<model_path> \
       actor_rollout_ref.model.trust_remote_code=True \
       actor_rollout_ref.rollout.name=<vllm|sglang> \
       actor_rollout_ref.rollout.temperature=<from intent> \
       actor_rollout_ref.rollout.top_p=<from intent> \
       actor_rollout_ref.rollout.top_k=<from intent> \
       actor_rollout_ref.rollout.prompt_length=<from intent> \
       actor_rollout_ref.rollout.response_length=<from intent> \
       actor_rollout_ref.rollout.tensor_model_parallel_size=<from intent or compute_choice> \
       actor_rollout_ref.rollout.gpu_memory_utilization=<from intent> \
       actor_rollout_ref.rollout.n=<from intent>
   ```
   Wrap in the same compute-target launcher as `launch_training` (local-direct: bash backgrounded; local-slurm: sbatch a job.slurm with these as the body).

4. **Hand-off point — confirm generation plan.** Present:
   - Checkpoint path (existence verified)
   - Prompts parquet path + row count
   - Sampling params summary (`n=<>, T=<>, top_p=<>, top_k=<>, max_tokens=<>`)
   - Compute target + estimated wall-clock (~0.5 sec per response × rows × n, divided by num_replicas)
   - Output path

   Skipped with `--no-hitl`.

5. **Execute the launch + monitor to terminal.** The monitor loop here is the same as `monitor_training`'s — poll squeue / pid, tail stdout, scan for vllm errors. Terminal markers for generation:
   - **success**: process exit 0 + the output parquet exists at `output_path` + has the expected row count.
   - **crashed**: process exit nonzero. Common: OOM at vllm load (lower `gpu_memory_utilization`), TP > available GPUs (`num_replicas=0` hang then timeout).
   - **timeout**: slurm time limit hit before the batch finished. Report partial output if the parquet was written incrementally.

6. **Write `workspace/generate/generate_report.md`** per the `run_generate` skill's template: source paths, sampling params, compute used, output stats, row-0 sample (prompt + ground_truth + response[0]).

## Skills

- skills/run_generate
- skills/compute_local
- skills/compute_slurm
- skills/compute_ssh_slurm
- skills/training_monitor          # for the polling cadences + log-anomaly patterns
- skills/builtin-tools
- skills/global

> Of the three `compute_*` skills, **read only the one matching the chosen target**.

## Hand-off Points

- **Confirm generation plan.** Step 4. Skipped with `--no-hitl`.

## Next States

### run_eval

**Condition:** Generation succeeded (`workspace/generate/generate_report.md` records `status: success`) AND `training_intent.md` records `chain_eval: true`.

**Deliverables:**

- generations_parquet: The path to the output parquet (also recorded in `generate_report.md`). `run_eval` will read it via `data.path`.

### finalize

**Condition:** Generation succeeded AND `chain_eval` is not set (the user wanted generations only).

**Deliverables:**

- generate_report: `workspace/generate/generate_report.md` — the successful generation report; `finalize` writes a thin terminal wrapper around it.

### finalize

**Condition:** Generation crashed or timed out. Same short-circuit shape as `launch_training` failure — record the failure mode and exit honestly.

**Deliverables:**

- generate_failed: A `workspace/generate/generate_failed.md` with the verbatim error tail and a one-line inferred cause + remediation.
