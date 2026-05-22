compute_local skill — provisioning, launch, monitor when the target is `local-direct`.

The agent's shell *is* the training shell. The training process runs as a child of this host.

## Provisioning checks (used by `provision_env`)

Run these locally. Record verbatim outputs in `workspace/env/env_state.md`.

### Env-collision pre-flight (REQUIRED before any verl launch)

verl's `_setup_env_cuda_visible_devices` (at `verl/single_controller/base/worker.py:256-267`) raises `ValueError("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.")` when both are set. This crash occurs inside the *Ray worker actor's `__init__`* — past every standard provisioning check (sbatch dry-run, `import verl/torch/vllm`). It is therefore not catchable by import-only probes; the harness must pre-flight the env explicitly.

For `local-direct`, the agent's own shell *is* the worker, so probe in-place:

```bash
env | grep -E '^(ROCR|HIP|CUDA)_VISIBLE_DEVICES' || echo '(none set)'
```

If both `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set, OR both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set with different values, the harness **must inject** the following lines into the head of the generated `launch_env.sh`:

```bash
unset ROCR_VISIBLE_DEVICES   # set by some site configs for cross-vendor portability;
                              # collides with verl's worker.py:267 guard on NVIDIA hosts.
unset HIP_VISIBLE_DEVICES    # defensive — same family of variables.
```

Record the collision (with the verbatim env-grep output) in `env_state.md` under a `## Env-collision pre-flight` heading.

### vLLM engine prerequisite (REQUIRED when recipe uses `rollout.name=vllm`)

verl's async rollout server (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:35`) imports `from vllm.v1.engine.async_llm import AsyncLLM` — **hardcoded V1**. With modern vllm (≥0.10.0) and certain config combinations (e.g., `enable_sleep_mode=True`, background-thread engine), vllm silently falls back to V0 unless `VLLM_USE_V1=1` is explicitly set. verl then refuses with `ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False.`

If `workspace/recipe/recipe.md` records `rollout.name=vllm`, the harness **must inject** into `launch_env.sh`:

```bash
export VLLM_USE_V1=1   # required by verl/workers/rollout/vllm_rollout/vllm_async_server.py (V1-only).
```

Record the injection (with the recipe-side trigger) in `env_state.md` under a `## vLLM engine prerequisite` heading.

### Standard checks (run after the pre-flights above)

```bash
# torch + cuda
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"

# nccl
python -c "import torch; print(torch.cuda.nccl.version())"

# verl
python -c "import sys; sys.path.insert(0, '$VERL_ROOT'); import verl; print(verl.__version__ if hasattr(verl,'__version__') else 'verl OK')"

# inference backend
python -c "import vllm; print('vllm', vllm.__version__)"     # or import sglang

# disk
df -h "$OUTPUT_DIR"

# huggingface cache
echo "HF_HOME=$HF_HOME"
huggingface-cli whoami       # only if HF_TOKEN is set
```

If `import verl` fails:

```bash
cd "$VERL_ROOT"
pip install -e .
```

Re-check. If still failing, write to `workspace/env/env_failed.md` and transition `provision_env → finalize`.

## Model weights resolution

If `model` is an HF id, decide whether to pre-download:

```bash
# Check if already cached
ls "$HF_HOME/hub/models--${model//\//--}" 2>/dev/null
```

If absent and the model is gated, ensure `HF_TOKEN` is set, then:

```bash
huggingface-cli download "$model" --local-dir "$HF_HOME/hub/models--${model//\//--}/snapshots/main"
```

If `model` is a local path, validate `config.json` + safetensor shards exist, and the shard count is plausible vs `config.json["num_hidden_layers"]`.

## Launch (used by `launch_training`)

Write `workspace/job/launch.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

source "<workspace>/env/launch_env.sh"
cd "$VERL_ROOT"

exec <the assembled command from recipe.md>   # either bash <script>.sh or python -m verl.trainer.main_<algo> ...
```

Run it in the background, capturing PID and writing stdout / stderr to known paths:

```bash
chmod +x workspace/job/launch.sh
nohup workspace/job/launch.sh \
  > workspace/job/stdout.log \
  2> workspace/job/stderr.log &
echo $! > workspace/job/pid
```

Record `workspace/job/job_info.md`:

```markdown
target: local-direct
started_at: <ISO8601>
pid: <pid>
output_dir: <abs path>
stdout_log: workspace/job/stdout.log
stderr_log: workspace/job/stderr.log
cmd: <full command, identical to the one inside launch.sh>
```

Also write `workspace/job/exit_code_watcher.sh` (one-shot script the user can run later if the launching session is lost) that does `wait <pid>; echo $? > workspace/job/exit_code`. The harness's own monitor loop is the primary source of truth; this is just a fallback for recovery.

## Monitor (used by `monitor_training`)

Polling cadence: **30 seconds**.

Each poll:

- `kill -0 "$(cat workspace/job/pid)" 2>/dev/null` — alive check. If returns 0, process still running.
- If dead, read `workspace/job/exit_code` (the launch script writes it on exit). If exit code is 0 and `<output_dir>/checkpoints/` is non-empty → success. If nonzero → crash.
- Tail new bytes from stdout / stderr logs, append to `workspace/logs/job_log.md`.

OOM detection — the kernel may have OOM-killed the process: check `dmesg | tail -50 | grep -i "killed process"`. Record matches.

## Cancellation

`scancel` doesn't apply. To cancel locally:

```bash
kill -INT "$(cat workspace/job/pid)"            # try clean shutdown first
sleep 10
kill -KILL "$(cat workspace/job/pid)" 2>/dev/null   # force if still alive
```

## Cost estimate (used by the cost gate)

Per-step throughput depends heavily on model size and rollout config; the recipe's own README usually has a rough number. Default heuristic for a sanity baseline (correct within 2×):

```
estimated_wall_clock_minutes ≈
    (total_epochs × train_steps_per_epoch) / steps_per_minute
where steps_per_minute ≈ {
    "7B fsdp":  ~5,
    "8B fsdp":  ~4,
    "32B fsdp": ~0.8,
    "70B megatron": ~0.3,
}[model_size_bucket]
```

These are *very* rough — the cost-gate UI must present them as an estimate, not a promise. The harness's own measured throughput (from the first 100 steps) supersedes the estimate after warmup.

## Things you must not do

- Do not start the training process inside the agent's foreground shell — it must be backgrounded so the harness retains control.
- Do not modify the verl repo. The harness reads it.
- Do not block on the training process. The monitor loop polls; it never `wait`s.
