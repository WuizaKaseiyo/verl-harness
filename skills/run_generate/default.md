run_generate skill — batch generation via `verl.trainer.main_generation_server`. Grounded in `examples/generation/run_*.sh` and the source at `verl/trainer/main_generation_server.py`.

## What the trainer module is, despite the name

`main_generation_server` is **not a long-running HTTP server**. It spins up vllm/sglang server replicas internally (each replica is a vllm `start_server` instance), submits the entire prompts dataset to them via async HTTP chat completions, collects results, writes a parquet, and exits. It is a **finite batch job** with a clear terminal status — fits the FSM model.

(The name comes from the fact that internally it uses a server protocol to amortise vllm engine startup across many requests. The user-facing semantics are batch generation, not service hosting.)

## Real CLI surface (from `examples/generation/run_deepseek_llm_7b.sh`)

```bash
python3 -m verl.trainer.main_generation_server \
    trainer.nnodes=<N> \
    trainer.n_gpus_per_node=<G> \
    data.train_files=<prompts_parquet>           # NOTE: field is `train_files` even for generation
    data.prompt_key=prompt                       # column name in the parquet
    +data.output_path=<output_parquet>           # where to write generations
    actor_rollout_ref.model.path=<checkpoint>
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.rollout.name=vllm          # or sglang
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_k=50
    actor_rollout_ref.rollout.top_p=0.7
    actor_rollout_ref.rollout.prompt_length=2048
    actor_rollout_ref.rollout.response_length=1024
    actor_rollout_ref.rollout.tensor_model_parallel_size=<TP>
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.n=1                # responses per prompt
```

## Input parquet shape

The input parquet must have a `data.prompt_key` column (default `prompt`) whose rows are list-of-`{role, content}` dicts (the canonical verl chat format). Other columns (e.g., `data_source`, `reward_model.ground_truth`, `extra_info`) are **preserved into the output parquet** — this matters because `main_eval` needs `data_source` and `reward_model` columns to score.

When the user wants to eval on the same dataset they trained on (the most common case), use the **train run's `val_files`** (the test split) as the prompts: `prompts_path = workspace/dataset/<name>/test.parquet`. Otherwise, accept the user's path.

## Output parquet shape

Same columns as input, **plus** a new `responses` column (list-of-strings, length = `rollout.n`) holding the generated outputs. `main_eval`'s default `response_key=responses` matches this verbatim.

## CLI overrides to surface in HITL

| Field | What it controls | Typical default |
|---|---|---|
| `actor_rollout_ref.rollout.n` | Responses per prompt | `1` (eval); `8` (best-of-N analysis) |
| `actor_rollout_ref.rollout.temperature` | Sampling temperature | `0.0` (deterministic / greedy) or `1.0` (diverse) — pick per use case |
| `actor_rollout_ref.rollout.top_p` / `top_k` | Nucleus / top-k truncation | `0.7` / `50` (matches the example recipe) |
| `actor_rollout_ref.rollout.response_length` | Max generated tokens | recipe-default or user override |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | TP for the rollout server | `2`+ for ≥ 7B models |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vllm memory cap | `0.8` for generation (no actor/critic competing for memory) |

## Failure modes

- **OOM at vllm load.** Generation often uses higher `gpu_memory_utilization` than training (no actor/critic share the GPU). If a recipe-default `0.5` was inherited from training, raise it; conversely if you're on a smaller GPU than training was sized for, lower it.
- **Prompt format mismatch.** The prompts parquet's `prompt` column must be the canonical chat-message list. If it's a raw string column, generation will fail — the harness should validate this at `sanity_rollout`-equivalent time (one trial generation on row 0 before submitting the whole batch).
- **Missing output_path.** verl uses `+data.output_path` (the `+` is Hydra append, because the base config doesn't declare `data.output_path`). Forgetting the `+` raises a config-key-missing error.
- **TP > num_gpus_per_replica.** `num_replicas = (n_gpus_per_node × nnodes) // tp_size`. If TP ≥ total GPUs, `num_replicas = 0` and generation hangs forever waiting for a server that was never started.

## Output: `workspace/generate/generate_report.md`

```markdown
# Generate report

## Source
- checkpoint: <ckpt path>
- prompts_path: <input parquet>
- output_path: <output parquet>

## Sampling params
- n: 1
- temperature: 0.0
- top_p: 0.7
- top_k: 50
- prompt_length: 2048
- response_length: 1024

## Compute
- target: local-slurm (from compute_choice.md)
- nodes: 1, gpus_per_node: 8, tp: 2 → 4 replicas
- wallclock: 12 min

## Output stats
- rows_in: 1319
- rows_out: 1319    (per-prompt; multiply by `rollout.n` for total responses)
- output_parquet_size: 8.4 MB
- columns: [prompt, data_source, ability, reward_model, extra_info, responses]   # `responses` is the new column

## Sample (row 0)
- prompt: …
- ground_truth: …
- response[0]: …
```

## Things you must not do

- Do not enable `algorithm.adv_estimator` or any `actor.*` / `critic.*` knobs — generation is rollout-only; those fields are ignored or produce errors.
- Do not generate without setting `+data.output_path` — silent OOM on the in-memory result list at large dataset sizes.
- Do not run generation co-located with a training job on the same GPUs; vllm will OOM or fight for KV cache.
