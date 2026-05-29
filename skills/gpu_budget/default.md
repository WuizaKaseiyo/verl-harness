gpu_budget skill — estimate the GPU footprint of a training run from model size +
algorithm, and recommend how many GPUs to request, BEFORE asking the scheduler.

Consumed by `select_compute` (to fill `--gpus-per-node` / `tensor_model_parallel_size`)
and feeds `launch_training`'s cost gate. The goal: never request too few and OOM a
queued job 20 minutes in, nor request more than the model needs.

## Why RL needs far more than the model weights

A naive "2 bytes × params" badly under-counts. A full fine-tune RL step holds several
copies on the GPU at once. Per-GPU footprint with **N-way FSDP data parallel** (ZeRO-3 /
full shard, verl's default):

| Component | Bytes | Sharded by N? | Notes |
|---|---|---|---|
| params (bf16) | 2P | yes | |
| grads (bf16) | 2P | yes | |
| Adam m, v (fp32) | 8P | yes | the big one |
| master weights (fp32) | 4P | yes | |
| **= training state** | **16P** | **→ 16P/N** | the classic "16 bytes/param" rule |
| ref policy (KL) | 2P | yes → 2P/N | RL only (frozen reference) |
| critic | 16P | yes → 16P/N | **PPO/gae only** (separate value model) |
| teacher | 2P | yes → 2P/N | distillation only |
| activations | ~batch·seq | per-GPU | ~10× less with gradient checkpointing |
| **rollout engine** | gpu_mem_util·VRAM | **reserved per GPU** | RL only — vLLM/sglang KV+weights, NOT sharded |

**Fits per GPU iff:** `16P/N + ref + critic + activations + gpu_mem_util·VRAM < VRAM − safety`.

The rollout reservation is the subtle part: at `gpu_memory_utilization=0.6` on an 80 GiB
card, vLLM reserves 48 GiB, leaving only ~32 GiB for the training state — so even a small
model can need ≥2 GPUs at high util. Lowering `gpu_mem_util` frees training memory but
shrinks the KV cache (slower / smaller rollouts). This trade-off is why the estimate takes
`gpu_mem_util` as an input.

SFT / distillation have **no** rollout reservation and no critic, so they fit on far fewer
GPUs than RL at the same model size.

## Tensor parallel (TP) escalation

If a single bf16 copy `2P` exceeds one GPU's VRAM, the model cannot even be held for
rollout on one card → tensor parallel is required: `TP = ceil(2P / (VRAM − safety))`, and
the GPU count must be a multiple of TP. Big models (≳30 B on 80 GiB) hit this; the
estimator reports it and, if it still doesn't fit within the cap, says so honestly
(options: multi-node, LoRA, quantized/8-bit optimizer, lower util).

## The estimator

A working tool lives at **`skills/gpu_budget/templates/estimate_gpu_budget.py`**. It reads
model size (parsed from the name like `Qwen3-1.7B`, or from `config.json` dims, or explicit
`--params`), the algorithm, per-GPU VRAM, `gpu_mem_util`, batch/seq, and the account GPU
**cap**, then prints per-N fit and a recommendation. `select_compute` runs it like:

```
python skills/gpu_budget/templates/estimate_gpu_budget.py \
    --model "<model path or id>" --algo <grpo|ppo|sft|distill> \
    --gpu <h100|a100-40|...> --vram-gb <from gpu.access probe> \
    --gpu-mem-util <recipe rollout.gpu_memory_utilization> \
    --train-batch-size <intent/recipe> --seq-len <max_prompt+response> \
    --cap <account GPU limit>
```

## Recommendation rule (what select_compute records)

1. **minimum-to-fit** `N_min` — the smallest GPU count where the run fits with headroom.
   If `N_min > cap` → **halt-and-advise** (model too big for the budget; offer TP/multi-node/
   LoRA/quantization/smaller model). Do not submit a doomed job.
2. **recommended** `N_rec = cap` once it fits — RL/SFT data-parallel throughput scales ~linearly
   with N, so for a fixed step budget, more GPUs = proportionally faster wall-clock, at the same
   node-hours-ish cost (1 node). Use the cap unless the user asks to conserve.
3. Record both + the per-GPU estimate + TP into `compute_choice.md`, and pass `N_rec` as
   `gpus_per_node` into the sbatch directives and the cost-gate node-hours estimate.

This is an **estimate, not a profiler** — always keep the safety headroom and tell the user
"if it OOMs, lower gpu_mem_util / train_batch_size or enable gradient checkpointing." Honesty
rule: never claim a precise number; report the assumptions (FSDP full shard, Adam, the util)
so the user can sanity-check.

## Worked examples (verified with the template)

- **Qwen3-1.7B GRPO, H100, util=0.6, cap=4** → fits at N=2 (N=1 is tight: 78.6/80), recommend 4 for throughput.
- **Qwen2.5-7B GRPO, H100, util=0.5, cap=4** → minimum-to-fit = 4; recommend 4.
- **70B GRPO, H100, cap=8** → one copy 140 GiB > 80 → TP=2; still OOM within 8 → halt-and-advise (multi-node / LoRA / quantize).
