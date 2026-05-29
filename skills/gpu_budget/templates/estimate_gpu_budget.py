#!/usr/bin/env python3
"""Estimate the GPU budget for a verl training run BEFORE requesting resources.

Given the model size + algorithm + a few batch/sequence knobs, estimate per-GPU
VRAM and recommend a GPU count: the *minimum to fit*, and a *throughput target*
capped by the user's account limit. Used by `select_compute` to right-size the
sbatch `--gpus-per-node` instead of guessing or maxing.

Reference implementation — deliberately a CONSERVATIVE heuristic, not a profiler.
It exists to avoid two failure modes: (a) requesting too few GPUs and OOMing 20
minutes into a queued job, (b) requesting more than the model needs. Always keep
the printed headroom; real footprint varies with framework, attention impl,
offload, and activation checkpointing.

Memory model (per GPU), for N-way FSDP data parallel (ZeRO-3 / full shard):
  training_state  = 16 * P / N        # bf16 params(2) + grads(2) + Adam m,v fp32(8) + master fp32(4)
  ref_policy      = 2  * P / N         # frozen KL reference (RL only; FSDP-sharded)
  critic          = 16 * P / N         # PPO/gae only (separate value model)
  activations     ~ act_gb_per_gpu     # batch/seq heuristic, ~10x less with grad ckpt
  rollout_reserve = gpu_mem_util * VRAM # vLLM/sglang KV+weights reservation (colocated)
  FITS if: training_state + ref + critic + activations + rollout_reserve < VRAM - safety

For SFT/distill there is no rollout_reserve, no ref/critic (distill adds a teacher
copy instead). If 2*P (one bf16 copy) alone exceeds one GPU's VRAM, tensor parallel
(TP) is required so the model shards for the rollout engine too.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

# bytes/param for mixed-precision Adam full fine-tune (params2 + grad2 + m4 + v4 + master4 = 16)
ADAM_BYTES_PER_PARAM = 16
BF16 = 2

# Common accelerator VRAM (GiB). Extend as needed.
GPU_VRAM = {"h100": 80, "h800": 80, "a100-80": 80, "a100-40": 40, "a800": 80,
            "l40s": 48, "l40": 48, "a6000": 48, "v100": 32, "4090": 24}


def params_billions(model: str, explicit: float | None, config_dir: str | None) -> tuple[float, str]:
    """Best-effort model size in billions. Order: explicit > name regex > config.json dims."""
    if explicit:
        return explicit, "explicit --params"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model or "")
    if m:
        return float(m.group(1)), f"parsed from name '{model}'"
    cfg = os.path.join(config_dir or model or "", "config.json")
    if os.path.isfile(cfg):
        c = json.load(open(cfg))
        h = c.get("hidden_size"); L = c.get("num_hidden_layers"); v = c.get("vocab_size")
        if h and L:
            # ~12*L*h^2 (attn+mlp) + 2*v*h (embed+lm_head)
            p = (12 * L * h * h + 2 * (v or 0) * h) / 1e9
            return round(p, 3), f"estimated from config.json (h={h}, L={L}, vocab={v})"
    raise SystemExit("cannot determine model size: pass --params <billions>")


def fits(P, N, vram, gpu_mem_util, algo, act_gb, safety_gb):
    state = ADAM_BYTES_PER_PARAM * P / N
    ref = (BF16 * P / N) if algo in RL_ALGOS else 0.0
    critic = (ADAM_BYTES_PER_PARAM * P / N) if algo == "ppo" else 0.0
    teacher = (BF16 * P / N) if algo in ("distill", "distillation") else 0.0
    rollout = (gpu_mem_util * vram) if algo in RL_ALGOS else 0.0
    need = state + ref + critic + teacher + act_gb
    return (need + rollout) < (vram - safety_gb), need, rollout


RL_ALGOS = {"grpo", "ppo", "gspo", "cispo", "gmpo", "sapo", "dppo", "rloo", "remax",
            "gpg", "opo", "reinforce_plus_plus", "gdpo", "grpo_passk"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF id / local path / name (size parsed from it)")
    ap.add_argument("--params", type=float, default=None, help="model size in billions (overrides parsing)")
    ap.add_argument("--config-dir", default=None, help="dir containing config.json")
    ap.add_argument("--algo", default="grpo", help="grpo|ppo|sft|distill|...")
    ap.add_argument("--gpu", default="h100", help=f"gpu type: {','.join(GPU_VRAM)}")
    ap.add_argument("--vram-gb", type=float, default=None, help="override per-GPU VRAM (GiB)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.5, help="rollout.gpu_memory_utilization (RL)")
    ap.add_argument("--cap", type=int, default=8, help="max GPUs you may request (account limit)")
    ap.add_argument("--train-batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=2048, help="prompt+response tokens")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing on (default off)")
    ap.add_argument("--safety-gb", type=float, default=6.0, help="reserved headroom per GPU")
    args = ap.parse_args()

    P, how = params_billions(args.model, args.params, args.config_dir)
    vram = args.vram_gb or GPU_VRAM.get(args.gpu.lower(), 80)
    algo = args.algo.lower()
    # crude activation estimate per GPU: ~ local_batch * seq * 2 bytes * hidden_factor, /10 if ckpt.
    local_batch = max(1, args.train_batch_size // max(1, args.cap))
    act_gb = local_batch * args.seq_len * 2 * 4 / 1e9          # ~4 resident activation copies
    if not args.grad_ckpt:
        act_gb *= 4                                            # no checkpointing -> keep more
    act_gb = round(min(act_gb, vram * 0.25), 2)                # cap the heuristic

    one_copy = BF16 * P
    tp = max(1, math.ceil(one_copy / (vram - args.safety_gb))) if one_copy > (vram - args.safety_gb) else 1

    print(f"# GPU budget estimate")
    print(f"model size     : {P} B  ({how})")
    print(f"algorithm      : {algo}   gpu: {args.gpu} ({vram} GiB)   cap: {args.cap} GPUs")
    print(f"one bf16 copy  : {one_copy:.1f} GiB" + (f"  -> needs TP={tp} (model too big for one GPU)" if tp > 1 else ""))
    print(f"activations~   : {act_gb} GiB/GPU (grad_ckpt={'on' if args.grad_ckpt else 'OFF'})")
    if algo in RL_ALGOS:
        print(f"rollout reserve: {args.gpu_mem_util:.2f} x {vram} = {args.gpu_mem_util*vram:.1f} GiB/GPU (vLLM)")
    print()

    min_fit = None
    for N in range(1, args.cap + 1):
        if N % tp != 0 and tp > 1:
            continue
        ok, need, rollout = fits(P, N, vram, args.gpu_mem_util, algo, act_gb, args.safety_gb)
        used = need + rollout
        tag = "FITS" if ok else "OOM "
        print(f"  N={N:>2}: per-GPU ~{used:5.1f}/{vram:.0f} GiB  [{tag}]"
              + ("" if ok else "  <- training_state {:.1f} dominates".format(need)))
        if ok and min_fit is None:
            min_fit = N

    print()
    if min_fit is None:
        print(f"RESULT: does NOT fit within {args.cap} GPUs on {args.gpu}.")
        print("  options: tensor-parallel + more GPUs / multi-node, smaller model, "
              "lower gpu_mem_util, enable --grad-ckpt, or LoRA/quantized training.")
        return
    rec = args.cap  # throughput: use the cap (DP scales ~linearly) once it fits
    print(f"RESULT")
    print(f"  minimum to fit : {min_fit} GPU(s)" + (f" (TP={tp})" if tp > 1 else ""))
    print(f"  recommended    : {rec} GPU(s)  (fits at {min_fit}; DP up to your cap={args.cap} for ~{rec}x throughput)")
    print(f"  request        : --gpus-per-node={rec}  --nodes=1"
          + (f"  tensor_model_parallel_size={tp}" if tp > 1 else "  tensor_model_parallel_size=1"))
    print(f"  note           : estimate only — keep the {args.safety_gb} GiB headroom; if it OOMs, "
          f"lower gpu_mem_util or train_batch_size, or enable grad checkpointing.")


if __name__ == "__main__":
    main()
