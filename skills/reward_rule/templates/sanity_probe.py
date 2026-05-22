"""Sanity probe: load model + sample one response + invoke reward fn.

Used by `states/sanity_rollout.md`. Generic across all four reward kinds — the
reward-invocation step branches on `reward_kind` from reward_config.md.

Run from the worker that will host training (provision_env's launch_env.sh
sourced first). Reads workspace paths from the WORKSPACE env var.
"""
from __future__ import annotations

import importlib
import json
import os
import statistics
import sys
import time
from typing import Any

import pyarrow.parquet as pq


WORKSPACE = os.environ["WORKSPACE"]


def _read_md_kv(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if ":" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition(":")
                out[k.strip().lstrip("-").strip()] = v.strip()
    return out


def load_actor() -> tuple[Any, Any, int]:
    """Returns (llm, tokenizer, max_response_length).

    Imported lazily — vLLM imports torch+CUDA, which can take ~10s.
    """
    from vllm import LLM
    recipe = _read_md_kv(os.path.join(WORKSPACE, "recipe", "recipe.md"))
    intent = _read_md_kv(os.path.join(WORKSPACE, "intake", "training_intent.md"))
    model_path = intent.get("model") or recipe.get("MODEL_PATH")
    max_prompt = int(recipe.get("data.max_prompt_length", "512"))
    max_response = int(recipe.get("data.max_response_length", "256"))
    tp = int(recipe.get("actor_rollout_ref.rollout.tensor_model_parallel_size", "1"))
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=tp,
        max_model_len=max_prompt + max_response,
        dtype="bfloat16",
    )
    return llm, llm.get_tokenizer(), max_response


def sample_response(llm, tokenizer, prompt: list[dict], max_response: int) -> str:
    from vllm import SamplingParams
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_response, n=1)
    return llm.generate([text], sp)[0].outputs[0].text


def invoke_reward(reward_config: dict[str, str], row: dict) -> dict | float:
    """Branch on reward_kind and call the resolved compute_score."""
    kind = reward_config.get("kind") or reward_config.get("Kind")
    data_source = row["data_source"]
    response = row["__response__"]  # injected by the caller
    ground_truth = row["reward_model"]["ground_truth"]
    extra_info = row.get("extra_info") or {}

    if kind in ("custom", "shaped"):
        sys.path.insert(0, os.path.join(WORKSPACE, "reward"))
        mod = importlib.import_module("compute_score")
        return mod.compute_score(data_source, response, ground_truth, extra_info)
    if kind == "rule":
        # The built-in path is recorded in reward_config.md.
        module_path = reward_config.get("module") or reward_config.get("built_in_path") or ""
        if module_path.endswith(".py"):
            spec = importlib.util.spec_from_file_location("reward_mod", module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        else:
            mod = importlib.import_module(module_path)
        return mod.compute_score(data_source, response, ground_truth, extra_info)
    if kind == "model":
        # RM-based reward: load the RM and score. Out of scope for this template
        # since RM loading is heavyweight — caller (sanity_rollout state) handles it.
        raise NotImplementedError("RM-based sanity invocation: caller-handled.")
    raise ValueError(f"Unknown reward kind: {kind!r}")


def main(num_rows: int = 10) -> dict:
    intent_path = os.path.join(WORKSPACE, "intake", "training_intent.md")
    dataset_path = os.path.join(WORKSPACE, "dataset", "dataset.md")
    reward_path = os.path.join(WORKSPACE, "reward", "reward_config.md")
    dataset = _read_md_kv(dataset_path)
    train_parquet = dataset.get("train_files", "").split()[0]
    table = pq.read_table(train_parquet).slice(0, num_rows).to_pylist()

    started = time.time()
    llm, tok, max_resp = load_actor()
    load_seconds = time.time() - started

    reward_config = _read_md_kv(reward_path)

    rows_out = []
    for row in table:
        response = sample_response(llm, tok, row["prompt"], max_resp)
        row["__response__"] = response
        reward = invoke_reward(reward_config, row)
        rows_out.append({
            "prompt_preview": row["prompt"][0]["content"][:200] if row["prompt"] else "",
            "response": response,
            "ground_truth": row["reward_model"]["ground_truth"],
            "reward": reward,
        })

    # Distribution stats on summed reward (dicts get summed for the stat).
    def _scalar(r):
        return sum(r.values()) if isinstance(r, dict) else float(r)
    scalars = [_scalar(r["reward"]) for r in rows_out]
    non_zero = sum(1 for v in scalars if abs(v) > 1e-9)
    report = {
        "rows": rows_out,
        "distribution": {
            "min": min(scalars), "median": statistics.median(scalars), "max": max(scalars),
            "non_zero_rate": non_zero / max(1, len(scalars)),
        },
        "load_seconds": load_seconds,
        "verdict": "green" if (non_zero / max(1, len(scalars))) >= 0.05 else "fail",
    }
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
