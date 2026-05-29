"""Helper used by summarize / finalize to pick the best checkpoint.

This is a reference implementation. Adapt the column-key fallbacks if the user's
verl logger emits a different namespace (some recipes log to wandb only, etc.).
"""
from __future__ import annotations

import csv
import math
import os
from typing import Optional


# Per-trainer-family canonical val metric. Format: (metric_key, higher_is_better).
# Keys are namespaced verl logger keys; the trainer-side log is the source of truth.
CANONICAL_VAL_METRIC = {
    "sft": ("val/loss", False),
    "ppo": ("val/reward/mean", True),                  # adv_estimator=gae
    "grpo": ("val/reward/mean", True),                 # adv_estimator=grpo / grpo_passk / grpo_vectorized
    "rloo": ("val/reward/mean", True),
    "remax": ("val/reward/mean", True),
    "gpg": ("val/reward/mean", True),
    "reinforce_plus_plus": ("val/reward/mean", True),
    "reinforce_plus_plus_baseline": ("val/reward/mean", True),
    "opo": ("val/reward/mean", True),
    "gdpo": ("val/reward/mean", True),
    "distill": ("val/reward/mean", True),              # when use_task_rewards=True
    # DPO / RM: not first-class in this verl; if external trainer used, expect:
    "dpo": ("val/reward_gap", True),
    "rm": ("val/accuracy", True),
}


def _is_finite(x) -> bool:
    try:
        f = float(x)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def parse_progress_csv(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def pick_best_checkpoint(
    progress_csv: str,
    output_dir: str,
    trainer_family: str,
) -> Optional[tuple[int, float, str]]:
    """Return (best_step, best_value, best_path) or None when unavailable.

    None cases:
    - No val metric was ever logged.
    - The best step's ckpt isn't on disk (save_freq missed it).
    """
    if trainer_family not in CANONICAL_VAL_METRIC:
        return None
    metric_key, higher_is_better = CANONICAL_VAL_METRIC[trainer_family]
    rows = parse_progress_csv(progress_csv)
    candidates = [r for r in rows if metric_key in r and _is_finite(r[metric_key])]
    if not candidates:
        return None
    best = (max if higher_is_better else min)(
        candidates, key=lambda r: float(r[metric_key])
    )
    try:
        best_step = int(float(best.get("training/global_step", "0")))
    except (TypeError, ValueError):
        return None
    # verl persists checkpoints at <output_dir>/global_step_<N>/ (NOT under
    # a `checkpoints/` subdir, despite older docs claiming so — verified
    # against verl 0.8.0.dev's _save_checkpoint at ray_trainer.py:935-975).
    best_path = os.path.join(output_dir, f"global_step_{best_step}")
    if not os.path.isdir(best_path):
        return None
    return (best_step, float(best[metric_key]), best_path)
