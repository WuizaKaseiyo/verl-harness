"""Template: custom_reward_function for verl.

Authored by verl-harness reward_custom skill.

The function signature is fixed by verl's reward dispatcher. Verl reads:

    reward.custom_reward_function.path=<path/to/this/file>
    reward.custom_reward_function.name=compute_score

…and calls `compute_score(data_source, solution_str, ground_truth, extra_info)`
once per rollout response on the rollout-worker process.

Return either:
- a `float` → used as the per-response scalar reward
- a `dict[str, float]` → verl sums the values for the per-response reward AND logs
  each key separately under `reward/<key>/*` (visible in progress.csv + dashboard).

Keep imports cheap at module load. Do NOT open files or load models at module level;
do that lazily inside the function with a module-level cache if you must.
"""
from __future__ import annotations

import re
from typing import Optional


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict:
    """Return component rewards as a dict; verl sums them."""
    correctness = _match_final_answer(solution_str, ground_truth)
    format_bonus = _check_format(solution_str)
    length_penalty = _length_penalty(solution_str)
    return {
        "correctness": correctness,
        "format_bonus": format_bonus,
        "length_penalty": length_penalty,
    }


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _match_final_answer(response: str, ground_truth: str) -> float:
    if not ground_truth:
        return 0.0
    m = _BOXED_RE.search(response)
    extracted = (m.group(1).strip() if m
                 else (response.strip().split()[-1] if response.strip() else ""))
    return 1.0 if _canonical(extracted) == _canonical(ground_truth) else 0.0


def _canonical(s: str) -> str:
    """Normalise '1/2' == '0.5' == '0.50' etc. — task-specific; extend as needed."""
    s = s.strip().rstrip(".")
    try:
        return f"{float(s):.6g}"
    except ValueError:
        return s


def _check_format(response: str) -> float:
    return 1.0 if _BOXED_RE.search(response) else 0.0


def _length_penalty(response: str, budget_tokens: int = 256) -> float:
    approx_tokens = len(response) // 4
    over = max(0, approx_tokens - budget_tokens)
    return -0.01 * over
