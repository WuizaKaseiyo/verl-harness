reward_shaping skill — `reward_kind: shaped`. Compose multiple reward components with weights, normalisation, and stability guards.

## When to use

- A single reward (rule or model) under-specifies the desired behaviour. Examples:
  - "Correct answer **and** in `\boxed{}` format **and** under 200 tokens."
  - "RM score **plus** a small bonus for citing a source."
  - "Math correctness **minus** a penalty for revealing the chain-of-thought outside `<think>` tags."
- The user wants explicit, separately-loggable component rewards rather than a black-box scalar.

## Composition pattern

A shaped reward is *almost always* realised as a `reward_custom` function whose `return` is a dict. The shaping is the choice of components and their weights:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    components = {
        "correctness":   1.0 * _correctness(solution_str, ground_truth),
        "format_bonus":  0.2 * _format_ok(solution_str),
        "length_penalty": -0.005 * max(0, _approx_tokens(solution_str) - 200),
        "style_bonus":   0.1 * _style_score(solution_str),
    }
    return components   # verl sums to a scalar, AND logs each separately
```

The weights are the design knob. `skills/reward_custom/default.md` covers the mechanics of the function itself; this skill is about **how to pick the weights and which components to use**.

## Common components

| Component | Source | Typical weight | Rationale |
|---|---|---|---|
| `correctness` | rule (regex / parser / exec) | 1.0 | The primary signal — never less than half the total magnitude |
| `format_bonus` | regex on response | 0.1 – 0.3 | Encourages the model to use the requested template; small so it doesn't dominate when format is right but content is wrong |
| `length_penalty` | per-token cost beyond budget | -0.005 to -0.05 per token | Counters length hacking from RM rewards; calibrate to `data.max_response_length` |
| `verifier_score` | model-based scorer (NLI, classifier) | 0.3 – 0.7 | Used when a rule-based check is unreliable; consider `reward_kind: model` instead if this dominates |
| `tool_use_bonus` | structured-output parser | 0.1 – 0.2 | For tool-agent training; reward emitting valid tool calls |

## Stability rules

Reward shaping is the most common source of *silent* policy collapse in PPO. Three guards:

1. **Component magnitude balance.** Largest component should not exceed ~3× the smallest non-zero component, otherwise small ones become noise. Print the per-component mean over the first 50 steps; if any component has |mean| > 5× another, retune weights.
2. **Sign discipline.** Penalties should be small relative to rewards. A net-negative reward signal teaches the policy to refuse — common failure mode is `length_penalty` so large that the model truncates to one token.
3. **Component logging.** Always return a dict (not a sum). verl logs each key separately under `reward/<component>/mean` etc.; the dashboard's progress chart can then show each component's trajectory. A run with sum-only logging is un-debuggable.

## Sanity rollout requirements

Beyond the standard `sanity_rollout` checks (`skills/reward_custom`), shaped rewards add:

- Print the per-component mean over 100 sampled responses.
- Plot a histogram of the *summed* reward on 100 responses; confirm it isn't degenerate (e.g., 90% mass on the same value).
- Show the highest- and lowest-rewarded responses verbatim — sanity-check the function actually rewards what the user wants.

## Configuration in `workspace/reward/reward_config.md`

```markdown
# Reward config

## Kind
shaped

## Components (sum to the per-response scalar reward)
| Component | Weight | Source |
|---|---|---|
| correctness | 1.0 | rule (boxed-answer regex) |
| format_bonus | 0.2 | regex: response contains `\boxed{...}` |
| length_penalty | -0.005 per token over 200 | char-count / 4 |

## Implementation
- file: workspace/reward/compute_score.py
- function: compute_score
- weights baked into the function (not in CLI)

## CLI injection
- reward.custom_reward_function.path=<workspace>/reward/compute_score.py
- reward.custom_reward_function.name=compute_score

## Stability sanity check (filled by sanity_rollout)
- per-component means over 100 samples:
  - correctness: 0.42
  - format_bonus: 0.18
  - length_penalty: -0.08
- summed reward histogram (binned): {-0.5..0.0: 25, 0.0..0.5: 38, 0.5..1.0: 27, 1.0..1.5: 10}
- balance check: max/min |mean| = 0.42 / 0.08 = 5.25 ⚠ (slight imbalance; consider raising length_penalty weight)
```

## Things you must not do

- Do not invent components from thin air. Every component must be checkable by `sanity_rollout` against row-0 data; if the component's signal is invisible at sanity time, it's invisible at training time.
- Do not compose by adding `reward_kind: rule` + `reward_kind: model` paths in the recipe — that asks verl to apply both, which depending on `reward_manager=naive|prime` may or may not do what you want. Compose at the function level (custom dict-returning function), not at the recipe level.
- Do not set a length penalty that, alone, makes the reward negative for any reasonable response. If `length_penalty` swamps `correctness` even for short correct answers, the policy will learn to refuse.
