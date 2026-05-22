algo_rm skill — Reward-model training. Honest take on what this verl supports.

## Important: RM training is not a first-class trainer in this verl checkout

Probing this verl shows:

- **No `verl.trainer.main_rm` / `main_reward_model` module.** RM training has no top-level entry point.
- **`verl/utils/reward_score/`** contains rule-based scorers (gsm8k, math, hellaswag, full_hh_rlhf, …) and `verl/workers/config/reward.py` configures RM **usage** during PPO/GRPO rollouts — but neither trains an RM.
- **`verl/experimental/reward_loop/reward_model.py`** is the closest thing — experimental RM-loop infrastructure for online reward-model updates *during* RL training, not standalone RM training.
- **`examples/data_preprocess/full_hh_rlhf.py`** prepares HH-RLHF pair data (`chosen`/`rejected`) — useful as RM input, but no trainer consumes it for RM training in this checkout.

## What the harness does when the user says `algorithm: rm` (or `reward_model`)

1. **Halt with "RM training is not a first-class trainer in this verl"** — same shape as the `algorithm: dpo` halt in `algo_dpo`.
2. **Point at the alternatives:**
   - Use a pre-trained off-the-shelf RM via `reward_kind: model` (see `skills/reward_model/`) — this is the supported "use an RM" flow.
   - Train an RM out-of-tree (e.g., `trl.RewardTrainer`, OpenRLHF's RM trainer) using the harness's `prepare_data` to prepare pair data (`full_hh_rlhf` template), then return to the harness with the trained RM for the PPO/GRPO run.
   - If the user wants to train an RM *jointly* with the policy (online RM updates), point at `verl/experimental/reward_loop/` and warn it is experimental and not driven by this harness.
3. **Do not lie.** The harness will not pretend to drive a missing trainer.

## If the user accepts the "external trainer" path

`configure_algorithm` records the user's plan (which external trainer, where the resulting RM will be saved) into `workspace/algorithm/algorithm_config.md` but **does not** transition to `launch_training`. Instead the FSM exits the algorithm-binding branch and the user re-enters the harness later for the policy training run with the now-existing RM:

```
intake (algorithm=rm)
  → locate_recipe (notes: out-of-tree RM training)
  → configure_algorithm (records the plan, halts the train branch)
  → finalize (with verdict "out-of-tree-training-required")
```

Then a fresh harness run:

```
intake (algorithm=ppo or grpo, reward_kind=model, reward_model.path=<path the user trained>)
  → ... normal train track
```

## Failure modes

- **User confuses RM training with RM use.** Most "I want to train with reward model X" intents mean "use this pre-trained X as the reward fn during PPO/GRPO". Verify before halting: ask "do you have an existing reward model you want to use, or do you want to train one from scratch?".
- **Data shape mismatch.** RM training expects chosen/rejected pair data. If the user's dataset only has `prompt` + `response` (SFT-shaped) or `prompt` + `ground_truth` (RL-shaped), RM training is not what they want — they probably want SFT or PPO/GRPO with a rule reward.

## When verl gains an RM trainer (future)

If a future verl release adds `verl.trainer.main_rm` (or similar) as a first-class trainer, this skill should be rewritten with:
- Module entry point
- `algorithm.loss_type` for RM (typically `"bradley_terry"`)
- Data column requirements (`chosen` / `rejected` of identical shape)
- Head choice (single scalar vs multi-objective)
- Per-pair vs in-batch normalisation

Until then, this skill's job is to halt honestly.

## Things you must not do

- Do not author a fake `main_rm.py` invocation. verl will not import it.
- Do not silently retarget the run to PPO when the user said RM training; ask them which they want.
- Do not refer the user to "verl's RM trainer" as if it existed; refer them to the experimental loop OR to an external project.
