reward_model skill — `reward_kind: model`. Score responses with a pre-trained reward model (RM).

## When to use

- The task lacks a deterministic correctness check (open-ended generation, dialogue, style alignment, helpfulness).
- A trained reward model exists — typically a Bradley-Terry-style model fine-tuned on preference pairs (e.g., `Skywork/Skywork-Reward-Llama-3.1-8B`, `OpenAssistant/reward-model-deberta-v3-base`, or a model the user trained themselves with verl's RM-training flow when Phase 2 lands).
- The RM's tokenizer can ingest the dataset's `prompt` + the rollout's response.

## How verl realises it

verl exposes a model-based reward through:

```
reward_model.enable=True
reward_model.path=<HF id or local path to the RM>
reward_model.input_tokenizer=<HF id>             # usually the same as reward_model.path
reward_model.micro_batch_size_per_gpu=<N>
reward_model.max_length=<N>                      # truncation for the RM input
reward_model.reward_manager=naive | prime        # how multiple RMs / signals combine
```

The RM is loaded once per training run and held in GPU memory alongside the actor; the rollout pipeline calls it after each rollout batch and writes the scalar reward into the trajectory. The `reward_model.style` column on the parquet row is typically `"model"` for this path.

## Resources cost

A 7B RM running alongside an 8B actor on a single 80GB H100 is **tight**. Plan for one of:
- A smaller RM (DeBERTa-class) on the same GPU.
- Reserve the RM to a separate GPU/node (`reward_model.megatron.*` or `reward_model.fsdp_config.*` placement directives — varies by verl version).
- Lower `rollout.gpu_memory_utilization` to leave RM headroom.

## Pitfalls

- **Tokenizer mismatch.** If `reward_model.input_tokenizer` is the actor's tokenizer instead of the RM's, scores are garbage. Always set it to the RM's HF id explicitly.
- **Score scale.** RMs return arbitrary scalars (Skywork-class: -10 to +10; Bradley-Terry: logits). `reward_manager=naive` uses raw values; for PPO, this often blows up the advantage variance. Normalise via `reward.reward_norm_type=mean_std` (or `clip`).
- **Length bias.** Most off-the-shelf RMs over-reward longer responses. Watch `response/length/mean` during training (see `training_monitor` anomaly thresholds); if it grows toward `data.max_response_length`, add a length penalty (shift to `reward_kind: shaped`).
- **OOM at RM load.** The RM weights load *after* the actor + critic; OOM here is invisible until the first rollout batch. Mitigation: `sanity_rollout` (Phase 1 state) loads everything in advance.

## Configuration in `workspace/reward/reward_config.md`

```markdown
# Reward config

## Kind
model

## RM source
- path: Skywork/Skywork-Reward-Llama-3.1-8B
- tokenizer: Skywork/Skywork-Reward-Llama-3.1-8B
- size_bucket: 8B   # for sanity check vs available GPU memory

## CLI injection
- reward_model.enable=True
- reward_model.path=<path>
- reward_model.input_tokenizer=<tokenizer>
- reward_model.micro_batch_size_per_gpu=4
- reward_model.max_length=2048
- reward_model.reward_manager=naive

## Normalisation
- reward.reward_norm_type=mean_std    # recommended for PPO with RM
- reward.reward_clip=20.0             # generous; tighten on instability

## Sanity check (filled by sanity_rollout state)
- model loaded: <duration> ms, peak GPU mem: <MB>
- row-0 prompt scored: prompt → response → reward = <value>
- 10-row distribution: min=<...>, p50=<...>, max=<...>
```

## Things you must not do

- Do not pick an RM the user did not name. If `reward_kind=model` and the intent file has no `reward_model.path`, halt and ask.
- Do not run RM-based rewards with `reward.reward_norm_type=none` for PPO. Variance is too high; surface the recommendation.
- Do not assume the RM is single-GPU-fittable. Verify in `provision_env` step 4 (model weights resolution) or at `sanity_rollout` (model load smoke).
