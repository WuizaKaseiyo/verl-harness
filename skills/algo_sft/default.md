algo_sft skill — SFT (Supervised Fine-Tuning) knobs and pitfalls. The only non-RL trainer in verl, with its own config surface.

## Trainer binding

Two trainer modules; pick by scale:

- **`verl.trainer.sft_trainer`** — single-node FSDP. Use when `nodes × gpus_per_node ≤ 8`.
- **`verl.trainer.sft_trainer_ray`** — multi-node via Ray. Use when `nodes ≥ 2` OR the user has stated a Ray-based fleet.

Config root: `<verl_root>/verl/trainer/config/sft_trainer_engine.yaml` + Hydra subdirs (`data/`, `model/`, `engine/fsdp.yaml`, `optim/fsdp.yaml`, `profiler/`).

## Knobs to surface in `configure_algorithm`

### Data
| Field | What it controls | Typical default |
|---|---|---|
| `data.train_batch_size` | Global batch size | `256` |
| `data.micro_batch_size_per_gpu` | Per-GPU micro-batch (also val batch size) | `4` |
| `data.use_dynamic_bsz` | Pack-by-token-budget rather than fixed micro-batch | `True` |
| `data.max_token_len_per_gpu` | Token budget per GPU when `use_dynamic_bsz=True` | `8192` |
| `data.max_length` | Truncation cap (used when `pad_mode != no_padding`) | `1024` |
| `data.pad_mode` | `"no_padding"` (packed sequences) or `"left"` / `"right"` | `"no_padding"` |
| `data.truncation` | `"error"` (halt on overflow) / `"truncate"` / `"keep"` | `"error"` |
| `data.train_max_samples` | Cap on training rows (-1 = all) | `-1` |
| `data.messages_key` | Column with the chat-message list (multi-turn) | `"messages"` |
| `data.tools_key` | Column with the tool-definition list (multi-turn + tool use) | `"tools"` |
| `data.enable_thinking_key` | Column controlling `<think>` block inclusion (per-row) | `"enable_thinking"` |
| `data.enable_thinking_default` | Default when the per-row key is absent: `none`/`true`/`false` | `"none"` |
| `data.ignore_input_ids_mismatch` | For multi-turn SFT — accept `input_ids` mismatch between per-turn and whole-conversation tokenisations | `False` |

The chat-template knobs (`messages_key`, `tools_key`, `enable_thinking_*`) only fire for **multi-turn** SFT data. Single-turn data uses the simpler `prompt` + `response` flow.

### Model
| Field | What it controls | Typical default |
|---|---|---|
| `model.path` | HF id or local path | — (from intake) |
| `model.use_remove_padding` | Strip pad tokens from packed sequences (cheap, recommended) | `True` |
| `model.enable_gradient_checkpointing` | Memory ↓, time ↑ ~30% | `True` for ≥ 4B models |
| `model.trust_remote_code` | Required for some custom models (e.g., Qwen2-VL) | `False` |

### Optim
| Field | What it controls | Typical default |
|---|---|---|
| `optim.lr` | Learning rate | `1e-5` (10× the RL actor's `1e-6`) |
| `optim.warmup_steps_ratio` | LR warmup as fraction of total steps | `0.03` |
| `optim.lr_scheduler` | `"cosine"` / `"linear"` / `"constant"` | `"cosine"` |
| `optim.weight_decay` | AdamW weight decay | `0.0`–`0.1` |
| `optim.total_training_steps` | Set by trainer if `trainer.total_epochs` is given | auto |

### Trainer / checkpoint
| Field | What it controls | Typical default |
|---|---|---|
| `trainer.total_epochs` | Number of passes over the dataset | `3` |
| `trainer.save_freq` | Save every N steps (-1 = end only) | -1 |
| `trainer.test_freq` | Val every N steps (0 disables) | `0` |
| `checkpoint.save_contents` | What to save: `["model"]`, `["model","optimizer"]`, `["model","optimizer","extra"]` | `["model","optimizer","extra"]` |

## Failure modes

- **Loss includes the prompt tokens.** SFT must train on assistant-turn tokens only. Verl handles this via the chat-template's `add_generation_prompt=True` boundary, BUT if the dataset's `messages_key` shape is wrong (e.g., the response is stuffed into the user turn, or there's no explicit assistant turn), the trainer learns to copy prompts. Catch this at `sanity_rollout` — print row-0 with the masked-out-tokens highlighted.
- **`ignore_input_ids_mismatch=False` causes hard error on Qwen Thinking models.** These models add `<think></think>` tags only to the last turn, so per-turn vs whole-conversation tokenisations differ. If you see `AssertionError: input_ids mismatch` on multi-turn SFT, flip to `True`.
- **`use_dynamic_bsz=True` + small `max_token_len_per_gpu` → tiny effective batches.** The packing tries to fit token budget; if budget is small, each step has < expected micro-batches and grad accumulation becomes flat. Aim for `max_token_len_per_gpu` ≥ `8 × data.max_length`.
- **`pad_mode="right"` with FSDP causes silent stride bugs on some torch versions.** Prefer `no_padding` (sequence packing); it's both faster and avoids the bug class.

## Canonical val metric (for `pick_best_checkpoint`)

`val/loss` — lower is better. Also surface `val/perplexity` (= `exp(val/loss)`) for the dashboard. SFT does not produce reward.

## CLI injection from `configure_algorithm`

```
data.train_batch_size=256
data.micro_batch_size_per_gpu=4
data.use_dynamic_bsz=True
data.max_token_len_per_gpu=8192
data.max_length=2048
data.pad_mode=no_padding
data.truncation=error
data.messages_key=messages                                 # if multi-turn
model.path=<from intake>
model.use_remove_padding=True
model.enable_gradient_checkpointing=True
optim.lr=1e-5
optim.warmup_steps_ratio=0.03
optim.lr_scheduler=cosine
trainer.total_epochs=3
trainer.save_freq=-1
trainer.test_freq=0                                        # 0 = no in-training eval; set > 0 if user wants per-step val
checkpoint.save_contents=["model","optimizer","extra"]
```

## Things you must not do

- Do not surface PPO-family knobs (`critic.*`, `algorithm.adv_estimator`, `rollout.*`) for SFT — those are RL-only and the SFT trainer rejects them.
- Do not run SFT with `pad_mode="no_padding"` AND a `data.max_length` so small that the packing budget falls below one sample — the trainer silently drops examples that don't fit.
- Do not assume the dataset uses `messages_key=messages`. Some datasets use `conversations` or `dialog`; verify against `dataset.md` row-0 sample.
