Intake skill — turn the user's free-form training request into a structured `training_intent.md` the rest of the harness consumes.

## Canonical fields

These are the fields every downstream state reads from `workspace/intake/training_intent.md`. The intake state's job is to populate them. Fields with no user-supplied value get a documented default (or are left blank for `locate_recipe` to inherit from the recipe).

| Field | Required | Source | Default |
|---|---|---|---|
| `verl_root` | yes | invocation arg, `$VERL_HOME`, or ask | none — must be resolved |
| `algorithm` | yes | user | none — must be supplied |
| `model` | yes | user (HF id or absolute local path) | none |
| `dataset` | yes | user (known name / HF id / parquet path) | none |
| `compute_pref` | no | user | `auto` |
| `nodes` | no | user | inherit recipe default at `locate_recipe` |
| `gpus_per_node` | no | user | inherit recipe default |
| `train_batch_size` | no | user | inherit recipe default |
| `mini_batch_size` | no | user | inherit recipe default |
| `max_prompt_length` | no | user | inherit recipe default |
| `max_response_length` | no | user | inherit recipe default |
| `total_epochs` | no | user | inherit recipe default |
| `output_dir` | no | user | `<verl_root>/outputs/<run_id>/` |
| `seed` | no | user | `1` |
| `wandb.enabled` | no | user | `false` |
| `wandb.project` | conditional | user | required if wandb.enabled |
| `wandb.run_name` | no | user | `<algorithm>-<model_slug>-<dataset_slug>` |
| `hf_token_source` | conditional | user / env | required only for gated models/datasets; default = `$HF_TOKEN` |
| `slurm.partition` | conditional | user | required for slurm targets |
| `slurm.account` | conditional | user | required for slurm targets |
| `slurm.time_limit` | conditional | user | required for slurm targets; recommend ≥ recipe-implied wall-clock |
| `ssh.alias` | conditional | user, `$VERL_HARNESS_REMOTE` | required for ssh-slurm target |

## Resolving `verl_root`

In order, until one succeeds:

1. Invocation arg (e.g., "run the harness with verl at /opt/verl"). The agent records this as the user-supplied path.
2. `$VERL_HOME` env var.
3. Ask the user. Do not guess; do not search the disk.

Once resolved, verify it looks like a verl checkout:

- `<verl_root>/verl/` exists and is a Python package
- `<verl_root>/examples/` exists
- `<verl_root>/requirements.txt` or `pyproject.toml` exists

If any of these checks fails, ask the user to confirm or fix the path.

## Conversation pattern

Default prompt to the user when intake is entered cold:

> I'm the verl-harness intake. To set up a training run I need:
> (1) the path to your verl checkout (or `$VERL_HOME` already set),
> (2) the trainer algorithm (e.g. ppo, grpo, sft),
> (3) the model (HF id or local path),
> (4) the dataset (a known verl name like `gsm8k`, a HuggingFace dataset id, or a local parquet path),
> (5) the compute target (`auto` lets me decide; or `local-direct` / `local-slurm` / `ssh-slurm`).
>
> Anything else (batch sizes, nodes, epochs, output dir, wandb project) — give it if you have a preference; otherwise I'll inherit verl's recipe defaults.

The intake state may pose follow-ups only when a required field is missing or ambiguous. Once all required fields are populated, the harness presents the normalised intent for confirmation (HITL checkpoint) and writes `training_intent.md`.

## Format of `training_intent.md`

```markdown
# Training intent

## Resolved
- verl_root: /opt/verl
- algorithm: grpo
- model: Qwen/Qwen3-4B
- dataset: gsm8k        # known verl name
- compute_pref: auto
- output_dir: /opt/verl/outputs/2026-05-21T13-50-00/
- seed: 1
- wandb.enabled: false

## User-supplied scale knobs
- nodes: 1
- gpus_per_node: 8
- train_batch_size: (inherit from recipe)
- total_epochs: 5

## Slurm fields (only if relevant)
- slurm.partition: (not yet set)
- slurm.account: (not yet set)

## HF token
- hf_token_source: $HF_TOKEN
```

`locate_recipe` and every downstream state read this file. It is the *only* authoritative record of the user's intent.

## Things you must not do

- Do not invent values for required fields (algorithm, model, dataset, verl_root). Ask the user.
- Do not search the disk for a verl checkout — ask the user where it is.
- Do not "helpfully" pick a model the user did not name.
- Do not silently downgrade `compute_pref`. If the user said `local-slurm` and the host has no slurm, `select_compute` halts with an error — intake does not paper over it by switching the preference.
