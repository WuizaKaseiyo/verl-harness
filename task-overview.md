# verl-harness — Drive an agent through a full verl training run

## Overview

This harness drives a FastHarness-compatible agent through the full lifecycle of a [verl](https://github.com/volcengine/verl) training run: it reads the user's training intent (which algorithm, which dataset, which model, what compute), locates the right recipe inside the verl repo, prepares the dataset (using verl's built-in preprocessing or auto-generating one for an unknown HuggingFace dataset), picks a compute target (local GPU, local Slurm login node, or remote Slurm via ssh), provisions the environment, launches the training job, monitors it to terminal status, and writes a summary report. It is a **Category B** harness — a workflow demo that produces a concrete output (a trained checkpoint + a run report).

**What this harness does not do.** It does not invent algorithms; the user names the trainer (e.g. PPO, GRPO, SFT) and the harness binds to whatever verl scripts exist for that trainer. It does not maintain a curated list of "supported verl trainers" — if the user names a trainer verl supports, the harness tries to drive it; if it doesn't, the harness halts honestly. It does not fabricate metrics: if the training job crashes, the run report records the crash, not a synthesised loss curve.

### When to use this harness

When you have a verl checkout, a dataset (named or referenceable by HuggingFace id), a model (HF id or local path), an algorithm, and either a GPU machine or a Slurm cluster, and you want an agent to drive the whole "go from spec to running job to trained checkpoint" pipeline without you sitting through it.

## Workflow Diagram

```
┌────────────┐
│   intake   │  parse the user's training intent — algorithm, dataset, model, compute pref,
│            │  scale knobs, where verl lives, output dir
└─────┬──────┘
      ▼
┌──────────────────┐
│  locate_recipe   │  find the verl example/recipe script that matches the trainer + model,
│                  │  or fall back to constructing a launch command from verl.trainer.main_*
└─────┬────────────┘
      ▼
┌──────────────────┐    unknown dataset    ┌────────────────────┐
│  prepare_data    │──────────────────────▶│ generate_preprocess │
│                  │◀──────────────────────│ (write a verl-style │
│  (verl preprocess script for known        │ preprocess script   │
│   datasets; produces parquet)             │ for an HF dataset)  │
└─────┬────────────┘                       └────────────────────┘
      ▼
┌──────────────────┐
│  select_compute  │  decide: local-direct | local-slurm | ssh-slurm
└─────┬────────────┘
      ▼
┌──────────────────┐
│  provision_env   │  python venv, install verl deps, check torch+CUDA, HF token,
│                  │  resolve model weights (local cache or HF download), wandb env
└─────┬────────────┘
      ▼
┌──────────────────┐
│ launch_training  │  three sub-paths:
│                  │   • local-direct: bash the training script in background
│                  │   • local-slurm: sbatch the slurm template here on the login node
│                  │   • ssh-slurm: ssh into a remote login node and sbatch
└─────┬────────────┘
      ▼
┌──────────────────┐
│ monitor_training │  loop: poll job status / tail logs / watch for OOM/NaN/crash.
│                  │  exits to summarize on (a) success, (b) crash, (c) user cancel.
└─────┬────────────┘
      ▼
┌──────────────────┐
│    summarize     │  collect final checkpoint path, training metrics, wandb URL, crash info,
│                  │  produce a run report
└─────┬────────────┘
      ▼
┌──────────────────┐
│    finalize      │  terminal — hand the run report and pointers to the user
└──────────────────┘
```

## Starting State
states/intake.md

## Human in the Loop
allowed

Several states pause for user confirmation by default — choosing a recipe when multiple match (`locate_recipe`), confirming dataset destination and quota (`prepare_data`), confirming the compute target picked (`select_compute`), and confirming the final launch command before it starts spending GPU time (`launch_training`). These checkpoints can be skipped by passing `--no-hitl` in the invocation; the harness records that escape in the run log.

## Required Capabilities

- filesystem.read
- filesystem.write
- shell.exec
- code.execute            # for running verl's Python preprocess scripts (e.g., gsm8k.py)
- web.search              # optional but recommended: finding HF dataset / model docs
- web.fetch               # fetching HF dataset cards, model cards, dataset schemas
- slurm.access            # CUSTOM token — host can run `sbatch` / `squeue` / `scancel` locally (i.e., this machine is a Slurm login node). If not present, the harness silently disables the `local-slurm` branch of `select_compute`.
- ssh.exec                # CUSTOM token — host has an ssh client and a configured remote login node (the agent runs `ssh <login> sbatch …`). If not present, the harness silently disables the `ssh-slurm` branch.
- gpu.access              # CUSTOM token — at least one CUDA-visible GPU on the host. If not present, the harness silently disables the `local-direct` branch.

At least one of `slurm.access` / `ssh.exec` / `gpu.access` must be present, or no training can run and `select_compute` halts with an error.

## Notes

- **Paths.** The harness assumes a verl checkout exists somewhere on disk and refers to it as `VERL_ROOT`. The user passes it in via the invocation (e.g., "run the harness with verl at /opt/verl") or by setting the `VERL_HOME` env var. If neither is set, `intake` asks. All references to verl's example scripts, slurm templates, and `verl.trainer.main_*` modules are anchored to `VERL_ROOT`.
- **Workspace layout.** All run artefacts live under `runs/<run_id>/workspace/`. Per-state deliverables go in well-known subdirectories: `workspace/intake/`, `workspace/recipe/`, `workspace/dataset/`, `workspace/compute/`, `workspace/env/`, `workspace/job/`, `workspace/logs/`, `workspace/summary/`. The training job's *own* output (checkpoints, the trainer's log files, the slurm `.out` / `.err` files) lives wherever the user pointed it (an `OUTPUT_DIR`) — we record the path under `workspace/job/output_dir.txt` rather than copying the entire checkpoint into the workspace.
- **Honesty over impressiveness.** If the training job OOM-crashes, the summary says so plainly. If only one step ran before a NaN, the summary reports the one step. The harness must never fabricate a metric, a checkpoint path, or a "success" verdict that didn't happen.
- **Cost awareness.** Training jobs can cost hundreds of GPU-hours. Before `launch_training` actually fires the job (if HITL is allowed), the agent must present the final launch command and the expected node-hours and ask the user to confirm.
- **Cancellation.** During `monitor_training`, if the user sends a stop signal (or `--cancel` is set externally), the agent issues `scancel` (Slurm) or kills the local process and writes a `cancelled` summary.
- **Reproducibility.** The harness records the exact training command, env-vars, seeds, and `git log -1` of the verl repo at launch time, so a second run can be re-launched verbatim.
- **American English in all written artefacts.**
