# verl-harness

A [FastHarness](https://github.com/nickhezy/FastHarness)-compatible harness that drives an agent through the full lifecycle of a [verl](https://github.com/volcengine/verl) training run: parse the user's intent, find the right recipe in the verl checkout, prepare the dataset (verl's preprocess scripts for known datasets; auto-generated for HuggingFace datasets that aren't in the registry), pick a compute target (local GPU, local Slurm login node, or remote Slurm via ssh), provision the env, launch the job, monitor it, and produce a run report.

This is a **Category B** harness — a workflow demo that produces a concrete output (a trained checkpoint + a structured run report). It does not optimise verl itself; it drives it.

## What it does

- **One agent session** walks the FSM from `intake` to `finalize`.
- **No registry of trainers.** The user names the algorithm (`ppo`, `grpo`, `gspo`, `sft`, …); the harness goes looking for a matching script under the verl checkout's `examples/<algo>_trainer/` or `recipe/<algo>_trainer/`, or falls back to a direct `python -m verl.trainer.main_<algo>` command. If the user names a trainer verl doesn't have, the harness halts honestly.
- **Two dataset paths.** Known datasets (gsm8k, math, hellaswag, full_hh_rlhf, aime, …) bind to verl's existing preprocess scripts. Unknown HuggingFace datasets route through `generate_preprocess`, which writes a verl-compatible preprocess script from one of verl's preprocess templates.
- **Three compute paths.** `local-direct` (run on this host's GPUs), `local-slurm` (sbatch from this Slurm login node), `ssh-slurm` (ssh to a remote login node and sbatch).
- **Monitor + report.** The `monitor_training` state polls until terminal (success / crashed / preempted / cancelled), tails logs, scans for OOM / NaN / NCCL / vLLM crashes, and parses progress lines into a CSV. `summarize` and `finalize` produce branch-aware reports — a crashed run is reported as crashed, with a specific remediation suggestion. The harness never fabricates a metric.

## FSM diagram

```
intake → locate_recipe → prepare_data ⇄ generate_preprocess
                                        ↓
                            select_compute → provision_env → launch_training → monitor_training → summarize → finalize
                                                                                                                   ↑
                                                          (provision_env failure / launch_training failure short-circuit here)
```

See `task-overview.md` for the full diagram and the parsing conventions.

## Layout

```
verl-harness/
├── task-overview.md
├── states/
│   ├── intake.md
│   ├── locate_recipe.md
│   ├── prepare_data.md
│   ├── generate_preprocess.md
│   ├── select_compute.md
│   ├── provision_env.md
│   ├── launch_training.md
│   ├── monitor_training.md
│   ├── summarize.md
│   └── finalize.md
├── skills/
│   ├── intake/             — canonical training-intent fields, how to elicit them
│   ├── verl_recipes/       — recipe scoring, direct-module fallback, recipe.md format
│   ├── dataset_registry/   — the ~14 known verl-preprocessable datasets + column conventions
│   ├── dataset_autogen/    — author a verl preprocess script from an HF dataset schema
│   ├── compute_select/     — capability probes (gpu/slurm/ssh) and target selection
│   ├── compute_local/      — local-direct provisioning, launch, monitoring
│   ├── compute_slurm/      — local-slurm provisioning, launch, monitoring
│   ├── compute_ssh_slurm/  — ssh-slurm provisioning, launch, monitoring
│   ├── training_monitor/   — polling cadences, terminal conditions, anomaly patterns, progress parsing
│   └── global/             — honesty principle, scope discipline, defaults
└── runs/                   — per-execution workspace dirs (gitignored)
```

## How to invoke

From any FastHarness-compatible host, invoke the FastHarness `run-harness` skill on this folder:

```
run-harness /Users/steven/verl-harness
```

The harness asks for `verl_root` (or reads `$VERL_HOME`), the algorithm, the model, the dataset, and the compute preference. From there it pauses at HITL checkpoints for confirmation — recipe selection (when multiple match), prepared-data confirmation, compute-target confirmation, and the cost gate before launching the actual training job. `--no-hitl` skips all pauses; the harness records the escape in the run log.

## Required capabilities

Declared in `task-overview.md`. At least one of `gpu.access`, `slurm.access`, `ssh.exec` must be present (otherwise no training can run). Other capabilities (`filesystem.read`/`write`, `shell.exec`, `code.execute`, `web.search`, `web.fetch`) are mandatory.

## What it does not do

- Does not modify the verl source tree (the verl repo is read-only from this harness's perspective).
- Does not curate "supported trainers" — whatever the user names, the harness tries to bind to a script or trainer module verl has.
- Does not run interactive ssh sessions or interactive Slurm srun shells.
- Does not invent metrics, checkpoints, or success verdicts.

## License

This harness is licensed under the Apache 2.0 License, matching the verl repo's license.
