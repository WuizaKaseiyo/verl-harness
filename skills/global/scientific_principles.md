Cross-cutting principles for the whole verl-harness run. These bind every state.

## Honesty is the cardinal rule

The output of this harness is a *training run*: real compute spent, real checkpoints written, real logs accumulated. Lying about what happened wastes more compute next time and erodes the user's trust in the harness's reports. Therefore:

- **Never report a checkpoint that does not exist.** If `<output_dir>/checkpoints/global_step_500/` is not on disk, the run did not produce that checkpoint. The summary may not name it.
- **Never report a metric the trainer did not log.** Every number in the summary and the final report traces to a row in `workspace/logs/progress.csv`, which itself was parsed from the trainer's stdout. If the trainer never printed a reward line, the summary says "reward not logged", not a guess.
- **A crashed run is reported as crashed.** Soften nothing. The crash report is itself a complete deliverable.
- **An empty / unhelpful slurm queue state is reported as that.** If `squeue` returns empty before the job appears, that is a real-world condition, not a bug to hide.
- **Costs are stated up-front.** Before `launch_training` actually fires the job, the user sees the expected node-hours and confirms. If the harness is wrong about the cost estimate, that's a known limitation, not a reason to skip the disclosure.

## Scope discipline

- All run artefacts live under `runs/<run_id>/workspace/`. Do not write outside it. The training job's own outputs (checkpoints, the trainer's logs, slurm `.out` files) live wherever the user pointed `output_dir`; the harness records *the path*, not a copy.
- The harness is the *runtime*, not the trainer. It does not invent algorithms, write training code, or modify the verl repo's source tree. The verl checkout is read-only from the harness's perspective.
- "Cheap before expensive" — every state runs cheap checks before triggering expensive actions. `provision_env` dry-runs sbatch before submission; `launch_training` shows the cost gate before spending GPU hours; `monitor_training` checks job status frequently enough that crashes are caught early.

## Defaults

Unless the user overrides them in `intake`:

- `compute_pref`: `auto` (let `select_compute` decide).
- `output_dir`: `<VERL_ROOT>/outputs/<run_id>/`.
- `seed`: 1.
- `wandb`: disabled.
- HITL: every documented checkpoint pauses. `--no-hitl` skips all of them and records the escape in the run log.
- Polling intervals during `monitor_training`: 30 s (local-direct), 60 s (local-slurm), 90 s (ssh-slurm).

## Tone

- American English.
- Quote tool output verbatim (squeue lines, srun output, trainer log lines). Never paraphrase a numerical result.
- Be precise about *who* did what: the harness reports its own actions; the trainer reports its own; do not blur the two.
