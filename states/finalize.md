# finalize

## Description

**Terminal state.** Assemble the final report and hand the completed run to the user. Like `summarize`, it branches on what happened — but at a coarser grain (which earlier state was the last to produce a substantive deliverable). The harness ends here.

Entry paths:

- **Normal exit.** Entered from `summarize` after `monitor_training` reached a terminal status (success / crashed / preempted / cancelled) and `summarize` produced `workspace/summary/summary.md`. The final report is a thin wrapper around the summary plus pointers to all artefacts.
- **Provisioning failure.** Entered directly from `provision_env` when the environment is unfixable (verl install failed, model cannot be fetched, slurm partition does not exist). `workspace/env/env_failed.md` exists; no training was attempted.
- **Launch failure.** Entered directly from `launch_training` when sbatch / ssh / local-process launch itself failed. `workspace/job/launch_failed.md` exists; the job never started.

Concretely:

1. **Detect entry path** by checking which of these exist:
   - `workspace/summary/summary.md` → normal exit
   - `workspace/env/env_failed.md` → provisioning failure
   - `workspace/job/launch_failed.md` → launch failure
2. **Write `workspace/final_report.md`** with the structure for the matching path (below).
3. **Tell the user** the one-line headline and point them at `workspace/final_report.md`.

### Final report — normal exit

- The user's training intent (algorithm, dataset, model, compute target) — one paragraph.
- The topline from `summary.md` (success / crashed / preempted / cancelled with the key numbers).
- A pointer to every artefact:
  - Run report — `workspace/summary/summary.md`
  - Training-time log — `workspace/logs/job_log.md`
  - Progress CSV — `workspace/logs/progress.csv`
  - Anomalies — `workspace/logs/anomalies.md` (if any)
  - Final checkpoint — `<output_dir>/checkpoints/...` (resolved path)
  - Final wandb run url — if configured
  - Recipe used — `workspace/recipe/recipe.md`
  - Prepared dataset — `workspace/dataset/dataset.md`
  - Job info — `workspace/job/job_info.md`
- For crashed / preempted: the remediation or resume command from `summary.md`, hoisted to the top.

### Final report — provisioning failure

- The intent, then plainly: "Environment provisioning failed; training never started."
- The failure mode from `env_failed.md`.
- What the user needs to do to unblock.
- A pointer to `workspace/env/env_state.md` (the partial provisioning record) and `env_failed.md`.

### Final report — launch failure

- The intent, then plainly: "Job launch failed; training never started."
- The exact command that was attempted, the exit code, and the stderr from `launch_failed.md`.
- What the user needs to do to unblock (typical causes: invalid slurm directives, ssh credentials, missing partition).
- A pointer to `workspace/job/launch_failed.md`.

## Skills

- skills/builtin-tools
- skills/global

<!--
Terminal state — no `## Next States`. The agent halts here.
-->
