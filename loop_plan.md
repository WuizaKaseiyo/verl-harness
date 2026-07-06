# loop_plan.md — Loop A (reflect) experiment runbook

Purpose: produce the real numbers for the paper's §5.6 "Closed-Loop
Refinement" (currently blue placeholders `[N]`, `[N_s/N]`, `[K]`, `[X]`,
`[Y]`) by running seeded-suboptimal training runs through the `reflect`
loop on a GPU/Slurm cluster. **Paper submission is Friday, July 10, 2026 —
numbers must exist by July 9 or §5.6 falls back to the mock-backend
validation wording.**

Branch: `feature/loop-a-refinement` (do not run this from `main` — the
loop only exists on the branch). The loop mechanics were already validated
end-to-end GPU-free against a mocked backend (2 iterations, plateaued →
kl-delta → target_met; all artifacts at canonical paths; dashboard parses
the cyclic state log), so any failure you hit here is environment/scale,
not FSM wiring.

---

## 0. Get the code onto the server

Either push the branch and clone:

    # on the laptop
    git push -u origin feature/loop-a-refinement
    # on the server
    git clone -b feature/loop-a-refinement https://github.com/WuizaKaseiyo/verl-harness.git

or clone straight from the laptop over ssh:

    git clone -b feature/loop-a-refinement ssh://<laptop>/Users/ding/work/ai_agent/verl_harness/verl-harness

First command on the server (must print exactly this):

    python3 tools/validate_harness.py .
    # -> OK: 16 states; schema, graph, loops, skills, and terminal contracts valid

## 1. Prerequisites checklist

- [ ] verl checkout on the server; set `VERL_HOME` or pass the path at intake.
      **Record `git -C $VERL_HOME log -1 --format=%h`** — the paper's §5.1 red
      placeholder needs it.
- [ ] **Record the agent model name/version** driving the harness (Claude Code
      `/model`) — second §5.1 placeholder.
- [ ] `Qwen/Qwen2.5-3B-Instruct` weights reachable (HF cache or local path).
- [ ] gsm8k + MATH-lighteval preprocessable (verl `examples/data_preprocess/`).
- [ ] Slurm partition + account + H100 GPU allocation (4 GPUs is enough; record
      the count for §5.1).
- [ ] Keep personal notes in `experiment_progress.md` (gitignored by design).
- [ ] Do NOT edit states/skills/tools mid-experiment. If an edit is unavoidable,
      re-run the validator, commit, and record the new harness commit — results
      must map to one harness version.

## 2. Seed matrix

Six runs were originally planned. **Descoped 2026-07-06** to a ~30 GPU-hour
demonstration budget (user decision at the S2 recipe gate, recorded in
`experiment_progress.md`): run **S2 → S6 → S5 only**; S1 is an optional
add-back (~5 GPU-h) if those three land under budget; **S3 and S4 are out of
scope**. §5.6 now reads as a single-seed demonstration (N=1, plus both
controls), not a four-fault-class study. S2 remains the headline.

| id | track | seeded fault (knob) | refine block | expected loop behaviour |
|---|---|---|---|---|
| S1 | SFT-3B / MATH | `optim.lr` 10× too high (e.g. `1e-4` vs `1e-5`) | target: final train loss ≤ **[calibrate from your Table 1 run: SFT-3B landed 0.4–0.8; pick ≈0.55]**, max_iterations 3 | diagnose `degraded_or_unstable`/`plateaued` → lower lr → target met |
| S2 | GRPO-3B / gsm8k | `actor_rollout_ref.actor.kl_loss_coef = 0.1` (vs `0.001`) | target: `critic/score/mean` ≥ 0.90, max_iterations 3 | diagnose `plateaued` (KL 100× over-tight) → restore 0.001 → target met |
| S3 | GRPO-3B / gsm8k | `actor_rollout_ref.rollout.n = 2` (vs 5+) | target: `critic/score/mean` ≥ 0.90, max_iterations 3 | diagnose `improving`-but-slow / noisy → raise `rollout.n` → target met |
| S4 | GRPO-3B / gsm8k | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` sized to OOM | same as S2 | iteration 1 **crashes**; diagnose `crashed` (OOM remediation from `training_monitor`) → shrink batch → recovers |
| S5 | GRPO-3B / gsm8k | none (default config) | target: `critic/score/mean` ≥ **0.995** (deliberately unreachable in the step budget), max_iterations 2 | honesty control: loop must stop `budget_exhausted`, finalize classifies **incomplete**, never success |
| S6 | GRPO-3B / gsm8k | none | **no refine block** | regression control: path must be exactly `summarize → finalize`, no reflect entry in `state_log.md` |

Knob names above follow verl's GRPO CLI surface — verify each against the
matched recipe's `## Key hyperparameters` at `configure_algorithm` time and
prefer the recipe's spelling if it differs.

## 3. Budget and horizons

- Caps (descoped 2026-07-06; calibrated from N11's measured ~70 s/step and its
  `critic/score/mean` crossing 0.90 at step 31, ≥ 0.918 after step 40):
  **S2 `total_training_steps=60`** (SAVE_FREQ/TEST_FREQ 30), **S5/S6
  `total_training_steps=40`** (SAVE_FREQ/TEST_FREQ 20) — ≈ 90 / 60 min
  wall-clock per iteration on 4 GPUs. The claim being tested is workflow
  validity, not optimizer quality — short horizons are legitimate; say so in
  the paper.
- Calibrate targets so the *seeded* config misses but the *corrected* config
  reaches within the cap (your existing paper runs are the calibration source:
  GRPO-3B previously reached ~0.95 reward).
- Expected ≈ 5 training iterations (S2×2, S5×2, S6×1) ≈ **26.5 GPU-h**
  including sanity probes; worst case (S2 needs its 3rd iteration) ≈ **33
  GPU-h**. One working day on the 4-GPU allocation.

## 4. Per-run protocol

1. Fresh intent to the agent, e.g. for S2:

   > Run GRPO on gsm8k with Qwen/Qwen2.5-3B-Instruct on the local slurm
   > cluster (partition <P>, account <A>, 1 node × 4 H100), output_dir
   > /scratch/<you>/loopa-S2. Cap training at 60 steps. Set
   > actor_rollout_ref.actor.kl_loss_coef=0.1. Refine: target
   > critic/score/mean ≥ 0.90 within 3 iterations.

   (The seeded fault goes in as a user knob override so `configure_algorithm`
   records it in the knob table — `reflect` may only delta knobs that appear
   there.)
2. Answer the HITL gates. Keep HITL **on** (approvals are part of the audit
   chain being demonstrated). Expected pauses per iteration: intent confirm
   (first iteration only), algorithm-config confirm, prepared-data confirm,
   sanity-rollout approval (always-on), cost gate only if ≥ 50 node-hours
   (won't fire at these horizons), and **reflect's delta approval** on every
   loop-back.
3. After each iteration, record in `experiment_progress.md`:

   | run | iter | diagnosis | delta proposed | approved? | metric value | wall-clock |

4. Artifact checklist per finished run (all must exist, honest content):
   `workspace/reflect/loop_state.json`, `refinement_plan.md` (per loop-back),
   `iter_<N>/` snapshots, `reflect_report.md`, `workspace/final_report.md`,
   `runs/<id>/meta.json.status` correct (`completed` only for genuine
   target_met), `state_log.md` showing the actual visit sequence.

## 5. Numbers for the paper (§5.6 placeholders)

- `[N]` = seeded runs attempted (descoped design: S2 only → N=1, S1 optional
  add-back; S5/S6 are controls, report separately).
- `[N_s/N]` = seeded runs whose target was reached within the bound.
- `[K]` = median iterations-to-target over successful runs (report max too).
- `[X] → [Y]` = target-metric mean at iteration 1 vs final iteration, averaged
  over successful seeded runs.
- Controls, reported as prose: S5 stopped `budget_exhausted` with no success
  claim (0 false successes); S6 never entered `reflect`.
- Also update §5.1: agent model/version, verl commit, GPU count per run.
- Replace the paper's blue bracket placeholders in `sections/experiments.tex`
  (§5.6) and delete the `% PLACEHOLDER RESULTS` comment; keep the text blue
  until the advisor pass.

## 6. Bring back

- `tar czf loopa-runs.tgz runs/` and copy to the laptop — `runs/*` is
  **gitignored by policy**, so workspaces travel out-of-band. These tars are
  also the replay fixtures for the demo video and the console screenshots
  (re-capture AFTER merging the branch so the header reads 16 states, in
  read-only mode).
- wandb links if enabled; the filled-in per-iteration table.

## 7. Troubleshooting

- Validator fails after any spec tweak → read the error list; undeclared-cycle
  errors mean a transition edit removed/added an edge without a `**Loop:**`
  declaration.
- `reflect` proposes an out-of-table knob → that's a spec violation worth
  recording as a finding; decline the approval and note it.
- `--no-hitl` is allowed for S6 only; for S1–S5 keep approvals on (the
  refinement approval is part of what §5.6 claims). The four always-on gates
  pause regardless.
- `refine.max_iterations` above 3 is clamped to the FSM bound (by design;
  recorded in `loop_state.json`).
- Cluster preemption mid-loop: resume via the normal `resume_train` path; the
  loop state lives in `workspace/reflect/` and survives.
