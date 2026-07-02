# verl-harness-web

Live dashboard for [verl-harness](https://github.com/WuizaKaseiyo/verl-harness) training runs. Dark cyberpunk-neon palette, FSM-aware, training-progress-aware.

```bash
uv tool install --from ./web verl-harness-web
verl-harness-web /path/to/verl-harness/
```

Opens `http://127.0.0.1:8766`.

## What it shows

- **FSM diagram** — Mermaid flowchart of the harness; the currently-active state pulses cyan, visited states are violet, terminal is slate, selected is magenta.
- **Skills tree** — every skill folder; skills belonging to the active state are highlighted magenta.
- **Inspector** — compiled markdown of any state / skill / overview, in-place edit + save for `task-overview.md`, `states/*.md`, `skills/**/*.md`.
- **Job card** — target (local-direct / local-slurm / ssh-slurm), PID or slurm jobid, output dir, started time, final-step / final-loss / final-reward / last-checkpoint once the run completes.
- **Progress chart** — live `progress.csv` rendered with Chart.js. Two y-axes: rewards/returns on the left (neon-lime), losses/KL/coefficients on the right (neon-cyan / magenta / violet / amber). Auto-detects the `env_steps` / `step` x-axis.
- **Anomalies** — `anomalies.md` entries colour-coded by severity: OOM/NaN → red glow, NCCL/vLLM/preempt → amber, others → cyan.
- **Log tail** — incremental `job_log.md` with per-line colouring (error red, warn amber, `step:` cyan, "training finished" green). Capped at ~400 KB rendered.

## Modes

- **Live** (default) — `watchfiles` over the harness folder, SSE push, panels auto-refresh every 5 s.
- **Static** (`--static`) — no watcher, polls only on explicit reload.

## Run

From the repo root:

```bash
uv run --project web verl-harness-web .
```

Or install once:

```bash
uv tool install --from ./web verl-harness-web
verl-harness-web ~/projects/some-other-verl-harness/
```

Options:

```bash
verl-harness-web <harness-path>            # live mode, opens browser at :8766
verl-harness-web <harness-path> --static
verl-harness-web <harness-path> --port 9000 --no-open
```

## Design

Observer + light editor. It does **not** execute the harness — it watches the workspace directory and renders what's there. The agent driving the FSM is whatever you point at this harness; `verl-harness-web` is the windshield. The only writes it permits are to state and skill `.md` files (and `task-overview.md`); everything else under `runs/` and outside the harness is read-only.

Endpoints under `/api/`:

| Endpoint                | Purpose                                        |
|-------------------------|------------------------------------------------|
| `/api/config`           | runtime config (live mode, root)               |
| `/api/harness`          | parsed harness (states, skills, Mermaid src)   |
| `/api/run`              | latest run state (current state, log, status)  |
| `/api/state/{name}`     | compiled state file + skills list              |
| `/api/skill?path=...`   | compiled skill folder                          |
| `/api/file?path=...`    | raw r/w of editable `.md`                      |
| `/api/progress`         | parsed `progress.csv` → series                 |
| `/api/anomalies`        | `anomalies.md` → severity-classified rows      |
| `/api/job`              | `job_info.md` + `job_status.md` → dict         |
| `/api/logs?since=N`     | incremental tail of `job_log.md`               |
| `/api/summary`          | `summary.md` / `final_report.md`               |
| `/events`               | SSE — filesystem changes                       |

## License

Apache-2.0 — matches the verl-harness repo.
