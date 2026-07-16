# verl-harness-runtime

Self-contained agent runtime that drives the verl-harness [FSM
specs](../states/) against any LLM backend — Anthropic, OpenAI,
OpenRouter, DeepSeek, Qwen, vLLM local, or any other OpenAI-compatible
endpoint.

Sits alongside the existing `web/` dashboard and shares its
[stream-json event format](../web/DESIGN.md), so the dashboard reads
runtime output with no changes.

## Status

**Milestone 2 — done.** The runtime now carries the entire 16-state FSM:

- Multi-state orchestration (`intake` → … → `finalize`)
- Workspace-contract enforcement (transition_to is rejected when the
  declared Deliverables aren't on disk; the model gets a retry)
- Loop cycle bounds (`reflect → configure_algorithm` clamps at
  `max_iterations: 3`; `prepare_data → generate_preprocess` at 1)
- HITL layer (`## Hand-off Points` parsed; the four always-on gates
  fire even under `--no-hitl`)
- `runs/<id>/meta.json` + `workspace/logs/state_log.md` in the exact
  format the existing dashboard reads
- Interrupt handling (Ctrl-C → `status=cancelled`) + `--resume-run`
- OpenAI-compatible wire (OpenRouter / OpenAI / DeepSeek / Qwen /
  vLLM / Together / Groq / anything speaking Chat Completions)

**176 tests green, 3 gated on API keys.**

## Not yet in this runtime

- **Live per-provider smoke** — only one live test ran (OpenRouter /
  gpt-4o-mini, intake state, single-state mode). Multi-state on a real
  model is opt-in via `test_live_multistate` when a key is present.
- **HITL dashboard UI** — approvals go to stdin/stderr today; the web
  approval button is M3.
- **Prompt caching** — Anthropic's `cache_control` isn't wired.
- **Per-provider rate-limit strategy** — best-effort retry only.
- **DPO / RM first-class trainers** — spec-level TODO; runtime treats
  them as "algo without first-class trainer", triggers `Always-on when
  the matching skill emits a halt condition`.

## Install

```bash
pip install -e ./harness
# or, without installing:
export PYTHONPATH=./harness/src
```

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-…      # or OPENROUTER_API_KEY, DEEPSEEK_API_KEY, …

verl-harness-runtime run . \
  --model anthropic/claude-opus-4-8 \
  --goal "train GRPO on gsm8k with Qwen2.5-3B-Instruct on 2xH100" \
  --run-id T15-30 \
  --no-hitl
```

Runs the entire FSM in the background. Watch progress with:

```bash
tail -f runs/T15-30/workspace/logs/state_log.md   # human-readable state transitions
cat runs/T15-30/meta.json                          # status + current_state
uv run --project web verl-harness-web .            # dashboard
```

### Resume an interrupted run

```bash
# Ctrl-C during a run marks status=cancelled + notes current_state.
verl-harness-runtime run . --resume-run T15-30
# Model + goal + verl_root inherited from runs/T15-30/meta.json.
# Runtime re-enters at the last recorded state fresh (workspace preserved).
```

### Single-state (M1 behaviour, for debugging one state at a time)

```bash
verl-harness-runtime run . --single-state \
  --model openrouter/openai/gpt-4o-mini \
  --goal "..." --state intake --run-id single-t
```

## Model spec

```
--model <provider>/<model_id>
```

Split at the FIRST `/` — the rest is the model id. Nested slashes stay
inside `model_id`, so OpenRouter routes like `anthropic/claude-opus-4`
work verbatim as `openrouter/anthropic/claude-opus-4`.

### Built-in providers

| provider | wire | env var | notes |
|---|---|---|---|
| `anthropic`  | anthropic | `ANTHROPIC_API_KEY`  | native |
| `openai`     | openai    | `OPENAI_API_KEY`     | |
| `openrouter` | openai    | `OPENROUTER_API_KEY` | live-smoke tested |
| `deepseek`   | openai    | `DEEPSEEK_API_KEY`   | |
| `qwen`       | openai    | `DASHSCOPE_API_KEY`  | |
| `together`   | openai    | `TOGETHER_API_KEY`   | |
| `groq`       | openai    | `GROQ_API_KEY`       | |
| `vllm`       | openai    | —                    | local `127.0.0.1:8000/v1` |

Override any of these or add your own in
`~/.verl-harness/providers.yaml` (or pass `--provider-config <path>`).
User config deep-merges on top of built-ins.

## Layout

```
harness/
├── pyproject.toml
├── README.md
├── src/harness/
│   ├── cli.py              # arg parsing + backend factory + orchestrator entry
│   ├── providers.py        # <provider>/<model_id> resolution
│   ├── providers.yaml      # built-in provider table
│   ├── context.py          # state.md + skill dirs → system prompt;
│   │                       # deliverable / loop-marker parser; workspace snapshot
│   ├── fsm.py              # loads all states/*.md; transition graph;
│   │                       # declared vs undeclared cycle detection
│   ├── contracts.py        # workspace-deliverable path extraction + enforcement
│   ├── hitl.py             # Hand-off Point parsing; always-on classifier; stdio prompter
│   ├── runlog.py           # meta.json + state_log.md primitives
│   ├── loop.py             # tool-use turn loop with control-tool short-circuit
│   ├── events.py           # Claude Code stream-json emitter
│   ├── state_driver.py     # drive_state(...) — one state end-to-end
│   ├── orchestrator.py     # walk the FSM until terminal / error / cancelled
│   ├── resume.py           # load_resume_plan + flip_status_to_running
│   ├── backends/
│   │   ├── base.py         # Backend abstract + RawEvent
│   │   ├── anthropic.py    # native Anthropic wire
│   │   ├── openai.py       # OpenAI + OpenAI-compatible endpoints
│   │   └── translate.py    # Anthropic ↔ OpenAI wire translation
│   └── tools/
│       ├── base.py         # Tool ABC + ToolContext + ToolError
│       ├── registry.py     # ToolRegistry + default_registry()
│       ├── bash.py
│       ├── read.py
│       ├── edit.py
│       ├── write.py
│       └── todo.py
└── tests/                  # 176 passing across 14 modules
    ├── conftest.py         # adds src/ to sys.path
    ├── test_providers.py           # 11
    ├── test_backends.py            # 2 gated on live key
    ├── test_tools.py               # 24
    ├── test_events.py              # 8
    ├── test_loop.py                # 8
    ├── test_context.py             # 24 (incl. transition + snapshot)
    ├── test_fsm.py                 # 13 (real 16 states + synthetic errors)
    ├── test_contracts.py           # 11
    ├── test_runlog.py              # 12
    ├── test_state_driver.py        # 8
    ├── test_loop_bounds.py         # 5
    ├── test_hitl.py                # 17
    ├── test_orchestrator.py        # 4
    ├── test_resume.py              # 6
    ├── test_full_fsm.py            # 6 (all 16 states covered)
    ├── test_translate.py           # 10
    └── test_cli.py                 # 9 subprocess + 1 live gated
```

## Running the tests

```bash
cd harness
python -m pytest tests/                       # 176 green
ANTHROPIC_API_KEY=sk-ant-… python -m pytest tests/
OPENROUTER_API_KEY=sk-or-… python -m pytest tests/test_cli.py::test_live_smoke_intake
```

## FSM contract (recap)

The runtime is intentionally spec-agnostic — it doesn't know what
"intake" or "GRPO" mean. It parses:

- `## Skills` → which `skills/<dir>/*.md` bundles to load into the prompt
- `## Next States` → target names + Conditions + Deliverables + `**Loop:**`
  back-edge markers
- `## Hand-off Points` → ordinary vs always-on classification (via
  `**Always-on**` / `**Always-on for X**` / `**Threshold-based**` markers)

Everything else (workspace file naming, deliverable file paths,
transition semantics) is enforced by parsing the spec at runtime. Change
the FSM by editing markdown — no runtime code change needed.
