"""Whole-FSM graph — loads every `states/*.md` and exposes the transition graph.

Cycles are allowed only when explicitly declared: the edge that closes the
cycle must carry `**Loop:** max_iterations: N` on its transition. This mirrors
the invariant in `tools/validate_harness.py`, but reimplemented here so the
runtime has no dependency on that older validator.

The graph minus declared loop edges must be a DAG in which every state
reaches at least one terminal state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from harness.context import (
    ContextError,
    StateContext,
    Transition,
    load_state_context,
)

DEFAULT_ENTRY = "intake"


class FSMError(Exception):
    """Malformed FSM (missing target, undeclared cycle, orphan state, …)."""


@dataclass(frozen=True)
class State:
    name: str
    context: StateContext

    @property
    def transitions(self) -> list[Transition]:
        return self.context.transitions

    @property
    def is_terminal(self) -> bool:
        return self.context.is_terminal


@dataclass
class FSM:
    """The parsed graph. Build via `FSM.load(workdir)` — never instantiate directly."""

    states: dict[str, State]
    entry: str = DEFAULT_ENTRY

    # ── graph queries ─────────────────────────────────────────────────────

    def state(self, name: str) -> State:
        try:
            return self.states[name]
        except KeyError as e:
            raise FSMError(f"unknown state {name!r}") from e

    def next_states(self, name: str) -> list[str]:
        return list(self.state(name).context.next_states)

    def is_terminal(self, name: str) -> bool:
        return self.state(name).is_terminal

    def terminal_states(self) -> list[str]:
        return sorted(n for n, s in self.states.items() if s.is_terminal)

    def transition(self, name: str, target: str) -> Transition | None:
        for t in self.state(name).transitions:
            if t.target == target:
                return t
        return None

    def loop_edges(self) -> set[tuple[str, str]]:
        """Directed edges declared as loop back-edges (with `**Loop:**` marker)."""
        out: set[tuple[str, str]] = set()
        for name, s in self.states.items():
            for t in s.transitions:
                if t.loop_max_iterations is not None:
                    out.add((name, t.target))
        return out

    def all_edges(self) -> set[tuple[str, str]]:
        return {
            (name, t.target)
            for name, s in self.states.items()
            for t in s.transitions
        }

    def acyclic_edges(self) -> set[tuple[str, str]]:
        """Edges after removing declared loop back-edges — should form a DAG."""
        return self.all_edges() - self.loop_edges()

    # ── validation ────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        errors: list[str] = []
        names = set(self.states)

        # 1. Every transition target must exist as a state file.
        for name, s in self.states.items():
            for t in s.transitions:
                if t.target not in names:
                    errors.append(
                        f"states/{name}.md: transition target {t.target!r} has no state file"
                    )

        # 2. On the acyclic subgraph, every state must be reachable from entry.
        if self.entry not in names:
            errors.append(f"entry state {self.entry!r} missing")
            return errors

        # Restrict adjacency to known targets so bogus edges don't crash the
        # downstream graph algorithms — those errors are already reported.
        adj_acyclic: dict[str, set[str]] = defaultdict(set)
        for src, dst in self.acyclic_edges():
            if dst in names:
                adj_acyclic[src].add(dst)
        adj_all: dict[str, set[str]] = defaultdict(set)
        for src, dst in self.all_edges():
            if dst in names:
                adj_all[src].add(dst)

        reachable = {self.entry}
        frontier = [self.entry]
        while frontier:
            cur = frontier.pop()
            for nxt in adj_all[cur]:
                if nxt not in reachable:
                    reachable.add(nxt)
                    frontier.append(nxt)
        for orphan in sorted(names - reachable):
            errors.append(f"states/{orphan}.md: unreachable from {self.entry!r}")

        # 3. The acyclic subgraph must be a DAG.
        cycle = _find_cycle(names, adj_acyclic)
        if cycle:
            errors.append(
                f"undeclared cycle on the loop-free subgraph: "
                + " → ".join(cycle)
                + " (add a **Loop:** max_iterations marker to break it)"
            )

        # 4. Every state must reach a terminal on the acyclic subgraph
        #    (once all loops eventually exhaust their bound).
        terms = set(self.terminal_states())
        if not terms:
            errors.append("no terminal state exists — every state must drain into one")
        else:
            reverse_adj: dict[str, set[str]] = defaultdict(set)
            for src, dst in self.acyclic_edges():
                if src in names and dst in names:
                    reverse_adj[dst].add(src)
            drainable = set(terms)
            frontier = list(terms)
            while frontier:
                cur = frontier.pop()
                for pred in reverse_adj[cur]:
                    if pred not in drainable:
                        drainable.add(pred)
                        frontier.append(pred)
            for name in sorted(names - drainable):
                errors.append(
                    f"states/{name}.md: no loop-free path reaches a terminal state"
                )

        return errors

    # ── loading ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, workdir: Path, *, entry: str = DEFAULT_ENTRY, validate: bool = True) -> "FSM":
        """Load every `states/*.md` under `workdir`. Optionally validates."""
        states_dir = workdir / "states"
        if not states_dir.is_dir():
            raise FSMError(f"states/ dir not found under {workdir}")

        states: dict[str, State] = {}
        for md_path in sorted(states_dir.glob("*.md")):
            name = md_path.stem
            try:
                ctx = load_state_context(workdir, name)
            except ContextError as e:
                raise FSMError(f"failed to load state {name!r}: {e}") from e
            states[name] = State(name=name, context=ctx)

        if entry not in states:
            raise FSMError(f"entry state {entry!r} not among loaded states")

        fsm = cls(states=states, entry=entry)
        if validate:
            errs = fsm.validate()
            if errs:
                raise FSMError(
                    f"FSM validation failed ({len(errs)} error(s)):\n  "
                    + "\n  ".join(errs)
                )
        return fsm


def _find_cycle(nodes: set[str], adj: dict[str, set[str]]) -> list[str] | None:
    """DFS cycle detection; returns node list if any cycle, else None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in nodes}
    parent: dict[str, str | None] = {n: None for n in nodes}

    def _dfs(u: str) -> list[str] | None:
        color[u] = GRAY
        for v in sorted(adj.get(u, ())):
            if color[v] == WHITE:
                parent[v] = u
                cyc = _dfs(v)
                if cyc:
                    return cyc
            elif color[v] == GRAY:
                # Back-edge — reconstruct the cycle u → ... → v → u
                cycle = [v]
                cur: str | None = u
                while cur is not None and cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.reverse()
                cycle.append(v)
                return cycle
        color[u] = BLACK
        return None

    for start in sorted(nodes):
        if color[start] == WHITE:
            cyc = _dfs(start)
            if cyc:
                return cyc
    return None
