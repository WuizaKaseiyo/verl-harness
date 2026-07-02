"""CLI entry: `verl-harness-web <harness-path>`."""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from .server import create_app


def main() -> None:
    p = argparse.ArgumentParser(
        prog="verl-harness-web",
        description="Live dashboard for verl-harness runs.",
    )
    p.add_argument("harness_path", type=Path, help="Path to a verl-harness folder.")
    p.add_argument("--port", type=int, default=8766,
                   help="Port to bind (default 8766).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--static", action="store_true",
                   help="Disable filesystem watcher (no live updates).")
    p.add_argument("--no-open", action="store_true",
                   help="Do not auto-open a browser tab.")
    args = p.parse_args()

    root = args.harness_path.resolve()
    if not (root / "task-overview.md").exists():
        print(f"error: {root} does not look like a harness folder "
              f"(no task-overview.md).", file=sys.stderr)
        sys.exit(2)

    app = create_app(root, live=not args.static)

    url = f"http://{args.host}:{args.port}"
    print(f"verl-harness-web — {root}")
    print(f"  serving at {url}")
    if not args.no_open:
        def _open():
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
