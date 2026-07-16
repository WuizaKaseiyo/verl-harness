"""Entry point: `python -m harness <subcommand> ...` and `verl-harness-runtime ...`."""

from __future__ import annotations

import sys

from harness.cli import main as _cli_main


def main(argv: list[str] | None = None) -> int:
    return _cli_main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
