"""Add the web package's src/ to sys.path so tests can import verl_harness_web
without requiring a package install. Mirrors harness/tests/conftest.py style.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
