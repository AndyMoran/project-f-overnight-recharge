"""
Pytest configuration.
=====================

This file exists so that `pytest` run from the project root discovers
both `src/` and `tests/` as importable packages without requiring the
user to set PYTHONPATH or install the project.

Without this file, `tests/test_matching.py` would fail to import
`src.matching` because pytest's default rootdir discovery doesn't
automatically add the project root to sys.path.
"""
import sys
from pathlib import Path

# Insert the project root (the directory containing this file) at the
# front of sys.path so `from src.xxx import ...` works in all tests.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
