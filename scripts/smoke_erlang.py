"""Runs the functional CPU smoke tests for ErlangMDLM."""

from pathlib import Path

import pytest


if __name__ == "__main__":
    test_file = Path(__file__).resolve().parents[1] / "tests/test_erlang_mdlm.py"
    raise SystemExit(pytest.main(["-q", str(test_file)]))
