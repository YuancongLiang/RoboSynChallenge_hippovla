#!/usr/bin/env python3
"""Compatibility entrypoint; implementation ships inside policy/hippovla."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "policy/hippovla/scripts/install_eval_adapter.py"), run_name="__main__")
