#!/usr/bin/env python3
"""Compatibility entrypoint for the composed policy runtime."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    bundle = Path(__file__).resolve().parent / "policy/hippovla"
    sys.path[:0] = [str(bundle / "transformer/src"), str(bundle)]
    runpy.run_path(str(bundle / "verify_install.py"), run_name="__main__")
