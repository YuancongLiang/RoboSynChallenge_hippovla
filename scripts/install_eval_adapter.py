#!/usr/bin/env python3
"""Copy this policy into an existing RoboSynChallenge checkout."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robosyn_root", type=Path)
    args = parser.parse_args()
    target_root = args.robosyn_root.resolve()
    if not (target_root / "scripts/eval_policy.py").is_file():
        parser.error("robosyn_root must contain scripts/eval_policy.py")
    target = target_root / "policy/hippovla"
    if target.exists():
        parser.error(f"Refusing to overwrite {target}; remove or back it up first")
    shutil.copytree(ROOT / "policy/hippovla", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Installed: {target}")


if __name__ == "__main__":
    main()
