#!/usr/bin/env python3
"""Copy this policy into an existing RoboSynChallenge checkout."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ignore_runtime_artifacts(directory, names):
    ignored = set(shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.egg-info", ".cache",
        "*.pt", "*.pth", "*.bin", "*.safetensors",
    )(directory, names))
    # Only omit downloaded snapshots at the package root. Transformers also
    # has a source directory named "models", which must be copied in full.
    if Path(directory) == ROOT:
        ignored.add("models")
    return ignored


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
    shutil.copytree(ROOT, target, ignore=ignore_runtime_artifacts)
    # Weights stay external; retain the bundled metadata and download notes.
    shutil.copytree(ROOT / "models", target / "models",
                    ignore=shutil.ignore_patterns("*.pt", "*.pth", "*.bin", "*.safetensors", ".cache"))
    print(f"Installed: {target}")


if __name__ == "__main__":
    main()
