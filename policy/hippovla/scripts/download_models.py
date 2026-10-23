#!/usr/bin/env python3
"""Download complete policy/base snapshots at the submission's immutable revisions."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    manifest = json.loads((ROOT / "submission.json").read_text())
    for repository, revision, folder in (
        (manifest["checkpoint_repository"], manifest["checkpoint_revision"], "hippovla"),
        (manifest["base_vlm_repository"], manifest["base_vlm_revision"], "RynnBrain8B"),
    ):
        target = args.models_dir.resolve() / folder
        snapshot_download(repo_id=repository, revision=revision, local_dir=str(target))
        print(f"[OK] {repository}@{revision} -> {target}")


if __name__ == "__main__":
    main()
