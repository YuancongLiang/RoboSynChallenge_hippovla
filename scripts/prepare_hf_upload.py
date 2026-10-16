#!/usr/bin/env python3
"""Stage checkpoint and matching metadata for a separate Hugging Face upload."""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; use a fresh directory")
    identity = json.loads((ROOT / "docs/checkpoint_identity.json").read_text())
    if not source.is_file() or source.stat().st_size != identity["size_bytes"]:
        parser.error("Checkpoint missing or size differs from the selected submission")
    for filename in ["config.yaml", "dataset_statistics.json"]:
        if (source.parents[1] / filename).read_bytes() != (ROOT / "models/hippovla" / filename).read_bytes():
            parser.error(f"Checkpoint metadata differs: {filename}")
    output.mkdir(parents=True, exist_ok=False)
    for filename in ["config.yaml", "dataset_statistics.json"]:
        shutil.copy2(ROOT / "models/hippovla" / filename, output / filename)
    shutil.copy2(ROOT / "docs/checkpoint_identity.json", output / "checkpoint_identity.json")
    shutil.copy2(ROOT / "docs/HF_MODEL_CARD.md", output / "README.md")
    target = output / "final_model/pytorch_model.pt"
    target.parent.mkdir()
    try:
        os.link(source, target)
        print("Staged using a hard link. Do not modify checkpoint contents in place.")
    except OSError:
        print("Hard link unavailable; copying the checkpoint.")
        shutil.copy2(source, target)
    print(f"Staged: {output}")
    print(f'hf upload YOUR_ACCOUNT/YOUR_MODEL "{output}" . --repo-type model')


if __name__ == "__main__":
    main()
