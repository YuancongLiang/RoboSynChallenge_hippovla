#!/usr/bin/env python3
"""Create a GitHub source ZIP and an SHA-256 manifest; never include weights."""
import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".codex", "__pycache__", ".venv", ".cache", "eval_runs", "hf_upload", "dist"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".bin", ".safetensors", ".mp4", ".log", ".zip"}


def files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name.startswith(".env") or path.suffix in SKIP_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError(f"Source archive must not contain symlinks: {relative}")
        if path.is_file():
            if path.stat().st_size >= 100_000_000:
                raise ValueError(f"Unexpected large file: {relative}")
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / f"{ROOT.name}_github.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("Place the archive outside the source repository")
    manifest = ROOT / "MANIFEST.sha256"
    entries = []
    for path in files():
        if path != manifest:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(ROOT).as_posix()}\n")
    manifest.write_text("".join(entries))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files():
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP integrity failure: {bad}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Archive: {output} ({output.stat().st_size:,} bytes)")
    print(f"Source files: {len(entries) + 1}; SHA-256: {digest}")


if __name__ == "__main__":
    main()
