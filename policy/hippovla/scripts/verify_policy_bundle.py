#!/usr/bin/env python3
"""Verify the composed policy's contents; --runtime also checks actual imports."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "click_bell", "drawer_open_place", "handle_basket", "item_assembly",
    "items_handover", "manipulate_pipette", "mixer_operating", "sample_loading",
    "table_rearrangement", "water_pouring",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    identity = json.loads((ROOT / "docs/inference_code_identity.json").read_text())
    for name, expected in identity["files"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen inference file missing or changed: {name}")
    for name in ("README.md", "install_policy_server.sh", "run_policy_server.sh",
                 "verify_install.py", "requirements-policy-server.txt", "requirements-eval.txt",
                 "eval.sh", "scripts/eval.py", "scripts/install_eval_adapter.py"):
        if not (ROOT / name).is_file():
            raise RuntimeError(f"Missing bundle entrypoint: {name}")
    for path in ROOT.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Bundle symlink escapes the policy: {path}")
    manifest = json.loads((ROOT / "submission.json").read_text())
    if set(manifest["tasks"]) != TASKS:
        raise RuntimeError("Submission must contain all ten task mappings")
    for field in ("checkpoint_revision", "base_vlm_revision"):
        revision = manifest[field]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise RuntimeError(f"Non-immutable revision: {field}")
    for task, mapping in manifest["tasks"].items():
        if mapping["checkpoint_file"] != manifest["checkpoint_file"]:
            raise RuntimeError(f"Changed shared checkpoint mapping: {task}")
        expected_url = (f'https://huggingface.co/{manifest["checkpoint_repository"]}/resolve/'
                        f'{manifest["checkpoint_revision"]}/{mapping["checkpoint_file"]}')
        if mapping["checkpoint_url"] != expected_url:
            raise RuntimeError(f"Invalid immutable checkpoint URL: {task}")
    print(f'[OK] {len(identity["files"])} frozen inference files; all ten mappings; bundle entrypoints')
    if args.runtime:
        sys.path[:0] = [str(ROOT / "transformer/src"), str(ROOT)]
        from deployment.model_server.server_policy import build_argparser
        from deployment.model_server.verify_robosyn_runtime import verify_robosyn_runtime
        from starVLA.model.framework.RynnBrainOFT import RynnBrain_OFT
        import deploy_policy

        assert build_argparser and RynnBrain_OFT and deploy_policy.get_model
        runtime = verify_robosyn_runtime(ROOT)
        checked = 0
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] not in {"transformers", "starVLA", "deployment", "deploy_policy"}:
                continue
            filename = getattr(module, "__file__", None)
            if filename:
                if not Path(filename).resolve().is_relative_to(ROOT):
                    raise RuntimeError(f"Project import escaped composed policy: {name}: {filename}")
                checked += 1
        print(f'[OK] {checked} project modules imported inside composed policy: {ROOT}')
        print(f'[OK] Custom memory implementation: {runtime["implementation"]}')


if __name__ == "__main__":
    main()
