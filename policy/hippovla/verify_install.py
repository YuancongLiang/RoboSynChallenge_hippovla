#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_TRANSFORMERS = PACKAGE_ROOT / "transformer" / "src" / "transformers"
sys.path[:0] = [str(PACKAGE_ROOT / "transformer" / "src"), str(PACKAGE_ROOT)]


def main() -> int:
    errors: list[str] = []

    try:
        import torch
        import torchvision

        print(f"[OK] torch={torch.__version__}, torchvision={torchvision.__version__}")
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is False")
        else:
            print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        errors.append(f"torch/torchvision import failed: {exc!r}")

    try:
        import transformers

        imported_path = Path(transformers.__file__).resolve()
        print(f"[OK] transformers={transformers.__version__}: {imported_path}")
        if EXPECTED_TRANSFORMERS.resolve() not in imported_path.parents:
            errors.append(
                "The bundled custom Transformers is not active. "
                "Run install_policy_server.sh or use run_policy_server.sh."
            )
    except Exception as exc:
        errors.append(f"transformers import failed: {exc!r}")

    for module_name in ("flash_attn", "accelerate", "omegaconf", "websockets", "msgpack", "timm"):
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:
            errors.append(f"import {module_name} failed: {exc!r}")

    try:
        from deployment.model_server.server_policy import build_argparser
        from deployment.model_server.verify_robosyn_runtime import verify_robosyn_runtime
        from starVLA.model.framework.RynnBrainOFT import RynnBrain_OFT

        assert build_argparser is not None
        assert RynnBrain_OFT is not None
        print("[OK] StarVLA RynnBrainOFT policy server imports")
        runtime = verify_robosyn_runtime(PACKAGE_ROOT)
        print(
            "[OK] RoboSyn memory implementation: "
            f"{runtime['implementation']} ({runtime['qwen3_vl']})"
        )
    except Exception as exc:
        errors.append(f"policy server import failed: {exc!r}")

    if errors:
        print("\nVerification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("\n[OK] Policy server environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
