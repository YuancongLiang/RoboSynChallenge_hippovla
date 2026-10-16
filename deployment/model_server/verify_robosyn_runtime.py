from __future__ import annotations

from pathlib import Path


EXPECTED_MEMORY_IMPLEMENTATION = "direct_mean_pool_v1"


def verify_robosyn_runtime(repo_root: str | Path | None = None) -> dict[str, str]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root).resolve()
    expected_transformers_root = (repo_root / "transformer" / "src" / "transformers").resolve()

    import transformers
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    transformers_path = Path(transformers.__file__).resolve()
    qwen_module_path = Path(modeling_qwen3_vl.__file__).resolve()
    implementation = getattr(modeling_qwen3_vl, "STARVLA_ROBOSYN_MEMORY_IMPLEMENTATION", None)
    path_is_bundled = (
        expected_transformers_root == transformers_path.parent
        and expected_transformers_root in qwen_module_path.parents
    )

    if not path_is_bundled or implementation != EXPECTED_MEMORY_IMPLEMENTATION:
        raise RuntimeError(
            "Incompatible Transformers/Qwen3-VL runtime. RoboSynChallenge requires the bundled "
            f"memory implementation {EXPECTED_MEMORY_IMPLEMENTATION!r}, but imported "
            f"implementation={implementation!r}, transformers={transformers_path}, "
            f"qwen3_vl={qwen_module_path}. Ensure <package>/transformer/src is first in "
            "PYTHONPATH and do not reuse the older OpenArm policy-server environment."
        )

    return {
        "implementation": implementation,
        "transformers": str(transformers_path),
        "qwen3_vl": str(qwen_module_path),
    }


def main() -> int:
    runtime = verify_robosyn_runtime()
    print(
        "[OK] RoboSyn Qwen3-VL runtime: "
        f"{runtime['implementation']} ({runtime['qwen3_vl']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
