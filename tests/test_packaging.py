"""Verify official-style composition, rather than imports from the source checkout."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "policy/hippovla"


class PackagingTests(unittest.TestCase):
    def test_installer_copies_complete_runtime_into_official_checkout(self):
        with tempfile.TemporaryDirectory(prefix="official composed policy ") as tmp:
            benchmark = Path(tmp) / "benchmark"
            (benchmark / "scripts").mkdir(parents=True)
            (benchmark / "scripts/eval_policy.py").touch()
            subprocess.run([sys.executable, str(ROOT / "scripts/install_eval_adapter.py"),
                            str(benchmark)], check=True, capture_output=True)
            copied = benchmark / "policy/hippovla"
            env = dict(os.environ, PYTHONPATH="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = subprocess.run(
                [sys.executable, str(copied / "scripts/verify_policy_bundle.py"), "--runtime"],
                cwd=tmp, env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("project modules imported inside composed policy", result.stdout)
            self.assertTrue((copied / "models/hippovla/config.yaml").is_file())
            self.assertTrue((copied / "transformer/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py").is_file())

    def test_submission_manifest_and_original_metadata_stay_consistent(self):
        self.assertEqual((ROOT / "submission.json").read_bytes(), (BUNDLE / "submission.json").read_bytes())
        manifest = json.loads((BUNDLE / "submission.json").read_text())
        self.assertEqual(len(manifest["tasks"]), 10)
        for name in ("config.yaml", "dataset_statistics.json"):
            self.assertEqual((ROOT / "models/hippovla" / name).read_bytes(),
                             (BUNDLE / "models/hippovla" / name).read_bytes())

    def test_server_launcher_resolves_caller_relative_weights_and_config(self):
        with tempfile.TemporaryDirectory(prefix="launch paths ") as tmp:
            folder = Path(tmp)
            (folder / "weights/final_model").mkdir(parents=True)
            (folder / "base model").mkdir()
            for name in ("weights/final_model/pytorch_model.pt", "weights/config.yaml",
                         "weights/dataset_statistics.json", "base model/config.json", "custom config.yaml"):
                (folder / name).touch()
            # Capture launcher paths without importing/loading any model.
            interpreter = folder / "capture python"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json,os,sys\n"
                "if sys.argv[1].endswith('server_policy.py'):\n"
                " print(json.dumps(dict(argv=sys.argv[1:],cwd=os.getcwd(),"
                "pythonpath=os.environ['PYTHONPATH'],gpu=os.environ['CUDA_VISIBLE_DEVICES'])))\n"
            )
            interpreter.chmod(0o755)
            result = subprocess.check_output(
                ["bash", str(BUNDLE / "run_policy_server.sh"), "weights/final_model/pytorch_model.pt",
                 "base model", "6", "5789", "custom config.yaml"], cwd=folder,
                env=dict(os.environ, STARVLA_PYTHON=str(interpreter), PYTHONPATH="/unbundled/source"), text=True,
            )
            actual = json.loads(result.splitlines()[-1])
            argv = actual["argv"]
            self.assertEqual(argv[argv.index("--ckpt_path") + 1], str(folder / "weights/final_model/pytorch_model.pt"))
            self.assertEqual(argv[argv.index("--config_path") + 1], str(folder / "custom config.yaml"))
            self.assertEqual(actual["cwd"], str(BUNDLE))
            self.assertEqual(actual["pythonpath"], f"{BUNDLE}/transformer/src:{BUNDLE}")
            self.assertEqual(actual["gpu"], "6")
            self.assertIn("--use_bf16", argv)
            self.assertEqual(argv[argv.index("--idle_timeout") + 1], "-1")


if __name__ == "__main__":
    unittest.main()
