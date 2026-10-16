"""Regression checks for deployment semantics without loading the 8B model."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from policy.hippovla.deploy_policy import ModelClient, encode_action

ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_new_chunk_uses_measured_state(self):
        model = ModelClient.__new__(ModelClient)
        model.prev_action = np.full(14, 1000, dtype=np.float32)
        measured = np.linspace(0, 1, 14, dtype=np.float32)
        delta = np.stack([np.full(14, .1), np.full(14, .2)])
        decoded = model._delta_to_absolute(delta, measured)
        np.testing.assert_allclose(decoded[0], measured + .1, atol=1e-6)
        np.testing.assert_allclose(decoded[1], measured + .3, atol=1e-6)
        model.prev_action = decoded[-1]
        next_measured = measured - .4
        np.testing.assert_allclose(model._delta_to_absolute(delta, next_measured)[0],
                                   next_measured + .1, atol=1e-6)
        with self.assertRaises(ValueError):
            model._delta_to_absolute(delta, None)

    def test_all_dimensions_continuous_despite_exported_mask(self):
        stats = json.loads((ROOT / "models/hippovla/dataset_statistics.json").read_text())
        model = ModelClient.__new__(ModelClient)
        model.action_norm_stats = stats["new_embodiment"]["action"]
        model.normalization_mode = "min_max"
        model.normalize_all_action_dims = True
        model.binary_action_indices = []
        model.binary_threshold = .5
        bounds = model.action_norm_stats
        normalized = np.stack([np.full(14, -1.), np.zeros(14), np.ones(14)])
        output = model.unnormalize_actions(normalized)
        np.testing.assert_allclose(output[0], bounds["min"], atol=1e-6)
        np.testing.assert_allclose(output[-1], bounds["max"], atol=1e-6)
        np.testing.assert_allclose(output[1], (np.array(bounds["min"]) + bounds["max"]) / 2,
                                   atol=1e-6)

    def test_gripper_mapping_preserves_arm_targets(self):
        class Robot:
            body_data = SimpleNamespace(qpos_limits=torch.tensor([[[0., .05]] * 14]))
            def get_joint_ids(self, name, remove_mimic):
                return [6 if name == "left_eef" else 13]
        unwrapped = SimpleNamespace(single_action_space=SimpleNamespace(shape=(14,)),
                                    active_joint_ids=list(range(14)), robot=Robot(), device="cpu")
        env = SimpleNamespace(unwrapped=unwrapped)
        action = np.arange(14, dtype=np.float32)
        action[6], action[13] = .5, 2.
        physical = encode_action(action, env, "ratio").numpy()[0]
        self.assertAlmostEqual(float(physical[6]), .025)
        self.assertAlmostEqual(float(physical[13]), .05)
        arms = list(range(6)) + list(range(7, 13))
        np.testing.assert_array_equal(physical[arms], action[arms])
        np.testing.assert_array_equal(encode_action(action, env, "physical").numpy()[0], action)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg/ffprobe")
    def test_three_camera_videos_encode_and_reset(self):
        model = ModelClient.__new__(ModelClient)
        with tempfile.TemporaryDirectory() as tmp:
            model.video_enabled = True
            model.video_dir = Path(tmp)
            model.video_episode_idx = -1
            model.video_closed = True
            model.video_episode_dir = None
            model.video_fps = 25
            model.video_format = "jpg"
            model.start_video_episode()
            keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
            obs = {"sensor": {key: {"color": np.full((32, 32, 3), 40*i, dtype=np.uint8)}
                               for i, key in enumerate(keys)}}
            for _ in range(3):
                model.record_video_obs(obs)
            model.close_video_episode(success=True)
            videos = sorted(Path(tmp).glob("*.mp4"))
            self.assertEqual(len(videos), 3)
            for video in videos:
                probe = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
                    "-show_entries", "stream=nb_read_frames", "-of", "json", str(video)], text=True)
                self.assertEqual(json.loads(probe)["streams"][0]["nb_read_frames"], "3")
            model.start_video_episode()
            self.assertEqual(model.video_episode_idx, 1)
            self.assertEqual(model.video_frame_idx, 0)
            model.close_video_episode()

    def test_all_task_launcher_uses_checkout_limits_and_does_not_change_output(self):
        tasks = json.loads((ROOT / "submission.json").read_text())["tasks"]
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            benchmark = temp / "RoboSynChallenge"
            (benchmark / "scripts").mkdir(parents=True)
            (benchmark / "scripts/eval_policy.py").touch()
            subprocess.run([sys.executable, str(ROOT / "scripts/install_eval_adapter.py"),
                            str(benchmark)], check=True, capture_output=True)
            for i, task in enumerate(tasks):
                folder = benchmark / "configs" / task / "random"
                folder.mkdir(parents=True)
                (folder / "gym_config.json").write_text(json.dumps({"max_episode_steps": 300+i}))
            run = temp / "model"
            (run / "final_model").mkdir(parents=True)
            ckpt = run / "final_model/pytorch_model.pt"
            ckpt.touch()
            for name in ["config.yaml", "dataset_statistics.json"]:
                shutil.copy2(ROOT / "models/hippovla" / name, run / name)
            output = temp / "output"
            result = subprocess.check_output([sys.executable, str(ROOT / "scripts/eval.py"), "all",
                "--robosyn-root", str(benchmark), "--checkpoint", str(ckpt),
                "--output", str(output), "--dry-run"], text=True)
            commands = json.loads(result)
            self.assertEqual(len(commands), 10)
            for i, command in enumerate(commands):
                self.assertEqual(command[command.index("--max_steps") + 1], str(300+i))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
