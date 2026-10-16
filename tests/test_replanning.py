"""Check replanning with simulated server responses, without loading model weights."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

from policy.hippovla import deploy_policy as adapter


class ReplanningTests(unittest.TestCase):
    def make_model(self, replan_steps=8, eval_steps=8, action_mode="abs", response_steps=16):
        server = Mock()
        server.get_server_metadata.return_value = {"action_chunk_size": 16}
        requests = []

        def predict(query):
            example = query["examples"][0]
            requests.append((int(example["step"]), int(example["image"][0][0, 0, 0])))
            if action_mode == "delta":
                actions = np.ones((response_steps, 14), dtype=np.float32)
            else:
                actions = np.repeat(
                    (100 * len(requests) + np.arange(response_steps))[:, None], 14, axis=1
                ).astype(np.float32)
            return {"data": {"actions": actions[None]}}

        server.predict_action.side_effect = predict
        with patch.object(adapter, "read_mode_config", return_value=({}, {})), \
                patch.object(adapter, "WebsocketClientPolicy", return_value=server), \
                contextlib.redirect_stdout(io.StringIO()):
            model = adapter.ModelClient(
                policy_ckpt_path="unused.pt", model_action_order="robosyn",
                replan_action_steps=replan_steps, eval_action_steps=eval_steps,
                action_mode=action_mode,
            )
        return model, requests

    @staticmethod
    def step(model, step, lang="test"):
        return model.step({
            "lang": lang,
            "image": [np.full((8, 8, 3), step, dtype=np.uint8)],
            "state": np.full(14, 10 * step, dtype=np.float32),
        }, step=step)

    def test_eight_step_prefix_and_fresh_observations(self):
        # The outer eval loop may execute 16 steps; replanning must still happen after 8.
        model, requests = self.make_model(eval_steps=16)
        actions = [self.step(model, step)[0] for step in range(20)]
        self.assertEqual(requests, [(0, 0), (8, 8), (16, 16)])
        self.assertEqual(actions, list(range(100, 108)) + list(range(200, 208)) + list(range(300, 304)))
        self.assertEqual(model.action_chunk_size, 16)

    def test_delta_replan_starts_from_latest_measured_state(self):
        model, requests = self.make_model(action_mode="delta")
        actions = [self.step(model, step)[0] for step in range(17)]
        np.testing.assert_allclose(actions, list(range(1, 9)) + list(range(81, 89)) + [161])
        self.assertEqual([step for step, _ in requests], [0, 8, 16])

    def test_default_replans_after_full_chunk(self):
        model, requests = self.make_model(replan_steps=None)
        for step in range(17):
            self.step(model, step)
        self.assertEqual([step for step, _ in requests], [0, 16])

    def test_reset_and_instruction_change_discard_remaining_actions(self):
        model, requests = self.make_model()
        for step in range(3):
            self.step(model, step)
        model.reset("test")
        self.assertEqual(self.step(model, 0)[0], 200)
        # Changing instructions mid-chunk must also start at the new chunk's first action.
        self.assertEqual(self.step(model, 1, lang="different")[0], 300)
        for step in range(2, 10):
            action = self.step(model, step, lang="different")
        self.assertEqual(action[0], 400)
        self.assertEqual([step for step, _ in requests], [0, 0, 1, 9])

    def test_short_responses_replan_instead_of_repeating_last_action(self):
        model, requests = self.make_model(response_steps=3)
        actions = [self.step(model, step)[0] for step in range(7)]
        self.assertEqual([step for step, _ in requests], [0, 3, 6])
        self.assertEqual(actions, [100, 101, 102, 200, 201, 202, 300])

    def test_invalid_replan_interval_is_rejected(self):
        for steps in (0, -1, 17):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "replan_action_steps"):
                self.make_model(replan_steps=steps)


if __name__ == "__main__":
    unittest.main()
