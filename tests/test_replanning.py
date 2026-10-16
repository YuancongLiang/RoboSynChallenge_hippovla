"""Check replanning with simulated server responses, without loading model weights."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

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

    @staticmethod
    def make_env(success_at=None, truncate_at=None, terminate_at=None):
        env = Mock()
        env._current_instruction = "test"
        env.unwrapped = SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(14,)),
            active_joint_ids=list(range(14)), device="cpu",
        )
        env.steps = 0

        def observation():
            return {
                "sensor": {name: {"color": np.full((1, 8, 8, 3), env.steps, dtype=np.uint8)}
                           for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
                "robot": {"qpos": np.full((1, 14), 10 * env.steps, dtype=np.float32)},
            }

        def step(action):
            env.steps += 1
            return (observation(), 0., torch.tensor([env.steps == terminate_at]),
                    torch.tensor([env.steps == truncate_at]),
                    {"elapsed_steps": torch.tensor([env.steps])})

        env.step.side_effect = step
        env.get_wrapper_attr.return_value = lambda: torch.tensor([env.steps == success_at])
        # Any access to private scoring or simulator object state must fail this test.
        env.unwrapped._evaluate_task_state = Mock(side_effect=AssertionError("private scoring"))
        env.unwrapped.sim = Mock(side_effect=AssertionError("simulator state"))
        return env, observation()

    def test_official_four_value_contract_and_four_step_replanning(self):
        model, requests = self.make_model(replan_steps=4, eval_steps=4, action_mode="delta")
        env, obs = self.make_env()
        executed = []
        original_step = env.step.side_effect
        env.step.side_effect = lambda action: (executed.append(float(action[0, 0])), original_step(action))[1]
        times = []
        for _ in range(3):
            # Exactly the unpacking and aggregation performed by the official runner.
            obs, info, truncated, new_times = adapter.eval(env, model, obs)
            times.extend(new_times)
            self.assertFalse(truncated.any())
        self.assertEqual(requests, [(0, 0), (4, 4), (8, 8)])
        np.testing.assert_allclose(executed, [1, 2, 3, 4, 41, 42, 43, 44, 81, 82, 83, 84])
        self.assertEqual(info["elapsed_steps"].item(), 12)
        self.assertEqual(len(times), 3)
        self.assertTrue(all(value > 0 for value in times))
        env.unwrapped._evaluate_task_state.assert_not_called()
        env.get_wrapper_attr.assert_called_with("is_task_success")

    def test_inference_timing_includes_preprocessing_but_excludes_execution_and_cache(self):
        model, requests = self.make_model(replan_steps=4, eval_steps=1)
        env, obs = self.make_env()
        clock = [0.]

        def advance(function, seconds):
            def call(*args, **kwargs):
                result = function(*args, **kwargs)
                clock[0] += seconds
                return result
            return call

        model.client.predict_action.side_effect = advance(model.client.predict_action.side_effect, .4)
        env.step.side_effect = advance(env.step.side_effect, 10.)
        model.record_video_obs = advance(model.record_video_obs, 20.)
        timings = []
        with patch.object(adapter.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(adapter, "encode_obs", side_effect=advance(adapter.encode_obs, .1)), \
                patch.object(adapter, "encode_action", side_effect=advance(adapter.encode_action, .2)):
            for _ in range(5):
                obs, info, truncated, new_times = adapter.eval(env, model, obs)
                timings.append(new_times)
        self.assertEqual([len(values) for values in timings], [1, 0, 0, 0, 1])
        np.testing.assert_allclose(timings[0] + timings[-1], [.7, .7])
        self.assertEqual([step for step, _ in requests], [0, 4])
        model.reset("test")
        self.assertFalse(model.last_step_did_infer)

    def test_eval_stops_on_official_success_timeout_or_termination(self):
        for outcome in ("success_at", "truncate_at", "terminate_at"):
            with self.subTest(outcome=outcome):
                model, requests = self.make_model(replan_steps=4, eval_steps=4)
                model.close_video_episode = Mock()
                env, obs = self.make_env(**{outcome: 2})
                obs, info, truncated, times = adapter.eval(env, model, obs)
                self.assertEqual(env.steps, 2)
                self.assertEqual(info["elapsed_steps"].item(), 2)
                self.assertEqual(len(times), 1)
                model.close_video_episode.assert_called_once_with(success=(outcome == "success_at"))

    def test_cuda_timing_excludes_pending_simulation_but_waits_for_action_transfer(self):
        model, requests = self.make_model(replan_steps=4, eval_steps=1)
        env, obs = self.make_env()
        env.unwrapped.device = "cuda:1"
        clock = [0.]
        pending_work = iter([100., .3])

        def synchronize(device):
            clock[0] += next(pending_work)

        predict = model.client.predict_action.side_effect
        def timed_predict(query):
            clock[0] += .4
            return predict(query)

        model.client.predict_action.side_effect = timed_predict
        encode_obs = adapter.encode_obs
        def timed_obs(*args, **kwargs):
            clock[0] += .1
            return encode_obs(*args, **kwargs)

        def timed_action(*args):
            clock[0] += .2
            return SimpleNamespace(is_cuda=True, device=torch.device("cuda:1"))

        with patch.object(adapter.torch.cuda, "synchronize", side_effect=synchronize), \
                patch.object(adapter.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(adapter, "encode_obs", side_effect=timed_obs), \
                patch.object(adapter, "encode_action", side_effect=timed_action):
            obs, info, truncated, times = adapter.eval(env, model, obs)
        # The 100 seconds of queued simulation work precede the timer; the
        # 0.3 seconds needed to finish transferring this action belong to inference.
        np.testing.assert_allclose(times, [1.])

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
