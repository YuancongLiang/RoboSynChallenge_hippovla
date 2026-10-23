#!/usr/bin/env python3
"""Exercise the real model server with the unchanged adapter and a mock environment."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5711)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import yaml
    import deploy_policy as adapter

    class MockEnvironment:
        def __init__(self):
            self.steps = 0
            self._current_instruction = "Move the object."
            self.unwrapped = SimpleNamespace(
                single_action_space=SimpleNamespace(shape=(14,)),
                active_joint_ids=list(range(14)), device="cpu",
            )

        def observation(self):
            return {
                "sensor": {name: {"color": np.zeros((1, 224, 224, 3), dtype=np.uint8)}
                           for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
                "robot": {"qpos": np.zeros((1, 14), dtype=np.float32)},
            }

        def step(self, action):
            if not torch.isfinite(action).all():
                raise RuntimeError("Non-finite executable action")
            self.steps += 1
            return (self.observation(), 0., torch.tensor([False]), torch.tensor([False]),
                    {"elapsed_steps": torch.tensor([self.steps])})

        def get_wrapper_attr(self, name):
            if name != "is_task_success":
                raise RuntimeError(f"Unexpected environment access: {name}")
            return lambda: torch.tensor([False])

    config = yaml.safe_load((ROOT / "deploy_policy.yml").read_text())
    # The synthetic environment has no robot gripper limits or video output.
    # Disable only test recording and use the task's existing physical mapping.
    config.update(policy_ckpt_path=str(args.checkpoint.resolve()), host=args.host,
                  port=args.port, task_name="table_rearrangement", starvla_eval_video_log=False)
    model = adapter.get_model(config)
    predictions, shapes = [], []
    original_predict = model.client.predict_action

    def predict(query):
        response = original_predict(query)
        if response.get("ok") is False:
            raise RuntimeError(f"Server inference failed: {response}")
        data = response["data"]
        actions = np.asarray(data["actions"] if "actions" in data else data["normalized_actions"])
        if actions.shape != (1, 16, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid server actions: {actions.shape}")
        predictions.append(query["examples"][0]["step"])
        shapes.append(list(actions.shape))
        return response

    model.client.predict_action = predict
    env = MockEnvironment()
    observation = env.observation()
    timings = []
    for _ in range(3):
        observation, info, truncated, new_timings = adapter.eval(env, model, observation)
        if truncated.any():
            raise RuntimeError("Unexpected truncation")
        timings.extend(new_timings)
    if predictions != [0, 4, 8] or env.steps != 12 or len(timings) != 3 or not all(t > 0 for t in timings):
        raise RuntimeError(f"Unexpected prediction schedule/timings: {predictions}, {timings}")
    result = dict(real_websocket_predictions=len(predictions), prediction_steps=predictions,
                  output_shapes=shapes, all_finite=True, four_value_contract="passed",
                  inference_times_s=timings, environment_steps=env.steps, simulation_episodes_run=0,
                  environment="synthetic RGB/qpos transitions; no simulator episodes")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
