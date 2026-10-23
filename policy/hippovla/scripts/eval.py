#!/usr/bin/env python3
"""Run official evaluation with task limits read from the benchmark checkout."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="One task name or 'all'")
    parser.add_argument("--robosyn-root", type=Path,
                        default=os.environ.get("ROBOSYNCHALLENGE_PATH"))
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models/hippovla/final_model/pytorch_model.pt")
    parser.add_argument("--host", default=os.environ.get("ROBOSYN_POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ROBOSYN_SERVER_PORT", "5711")))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh output directory; existing directories are rejected")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "submission.json").read_text())
    tasks = list(manifest["tasks"]) if args.task == "all" else [args.task]
    if any(task not in manifest["tasks"] for task in tasks):
        parser.error("Unknown task; see submission.json")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.robosyn_root is None:
        parser.error("Set ROBOSYNCHALLENGE_PATH or --robosyn-root")
    benchmark = args.robosyn_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    for path in [benchmark / "scripts/eval_policy.py",
                 benchmark / "policy/hippovla/deploy_policy.py",
                 checkpoint, checkpoint.parents[1] / "config.yaml",
                 checkpoint.parents[1] / "dataset_statistics.json"]:
        if not path.is_file():
            parser.error(f"Missing {path}")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    # Do not let a policy-server CUDA mask hide the simulator's physical GPU.
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    pythonpath = [str(benchmark)]
    if env.get("EMBODICHAIN_ROOT"):
        pythonpath += [env["EMBODICHAIN_ROOT"], str(Path(env["EMBODICHAIN_ROOT"]) / "embodichain_tasks")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    commands = []
    for task in tasks:
        config_path = benchmark / "configs" / task / "random/gym_config.json"
        gym_config = json.loads(config_path.read_text())
        steps = int(gym_config["max_episode_steps"])
        task_output = output / task
        command = [sys.executable, "-u", str(benchmark / "scripts/eval_policy.py"),
                   "--config", str(benchmark / "policy/hippovla/deploy_policy.yml"),
                   "--overrides", "--task_name", task, "--setting", "random",
                   "--model_name", output.name, "--seed", str(args.seed),
                   "--gpu_id", str(args.gpu), "--max_episodes", str(args.episodes),
                   "--max_steps", str(steps), "--policy_ckpt_path", str(checkpoint),
                   "--host", args.host, "--port", str(args.port),
                   "--starvla_video_dir", str(task_output / "videos")]
        commands.append((task, task_output, command, gym_config))
    if args.dry_run:
        print(json.dumps([item[2] for item in commands], indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    summary = {"execution_state": "running", "tasks": {}}
    def save():
        tmp = output / "summary.tmp"
        tmp.write_text(json.dumps(summary, indent=2) + "\n")
        tmp.replace(output / "summary.json")
    save()
    for task, task_output, command, gym_config in commands:
        task_output.mkdir()
        (task_output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        (task_output / "gym_config_source.json").write_text(json.dumps(gym_config, indent=2) + "\n")
        summary["tasks"][task] = {"execution_state": "running"}
        save()
        with (task_output / "eval.log").open("w") as log:
            result = subprocess.run(command, cwd=benchmark, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        summary["tasks"][task] = {"exit_code": result.returncode,
                                    "execution_state": "exited",
                                    "log": str(task_output / "eval.log")}
        save()
    summary["execution_state"] = "exited"
    save()
    print(f"Logs: {output}; inspect official episode counts and success rates in eval.log")
    if any(task["exit_code"] != 0 for task in summary["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
