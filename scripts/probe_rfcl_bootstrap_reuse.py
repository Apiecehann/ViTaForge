"""Check whether one live InsertUSB handoff can serve repeated RFCL resets."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--step-limit", type=int, default=200)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    env = None
    report = {
        "schema": "rfcl_bootstrap_reuse_probe_v1",
        "snapshot_root": str(args.snapshot_root),
        "demo_index": int(args.demo_index),
        "state_index": int(args.state_index),
        "repeats": [],
    }
    try:
        from policy.RL.rfcl_env import RFCLPrivilegedEnv
        from policy.RL.rfcl_snapshot import runtime_snapshot_diagnostics
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.snapshot_root,
            video_frequency=0,
            step_limit=args.step_limit,
            mode="eval_test",
            save_pre_move=False,
            device=args.device,
        )
        env = RFCLPrivilegedEnv(
            task,
            args.snapshot_root,
            action_repeat=1,
            snapshot_sync_steps=0,
            demo_horizon_to_max_steps_ratio=1.0,
            seed=0,
        )
        trajectory = env.curriculum.demos[int(args.demo_index)]
        seed = env.dataset.demos[int(args.demo_index)].get("seed")
        task.reset(seed=None if seed is None else int(seed))
        task._prepare_usb_standard()

        for repeat in range(int(args.repeats)):
            _state, reset_info = env.reset(
                options={
                    "demo_index": int(args.demo_index),
                    "state_index": int(args.state_index),
                    "skip_task_reset": True,
                }
            )
            terminated = False
            truncated = False
            reward = 0.0
            steps = 0
            info = {}
            for action in trajectory.actions[int(args.state_index) :]:
                _state, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            report["repeats"].append(
                {
                    "repeat": repeat,
                    "success": bool(info.get("success", False)),
                    "reward": float(reward),
                    "steps": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "reset_info": {
                        "demo_id": reset_info["demo_id"],
                        "raw_state_index": reset_info["raw_state_index"],
                    },
                    "runtime": runtime_snapshot_diagnostics(task),
                }
            )
            print(json.dumps(report["repeats"][-1], ensure_ascii=True), flush=True)
    except BaseException as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
