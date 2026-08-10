"""Replay RFCL checkpoints in the InsertUSB simulator.

For each selected Motion Plan demo, this script resets the same task seed,
replays the measured qpos prefix to three checkpoints, records the privileged
state error, and then replays the remaining suffix.  It is a state-recovery
probe only; it does not train an RL policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="insert_usb")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--data-dir", type=Path, default=Path("data/insert_usb/gelsight/hdf5"))
    parser.add_argument("--demo-ids", type=int, nargs="+", default=(10, 100, 198))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/analysis/rfcl_replay_probe"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def _pose_snapshot(task) -> dict[str, np.ndarray]:
    joint = task._robot_manager.get_observations(["joint"])["joint"]
    if torch.is_tensor(joint):
        joint = joint.detach().cpu().numpy()
    ee = task._robot_manager.get_ee_pose().tolist()
    usb = task.prism.get_pose().tolist()
    slot = task.slot.get_pose().tolist()
    return {
        "joint": np.asarray(joint, dtype=np.float32).reshape(-1),
        "ee": np.asarray(ee, dtype=np.float32),
        "usb": np.asarray(usb, dtype=np.float32),
        "slot": np.asarray(slot, dtype=np.float32),
    }


def _position_error(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "joint_max_abs": float(np.max(np.abs(actual["joint"][:7] - expected["joint"][:7]))),
        "ee_position_l2": float(np.linalg.norm(actual["ee"][:3] - expected["ee"][:3])),
        "usb_position_l2": float(np.linalg.norm(actual["usb"][:3] - expected["usb"][:3])),
        "slot_position_l2": float(np.linalg.norm(actual["slot"][:3] - expected["slot"][:3])),
    }


def _load_demo(path: Path) -> dict[str, np.ndarray | list[str]]:
    with h5py.File(path, "r") as handle:
        return {
            "joint": np.asarray(handle["embodiment/joint"][()], dtype=np.float32),
            "ee": np.asarray(handle["embodiment/ee"][()], dtype=np.float32),
            "usb": np.asarray(handle["actor/prism"][()], dtype=np.float32),
            "slot": np.asarray(handle["actor/slot"][()], dtype=np.float32),
            "tags": handle["atom/tag"][()].astype(str).tolist(),
        }


def _checkpoint_indices(tags: list[str]) -> list[tuple[str, int]]:
    start_matches = [index for index, tag in enumerate(tags) if tag == "move_usb_to_pre_insert"]
    if not start_matches:
        raise ValueError("Demo has no move_usb_to_pre_insert tag")
    start = start_matches[-1]
    insertion = [index for index in range(start, len(tags)) if tags[index] == "insert_usb_into_slot"]
    insertion_middle = insertion[len(insertion) // 2] if insertion else start
    return [
        ("handoff_end", start),
        ("insertion_middle", insertion_middle),
        ("near_terminal", max(start, len(tags) - 2)),
    ]


def _replay_qpos(task, joint: np.ndarray, start: int, end: int) -> None:
    # HDF5 stores two gripper joints for Panda; BaseTask's qpos action expects
    # seven arm joints plus one gripper target, matching the existing replay.py.
    for frame_index in range(start, end + 1):
        task.take_action(
            torch.as_tensor(joint[frame_index, :8], dtype=torch.float32, device=task.device),
            action_type="qpos",
            force=True,
        )
        if task.eval_success:
            break


def _expected_snapshot(demo: dict[str, object], frame_index: int) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value[frame_index], dtype=np.float32)
        for key, value in demo.items()
        if key in ("joint", "ee", "usb", "slot")
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("[rfcl-probe] launching Isaac application", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[rfcl-probe] Isaac application ready", flush=True)
    task = None
    report: dict[str, object] = {
        "schema": "rfcl_replay_probe_v1",
        "task": args.task_name,
        "task_config": args.task_config,
        "demos": [],
    }
    try:
        from policy.RL.task_factory import create_task

        print("[rfcl-probe] creating task", flush=True)
        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.save_dir,
            video_frequency=0,
            step_limit=500,
            mode="eval",
            save_pre_move=False,
            device=args.device,
        )
        print("[rfcl-probe] task ready", flush=True)
        for demo_id in args.demo_ids:
            path = args.data_dir / f"{int(demo_id)}.hdf5"
            demo = _load_demo(path)
            checkpoints = _checkpoint_indices(demo["tags"])
            demo_report = {
                "demo_id": int(demo_id),
                "path": str(path),
                "checkpoints": [],
            }
            for label, frame_index in checkpoints:
                reset_start = time.perf_counter()
                task.reset(seed=int(demo_id))
                _replay_qpos(task, demo["joint"], 0, frame_index)
                actual = _pose_snapshot(task)
                expected = _expected_snapshot(demo, frame_index)
                error = _position_error(actual, expected)
                suffix_start = frame_index
                _replay_qpos(task, demo["joint"], suffix_start + 1, len(demo["joint"]) - 1)
                demo_report["checkpoints"].append(
                    {
                        "label": label,
                        "frame_index": int(frame_index),
                        "tag": str(demo["tags"][frame_index]),
                        "replay_seconds": time.perf_counter() - reset_start,
                        "state_error": error,
                        "suffix_eval_success": bool(task.eval_success),
                        "terminal_reason": getattr(task, "terminal_reason", None),
                        "actual_joint": actual["joint"][:7].tolist(),
                        "expected_joint": expected["joint"][:7].tolist(),
                    }
                )
            report["demos"].append(demo_report)
            print(json.dumps(demo_report, indent=2), flush=True)
    except BaseException as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        rendered = json.dumps(report, indent=2, ensure_ascii=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        if task is not None:
            task.close()
        simulation_app.close()

    print(rendered)


if __name__ == "__main__":
    main()
