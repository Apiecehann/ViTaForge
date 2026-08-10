"""Replay exact RFCL demo actions from restored simulator snapshots.

This probe does not train.  It checks whether the privileged state restore and
the inferred per-joint action scale can execute one saved Motion Plan
transition without learning or visual observations.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument(
        "--snapshot-sync-steps",
        type=int,
        default=1,
        help="Action-free physics steps applied after snapshot recovery.",
    )
    parser.add_argument(
        "--snapshot-sync-constraint-strength",
        type=float,
        default=1.0e5,
        help="Temporary USB pose-constraint strength during snapshot settling.",
    )
    parser.add_argument(
        "--no-rerestore-after-sync",
        action="store_true",
        help="Keep the settled state instead of recovering the exact frame again.",
    )
    parser.add_argument(
        "--demo-horizon-to-max-steps-ratio",
        type=float,
        default=1.0,
        help=(
            "RFCL demo-horizon divisor. The stride=1 low-level replay needs "
            "at least one policy step per saved transition."
        ),
    )
    parser.add_argument("--demo-indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--local-state-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override checkpoint selection with RFCL-local state indices.",
    )
    parser.add_argument(
        "--checkpoints",
        choices=("representative", "actions", "all"),
        default="actions",
        help="Probe representative states plus non-zero demo transitions, or all.",
    )
    parser.add_argument(
        "--sequential-rollout",
        action="store_true",
        help=(
            "Restore only local state zero, then replay the complete demo "
            "suffix without any intermediate checkpoint recovery."
        ),
    )
    parser.add_argument(
        "--raw-command-replay",
        action="store_true",
        help=(
            "In sequential mode, replay the next snapshot's recorded low-level "
            "position/velocity targets directly instead of decoding an RFCL action."
        ),
    )
    parser.add_argument(
        "--bootstrap-motion-plan",
        action="store_true",
        help=(
            "Run the task's grasp/pre-insert Motion Plan in this process before "
            "restoring the RFCL suffix; preserves the live gripper contact manifold."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Record actor, attachment, velocity, and relative-pose diagnostics.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def _representative_indices(state_count: int) -> list[int]:
    transition_count = max(0, state_count - 1)
    if transition_count <= 0:
        return []
    return sorted({0, transition_count // 2, transition_count - 1})


def _action_probe_indices(trajectory) -> list[int]:
    indices = set(_representative_indices(len(trajectory.states)))
    indices.update(
        int(index)
        for index in np.flatnonzero(np.max(np.abs(trajectory.actions), axis=1) > 1e-6)
    )
    return sorted(indices)


def _max_abs(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def _pose_snapshot(task) -> dict[str, np.ndarray]:
    from policy.RL.rfcl_snapshot import read_robot_physics_state

    physics_state = read_robot_physics_state(task)
    return {
        "joint": physics_state["joint_pos"],
        "ee": physics_state["ee"],
        "usb": np.asarray(task.prism.get_pose().tolist(), dtype=np.float32),
        "slot": np.asarray(task.slot.get_pose().tolist(), dtype=np.float32),
    }


def _pose_errors(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "joint_arm_max_abs": _max_abs(actual["joint"][:7], expected["joint"][:7]),
        "ee_position_l2": float(np.linalg.norm(actual["ee"][:3] - expected["ee"][:3])),
        "usb_position_l2": float(np.linalg.norm(actual["usb"][:3] - expected["usb"][:3])),
        "slot_position_l2": float(np.linalg.norm(actual["slot"][:3] - expected["slot"][:3])),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    env = None
    report: dict[str, object] = {
        "schema": "rfcl_demo_action_probe_v1",
        "snapshot_root": str(args.snapshot_root),
        "action_repeat": int(args.action_repeat),
        "demos": [],
    }
    try:
        from policy.RL.rfcl_env import RFCLPrivilegedEnv
        from policy.RL.rfcl_snapshot import runtime_snapshot_diagnostics
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            # UIPC frame dumps are part of the snapshot dataset workspace;
            # recovery must use the exact same scene/dump directory.
            save_dir=args.snapshot_root,
            video_frequency=0,
            step_limit=int(args.step_limit),
            mode="eval_test",
            save_pre_move=False,
            device=args.device,
        )
        env = RFCLPrivilegedEnv(
            task,
            args.snapshot_root,
            action_repeat=int(args.action_repeat),
            snapshot_sync_steps=int(args.snapshot_sync_steps),
            snapshot_sync_constraint_strength=float(
                args.snapshot_sync_constraint_strength
            ),
            snapshot_rerestore_after_sync=not args.no_rerestore_after_sync,
            demo_horizon_to_max_steps_ratio=float(
                args.demo_horizon_to_max_steps_ratio
            ),
            seed=0,
        )
        trajectories = env.curriculum.demos
        demo_indices = list(range(len(trajectories))) if args.demo_indices is None else [int(x) for x in args.demo_indices]
        for demo_index in demo_indices:
            trajectory = trajectories[demo_index]
            if args.bootstrap_motion_plan:
                if not args.sequential_rollout:
                    raise ValueError("--bootstrap-motion-plan requires --sequential-rollout")
                seed = env.dataset.demos[demo_index].get("seed")
                task.reset(seed=None if seed is None else int(seed))
                prepare_handoff = getattr(task, "_prepare_usb_standard", None)
                if not callable(prepare_handoff):
                    raise RuntimeError(
                        "--bootstrap-motion-plan is only supported by InsertUSB tasks"
                    )
                prepare_handoff()
            if args.sequential_rollout:
                state_indices = list(range(trajectory.transition_count))
            elif args.local_state_indices is not None:
                state_indices = [int(index) for index in args.local_state_indices]
            elif args.checkpoints == "all":
                state_indices = list(range(trajectory.transition_count))
            elif args.checkpoints == "actions":
                state_indices = _action_probe_indices(trajectory)
            else:
                state_indices = _representative_indices(len(trajectory.states))
            demo_report = {
                "demo_index": demo_index,
                "sequential_rollout": bool(args.sequential_rollout),
                "checks": [],
            }
            if args.sequential_rollout:
                state, reset_info = env.reset(
                    options={
                        "demo_index": demo_index,
                        "state_index": 0,
                        "skip_task_reset": bool(args.bootstrap_motion_plan),
                    }
                )
            for state_index in state_indices:
                snapshot = env.dataset.snapshot(demo_index, state_index)
                expected_next = env.dataset.snapshot(demo_index, state_index + 1)
                if not args.sequential_rollout:
                    state, reset_info = env.reset(
                        options={"demo_index": demo_index, "state_index": state_index}
                    )
                initial_state_error = _max_abs(state, snapshot.privileged_state)
                initial_pose = _pose_snapshot(task)
                restore_runtime = (
                    runtime_snapshot_diagnostics(task, snapshot)
                    if args.diagnostics
                    else None
                )
                action = trajectory.actions[state_index]
                if args.raw_command_replay:
                    if not args.sequential_rollout:
                        raise ValueError("--raw-command-replay requires --sequential-rollout")
                    # A false marker means that the original Motion Plan was
                    # in a delay/no-command step.  Preserve that timing by
                    # advancing physics without writing a fresh PD target.
                    force = bool(expected_next.force_position_write)
                    manager = task._robot_manager
                    if force:
                        device = task.device
                        command_pos = torch.as_tensor(
                            np.asarray(expected_next.robot_state["joint_pos_target"])
                            .reshape(-1),
                            dtype=torch.float32,
                            device=device,
                        )
                        command_vel = torch.as_tensor(
                            np.asarray(expected_next.robot_state["joint_vel_target"])
                            .reshape(-1),
                            dtype=torch.float32,
                            device=device,
                        )
                        manager.set_arm(
                            command_pos[:7],
                            vel=command_vel[:7],
                            force=True,
                        )
                        # GelSight uses the two Panda finger joints.  Passing
                        # the first target lets RobotManager apply its normal
                        # mimic mapping and matches Motion Plan calls.
                        gripper_id = int(manager._gripper_ids[0])
                        manager.set_gripper(
                            command_pos[gripper_id],
                            force=True,
                        )
                    task._step(is_save=False)
                    task.take_action_cnt += 1
                    task.policy_step_count += 1
                    success = bool(task.check_success())
                    if success:
                        task.eval_success = True
                    reward = 1.0 if success else 0.0
                    terminated = bool(success)
                    truncated = bool(
                        (not success)
                        and task.take_action_cnt >= int(args.step_limit)
                    )
                    info = {
                        "success": success,
                        "force_position_write": force,
                        "raw_command_replay": True,
                    }
                else:
                    state, reward, terminated, truncated, info = env.step(action)
                actual_pose = _pose_snapshot(task)
                expected_pose = {
                    key: np.asarray(expected_next.poses[key]).reshape(-1)
                    for key in ("ee", "usb", "slot")
                }
                expected_pose["joint"] = np.asarray(expected_next.robot_state["joint_pos"]).reshape(-1)
                check = {
                        "state_index": state_index,
                        "raw_state_index": int(snapshot.state_index),
                        "snapshot_id": snapshot.snapshot_id,
                        "next_snapshot_id": expected_next.snapshot_id,
                        "initial_state_max_abs_error": initial_state_error,
                        "initial_pose_errors": _pose_errors(
                            initial_pose,
                            {
                                key: np.asarray(snapshot.poses[key]).reshape(-1)
                                for key in ("ee", "usb", "slot")
                            }
                            | {
                                "joint": np.asarray(
                                    snapshot.robot_state["joint_pos"]
                                ).reshape(-1)
                            },
                        ),
                        "action_max_abs": float(np.max(np.abs(action))),
                        "force_position_write": bool(expected_next.force_position_write),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "pose_errors": _pose_errors(actual_pose, expected_pose),
                        "terminal_reason": info.get("terminal_reason"),
                    }
                if args.diagnostics:
                    check["restore_runtime"] = restore_runtime
                    check["next_runtime"] = runtime_snapshot_diagnostics(task, expected_next)
                demo_report["checks"].append(check)
                if args.sequential_rollout and (terminated or truncated):
                    break
            demo_report["handoff"] = env.dataset.handoff_diagnostics(demo_index)
            demo_report["local_state_count"] = env.dataset.state_count(demo_index)
            report["demos"].append(demo_report)
    except BaseException as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        simulation_app.close()
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
