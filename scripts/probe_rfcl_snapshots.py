"""Validate full-state RFCL checkpoints with a real Motion Plan rollout.

The probe executes one successful InsertUSB Motion Plan trajectory, dumps the
UIPC world at three suffix checkpoints, and records the low-level articulation
targets used by the planner.  It then restores each checkpoint and replays the
exact remaining target sequence.  This isolates checkpoint fidelity from RL
and from motion replanning.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/rfcl_snapshot_probe_seed10.json"),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/analysis/rfcl_snapshot_probe_seed10"),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _optional_robot_value(robot: Any, name: str, fallback: np.ndarray) -> np.ndarray:
    value = getattr(robot.data, name, None)
    if value is None:
        return np.asarray(fallback).copy()
    return _to_numpy(value)


def _actor_control_state(task: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, actor in task._actor_manager.actors.items():
        result[name] = {
            "next_status": actor.next_status,
            "next_pts": copy.deepcopy(actor.next_pts),
            "next_mat": copy.deepcopy(actor.next_mat),
            "next_mask": copy.deepcopy(actor.next_mask),
        }
    return result


def _restore_actor_control_state(task: Any, state: dict[str, dict[str, Any]]) -> None:
    for name, values in state.items():
        actor = task._actor_manager.actors[name]
        actor.next_status = values["next_status"]
        actor.next_pts = copy.deepcopy(values["next_pts"])
        actor.next_mat = copy.deepcopy(values["next_mat"])
        actor.next_mask = copy.deepcopy(values["next_mask"])


def _tactile_attachment_state(task: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, tactile in task._tactile_manager.tactiles.items():
        attachment = tactile.attachment
        result[name] = {
            "aim_positions": copy.deepcopy(attachment.aim_positions),
            "attachment_offsets": copy.deepcopy(attachment.attachment_offsets),
            "obj_pose": copy.deepcopy(getattr(attachment, "obj_pose", None)),
        }
    return result


def _restore_tactile_attachment_state(
    task: Any, state: dict[str, dict[str, Any]]
) -> None:
    for name, values in state.items():
        attachment = task._tactile_manager.tactiles[name].attachment
        attachment.aim_positions = copy.deepcopy(values["aim_positions"])
        attachment.attachment_offsets = copy.deepcopy(values["attachment_offsets"])
        if values["obj_pose"] is not None:
            attachment.obj_pose = copy.deepcopy(values["obj_pose"])


def _pose_snapshot(task: Any) -> dict[str, np.ndarray]:
    manager = task._robot_manager
    robot = manager.robot
    return {
        "joint_pos": _to_numpy(robot.data.joint_pos).reshape(-1),
        "joint_vel": _to_numpy(robot.data.joint_vel).reshape(-1),
        "ee": np.asarray(manager.get_ee_pose().tolist(), dtype=np.float64),
        "usb": np.asarray(task.prism.get_pose().tolist(), dtype=np.float64),
        "slot": np.asarray(task.slot.get_pose().tolist(), dtype=np.float64),
    }


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)))


def _state_error(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> dict[str, float]:
    result = {
        "joint_pos_max_abs_rad": float(
            np.max(np.abs(actual["joint_pos"] - expected["joint_pos"]))
        ),
        "joint_vel_max_abs_rad_s": float(
            np.max(np.abs(actual["joint_vel"] - expected["joint_vel"]))
        ),
    }
    for name in ("ee", "usb", "slot"):
        result[f"{name}_position_l2_m"] = float(
            np.linalg.norm(actual[name][:3] - expected[name][:3])
        )
        result[f"{name}_rotation_rad"] = _quaternion_angle(
            actual[name][3:7], expected[name][3:7]
        )
    return result


@dataclass
class LowLevelCommand:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_pos_target: np.ndarray
    joint_vel_target: np.ndarray
    force_position_write: bool
    atom_id: int
    atom_tag: str


@dataclass
class FullCheckpoint:
    label: str
    uipc_frame: int
    trace_index: int
    task_state: dict[str, Any]
    robot_state: dict[str, np.ndarray]
    actor_control_state: dict[str, dict[str, Any]]
    tactile_attachment_state: dict[str, dict[str, Any]]
    poses: dict[str, np.ndarray]

    def manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "uipc_frame": self.uipc_frame,
            "trace_index": self.trace_index,
            "sim_step": self.task_state["step_count"],
            "atom_id": self.task_state["atom_id"],
            "atom_tag": self.task_state["atom_tag"],
            "usb_pose": self.poses["usb"].tolist(),
            "slot_pose": self.poses["slot"].tolist(),
            "ee_pose": self.poses["ee"].tolist(),
            "joint_pos": self.robot_state["joint_pos"].reshape(-1).tolist(),
            "joint_vel": self.robot_state["joint_vel"].reshape(-1).tolist(),
            "actor_constraint_status": {
                name: values["next_status"]
                for name, values in self.actor_control_state.items()
            },
            "tactile_attachment_points": {
                name: int(np.asarray(values["aim_positions"]).reshape(-1, 3).shape[0])
                for name, values in self.tactile_attachment_state.items()
            },
        }


class MotionPlanRecorder:
    def __init__(self, task: Any) -> None:
        self.task = task
        self.commands: list[LowLevelCommand] = []
        self.checkpoints: dict[str, FullCheckpoint] = {}
        self._recording = False
        self._insertion_start_z: float | None = None
        self._original_step: Callable[..., Any] | None = None
        self._original_move: Callable[..., Any] | None = None
        self._original_set_arm: Callable[..., Any] | None = None
        self._original_set_gripper: Callable[..., Any] | None = None
        self._force_position_write_pending = False

    def install(self) -> None:
        self._original_step = self.task._step
        self._original_move = self.task.move
        manager = self.task._robot_manager
        self._original_set_arm = manager.set_arm
        self._original_set_gripper = manager.set_gripper

        def recorded_set_arm(*args: Any, **kwargs: Any) -> Any:
            force = bool(kwargs.get("force", True))
            if self._recording and force:
                self._force_position_write_pending = True
            return self._original_set_arm(*args, **kwargs)

        def recorded_set_gripper(*args: Any, **kwargs: Any) -> Any:
            force = bool(kwargs.get("force", True))
            if self._recording and force:
                self._force_position_write_pending = True
            return self._original_set_gripper(*args, **kwargs)

        def recorded_step(*args: Any, **kwargs: Any) -> Any:
            if self._recording:
                robot = self.task._robot_manager.robot
                joint_pos = _to_numpy(robot.data.joint_pos)
                self.commands.append(
                    LowLevelCommand(
                        joint_pos=joint_pos,
                        joint_vel=_to_numpy(robot.data.joint_vel),
                        joint_pos_target=_optional_robot_value(
                            robot, "joint_pos_target", joint_pos
                        ),
                        joint_vel_target=_optional_robot_value(
                            robot, "joint_vel_target", np.zeros_like(joint_pos)
                        ),
                        force_position_write=self._force_position_write_pending,
                        atom_id=int(self.task.atom_id),
                        atom_tag=str(self.task.atom_tag),
                    )
                )
            result = self._original_step(*args, **kwargs)
            if self._recording:
                self._maybe_capture_insertion_middle()
                self._force_position_write_pending = False
            return result

        def recorded_move(*args: Any, **kwargs: Any) -> Any:
            tag = str(kwargs.get("tag", "move"))
            result = self._original_move(*args, **kwargs)
            if self._recording and result:
                if tag == "move_usb_to_pre_insert":
                    self.capture("handoff_end")
                elif tag == "insert_USB_into_slot":
                    self.capture("near_terminal")
            return result

        self.task._step = recorded_step
        self.task.move = recorded_move
        manager.set_arm = recorded_set_arm
        manager.set_gripper = recorded_set_gripper

    def uninstall(self) -> None:
        if self._original_step is not None:
            self.task._step = self._original_step
        if self._original_move is not None:
            self.task.move = self._original_move
        manager = self.task._robot_manager
        if self._original_set_arm is not None:
            manager.set_arm = self._original_set_arm
        if self._original_set_gripper is not None:
            manager.set_gripper = self._original_set_gripper

    def start(self) -> None:
        self._recording = True

    def stop(self) -> None:
        self._recording = False

    def _maybe_capture_insertion_middle(self) -> None:
        if str(self.task.atom_tag) != "insert_USB_into_slot":
            return
        current_z = float(self.task.prism.get_pose().p[2])
        if self._insertion_start_z is None:
            self._insertion_start_z = current_z
            return
        if "insertion_middle" in self.checkpoints:
            return
        target_z = float(self.task.target_pose.p[2])
        total = self._insertion_start_z - target_z
        if total <= 1e-9:
            return
        progress = (self._insertion_start_z - current_z) / total
        if progress >= 0.5:
            self.capture("insertion_middle")

    def capture(self, label: str) -> None:
        if label in self.checkpoints:
            return
        world = self.task.uipc_sim.world
        if not bool(world.dump()):
            raise RuntimeError(f"UIPC world.dump() failed for checkpoint {label!r}")
        robot = self.task._robot_manager.robot
        joint_pos = _to_numpy(robot.data.joint_pos)
        joint_vel = _to_numpy(robot.data.joint_vel)
        checkpoint = FullCheckpoint(
            label=label,
            uipc_frame=int(world.frame()),
            trace_index=len(self.commands),
            task_state={
                "step_count": int(self.task.step_count),
                "take_action_cnt": int(self.task.take_action_cnt),
                "policy_step_count": int(self.task.policy_step_count),
                "phase_id": int(self.task.phase_id),
                "atom_id": int(self.task.atom_id),
                "atom_tag": str(self.task.atom_tag),
                "eval_success": bool(self.task.eval_success),
                "plan_success": bool(self.task.plan_success),
                "terminal_reason": self.task.terminal_reason,
                "last_render": int(self.task.last_render),
            },
            robot_state={
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "joint_pos_target": _optional_robot_value(
                    robot, "joint_pos_target", joint_pos
                ),
                "joint_vel_target": _optional_robot_value(
                    robot, "joint_vel_target", np.zeros_like(joint_vel)
                ),
            },
            actor_control_state=_actor_control_state(self.task),
            tactile_attachment_state=_tactile_attachment_state(self.task),
            poses=_pose_snapshot(self.task),
        )
        self.checkpoints[label] = checkpoint
        print(
            f"\n[rfcl-snapshot] captured {label}: frame={checkpoint.uipc_frame}, "
            f"trace_index={checkpoint.trace_index}, usb_z={checkpoint.poses['usb'][2]:.6f}",
            flush=True,
        )


def _write_robot_state(task: Any, state: dict[str, np.ndarray]) -> None:
    robot = task._robot_manager.robot
    device = robot.device

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(state[name], dtype=torch.float32, device=device)

    joint_pos = tensor("joint_pos")
    joint_vel = tensor("joint_vel")
    joint_pos_target = tensor("joint_pos_target")
    joint_vel_target = tensor("joint_vel_target")
    robot.set_joint_position_target(joint_pos_target)
    robot.set_joint_velocity_target(joint_vel_target)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot._physics_sim_view.update_articulations_kinematic()


def restore_checkpoint(task: Any, checkpoint: FullCheckpoint) -> None:
    world = task.uipc_sim.world
    if not bool(world.recover(checkpoint.uipc_frame)):
        raise RuntimeError(
            f"UIPC world.recover({checkpoint.uipc_frame}) failed for {checkpoint.label!r}"
        )
    world.retrieve()
    _restore_actor_control_state(task, checkpoint.actor_control_state)
    _write_robot_state(task, checkpoint.robot_state)
    # UIPC dumps generalized positions/velocities, but the gelpad attachment
    # target arrays live in Python and otherwise retain their terminal values.
    _restore_tactile_attachment_state(task, checkpoint.tactile_attachment_state)

    for name, value in checkpoint.task_state.items():
        setattr(task, name, value)
    task.render_outdated = True
    task.uipc_sim._contact_grad_cache = None
    task.scene.update(dt=0.0)
    task._actor_manager.update(dt=0.0)
    task.uipc_sim.update_render_meshes()
    task._actor_manager.sync_visuals()


def apply_command(task: Any, command: LowLevelCommand) -> None:
    robot = task._robot_manager.robot
    joint_pos_target = torch.as_tensor(
        command.joint_pos_target, dtype=torch.float32, device=robot.device
    )
    joint_vel_target = torch.as_tensor(
        command.joint_vel_target, dtype=torch.float32, device=robot.device
    )
    robot.set_joint_position_target(joint_pos_target)
    robot.set_joint_velocity_target(joint_vel_target)
    # Motion Plan force-writes qpos on new arm/gripper control points, but lets
    # the articulation evolve under physics during delay/contact steps.
    if command.force_position_write:
        robot.root_physx_view.set_dof_positions(
            robot._data.joint_pos_target,
            robot._ALL_INDICES,
        )
    task.atom_id = int(command.atom_id)
    task.atom_tag = str(command.atom_tag)
    task._step(is_save=False)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print("[rfcl-snapshot] launching Isaac application", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    report: dict[str, Any] = {
        "schema": "rfcl_snapshot_probe_v1",
        "task": args.task_name,
        "task_config": args.task_config,
        "seed": int(args.seed),
        "purpose": "full_state_restore_then_exact_motion_plan_suffix_replay",
        "checkpoints": [],
    }
    try:
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.save_dir,
            video_frequency=0,
            step_limit=500,
            mode="collect",
            save_pre_move=False,
            device=args.device,
        )
        # This probe records low-level controls directly and does not need RGB,
        # tactile, or temporary HDF5 observations during the Motion Plan run.
        task.cfg.save_frequency = 0
        print(f"[rfcl-snapshot] resetting seed {args.seed}", flush=True)
        task.reset(seed=int(args.seed))

        recorder = MotionPlanRecorder(task)
        recorder.install()
        recorder.start()
        rollout_start = time.perf_counter()
        task.play_once()
        recorder.stop()
        recorder.uninstall()

        baseline_success = bool(task.plan_success and task.check_success())
        baseline_terminal = _pose_snapshot(task)
        report["baseline"] = {
            "success": baseline_success,
            "plan_success": bool(task.plan_success),
            "elapsed_seconds": time.perf_counter() - rollout_start,
            "low_level_steps": len(recorder.commands),
            "terminal_usb_pose": baseline_terminal["usb"].tolist(),
        }
        required = ("handoff_end", "insertion_middle", "near_terminal")
        missing = [label for label in required if label not in recorder.checkpoints]
        if missing:
            raise RuntimeError(f"Motion Plan did not produce checkpoints: {missing}")
        if not baseline_success:
            raise RuntimeError("Seed Motion Plan rollout was not successful")

        report["manifests"] = [
            recorder.checkpoints[label].manifest() for label in required
        ]
        for label in required:
            checkpoint = recorder.checkpoints[label]
            restore_start = time.perf_counter()
            restore_checkpoint(task, checkpoint)
            restored = _pose_snapshot(task)
            restore_error = _state_error(restored, checkpoint.poses)

            for command in recorder.commands[checkpoint.trace_index :]:
                apply_command(task, command)
            suffix_success = bool(task.plan_success and task.check_success())
            suffix_terminal = _pose_snapshot(task)
            entry = {
                "label": label,
                "uipc_frame": checkpoint.uipc_frame,
                "trace_index": checkpoint.trace_index,
                "suffix_steps": len(recorder.commands) - checkpoint.trace_index,
                "restore_error": restore_error,
                "suffix_success": suffix_success,
                "terminal_error_vs_baseline": _state_error(
                    suffix_terminal, baseline_terminal
                ),
                "elapsed_seconds": time.perf_counter() - restore_start,
            }
            report["checkpoints"].append(entry)
            print(f"\n{json.dumps(entry, indent=2)}", flush=True)
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
        if task is not None:
            task.close()
        simulation_app.close()

    print(f"[rfcl-snapshot] report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
