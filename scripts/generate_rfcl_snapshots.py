"""Generate persistent full-state RFCL demos from Motion Plan episodes.

The recorder keeps the final pre-insert checkpoint, then dumps every
``--stride`` simulator step until the first frame satisfying the task's
original success predicate.  It deliberately stops before the scripted
gripper release, so the resulting demos match the RFCL environment's fixed
closed-gripper action space.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher

from policy.RL.checkpoint import extract_action_scale_from_bc_checkpoint
from policy.RL.rfcl_snapshot import (
    SNAPSHOT_DATASET_SCHEMA,
    SNAPSHOT_MANIFEST_NAME,
    RFCLSnapshot,
    archive_uipc_frame,
    capture_snapshot,
    write_snapshot_sidecar,
)
from policy.RL.rfcl import DEFAULT_GRIPPER_OFFSET, handoff_xy_error


START_TAG = "move_usb_to_pre_insert"
ALLOWED_TAGS = {
    "move_usb_to_pre_insert",
    "move_usb_to_play_pre_insert",
    "insert_usb_into_slot",
    "delay",
    "open_gripper_after_insert",
    "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--action-mode",
        choices=("qpos_delta", "target_pos_vel", "target_pos_vel_force"),
        default="qpos_delta",
        help="RFCL action representation stored in the manifest.",
    )
    parser.add_argument("--step-limit", type=int, default=600)
    parser.add_argument(
        "--skip-failed-demos",
        action="store_true",
        help=(
            "Record failed Motion Plan seeds as diagnostics and continue; only "
            "successful demos are added to the RFCL manifest."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rfcl/snapshot_pilot_v1"),
    )
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument(
        "--action-scale",
        type=float,
        nargs="+",
        metavar="SCALE",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def _action_scale(args: argparse.Namespace) -> np.ndarray:
    if args.action_scale is not None and args.bc_checkpoint is not None:
        raise ValueError("Pass either --action-scale or --bc-checkpoint, not both")
    if args.action_scale is not None:
        scale = np.asarray(args.action_scale, dtype=np.float32)
    elif args.bc_checkpoint is not None:
        checkpoint = torch.load(
            args.bc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        scale = extract_action_scale_from_bc_checkpoint(
            checkpoint,
            expected_action_dim=7,
            device="cpu",
        ).numpy().astype(np.float32)
    else:
        # This is the validated scale used by the existing qpos BC datasets.
        scale = np.asarray(
            [
                0.000938745797611773,
                0.0023768171668052673,
                0.0007808265509083867,
                0.0016817807918414474,
                0.0012212832225486636,
                0.00456193694844842,
                0.001278358744457364,
            ],
            dtype=np.float32,
        )
    if args.action_mode in ("target_pos_vel", "target_pos_vel_force"):
        if scale.shape == (7,):
            # Planner velocity targets are in rad/s.  This range covers the
            # saved USB insertion commands while leaving exploration headroom.
            scale = np.concatenate((scale, np.full(7, 0.25, dtype=np.float32)))
        if scale.shape == (14,) and args.action_mode == "target_pos_vel_force":
            scale = np.concatenate((scale, np.ones(1, dtype=np.float32)))
        expected = 15 if args.action_mode == "target_pos_vel_force" else 14
        if scale.shape != (expected,):
            raise ValueError(f"{args.action_mode} action scale must have shape ({expected},), got {scale.shape}")
    elif scale.shape != (7,):
        raise ValueError(f"qpos_delta action scale must have shape (7,), got {scale.shape}")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"action scale must be finite positive, got {scale}")
    return scale


def _tag(task: Any) -> str:
    return str(task.atom_tag).strip().lower()


class SnapshotRecorder:
    def __init__(
        self,
        task: Any,
        root: Path,
        *,
        demo_index: int,
        seed: int,
        stride: int,
    ) -> None:
        self.task = task
        self.root = root
        self.demo_index = int(demo_index)
        self.seed = int(seed)
        self.demo_id = str(seed)
        self.stride = int(stride)
        self.snapshots: list[dict[str, Any]] = []
        self.snapshot_objects: list[RFCLSnapshot] = []
        self.pending_start: RFCLSnapshot | None = None
        self.suffix_started = False
        self.terminal_success = False
        self._original_step = None
        self._original_set_arm = None
        self._original_set_gripper = None
        self._force_position_write_pending = False
        self._current_force_position_write = True
        self._command_arm_pos = None
        self._command_arm_vel = None

    def install(self) -> None:
        self._original_step = self.task._step
        manager = self.task._robot_manager
        self._original_set_arm = manager.set_arm
        self._original_set_gripper = manager.set_gripper

        def recorded_set_arm(*args: Any, **kwargs: Any) -> Any:
            if bool(kwargs.get("force", True)):
                self._force_position_write_pending = True
            if args:
                self._command_arm_pos = np.asarray(
                    args[0].detach().cpu().numpy()
                    if torch.is_tensor(args[0])
                    else args[0],
                    dtype=np.float32,
                ).reshape(-1)[:7].copy()
                velocity = kwargs.get("vel", args[1] if len(args) > 1 else None)
                if velocity is not None:
                    self._command_arm_vel = np.asarray(
                        velocity.detach().cpu().numpy()
                        if torch.is_tensor(velocity)
                        else velocity,
                        dtype=np.float32,
                    ).reshape(-1)[:7].copy()
            return self._original_set_arm(*args, **kwargs)

        def recorded_set_gripper(*args: Any, **kwargs: Any) -> Any:
            if bool(kwargs.get("force", True)):
                self._force_position_write_pending = True
            return self._original_set_gripper(*args, **kwargs)

        def recorded_step(*args: Any, **kwargs: Any) -> Any:
            result = self._original_step(*args, **kwargs)
            self._current_force_position_write = self._force_position_write_pending
            self._force_position_write_pending = False
            self._observe_step()
            return result

        self.task._step = recorded_step
        manager.set_arm = recorded_set_arm
        manager.set_gripper = recorded_set_gripper

    def uninstall(self) -> None:
        if self._original_step is not None:
            self.task._step = self._original_step
        manager = self.task._robot_manager
        if self._original_set_arm is not None:
            manager.set_arm = self._original_set_arm
        if self._original_set_gripper is not None:
            manager.set_gripper = self._original_set_gripper

    def _capture(self, success: bool) -> RFCLSnapshot:
        snapshot_id = f"demo_{self.seed:06d}_state_{len(self.snapshot_objects):04d}"
        snapshot = capture_snapshot(
            self.task,
            snapshot_id=snapshot_id,
            demo_id=self.demo_id,
            state_index=-1,
            success=bool(success),
            force_position_write=self._current_force_position_write,
        )
        if self._command_arm_pos is not None:
            snapshot.robot_state["joint_pos_target"] = np.asarray(
                snapshot.robot_state["joint_pos_target"]
            ).copy()
            snapshot.robot_state["joint_pos_target"].reshape(-1)[:7] = (
                self._command_arm_pos
            )
        if self._command_arm_vel is not None:
            snapshot.robot_state["joint_vel_target"] = np.asarray(
                snapshot.robot_state["joint_vel_target"]
            ).copy()
            snapshot.robot_state["joint_vel_target"].reshape(-1)[:7] = (
                self._command_arm_vel
            )
        # Each reset can reuse low world.frame() values.  Reserve one million
        # logical frames per demo so all checkpoints remain addressable in the
        # single UIPC workspace used by the training environment.  Repeated
        # pending handoff captures intentionally replace local state zero.
        if len(self.snapshot_objects) >= 1_000_000:
            raise RuntimeError("RFCL demo exceeds its logical UIPC frame range")
        logical_frame = (
            (self.demo_index + 1) * 1_000_000 + len(self.snapshot_objects)
        )
        archive_uipc_frame(
            self.root / "scene" / "dump",
            snapshot.uipc_frame,
            logical_frame,
        )
        snapshot.uipc_frame = logical_frame
        return snapshot

    def _emit(self, snapshot: RFCLSnapshot) -> None:
        snapshot.state_index = len(self.snapshot_objects)
        metadata = write_snapshot_sidecar(self.root, snapshot)
        self.snapshot_objects.append(snapshot)
        self.snapshots.append(metadata)

    def _observe_step(self) -> None:
        if self.terminal_success:
            return
        current_tag = _tag(self.task)

        # The HDF5 suffix uses the last saved pre-insert frame. Keep replacing
        # this pending checkpoint until the planner leaves that atom.
        if not self.suffix_started:
            if current_tag == START_TAG:
                if int(self.task.step_count) % self.stride == 0:
                    self.pending_start = self._capture(success=False)
                return
            if self.pending_start is None:
                return
            self.suffix_started = True
            self._emit(self.pending_start)

        if current_tag not in ALLOWED_TAGS:
            raise RuntimeError(
                f"Unexpected RFCL suffix tag {current_tag!r} at step "
                f"{self.task.step_count}"
            )

        diagnostics = self.task._get_success_diagnostics()
        success = bool(
            diagnostics["xy_ok"]
            and diagnostics["z_ok"]
            and diagnostics["ee_z_ok"]
            and diagnostics["angle_ok"]
        )
        due = int(self.task.step_count) % self.stride == 0
        if due or success:
            self._emit(self._capture(success=success))
        if success:
            self.terminal_success = True

    def finish(self) -> dict[str, Any]:
        if self.pending_start is not None and not self.suffix_started:
            raise RuntimeError("Motion Plan never left the pre-insert atom")
        if len(self.snapshot_objects) < 2:
            raise RuntimeError(
                f"RFCL demo {self.demo_id} has too few snapshots: "
                f"{len(self.snapshot_objects)}"
            )
        if not self.terminal_success:
            raise RuntimeError(f"RFCL demo {self.demo_id} never reached success")
        handoff_errors = [
            handoff_xy_error(
                snapshot.poses["ee"],
                snapshot.poses["usb"],
                gripper_offset=DEFAULT_GRIPPER_OFFSET,
            )
            for snapshot in self.snapshot_objects
        ]
        valid_handoffs = [
            index for index, error in enumerate(handoff_errors) if error <= 0.01
        ]
        if not valid_handoffs or valid_handoffs[0] >= len(self.snapshot_objects) - 1:
            raise RuntimeError(
                f"RFCL demo {self.demo_id} has no valid handoff before terminal: "
                f"min_xy_error={min(handoff_errors):.6f} m"
            )
        policy_start = int(valid_handoffs[0])
        return {
            "demo_id": self.demo_id,
            "seed": self.seed,
            "snapshot_ids": [snapshot["snapshot_id"] for snapshot in self.snapshots],
            "state_count": len(self.snapshots),
            "transition_count": len(self.snapshots) - 1,
            "terminal_snapshot_id": self.snapshots[-1]["snapshot_id"],
            "terminal_sim_step": self.snapshots[-1]["sim_step"],
            "policy_start_state_index": policy_start,
            "handoff_xy_error_m": float(handoff_errors[policy_start]),
            "handoff_xy_tolerance_m": 0.01,
        }


def _write_manifest(
    root: Path,
    *,
    args: argparse.Namespace,
    action_scale: np.ndarray,
    demos: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "schema": SNAPSHOT_DATASET_SCHEMA,
        "task": args.task_name,
        "task_config": args.task_config,
        "stride": int(args.stride),
        "action_repeat": int(args.stride),
        "action_mode": str(args.action_mode),
        "action_dim": 7,
        "action_scale": action_scale.tolist(),
        "gripper_offset": DEFAULT_GRIPPER_OFFSET,
        "handoff_xy_tolerance": 0.01,
        "reward": "success ? 1 : 0",
        "terminal_definition": "first_frame_satisfying_insert_USB.check_success",
        "gripper_control": "held_closed",
        "demos": demos,
        "snapshots": snapshots,
        "failed_demos": failures or [],
    }
    (root / SNAPSHOT_MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / SNAPSHOT_MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing RFCL dataset: {manifest_path}"
        )
    action_scale = _action_scale(args)
    print(f"[rfcl-snapshot] output={args.output}", flush=True)
    print(f"[rfcl-snapshot] seeds={args.seeds} stride={args.stride}", flush=True)
    print(f"[rfcl-snapshot] action_scale={action_scale.tolist()}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    demos: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    try:
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.output,
            video_frequency=0,
            step_limit=int(args.step_limit),
            mode="eval_test",
            save_pre_move=False,
            device=args.device,
        )
        task.cfg.save_frequency = 0
        task.cfg.video_frequency = 0

        if len(set(map(int, args.seeds))) != len(args.seeds):
            raise ValueError("--seeds must not contain duplicates")

        for demo_index, seed in enumerate(args.seeds):
            seed = int(seed)
            print(f"\n[rfcl-snapshot] seed={seed} reset", flush=True)
            task.reset(seed=seed)
            try:
                recorder = SnapshotRecorder(
                    task,
                    args.output,
                    demo_index=demo_index,
                    seed=seed,
                    stride=int(args.stride),
                )
                recorder.install()
                try:
                    task.play_once()
                finally:
                    recorder.uninstall()
                # ``play_once`` continues with the scripted gripper release and a
                # delay after the insertion predicate first becomes true.  The
                # recorder intentionally stops at that first success frame, so a
                # final check after ``play_once`` would inspect the released USB
                # and incorrectly mark an otherwise valid demo as failed.
                if not recorder.terminal_success:
                    diagnostics = task._get_success_diagnostics()
                    raise RuntimeError(
                        f"Motion Plan seed {seed} failed before first success: "
                        f"{diagnostics}"
                    )
                demo = recorder.finish()
                demos.append(demo)
                snapshots.update({item["snapshot_id"]: item for item in recorder.snapshots})
                _write_manifest(
                    args.output,
                    args=args,
                    action_scale=action_scale,
                    demos=demos,
                    snapshots=snapshots,
                    failures=failures,
                )
                print(
                    f"[rfcl-snapshot] seed={seed} success; states={demo['state_count']} "
                    f"terminal_step={demo['terminal_sim_step']}",
                    flush=True,
                )
            except BaseException as exc:
                if not args.skip_failed_demos:
                    raise
                diagnostics = None
                try:
                    diagnostics = task._get_success_diagnostics()
                except Exception:
                    pass
                failure = {
                    "seed": seed,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "diagnostics": diagnostics,
                }
                failures.append(failure)
                print(
                    f"[rfcl-snapshot] seed={seed} skipped after failure: "
                    f"{failure['message']}",
                    flush=True,
                )
                _write_manifest(
                    args.output,
                    args=args,
                    action_scale=action_scale,
                    demos=demos,
                    snapshots=snapshots,
                    failures=failures,
                )
                continue
        if not demos:
            raise RuntimeError("No successful RFCL demos were generated")
    except BaseException as exc:
        error_path = args.output / "rfcl_generation_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed_demos": demos,
                    "failed_demos": failures,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if task is not None:
            task.close()
        simulation_app.close()

    print(f"[rfcl-snapshot] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
