"""Static RFCL demo/checkpoint probe for InsertUSB HDF5 episodes.

This intentionally does not launch Isaac Sim.  It validates the part of RFCL
that can be checked from the saved demonstrations before implementing UIPC
state restoration: suffix boundaries, privileged-state dimensions, normalized
actions, and per-demo reverse-curriculum checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from policy.RL.action import action_to_target_qpos
from policy.RL.checkpoint import extract_action_scale_from_bc_checkpoint
from policy.RL.rfcl import (
    PRIVILEGED_STATE_LAYOUT,
    ReverseCurriculum,
    load_insert_usb_demo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/insert_usb/gelsight/hdf5"))
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument(
        "--action-scale",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    )
    parser.add_argument(
        "--demo-ids",
        type=int,
        nargs="+",
        default=(10, 100, 198),
        help="Representative HDF5 stems to inspect.",
    )
    parser.add_argument("--sim-dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--reverse-step-size", type=int, default=2)
    parser.add_argument("--geometric-p", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true", help="Write JSON without printing the full report.")
    return parser.parse_args()


def _action_scale(args: argparse.Namespace) -> np.ndarray:
    if args.action_scale is not None and args.bc_checkpoint is not None:
        raise ValueError("Pass either --action-scale or --bc-checkpoint, not both")
    if args.action_scale is not None:
        scale = np.asarray(args.action_scale, dtype=np.float32)
    elif args.bc_checkpoint is not None:
        checkpoint = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=False)
        scale = (
            extract_action_scale_from_bc_checkpoint(
                checkpoint,
                expected_action_dim=7,
                device="cpu",
            )
            .numpy()
            .astype(np.float32)
        )
    else:
        raise ValueError("One of --action-scale or --bc-checkpoint is required")
    if scale.shape != (7,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(f"action scale must be finite positive shape (7,), got {scale}")
    return scale


def _checkpoint_indices(demo) -> list[dict[str, object]]:
    tags = np.asarray(demo.tags, dtype=str)
    insertion = np.flatnonzero(tags == "insert_usb_into_slot")
    candidates = [0]
    if len(insertion):
        candidates.append(int(insertion[len(insertion) // 2]))
    candidates.append(max(0, demo.transition_count - 1))
    labels = ("handoff_end", "insertion_middle", "near_terminal")
    result = []
    for label, index in zip(labels, candidates):
        result.append(
            {
                "label": label,
                "state_index": int(index),
                "frame_index": int(demo.frame_indices[index]),
                "tag": str(demo.tags[index]),
                "usb_in_slot_xyz_mm": (
                    demo.states[index, PRIVILEGED_STATE_LAYOUT.usb_in_slot][:3]
                    * 1000.0
                ).tolist(),
            }
        )
    return result


def _probe_demo(demo, action_scale: np.ndarray) -> dict[str, object]:
    reconstructed = np.stack(
        [
            action_to_target_qpos(
                demo.states[index, PRIVILEGED_STATE_LAYOUT.joint][:7],
                demo.actions[index],
                action_scale,
            )
            for index in range(demo.transition_count)
        ],
        axis=0,
    )
    measured = demo.states[1:, PRIVILEGED_STATE_LAYOUT.joint][:, :7]
    delta_error = np.abs(reconstructed - measured)
    clipped = np.any(np.abs(demo.actions) >= 0.999999, axis=1)
    return {
        "demo_id": demo.demo_id,
        "path": str(demo.path),
        "state_count": int(len(demo.frame_indices)),
        "transition_count": demo.transition_count,
        "state_dim": int(demo.states.shape[1]),
        "action_dim": int(demo.actions.shape[1]) if demo.actions.ndim == 2 else 0,
        "velocity_source": demo.velocity_source,
        "frame_start": int(demo.frame_indices[0]),
        "frame_end": int(demo.frame_indices[-1]),
        "max_normalized_action": float(np.max(np.abs(demo.actions))),
        "clipped_transition_count": int(clipped.sum()),
        "reconstructed_next_qpos_max_abs_error": float(delta_error.max()),
        "checkpoints": _checkpoint_indices(demo),
    }


def main() -> None:
    args = parse_args()
    action_scale = _action_scale(args)
    demos = []
    for demo_id in args.demo_ids:
        path = args.data_dir / f"{int(demo_id)}.hdf5"
        if not path.exists():
            raise FileNotFoundError(path)
        demos.append(
            load_insert_usb_demo(
                path,
                action_scale=action_scale,
                sim_dt=args.sim_dt,
            )
        )

    curriculum = ReverseCurriculum(
        demos,
        reverse_step_size=args.reverse_step_size,
        geometric_p=args.geometric_p,
        seed=0,
    )
    curriculum_before = curriculum.state()
    # This is a deterministic state-machine check, not an invented success:
    # official RFCL advances after three successful frontier rollouts.
    for demo_index, demo in enumerate(demos):
        for _ in range(curriculum.per_demo_buffer_size):
            curriculum.record_result(
                demo_index,
                demo.states.shape[0] - 1,
                success=True,
            )
    curriculum_after_success_window = curriculum.state()
    report = {
        "schema": "rfcl_static_probe_v1",
        "action_scale": action_scale.tolist(),
        "state_layout": {
            "joint": [PRIVILEGED_STATE_LAYOUT.joint.start, PRIVILEGED_STATE_LAYOUT.joint.stop],
            "joint_delta": [PRIVILEGED_STATE_LAYOUT.joint_delta.start, PRIVILEGED_STATE_LAYOUT.joint_delta.stop],
            "ee_pose": [PRIVILEGED_STATE_LAYOUT.ee_pose.start, PRIVILEGED_STATE_LAYOUT.ee_pose.stop],
            "usb_pose": [PRIVILEGED_STATE_LAYOUT.usb_pose.start, PRIVILEGED_STATE_LAYOUT.usb_pose.stop],
            "slot_pose": [PRIVILEGED_STATE_LAYOUT.slot_pose.start, PRIVILEGED_STATE_LAYOUT.slot_pose.stop],
            "usb_in_slot": [PRIVILEGED_STATE_LAYOUT.usb_in_slot.start, PRIVILEGED_STATE_LAYOUT.usb_in_slot.stop],
            "dim": PRIVILEGED_STATE_LAYOUT.dim,
        },
        "demos": [_probe_demo(demo, action_scale) for demo in demos],
        "curriculum": {
            "initial_frontiers": curriculum_before["frontiers"].tolist(),
            "frontiers_after_terminal_success_window": curriculum_after_success_window["frontiers"].tolist(),
            "success_counts": curriculum_after_success_window["success_counts"].tolist(),
            "reverse_step_size": args.reverse_step_size,
            "geometric_p": args.geometric_p,
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    if not args.quiet:
        print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
