"""Record full Motion Plan demonstrations used as RFCL trajectory prefixes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import h5py

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


REQUIRED_DATASETS = (
    "observation/head/rgb",
    "observation/wrist/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
    "embodiment/joint",
    "actor/prism",
    "actor/slot",
    "step",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default="insert_USB")
    parser.add_argument("--task-config", default="gelsight")
    parser.add_argument("--step-limit", type=int, default=600)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def validate(path: Path) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        lengths = {name: len(handle[name]) for name in REQUIRED_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Dataset lengths disagree in {path}: {lengths}")
        return {"frames": next(iter(lengths.values()))}


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.snapshot_root / "rfcl_manifest.json").read_text(encoding="utf-8")
    )
    seeds = [int(demo["seed"]) for demo in manifest["demos"]]
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "recording_status.jsonl"
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task = None
    completed = 0
    failed = 0
    try:
        from policy.RL.task_factory import create_task

        task = create_task(
            args.task_name,
            args.task_config,
            save_dir=args.output,
            video_frequency=0,
            step_limit=args.step_limit,
            mode="collect",
            save_pre_move=False,
            device=args.device,
        )
        for seed in seeds:
            destination = args.output / "hdf5" / f"{seed}.hdf5"
            if args.resume and destination.is_file():
                try:
                    details = validate(destination)
                except Exception as exc:
                    print(f"[prefix-record] invalid existing seed={seed}: {exc}", flush=True)
                else:
                    completed += 1
                    print(
                        f"[prefix-record] skip seed={seed} frames={details['frames']}",
                        flush=True,
                    )
                    continue
            row = {
                "seed": seed,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
            try:
                cache = args.output / ".cache" / str(seed)
                if cache.exists():
                    shutil.rmtree(cache)
                destination.unlink(missing_ok=True)
                task.reset(seed=seed)
                task._update_render()
                task.save_observations(task._get_observations())
                task.play_once()
                success = bool(
                    task.plan_success
                    and task.check_success()
                    and not task.check_early_stop()
                )
                if not success:
                    raise RuntimeError(
                        f"Motion Plan failed: plan={task.plan_success}, "
                        f"success={task.check_success()}"
                    )
                task.save_to_hdf5()
                details = validate(destination)
                task.clean_cache(result="success")
                row.update({"success": True, **details, "output": str(destination)})
                completed += 1
            except BaseException as exc:
                failed += 1
                row.update(
                    {
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                task.clean_cache(result="error")
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(
                f"[prefix-record] seed={seed} success={row['success']} "
                f"frames={row.get('frames')}",
                flush=True,
            )
        summary = {
            "schema": "rfcl_motion_plan_prefixes_v1",
            "snapshot_root": str(args.snapshot_root.resolve()),
            "seeds": seeds,
            "completed": completed,
            "failed": failed,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    finally:
        if task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
