import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SUPPORTED_SCHEMA_VERSIONS = (1, 2)


REQUIRED_DATASETS = (
    "step",
    "embodiment/joint",
    "observation/head/rgb",
    "observation/wrist/rgb",
    "phase/id",
    "phase/name",
    "phase/policy_step",
    "phase/is_boundary",
)


def validate_episode(path):
    with h5py.File(path, "r") as hdf5_file:
        missing = [name for name in REQUIRED_DATASETS if name not in hdf5_file]
        if missing:
            raise KeyError(f"{path}: missing {missing}")
        tactile_pairs = (
            (
                "tactile/left_tactile/rgb_marker",
                "tactile/right_tactile/rgb_marker",
            ),
            (
                "tactile/left_gsmini/rgb_marker",
                "tactile/right_gsmini/rgb_marker",
            ),
        )
        tactile_paths = next(
            (pair for pair in tactile_pairs if all(name in hdf5_file for name in pair)),
            None,
        )
        if tactile_paths is None:
            raise KeyError(f"{path}: missing left/right rgb_marker")
        frame_count = len(hdf5_file["step"])
        for name in REQUIRED_DATASETS[1:] + tactile_paths:
            if len(hdf5_file[name]) != frame_count:
                raise ValueError(
                    f"{path}: {name} has {len(hdf5_file[name])} frames, "
                    f"expected {frame_count}"
                )
        phase_ids = hdf5_file["phase/id"][()]
        boundaries = np.flatnonzero(hdf5_file["phase/is_boundary"][()])
        if set(np.unique(phase_ids).tolist()) != {0, 1}:
            raise ValueError(f"{path}: invalid phase ids {np.unique(phase_ids)}")
        if boundaries.tolist() != [0, int(np.flatnonzero(phase_ids == 1)[0])]:
            raise ValueError(f"{path}: invalid boundaries {boundaries.tolist()}")
        attrs = hdf5_file["phase"].attrs
        schema_version = int(attrs.get("schema_version", -1))
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"{path}: unsupported phase schema {schema_version}; "
                f"expected one of {SUPPORTED_SCHEMA_VERSIONS}"
            )
        if int(attrs["policy_start_saved_index"]) != int(boundaries[1]):
            raise ValueError(f"{path}: policy boundary attribute mismatch")
        action_pairs = int(np.sum((phase_ids[:-1] == 1) & (phase_ids[1:] == 1)))
        return {
            "frames": frame_count,
            "pre_move_frames": int(np.sum(phase_ids == 0)),
            "action_frames": int(np.sum(phase_ids == 1)),
            "action_pairs": action_pairs,
            "schema_version": schema_version,
            "bytes": path.stat().st_size,
        }


def main():
    parser = argparse.ArgumentParser(description="Validate phased GelSight episodes.")
    parser.add_argument("dataset_root")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    hdf5_root = root / "hdf5" if (root / "hdf5").exists() else root
    paths = sorted(hdf5_root.glob("*.hdf5"), key=lambda path: int(path.stem))
    if args.expected_episodes is not None and len(paths) != args.expected_episodes:
        raise ValueError(
            f"Found {len(paths)} episodes, expected {args.expected_episodes}"
        )
    episodes = {path.stem: validate_episode(path) for path in paths}
    videos = list(root.joinpath("video").glob("*_success.mp4"))
    summary = {
        "episode_count": len(episodes),
        "success_video_count": len(videos),
        "total_frames": sum(item["frames"] for item in episodes.values()),
        "total_action_pairs": sum(item["action_pairs"] for item in episodes.values()),
        "total_bytes": sum(item["bytes"] for item in episodes.values()),
        "episodes": episodes,
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)


if __name__ == "__main__":
    main()
