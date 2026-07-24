from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetLayout:
    camera_paths: dict[str, str]
    tactile_paths: dict[str, str]


def _find_layout(path: Path, camera_keys, tactile_keys):
    with h5py.File(path, "r") as hdf5_file:
        camera_paths = {}
        camera_mapping = {"cam_high": "head", "cam_wrist": "wrist"}
        for key in camera_keys:
            source = camera_mapping[key]
            hdf5_path = f"observation/{source}/rgb"
            if hdf5_path not in hdf5_file:
                raise KeyError(f"Missing {hdf5_path} in {path}")
            camera_paths[key] = hdf5_path
        tactile_paths = {}
        side_mapping = {"tac_left": "left", "tac_right": "right"}
        for key in tactile_keys:
            side = side_mapping[key]
            candidates = (
                f"tactile/{side}_tactile/rgb_marker",
                f"tactile/{side}_gsmini/rgb_marker",
            )
            tactile_paths[key] = next(
                (candidate for candidate in candidates if candidate in hdf5_file),
                None,
            )
            if tactile_paths[key] is None:
                raise KeyError(f"Missing rgb_marker for {side} tactile in {path}")
    return DatasetLayout(camera_paths, tactile_paths)


def _decode_image(encoded, image_size):
    encoded_array = np.frombuffer(bytes(encoded), dtype=np.uint8)
    image = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode JPEG observation")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class ActionPhaseDataset(Dataset):
    def __init__(
        self,
        hdf5_paths,
        image_size=224,
        camera_keys=("cam_high", "cam_wrist"),
        tactile_keys=("tac_left", "tac_right"),
        require_phase=True,
    ):
        self.paths = [Path(path) for path in hdf5_paths]
        if not self.paths:
            raise ValueError("No HDF5 episodes were provided")
        self.image_size = int(image_size)
        self.camera_keys = list(camera_keys)
        self.tactile_keys = list(tactile_keys)
        self.layout = _find_layout(self.paths[0], self.camera_keys, self.tactile_keys)
        self.records = []
        for episode_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as hdf5_file:
                frame_count = len(hdf5_file["embodiment/joint"])
                pair_indices = np.arange(frame_count - 1)
                if "phase/id" in hdf5_file:
                    phase_ids = hdf5_file["phase/id"][()]
                    pair_indices = pair_indices[
                        (phase_ids[:-1] == 1) & (phase_ids[1:] == 1)
                    ]
                elif require_phase:
                    raise KeyError(f"Missing phase/id in {path}")
                self.records.extend(
                    (episode_index, int(frame_index))
                    for frame_index in pair_indices
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        episode_index, frame_index = self.records[index]
        path = self.paths[episode_index]
        with h5py.File(path, "r") as hdf5_file:
            observation = {
                "qpos": torch.from_numpy(
                    hdf5_file["embodiment/joint"][frame_index, :8].astype(np.float32)
                )
            }
            for key, hdf5_path in self.layout.camera_paths.items():
                observation[key] = _decode_image(
                    hdf5_file[hdf5_path][frame_index],
                    self.image_size,
                )
            for key, hdf5_path in self.layout.tactile_paths.items():
                observation[key] = _decode_image(
                    hdf5_file[hdf5_path][frame_index],
                    self.image_size,
                )
            action = torch.from_numpy(
                hdf5_file["embodiment/joint"][frame_index + 1, :8].astype(np.float32)
            )
        return observation, action


def split_episode_paths(dataset_root, validation_fraction=0.1, seed=0):
    paths = sorted(
        Path(dataset_root).glob("*.hdf5"),
        key=lambda path: int(path.stem),
    )
    if not paths:
        paths = sorted(
            Path(dataset_root).joinpath("hdf5").glob("*.hdf5"),
            key=lambda path: int(path.stem),
        )
    if not paths:
        raise FileNotFoundError(f"No HDF5 episodes found under {dataset_root}")
    generator = np.random.default_rng(seed)
    shuffled = list(paths)
    generator.shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    if len(shuffled) == 1:
        return shuffled, shuffled
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def compute_joint_statistics(paths):
    qpos_values = []
    action_values = []
    for path in paths:
        with h5py.File(path, "r") as hdf5_file:
            joints = hdf5_file["embodiment/joint"][:, :8].astype(np.float32)
            pair_indices = np.arange(len(joints) - 1)
            if "phase/id" in hdf5_file:
                phases = hdf5_file["phase/id"][()]
                pair_indices = pair_indices[
                    (phases[:-1] == 1) & (phases[1:] == 1)
                ]
            qpos_values.append(joints[pair_indices])
            action_values.append(joints[pair_indices + 1])
    qpos = np.concatenate(qpos_values)
    action = np.concatenate(action_values)
    delta = action - qpos
    return {
        "qpos_mean": qpos.mean(axis=0),
        "qpos_std": np.maximum(qpos.std(axis=0), 1e-4),
        "delta_mean": delta.mean(axis=0),
        "delta_std": np.maximum(delta.std(axis=0), 1e-5),
        "joint_min": action.min(axis=0),
        "joint_max": action.max(axis=0),
    }
