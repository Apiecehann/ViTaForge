from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


def _task_config_path(task_config: str | Path) -> Path:
    path = Path(task_config)
    if path.suffix in (".yml", ".yaml"):
        return path
    return Path("task_config") / f"{task_config}.yml"


def _set_if_present(target: object, name: str, value: Any) -> None:
    if hasattr(target, name):
        setattr(target, name, value)


def create_task(
    task_name: str,
    task_config: str | Path,
    save_dir: str | Path,
    *,
    video_frequency: int = 0,
    step_limit: int = 120,
    mode: str = "eval",
    save_pre_move: bool | None = None,
    insert_usb_fixed_target_slot: bool = False,
    device: str | None = None,
):
    config_path = _task_config_path(task_config)
    with config_path.open("r", encoding="utf-8") as config_file:
        source = yaml.load(config_file, Loader=yaml.FullLoader) or {}

    task_module = importlib.import_module(f"envs.{task_name}")
    env_config = task_module.TaskCfg()

    env_config.tactile_sensor_type = source.get(
        "sensor_type",
        env_config.tactile_sensor_type,
    )
    env_config.save_dir = Path(save_dir)
    env_config.decimation = int(source.get("decimation", env_config.decimation))
    env_config.save_frequency = int(
        source.get("save_frequency", env_config.save_frequency)
    )
    env_config.video_frequency = int(video_frequency)
    env_config.render_frequency = 0
    env_config.obs_data_type = source.get("observations", {})
    env_config.random_texture = bool(source.get("random_texture", False))
    env_config.step_lim = int(step_limit)
    env_config.scene.num_envs = 1
    env_config.eval_start_delay_steps = 0
    _set_if_present(
        env_config,
        "fixed_target_slot",
        bool(insert_usb_fixed_target_slot),
    )

    if device is not None:
        _set_if_present(env_config.sim, "device", str(device))

    if save_pre_move is None:
        save_pre_move = source.get("save_pre_move", False)
    env_config.save_pre_move = bool(save_pre_move)
    env_config.skip_pre_move = bool(
        source.get("skip_pre_move", getattr(env_config, "skip_pre_move", False))
    )
    env_config.eval_start_delay_steps = int(source.get("eval_start_delay_steps", 0))

    for name, caster in (
        ("dense_gelpad", bool),
        ("use_adaptive_grasp", bool),
        ("adaptive_grasp_depth_threshold", float),
        ("tactile_video_key", str),
        ("reset_after_actor_steps", int),
        ("reset_time_limit", float),
    ):
        if name in source and hasattr(env_config, name):
            setattr(env_config, name, caster(source[name]))

    for name in ("block_base_pose_indices", "cup_base_pose_indices"):
        if name in source and hasattr(env_config, name):
            setattr(env_config, name, tuple(int(index) for index in source[name]))

    for name in ("rough_block_side", "initial_grasp_side"):
        if name in source and hasattr(env_config, name):
            setattr(env_config, name, str(source[name]))

    return task_module.Task(env_config, mode=mode)
