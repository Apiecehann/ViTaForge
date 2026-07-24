from __future__ import annotations

import importlib
from pathlib import Path

import yaml


def create_task(task_name, task_config, save_dir, video_frequency=0, step_limit=300):
    config_path = Path(task_config)
    if config_path.suffix not in (".yml", ".yaml"):
        config_path = Path("task_config") / f"{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        source = yaml.load(config_file, Loader=yaml.FullLoader)
    task_module = importlib.import_module(f"envs.{task_name}")
    env_config = task_module.TaskCfg()
    env_config.tactile_sensor_type = source.get("sensor_type", "gsmini")
    env_config.dense_gelpad = bool(source.get("dense_gelpad", env_config.dense_gelpad))
    env_config.save_dir = Path(save_dir)
    env_config.decimation = int(source.get("decimation", env_config.decimation))
    env_config.save_frequency = int(source.get("save_frequency", env_config.save_frequency))
    env_config.video_frequency = int(video_frequency)
    env_config.render_frequency = 0
    env_config.obs_data_type = source.get("observations", {})
    env_config.random_texture = bool(source.get("random_texture", False))
    env_config.save_pre_move = False
    env_config.eval_start_delay_steps = 0
    env_config.step_lim = int(step_limit)
    env_config.tactile_video_key = source.get(
        "tactile_video_key",
        env_config.tactile_video_key,
    )
    if "use_adaptive_grasp" in source:
        env_config.use_adaptive_grasp = bool(source["use_adaptive_grasp"])
    if "adaptive_grasp_depth_threshold" in source:
        env_config.adaptive_grasp_depth_threshold = float(
            source["adaptive_grasp_depth_threshold"]
        )
    env_config.scene.num_envs = 1
    return task_module.Task(env_config, mode="eval")
