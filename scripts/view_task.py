"""Bring up a task's scene in the Isaac Sim GUI for visual inspection only.

Builds the full task scene (robot + tactile gelpads + table + slot + peg) but
skips the robot script / motion planner, so you just see the initial scene and
can orbit around it. Close the window to exit.

Usage:
    python scripts/view_task.py insert_USB
"""

import sys
import argparse
from pathlib import Path

sys.path.append('.')

parser = argparse.ArgumentParser(description="View a UniVTAC task scene in Isaac Sim.")
parser.add_argument("task", type=str, help="Task file name under envs/ (e.g. insert_USB)")
parser.add_argument("--seed", type=int, default=0)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1
args_cli.headless = False  # GUI

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib

task_module = importlib.import_module(f"envs.{args_cli.task}")
env_cfg = task_module.TaskCfg()
env_cfg.tactile_sensor_type = getattr(env_cfg, "tactile_sensor_type", "gsmini") or "gsmini"
env_cfg.save_dir = str(Path("/tmp/view_task") / args_cli.task)
env_cfg.scene.num_envs = 1
env_cfg.video_frequency = 0
env_cfg.render_frequency = 1

task = task_module.Task(env_cfg, mode="collect")

# Static view: replace the robot script with a no-op so reset() does not invoke
# the motion planner / grasp sequence.
task.pre_move = lambda: task.delay(5)

print("[view_task] building scene ...", flush=True)
task.reset(seed=args_cli.seed)
print("[view_task] scene ready — orbit with the mouse, close the window to exit.", flush=True)

while simulation_app.is_running():
    task._step(is_save=False)
    task._update_render()

simulation_app.close()
