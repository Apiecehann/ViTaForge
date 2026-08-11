from shutil import ExecError
import sys

sys.path.append(".")
sys.path.append(f"./policy")

import os
import time
import json
import yaml
import torch
import argparse
import traceback
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Eval Policy"
)
parser.add_argument(
    "task_name",
    type=str,
    help="Task name",
)
parser.add_argument(
    "task_config",
    type=str,
    help="Task config",
)
parser.add_argument(
    "deploy_config",
    type=str,
    help="Deploy file name",
)
parser.add_argument(
    "--expert_check",
    action='store_true',
    help="Whether to do expert check before eval"
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1
)
parser.add_argument(
    "--total_num",
    type=int,
    default=100
)
parser.add_argument(
    "--eval_step_timeout_seconds",
    type=float,
    default=None,
    help="Fail an eval episode if one observation->policy->action iteration exceeds this many seconds.",
)
parser.add_argument(
    "--openpi_host",
    type=str,
    default=None,
    help="Override openpi.host in the deploy config.",
)
parser.add_argument(
    "--openpi_port",
    type=int,
    default=None,
    help="Override openpi.port in the deploy config.",
)
parser.add_argument(
    "--print_only",
    action='store_true',
)
parser.add_argument(
    "--tactile_sensor",
    type=str,
    default=None,
    choices=("gelsight", "xense", "neote", "neote_force_field", "gsmini", "xensews"),
    help=(
        "Override tactile sensor for eval. User-facing names gelsight/xense/neote "
        "are mapped to task cfg sensor_type gsmini/xensews/neote."
    ),
)
parser.add_argument(
    "--target_block",
    type=str,
    default=None,
    help="Override env_cfg.target_block when the selected task supports it.",
)
parser.add_argument(
    "--block_base_pose_indices",
    type=str,
    default=None,
    help=(
        "Override env_cfg.block_base_pose_indices when supported. "
        "Accepts comma-separated values such as 0,1,4 or a YAML/JSON list in the config."
    ),
)
parser.add_argument(
    "--target_cup",
    type=str,
    default=None,
    help="Override env_cfg.target_cup when the selected task supports it.",
)
parser.add_argument(
    "--reference_cup",
    type=str,
    default=None,
    help="Override env_cfg.reference_cup when the selected task supports it.",
)
parser.add_argument(
    "--placement_side",
    type=str,
    default=None,
    help="Override env_cfg.placement_side when the selected task supports it.",
)
parser.add_argument(
    "--cup_base_pose_indices",
    type=str,
    default=None,
    help=(
        "Override env_cfg.cup_base_pose_indices when supported. "
        "Accepts comma-separated values such as 0,1,2 or a YAML/JSON list in the config."
    ),
)
parser.add_argument(
    "--target_area",
    type=str,
    default=None,
    help="Override env_cfg.target_area when the selected task supports it.",
)
parser.add_argument(
    "--frame_order",
    type=str,
    default=None,
    help="Override env_cfg.frame_order when the selected task supports it.",
)
parser.add_argument(
    "--rough_block_side",
    type=str,
    default=None,
    choices=("random", "left", "right"),
    help="Override env_cfg.rough_block_side when the selected task supports it.",
)
parser.add_argument(
    "--initial_grasp_side",
    type=str,
    default=None,
    choices=("random", "left", "right"),
    help="Override env_cfg.initial_grasp_side when the selected task supports it.",
)
parser.add_argument(
    "--weight_label",
    type=str,
    default=None,
    choices=("random", "light", "heavy"),
    help="Override env_cfg.weight_label when the selected task supports it.",
)
parser.add_argument(
    "--roughness_label",
    type=str,
    default=None,
    choices=("random", "smooth", "rough"),
    help="Override env_cfg.roughness_label when the selected task supports it.",
)
parser.add_argument(
    "--hardness_label",
    type=str,
    default=None,
    choices=("random", "soft", "hard"),
    help="Override env_cfg.hardness_label when the selected task supports it.",
)
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.livestream = 2
args_cli.num_envs = 1

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg
    from policy._base_policy import BasePolicy

log_path = Path('./log')
def log(msg):
    global log_path, args_cli
    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    if not args_cli.print_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    print(msg)


def save_eval_timeout_images(observation: dict, output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    def save_image(value, stem: str):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            image = value.detach().cpu().numpy()
        else:
            image = np.asarray(value)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[-1] != 3:
            return
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        path = output_dir / f"{prefix}_{stem}.png"
        Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(path)
        saved_paths.append(path)

    camera_obs = observation.get("observation", {})
    for camera_name in ("head", "wrist"):
        camera_data = camera_obs.get(camera_name, {})
        save_image(camera_data.get("rgb"), f"{camera_name}_image")

    tactile_obs = observation.get("tactile", {})
    for tactile_name in ("left_tactile", "right_tactile"):
        tactile_data = tactile_obs.get(tactile_name, {})
        for image_key in ("rgb_marker", "gel_particle", "force_field_img", "marker_force_img", "rgb"):
            if image_key in tactile_data:
                save_image(tactile_data[image_key], f"{tactile_name}_{image_key}")
                break

    return saved_paths


class StepTimeoutError(RuntimeError):
    pass


def eval_policy(
    task: 'BaseTask', policy: 'BasePolicy', expert_check,
    start_seed, max_seed, test_total_num, instructions, instruciton_type:Literal['seen', 'unseen']='seen'
):
    test_num, succ_num, seed = 0, 0, start_seed

    seed_path = task.save_root.parent / 'seeds.json'
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    if seed_path.exists():
        with open(seed_path, 'r') as f:
            seed_status = json.load(f)
    else:
        seed_status = {}
 
    while test_num < test_total_num and (max_seed == -1 or seed <= max_seed):
        if not seed_status.get(str(seed), True):
            seed += 1
            continue
        
        if expert_check and str(seed) not in seed_status:
            test_start = time.perf_counter()
            task.mode = 'eval_test'
            try:
                task.reset(seed=seed)
                task.play_once()
                if not task.check_success() or not task.plan_success:
                    raise ExecError(f'seed {seed} Expert check failed, check {task.check_success()}, plan {task.plan_success}.')
                else:
                    seed_status[seed] = True
                    with open(seed_path, 'w') as f:
                        json.dump(seed_status, f)
                test_cost = time.perf_counter() - test_start
                task.clean_cache(result='test_success')
                log(f'Expert check succ, seed {seed}, cost {test_cost:.2f}s')
            except Exception as e:
                test_cost = time.perf_counter() - test_start
                log(f'Expert check failed, seed {seed}, cost {test_cost:.2f}s, with exception {e}')
                task.clean_cache(result='test_fail')
                seed_status[seed] = False
                with open(seed_path, 'w') as f:
                    json.dump(seed_status, f)
                seed += 1
                continue
        test_num += 1

        succ = False
        eval_start = time.perf_counter()
        task.mode = 'eval'
        try:
            if instructions is None:
                task.reset(seed=seed)
            else:
                task.reset(seed=seed, instructions=instructions[instruciton_type])
            task.mean_steps = task.cfg.step_lim
            policy.reset()
            while task.take_action_cnt < task.cfg.step_lim:
                step_start = time.perf_counter()
                observation = task._get_observations()
                policy.eval(task, observation)
                step_cost = time.perf_counter() - step_start
                timeout = args_cli.eval_step_timeout_seconds
                if timeout is None:
                    timeout = float(getattr(task.cfg, "eval_step_timeout_seconds", 0.0) or 0.0)
                if timeout > 0.0 and step_cost > timeout:
                    timeout_dir = task.save_root / "step_timeout"
                    prefix = f"seed_{seed}_step_{task.step_count}_cost_{step_cost:.2f}s"
                    saved_paths = save_eval_timeout_images(observation, timeout_dir, prefix)
                    raise StepTimeoutError(
                        f"{step_cost:.2f}s > {timeout:.2f}s, saved {saved_paths}"
                    )
                if task.eval_success:
                    succ = True
                    break
                if task.check_early_stop():
                    break
        except Exception as e:
            if e.__class__.__name__ == "StepTimeoutError":
                log(f"[{test_num:<3d}] Seed {seed} step timeout, mark error: {e}")
                succ_status = 'error'
                task.clean_cache(result=succ_status)
                test_num -= 1
                continue
            log(f"[{test_num:<3d}] Seed {seed} occurred exception: {e}\n{traceback.format_exc()}")
            succ_status = 'error'
            task.clean_cache(result=succ_status)
            test_num -= 1
        else:
            eval_cost = time.perf_counter() - eval_start
            
            if succ:
                succ_num += 1
            succ_status = 'success' if succ else 'failed'
            task.clean_cache(result=succ_status)
            log(f"[{test_num:<3d}] Seed {seed} {succ_status} after {eval_cost:.2f} s.\n"
                f"steps: {task.step_count:<5d}, actions: {task.take_action_cnt:<5d}.\n"
                f"Instruction: {task.instruction}\n"
                f"Total {succ_num}/{test_num}({succ_num/test_num*100:.2f}%) success.")
        finally:
            seed += 1
    
    return {
        'test_num': test_num,
        'succ_num': succ_num
    }

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file


def parse_int_tuple(value, name: str):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            value = json.loads(text)
        else:
            value = text.replace(",", " ").split()
    try:
        parsed = tuple(int(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a list or comma-separated string, got {value!r}") from exc
    except ValueError as exc:
        raise ValueError(f"{name} must contain only integers, got {value!r}") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def tactile_sensor_type_from_override(value: str | None) -> str | None:
    if value is None:
        return None
    mapping = {
        "gelsight": "gsmini",
        "xense": "xensews",
        "neote": "neote",
        "neote_force_field": "neote",
        "gsmini": "gsmini",
        "xensews": "xensews",
    }
    return mapping[value]


task_module, policy_module = None, None
def main():
    global args_cli, task_module, policy_module, log_path

    task_file_name = args_cli.task_name
    task_config, task_config_file = get_config(
        args_cli.task_config, default_root=Path(__file__).parent.parent / 'task_config', type='yaml'
    )
    deploy_config, deploy_config_file = get_config(
        args_cli.deploy_config, default_root=Path(__file__).parent.parent / 'policy', type='yaml'
    )
    if args_cli.openpi_host is not None or args_cli.openpi_port is not None:
        openpi_config = deploy_config.setdefault("openpi", {})
        if args_cli.openpi_host is not None:
            openpi_config["host"] = args_cli.openpi_host
        if args_cli.openpi_port is not None:
            openpi_config["port"] = args_cli.openpi_port
    if "openpi_debug_dump_first_n_obs" in task_config:
        deploy_config.setdefault("openpi", {})["debug_dump_first_n_obs"] = int(
            task_config["openpi_debug_dump_first_n_obs"]
        )
    policy_name = deploy_config['policy_name']
    deploy_config['task_name'] = task_file_name
    deploy_config['task_config'] = task_config_file.stem
 
    deploy_config['instuction_file'] = deploy_config.get(
        'instruction_file',
        deploy_config.get('instuction_file', task_file_name),
    )
    if deploy_config['instuction_file'] is not None:
        instructions, _ = get_config(
            deploy_config['instuction_file'], default_root=Path(__file__).parent.parent / 'instructions', type='json'
        )
    else:
        instructions = None

    task_module = importlib.import_module(f"envs.{task_file_name}")
    policy_module = importlib.import_module(f"policy.{policy_name}")
    
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')

    env_cfg:BaseTaskCfg = task_module.TaskCfg()
    env_cfg.save_dir = Path('eval_result') / policy_name / task_file_name / deploy_config_file.stem / curr_time
    eval_case_name = task_config.get("eval_case_name", None)
    if eval_case_name:
        env_cfg.save_dir = env_cfg.save_dir / str(eval_case_name)

    if hasattr(env_cfg, "target_block"):
        target_block = args_cli.target_block
        if target_block is None:
            target_block = task_config.get("target_block", None)
        if target_block is not None:
            env_cfg.target_block = str(target_block)
            task_config["target_block"] = env_cfg.target_block
    if hasattr(env_cfg, "block_base_pose_indices"):
        pose_indices = args_cli.block_base_pose_indices
        if pose_indices is None:
            pose_indices = task_config.get("block_base_pose_indices", None)
        pose_indices = parse_int_tuple(pose_indices, "block_base_pose_indices")
        if pose_indices is not None:
            env_cfg.block_base_pose_indices = pose_indices
            task_config["block_base_pose_indices"] = list(pose_indices)
    for key in ("target_cup", "reference_cup", "placement_side", "target_area", "frame_order"):
        if hasattr(env_cfg, key):
            value = getattr(args_cli, key)
            if value is None:
                value = task_config.get(key, None)
            if value is not None:
                setattr(env_cfg, key, str(value))
                task_config[key] = str(value)
    if hasattr(env_cfg, "cup_base_pose_indices"):
        pose_indices = args_cli.cup_base_pose_indices
        if pose_indices is None:
            pose_indices = task_config.get("cup_base_pose_indices", None)
        pose_indices = parse_int_tuple(pose_indices, "cup_base_pose_indices")
        if pose_indices is not None:
            env_cfg.cup_base_pose_indices = pose_indices
            task_config["cup_base_pose_indices"] = list(pose_indices)

    tactile_sensor_type = tactile_sensor_type_from_override(args_cli.tactile_sensor)
    if tactile_sensor_type is None:
        tactile_sensor_type = task_config.get('sensor_type', 'gsmini')
    env_cfg.tactile_sensor_type = tactile_sensor_type
    task_config["sensor_type"] = tactile_sensor_type
    env_cfg.dense_gelpad = bool(task_config.get('dense_gelpad', getattr(env_cfg, 'dense_gelpad', False)))
    env_cfg.force_field_grid = tuple(task_config.get('force_field_grid', env_cfg.force_field_grid))
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.render_frequency = task_config.get("render_frequency", env_cfg.render_frequency)
    if "reset_time_limit" in task_config:
        env_cfg.reset_time_limit = float(task_config["reset_time_limit"])
    if args_cli.eval_step_timeout_seconds is not None:
        env_cfg.eval_step_timeout_seconds = float(args_cli.eval_step_timeout_seconds)
    elif "eval_step_timeout_seconds" in task_config:
        env_cfg.eval_step_timeout_seconds = float(task_config["eval_step_timeout_seconds"])
    elif "eval_step_timeout_seconds" in deploy_config:
        env_cfg.eval_step_timeout_seconds = float(deploy_config["eval_step_timeout_seconds"])
    if "video_size" in task_config:
        env_cfg.video_size = tuple(task_config["video_size"])
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.save_pre_move = task_config.get("save_pre_move", getattr(env_cfg, "save_pre_move", False))
    default_skip_pre_move = policy_name == "openpi"
    env_cfg.skip_pre_move = bool(task_config.get("skip_pre_move", default_skip_pre_move))
    default_eval_start_delay_steps = 0 if env_cfg.skip_pre_move else getattr(env_cfg, "eval_start_delay_steps", 20)
    env_cfg.eval_start_delay_steps = int(
        task_config.get("eval_start_delay_steps", default_eval_start_delay_steps)
    )
    env_cfg.tactile_video_key = task_config.get("tactile_video_key", env_cfg.tactile_video_key)
    if "use_adaptive_grasp" in task_config:
        env_cfg.use_adaptive_grasp = bool(task_config["use_adaptive_grasp"])
    if "adaptive_grasp_depth_threshold" in task_config:
        env_cfg.adaptive_grasp_depth_threshold = float(task_config["adaptive_grasp_depth_threshold"])
    for key in (
        "rough_block_side",
        "initial_grasp_side",
        "weight_label",
        "roughness_label",
        "hardness_label",
    ):
        if hasattr(env_cfg, key):
            value = getattr(args_cli, key, None)
            if value is None:
                value = task_config.get(key, None)
            if value is not None:
                setattr(env_cfg, key, str(value))
    xense_tuning_types = {
        "xense_usb_close_percent": float,
        "xense_half_cylinder_close_percent": float,
        "xense_insert_half_cylinder_close_percent": float,
        "xense_cube_close_percent": float,
        "xense_cup_close_percent": float,
        "xense_cup_min_principal_ratio": float,
        "xense_cup_max_nonrigid_error": float,
        "xense_pour_cup_close_percent": float,
        "xense_pour_ball_friction_ratio": float,
        "xense_pour_grip_friction_ratio": float,
        "xense_pour_wrist_angle_deg": float,
        "xense_pour_wrist_steps": int,
        "xense_pour_wrist_translation_x": float,
        "xense_pour_wrist_translation_y": float,
        "xense_pour_wrist_translation_z": float,
        "xense_pour_actor_tilt_deg": float,
        "xense_pour_actor_tilt_axis_x": float,
        "xense_pour_actor_tilt_axis_y": float,
        "xense_pour_actor_tilt_axis_z": float,
        "xense_pour_carry_segments": int,
        "xense_pour_carry_settle_steps": int,
        "xense_pour_hold_actor_during_carry": bool,
        "xense_pour_target_y_offset": float,
        "xense_pour_target_z_offset": float,
        "xense_pour_release_lift": float,
        "xense_pour_release_snap_angle_deg": float,
        "xense_pour_release_snap_steps": int,
        "xense_pour_release_snap_cycles": int,
        "xense_pour_fix_cup_during_release": bool,
        "xense_pour_release_retract_x": float,
        "xense_pour_release_carry_y": float,
        "xense_drawer_close_percent": float,
        "xense_gear_close_percent": float,
        "xense_half_cylinder_grasp_height_bias": float,
        "xense_insert_half_cylinder_grasp_height_bias": float,
        "xense_cube_grasp_height_bias": float,
        "xense_cup_grasp_height_bias": float,
        "xense_pour_cup_grasp_height_bias": float,
        "xense_pour_cup_grasp_world_x_bias": float,
        "xense_drawer_grasp_z_bias": float,
        "xense_half_cylinder_grasp_world_y_bias": float,
        "xense_insert_half_cylinder_grasp_world_y_bias": float,
        "xense_cube_grasp_world_y_bias": float,
        "xense_initial_settle_steps": int,
        "xense_half_cylinder_initial_settle_steps": int,
        "xense_cup_initial_settle_steps": int,
        "xense_pour_initial_settle_steps": int,
        "xense_drawer_initial_settle_steps": int,
        "xense_insert_half_cylinder_initial_settle_steps": int,
        "xense_cube_initial_settle_steps": int,
        "xense_gear_grasp_world_y_bias": float,
        "xense_carry_time_dilation": float,
        "xense_carry_segments": int,
        "xense_carry_max_step": float,
        "xense_post_close_settle_steps": int,
        "xense_adaptive_grasp_max_steps": int,
        "xense_adaptive_grasp_tail_steps": int,
        "xense_adaptive_grasp_check_interval": int,
        "xense_adaptive_grasp_target_tolerance": float,
        "xense_adaptive_grasp_hold_margin": float,
        "xense_adaptive_grasp_hold_velocity": float,
        "xense_usb_post_close_settle_steps": int,
        "xense_adaptive_grasp_min_steps_before_contact": int,
        "xense_adaptive_grasp_min_travel": float,
        "xense_adaptive_grasp_require_both_contacts": bool,
        "xense_usb_adaptive_grasp_require_both_contacts": bool,
        "xense_half_cylinder_adaptive_grasp_require_both_contacts": bool,
        "xense_insert_half_cylinder_adaptive_grasp_require_both_contacts": bool,
        "xense_cube_adaptive_grasp_require_both_contacts": bool,
        "xense_cup_adaptive_grasp_require_both_contacts": bool,
        "xense_pour_cup_adaptive_grasp_require_both_contacts": bool,
        "xense_drawer_adaptive_grasp_require_both_contacts": bool,
        "xense_gear_adaptive_grasp_require_both_contacts": bool,
        "xense_usb_adaptive_grasp_depth_threshold": float,
        "xense_half_cylinder_adaptive_grasp_depth_threshold": float,
        "xense_insert_half_cylinder_adaptive_grasp_depth_threshold": float,
        "xense_cube_adaptive_grasp_depth_threshold": float,
        "xense_cup_adaptive_grasp_depth_threshold": float,
        "xense_pour_cup_adaptive_grasp_depth_threshold": float,
        "xense_drawer_adaptive_grasp_depth_threshold": float,
        "xense_gear_adaptive_grasp_depth_threshold": float,
    }
    for key, value_type in xense_tuning_types.items():
        if key in task_config:
            setattr(env_cfg, key, value_type(task_config[key]))

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None \
        else env_cfg.sim.device
    seed = deploy_config.get("seed", 0)

    init_start = time.perf_counter()
    policy:BasePolicy = policy_module.Policy(deploy_config)
    policy_init_cost = time.perf_counter() - init_start

    init_start = time.perf_counter()
    task:BaseTask = task_module.Task(env_cfg, mode='eval')
    task_init_cost = time.perf_counter() - init_start
    
    if os.environ.get('TRAIN_CONFIG'):
        deploy_config['train_config'] = os.environ['TRAIN_CONFIG']
    
    log_path = task.save_root / f"log.log"
    log(f"Task Name: {task_file_name}")
    log(f"Task Config: {task_config_file.absolute()}") 
    log(f"Eval Config: {json.dumps(deploy_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n") 
    log(f"Task init finish in {task_init_cost:.2f} seconds.")
    log(f"Policy init finish in {policy_init_cost:.2f} seconds.")

    results = eval_policy(
        task=task, policy=policy,
        expert_check=args_cli.expert_check,
        start_seed=1000000 * (1 + seed) if args_cli.start_seed == -1 else args_cli.start_seed,
        max_seed=args_cli.max_seed,
        test_total_num=args_cli.total_num,
        instructions=instructions,
        instruciton_type=deploy_config.get("instruction_type", "seen")
    )
    log(f"Final Result: {results['succ_num']}/{results['test_num']}({results['succ_num']/results['test_num']*100:.2f}%) success.")
    
    task.close()
    policy.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
