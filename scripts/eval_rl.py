import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate SFT, SAC, PPO, or legacy BC.")
parser.add_argument("task_name")
parser.add_argument("task_config")
parser.add_argument("bc_checkpoint")
parser.add_argument("output_dir")
parser.add_argument("--algorithm", choices=("sft", "bc", "sac", "ppo"), required=True)
parser.add_argument("--model-path", default=None)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--start-seed", type=int, default=20000)
parser.add_argument("--image-size", type=int, default=128)
parser.add_argument("--residual-scale", type=float, default=0.5)
parser.add_argument(
    "--control-mode",
    choices=("direct", "residual"),
    default="direct",
)
parser.add_argument("--action-repeat", type=int, default=2)
parser.add_argument("--control-gripper", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--force-control", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--step-limit", type=int, default=100)
parser.add_argument("--save-traces", action="store_true")
help_requested = any(argument in ("-h", "--help") for argument in sys.argv[1:])
if help_requested:
    parser.print_help()
    raise SystemExit(0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from stable_baselines3 import PPO, SAC

from policy.RL.gym_env import TactileControlEnv
from policy.RL.task_factory import create_task


def main():
    output_dir = Path(args.output_dir) / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    task = create_task(
        args.task_name,
        args.task_config,
        save_dir=output_dir,
        video_frequency=2,
        step_limit=args.step_limit,
    )
    environment = TactileControlEnv(
        task,
        args.bc_checkpoint,
        image_size=args.image_size,
        residual_scale=args.residual_scale,
        action_repeat=args.action_repeat,
        control_gripper=args.control_gripper,
        force_control=args.force_control,
        control_mode=args.control_mode,
        seed=args.start_seed,
        device="cuda:0",
    )
    model = None
    if args.algorithm not in ("bc", "sft"):
        if not args.model_path:
            raise ValueError("--model-path is required for SAC and PPO evaluation")
        model_class = SAC if args.algorithm == "sac" else PPO
        model = model_class.load(args.model_path, env=environment, device="cuda:0")
    results = []
    for episode_index in range(args.episodes):
        seed = args.start_seed + episode_index
        observation, reset_info = environment.reset(seed=seed)
        terminated = truncated = False
        episode_reward = 0.0
        last_info = {}
        trace = []
        while not terminated and not truncated:
            if model is None:
                if args.control_mode == "direct":
                    policy_action = environment.sft_action(observation)
                else:
                    policy_action = np.zeros(
                        environment.action_space.shape,
                        dtype=np.float32,
                    )
            else:
                policy_action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, last_info = environment.step(
                policy_action
            )
            episode_reward += reward
            if args.save_traces:
                trace.append(
                    {
                        "action_index": len(trace),
                        "qpos": observation["qpos"].tolist(),
                        "policy_step": observation["policy_step"].tolist(),
                        "bc_delta": (
                            np.asarray(last_info["bc_delta"]).tolist()
                            if last_info["bc_delta"] is not None
                            else None
                        ),
                        "policy_action": np.asarray(
                            last_info["policy_action"]
                        ).tolist(),
                        "final_action": np.asarray(last_info["final_action"]).tolist(),
                        "reward": float(reward),
                        "metrics": last_info.get("metrics", {}),
                    }
                )
        success = bool(last_info.get("success", False))
        task.clean_cache(result="success" if success else "failed")
        result = {
            "episode": episode_index,
            "seed": seed,
            "success": success,
            "reward": float(episode_reward),
            "actions": int(task.take_action_cnt),
            "initial_metrics": reset_info.get("metrics", {}),
            "metrics": last_info.get("metrics", {}),
        }
        if args.save_traces:
            result["trace"] = trace
        print(json.dumps(result))
        results.append(result)
    summary = {
        "algorithm": args.algorithm,
        "episodes": len(results),
        "successes": sum(result["success"] for result in results),
        "success_rate": sum(result["success"] for result in results) / len(results),
        "mean_reward": float(np.mean([result["reward"] for result in results])),
        "results": results,
    }
    with open(output_dir / "evaluation.json", "w", encoding="utf-8") as result_file:
        json.dump(summary, result_file, indent=2)
    environment.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
