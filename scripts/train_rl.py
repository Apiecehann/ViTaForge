import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train residual SAC or PPO in UniVTAC.")
parser.add_argument("task_name")
parser.add_argument("task_config")
parser.add_argument("bc_checkpoint")
parser.add_argument("output_dir")
parser.add_argument("--algorithm", choices=("sac", "ppo"), required=True)
parser.add_argument("--total-timesteps", type=int, default=10000)
parser.add_argument("--image-size", type=int, default=128)
parser.add_argument("--residual-scale", type=float, default=0.5)
parser.add_argument("--action-repeat", type=int, default=2)
parser.add_argument("--control-gripper", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--step-limit", type=int, default=100)
parser.add_argument("--seed", type=int, default=10000)
parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from policy.RL.gym_env import ResidualTactileEnv
from policy.RL.sb3_features import BCFeatureExtractor
from policy.RL.task_factory import create_task


def main():
    output_dir = Path(args.output_dir) / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    task = create_task(
        args.task_name,
        args.task_config,
        save_dir=output_dir / "environment",
        video_frequency=0,
        step_limit=args.step_limit,
    )
    environment = Monitor(
        ResidualTactileEnv(
            task,
            args.bc_checkpoint,
            image_size=args.image_size,
            residual_scale=args.residual_scale,
            action_repeat=args.action_repeat,
            control_gripper=args.control_gripper,
            seed=args.seed,
            device="cuda:0",
        ),
        filename=str(output_dir / "monitor.csv"),
        info_keywords=("success",),
    )
    extractor_kwargs = {
        "bc_checkpoint": args.bc_checkpoint,
        "freeze": args.freeze_encoder,
    }
    policy_kwargs = {
        "features_extractor_class": BCFeatureExtractor,
        "features_extractor_kwargs": extractor_kwargs,
        "normalize_images": False,
        "net_arch": [256, 256],
        "share_features_extractor": True,
    }
    if args.algorithm == "sac":
        model = SAC(
            "MultiInputPolicy",
            environment,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=10000,
            learning_starts=500,
            batch_size=128,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            seed=args.seed,
            device="cuda:0",
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            environment,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            seed=args.seed,
            device="cuda:0",
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
        )
    callback = CheckpointCallback(
        save_freq=2000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=args.algorithm,
        save_replay_buffer=args.algorithm == "sac",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(output_dir / "final_model")
    summary = {
        "algorithm": args.algorithm,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "bc_checkpoint": str(Path(args.bc_checkpoint).resolve()),
        "total_timesteps": args.total_timesteps,
        "residual_scale": args.residual_scale,
        "action_repeat": args.action_repeat,
        "control_gripper": args.control_gripper,
        "seed": args.seed,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    environment.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
