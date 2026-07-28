import argparse
import json
import sys
import time
from pathlib import Path

from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train SFT-initialized SAC or PPO in UniVTAC.")
parser.add_argument("task_name")
parser.add_argument("task_config")
parser.add_argument("bc_checkpoint")
parser.add_argument("output_dir")
parser.add_argument("--algorithm", choices=("sac", "ppo"), required=True)
parser.add_argument("--total-timesteps", type=int, default=10000)
parser.add_argument("--learning-starts", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--train-frequency", type=int, default=4)
parser.add_argument("--gradient-steps", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--ent-coef", type=float, default=1e-3)
parser.add_argument("--bc-dataset-root")
parser.add_argument("--online-bc-regularization", type=float, default=10.0)
parser.add_argument("--offline-bc-regularization", type=float, default=100.0)
parser.add_argument(
    "--share-features-extractor",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--save-replay-buffer",
    action=argparse.BooleanOptionalAction,
    default=False,
)
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
parser.add_argument("--seed", type=int, default=10000)
parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--initialize-actor",
    action=argparse.BooleanOptionalAction,
    default=True,
)
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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from policy.RL.gym_env import TactileControlEnv
from policy.RL.sac_bc import SFTRegularizedSAC
from policy.RL.sb3_features import BCFeatureExtractor, initialize_sac_actor_from_sft
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
        TactileControlEnv(
            task,
            args.bc_checkpoint,
            image_size=args.image_size,
            residual_scale=args.residual_scale,
            action_repeat=args.action_repeat,
            control_gripper=args.control_gripper,
            force_control=args.force_control,
            control_mode=args.control_mode,
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
        "activation_fn": nn.GELU,
        "log_std_init": -3.0,
        "use_expln": True,
        "clip_mean": 20.0,
        "share_features_extractor": args.share_features_extractor,
    }
    actor_initialization = None
    if args.algorithm == "sac":
        model_started_at = time.perf_counter()
        print("[RL init] constructing SAC model", flush=True)
        use_bc_regularization = (
            args.online_bc_regularization > 0
            or args.offline_bc_regularization > 0
        )
        sac_class = SFTRegularizedSAC if use_bc_regularization else SAC
        sac_kwargs = {}
        if sac_class is SFTRegularizedSAC:
            if not args.bc_dataset_root:
                raise ValueError("SFT-regularized SAC requires --bc-dataset-root")
            sac_kwargs.update(
                bc_checkpoint=args.bc_checkpoint,
                bc_dataset_root=args.bc_dataset_root,
                online_bc_regularization=args.online_bc_regularization,
                offline_bc_regularization=args.offline_bc_regularization,
                bc_image_size=args.image_size,
            )
        model = sac_class(
            "MultiInputPolicy",
            environment,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            buffer_size=10000,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            train_freq=args.train_frequency,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            use_sde=True,
            sde_sample_freq=4,
            use_sde_at_warmup=True,
            seed=args.seed,
            device="cuda:0",
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard"),
            **sac_kwargs,
        )
        print(
            f"[RL init] SAC model ready in "
            f"{time.perf_counter() - model_started_at:.2f}s",
            flush=True,
        )
        if args.initialize_actor:
            if args.control_mode != "direct":
                raise ValueError("SFT Actor initialization requires direct control")
            if not args.control_gripper:
                raise ValueError("The 8D SFT Actor requires --control-gripper")
            actor_started_at = time.perf_counter()
            actor_initialization = initialize_sac_actor_from_sft(
                model,
                args.bc_checkpoint,
            )
            print(
                f"[RL init] SFT actor copied in "
                f"{time.perf_counter() - actor_started_at:.2f}s",
                flush=True,
            )
    else:
        if args.initialize_actor:
            raise NotImplementedError(
                "Exact bounded SFT Actor initialization is currently implemented "
                "for SAC only"
            )
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
        save_replay_buffer=args.algorithm == "sac" and args.save_replay_buffer,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(output_dir / "final_model")
    summary = {
        "algorithm": args.algorithm,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "bc_checkpoint": str(Path(args.bc_checkpoint).resolve()),
        "total_timesteps": args.total_timesteps,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "train_frequency": args.train_frequency,
        "gradient_steps": args.gradient_steps,
        "learning_rate": args.learning_rate,
        "ent_coef": args.ent_coef,
        "bc_dataset_root": args.bc_dataset_root,
        "online_bc_regularization": args.online_bc_regularization,
        "offline_bc_regularization": args.offline_bc_regularization,
        "share_features_extractor": args.share_features_extractor,
        "save_replay_buffer": args.save_replay_buffer,
        "residual_scale": args.residual_scale,
        "control_mode": args.control_mode,
        "action_repeat": args.action_repeat,
        "control_gripper": args.control_gripper,
        "force_control": args.force_control,
        "seed": args.seed,
        "freeze_encoder": args.freeze_encoder,
        "actor_initialization": actor_initialization,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    environment.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
