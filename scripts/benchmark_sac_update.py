from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from stable_baselines3.common.logger import configure

from policy.RL.sac_bc import SFTRegularizedSAC
from policy.RL.sb3_features import BCFeatureExtractor, initialize_sac_actor_from_sft


class DummyTactileEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, checkpoint_path: str, image_size: int):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint["model_config"]
        observation_spaces = {
            "qpos": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            "policy_step": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
        }
        for key in config["camera_keys"] + config["tactile_keys"]:
            observation_spaces[key] = spaces.Box(
                0,
                255,
                shape=(3, image_size, image_size),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self._observation = {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in observation_spaces.items()
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation, {}

    def step(self, action):
        return self._observation, 0.0, False, False, {}


def batched_observation(environment: gym.Env):
    observation, _ = environment.reset()
    return {
        key: np.expand_dims(value, axis=0)
        for key, value in observation.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("dataset_root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()

    environment = DummyTactileEnv(args.checkpoint, args.image_size)
    policy_kwargs = {
        "features_extractor_class": BCFeatureExtractor,
        "features_extractor_kwargs": {
            "bc_checkpoint": args.checkpoint,
            "freeze": False,
        },
        "normalize_images": False,
        "net_arch": [256, 256],
        "activation_fn": nn.GELU,
        "log_std_init": -3.0,
        "use_expln": True,
        "clip_mean": 20.0,
        "share_features_extractor": False,
    }
    model = SFTRegularizedSAC(
        "MultiInputPolicy",
        environment,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=max(256, args.batch_size * 2),
        learning_starts=0,
        batch_size=args.batch_size,
        train_freq=4,
        gradient_steps=1,
        ent_coef=1e-3,
        use_sde=True,
        sde_sample_freq=4,
        use_sde_at_warmup=True,
        seed=10000,
        device="cuda:0",
        verbose=0,
        bc_checkpoint=args.checkpoint,
        bc_dataset_root=args.dataset_root,
        online_bc_regularization=10.0,
        offline_bc_regularization=100.0,
        bc_image_size=args.image_size,
    )
    actor_initialization = initialize_sac_actor_from_sft(model, args.checkpoint)
    replay_device = str(model.replay_buffer.device)

    observation = batched_observation(environment)
    action = np.zeros((1, 8), dtype=np.float32)
    reward = np.zeros(1, dtype=np.float32)
    done = np.zeros(1, dtype=np.float32)
    for _ in range(max(64, args.batch_size * 2)):
        model.replay_buffer.add(
            observation,
            observation,
            action,
            reward,
            done,
            [{}],
        )

    model.set_logger(configure(tempfile.mkdtemp(prefix="sac_benchmark_"), []))
    model._current_progress_remaining = 1.0
    encoder_parameter = next(model.actor.features_extractor.encoder.parameters())
    encoder_before = encoder_parameter.detach().clone()
    update_seconds = []
    peak_memory_bytes = []
    for update_index in range(args.updates):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        model.train(gradient_steps=1, batch_size=args.batch_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at
        update_seconds.append(elapsed)
        peak_memory_bytes.append(torch.cuda.max_memory_allocated())
        print(
            f"update={update_index + 1} seconds={elapsed:.3f} "
            f"peak_gib={peak_memory_bytes[-1] / 2**30:.3f}",
            flush=True,
        )

    encoder_max_abs_change = float(
        (encoder_parameter.detach() - encoder_before).abs().max()
    )
    summary = {
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "updates": args.updates,
        "update_seconds": update_seconds,
        "mean_update_seconds": float(np.mean(update_seconds)),
        "steady_state_mean_seconds": float(np.mean(update_seconds[1:])),
        "peak_memory_gib": [value / 2**30 for value in peak_memory_bytes],
        "model_device": str(model.device),
        "replay_device": replay_device,
        "encoder_max_abs_change": encoder_max_abs_change,
        "actor_initialization": actor_initialization,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if encoder_max_abs_change <= 0.0:
        raise RuntimeError("The shared encoder was not updated")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_output)


if __name__ == "__main__":
    main()
