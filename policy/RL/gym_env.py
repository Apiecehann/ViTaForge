from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .bc import load_bc_checkpoint


class ResidualTactileEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        task,
        bc_checkpoint,
        image_size=224,
        residual_scale=0.5,
        action_repeat=2,
        seed=0,
        device="cuda:0",
    ):
        super().__init__()
        self.task = task
        self.image_size = int(image_size)
        self.residual_scale = float(residual_scale)
        self.action_repeat = int(action_repeat)
        self.next_seed = int(seed)
        self.device = torch.device(device)
        self.bc_model, self.bc_checkpoint = load_bc_checkpoint(
            bc_checkpoint,
            device=self.device,
        )
        config = self.bc_checkpoint["model_config"]
        self.camera_keys = list(config["camera_keys"])
        self.tactile_keys = list(config["tactile_keys"])
        observation_spaces = {
            "qpos": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        }
        for key in self.camera_keys + self.tactile_keys:
            observation_spaces[key] = spaces.Box(
                0,
                255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.last_observation = None

    def _image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = cv2.resize(
            np.asarray(image),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.uint8)

    @staticmethod
    def _sensor(raw_observation, side):
        tactile = raw_observation["tactile"]
        for key in (f"{side}_tactile", f"{side}_gsmini"):
            if key in tactile:
                return tactile[key]
        raise KeyError(f"Missing {side} tactile observation: {list(tactile)}")

    def encode_observation(self, raw_observation):
        joint = raw_observation["embodiment"]["joint"]
        if isinstance(joint, torch.Tensor):
            joint = joint.detach().cpu().numpy()
        encoded = {"qpos": np.asarray(joint[:8], dtype=np.float32)}
        if "cam_high" in self.camera_keys:
            encoded["cam_high"] = self._image(
                raw_observation["observation"]["head"]["rgb"]
            )
        if "cam_wrist" in self.camera_keys:
            encoded["cam_wrist"] = self._image(
                raw_observation["observation"]["wrist"]["rgb"]
            )
        if "tac_left" in self.tactile_keys:
            encoded["tac_left"] = self._image(
                self._sensor(raw_observation, "left")["rgb_marker"]
            )
        if "tac_right" in self.tactile_keys:
            encoded["tac_right"] = self._image(
                self._sensor(raw_observation, "right")["rgb_marker"]
            )
        return encoded

    def _bc_action(self, observation):
        batch = {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in observation.items()
        }
        with torch.no_grad():
            action = self.bc_model(batch)[0]
        return action.detach().cpu().numpy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = self.next_seed if seed is None else int(seed)
        self.next_seed = episode_seed + 1
        self.task.reset(seed=episode_seed)
        raw_observation = self.task._get_observations()
        self.last_observation = self.encode_observation(raw_observation)
        return self.last_observation, {"seed": episode_seed}

    def step(self, residual_action):
        residual_action = np.asarray(residual_action, dtype=np.float32)
        bc_action = self._bc_action(self.last_observation)
        action_std = self.bc_model.action_std.detach().cpu().numpy()
        final_action = bc_action + self.residual_scale * action_std * residual_action
        action_min = self.bc_model.action_min.detach().cpu().numpy()
        action_max = self.bc_model.action_max.detach().cpu().numpy()
        safety_margin = np.maximum(action_std, 1e-4)
        final_action = np.clip(
            final_action,
            action_min - safety_margin,
            action_max + safety_margin,
        )
        raw_observation, reward, terminated, truncated, info = self.task.env_step(
            final_action,
            action_type="qpos",
            force=True,
            action_repeat=self.action_repeat,
        )
        self.last_observation = self.encode_observation(raw_observation)
        info.update(
            {
                "bc_action": bc_action,
                "residual_action": residual_action,
                "final_action": final_action,
            }
        )
        return self.last_observation, reward, terminated, truncated, info

    def close(self):
        self.task.close()
