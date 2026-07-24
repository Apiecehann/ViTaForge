from __future__ import annotations

import torch
from torch import nn

from .encoders import MultiModalEncoder


class MultiModalBC(nn.Module):
    def __init__(self, model_config, statistics):
        super().__init__()
        self.model_config = dict(model_config)
        self.encoder = MultiModalEncoder(**self.model_config)
        feature_dim = int(self.model_config["feature_dim"])
        action_dim = int(self.model_config.get("qpos_dim", 8))
        self.action_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )
        for name, value in statistics.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def normalized_observation(self, observation):
        normalized = dict(observation)
        normalized["qpos"] = (
            observation["qpos"].float() - self.qpos_mean
        ) / self.qpos_std
        return normalized

    def forward_normalized(self, observation):
        features = self.encoder(self.normalized_observation(observation))
        return self.action_head(features)

    def forward(self, observation):
        normalized_delta = self.forward_normalized(observation)
        delta = normalized_delta * self.delta_std + self.delta_mean
        return observation["qpos"].float() + delta

    def checkpoint(self, metadata=None):
        statistics = {
            name: getattr(self, name).detach().cpu().numpy()
            for name in (
                "qpos_mean",
                "qpos_std",
                "delta_mean",
                "delta_std",
                "joint_min",
                "joint_max",
            )
        }
        return {
            "model_config": self.model_config,
            "action_representation": "delta_qpos_v1",
            "statistics": statistics,
            "model_state": self.state_dict(),
            "metadata": metadata or {},
        }


def load_bc_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiModalBC(
        checkpoint["model_config"],
        checkpoint["statistics"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
