from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .bc import load_bc_checkpoint


class BCFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, bc_checkpoint, freeze=True):
        model, _ = load_bc_checkpoint(bc_checkpoint, device="cpu")
        super().__init__(observation_space, features_dim=model.encoder.feature_dim)
        self.encoder = model.encoder
        self.register_buffer("qpos_mean", model.qpos_mean.detach().clone())
        self.register_buffer("qpos_std", model.qpos_std.detach().clone())
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, observations):
        normalized = dict(observations)
        normalized["qpos"] = (
            observations["qpos"].float() - self.qpos_mean
        ) / self.qpos_std
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            with torch.no_grad():
                return self.encoder(normalized)
        return self.encoder(normalized)
