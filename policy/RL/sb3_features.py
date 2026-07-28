from __future__ import annotations

import time

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .bc import load_bc_checkpoint


class BCFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, bc_checkpoint, freeze=True):
        started_at = time.perf_counter()
        print(f"[RL init] loading feature extractor from {bc_checkpoint}", flush=True)
        model, _ = load_bc_checkpoint(bc_checkpoint, device="cpu")
        super().__init__(observation_space, features_dim=model.encoder.feature_dim)
        self.encoder = model.encoder
        self.register_buffer("qpos_mean", model.qpos_mean.detach().clone())
        self.register_buffer("qpos_std", model.qpos_std.detach().clone())
        self.register_buffer(
            "policy_step_scale",
            model.policy_step_scale.detach().clone(),
        )
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        print(
            f"[RL init] feature extractor ready in "
            f"{time.perf_counter() - started_at:.2f}s",
            flush=True,
        )

    def forward(self, observations):
        normalized = dict(observations)
        normalized["qpos"] = (
            observations["qpos"].float() - self.qpos_mean
        ) / self.qpos_std
        if "policy_step" in observations:
            normalized["policy_step"] = torch.clamp(
                observations["policy_step"].float() / self.policy_step_scale,
                min=0.0,
                max=1.5,
            )
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            with torch.no_grad():
                return self.encoder(normalized)
        return self.encoder(normalized)


def initialize_sac_actor_from_sft(model, checkpoint_path):
    print("[RL init] loading SFT action head", flush=True)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    print("[RL init] SFT action head loaded", flush=True)
    if checkpoint.get("action_representation") != "bounded_delta_qpos_v2":
        raise ValueError(
            "SAC Actor initialization requires bounded_delta_qpos_v2 SFT weights"
        )
    source_state = checkpoint["model_state"]
    source_weight_keys = sorted(
        (
            key
            for key in source_state
            if key.startswith("action_head.") and key.endswith(".weight")
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    target_layers = [
        layer for layer in model.actor.latent_pi if isinstance(layer, nn.Linear)
    ] + [
        layer for layer in model.actor.mu.modules() if isinstance(layer, nn.Linear)
    ]
    if len(source_weight_keys) != len(target_layers):
        raise ValueError(
            f"SFT/SAC action head depth mismatch: {len(source_weight_keys)} != "
            f"{len(target_layers)}"
        )
    copied_shapes = []
    copied_max_abs_errors = []
    with torch.no_grad():
        for layer_index, (weight_key, target) in enumerate(
            zip(source_weight_keys, target_layers)
        ):
            source_weight = source_state[weight_key]
            source_bias = source_state[weight_key.removesuffix("weight") + "bias"]
            if source_weight.shape != target.weight.shape:
                raise ValueError(
                    f"SFT/SAC action head shape mismatch: "
                    f"{tuple(source_weight.shape)} != {tuple(target.weight.shape)}"
                )
            print(
                f"[RL init] copying action layer {layer_index}: "
                f"{tuple(source_weight.shape)} to {target.weight.device}",
                flush=True,
            )
            target.weight.copy_(source_weight.to(target.weight.device))
            target.bias.copy_(source_bias.to(target.bias.device))
            max_abs_error = max(
                float((target.weight.detach().cpu() - source_weight).abs().max()),
                float((target.bias.detach().cpu() - source_bias).abs().max()),
            )
            print(f"[RL init] action layer {layer_index} copied", flush=True)
            copied_shapes.append(tuple(source_weight.shape))
            copied_max_abs_errors.append(max_abs_error)
    return {
        "checkpoint": str(checkpoint_path),
        "action_representation": checkpoint["action_representation"],
        "copied_linear_shapes": copied_shapes,
        "copied_max_abs_errors": copied_max_abs_errors,
    }
