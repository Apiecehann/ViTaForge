from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _load_partial(module: nn.Module, checkpoint_path: str | None):
    if not checkpoint_path:
        return
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "encoder.", "backbone.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value
    module.load_state_dict(cleaned, strict=False)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "act_resnet18",
        pretrained: bool = False,
        checkpoint_path: str | None = None,
        imagenet_normalize: bool = True,
    ):
        super().__init__()
        self.imagenet_normalize = imagenet_normalize
        normalization_mean = (0.485, 0.456, 0.406)
        normalization_std = (0.229, 0.224, 0.225)
        if backbone == "act_resnet18":
            model = resnet18(weights="DEFAULT" if pretrained else None)
            self.output_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.model = model
        elif backbone.startswith("timm:"):
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "timm is required for CLIP, DINOv3, and LingBot adapters. "
                    "Install policy/RL/requirements.txt."
                ) from exc
            model_name = backbone.split(":", 1)[1]
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            self.output_dim = int(self.model.num_features)
            normalization_mean = self.model.pretrained_cfg.get(
                "mean",
                normalization_mean,
            )
            normalization_std = self.model.pretrained_cfg.get(
                "std",
                normalization_std,
            )
        else:
            raise ValueError(f"Unsupported image backbone: {backbone}")
        _load_partial(self.model, checkpoint_path)
        self.register_buffer(
            "mean",
            torch.tensor(normalization_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(normalization_std).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image: torch.Tensor):
        image = image.float()
        if image.detach().amax() > 1.5:
            image = image / 255.0
        if self.imagenet_normalize:
            image = (image - self.mean) / self.std
        return self.model(image)


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        qpos_dim: int,
        policy_step_dim: int,
        camera_keys: list[str],
        tactile_keys: list[str],
        feature_dim: int = 512,
        visual_backbone: str = "act_resnet18",
        tactile_backbone: str = "act_resnet18",
        visual_pretrained: bool = False,
        tactile_pretrained: bool = False,
        visual_checkpoint: str | None = None,
        tactile_checkpoint: str | None = None,
    ):
        super().__init__()
        self.qpos_dim = qpos_dim
        self.camera_keys = list(camera_keys)
        self.tactile_keys = list(tactile_keys)
        self.feature_dim = feature_dim
        self.visual_encoder = None
        self.tactile_encoder = None
        input_dim = qpos_dim + policy_step_dim
        if self.camera_keys:
            self.visual_encoder = ImageEncoder(
                backbone=visual_backbone,
                pretrained=visual_pretrained,
                checkpoint_path=visual_checkpoint,
                imagenet_normalize=True,
            )
            input_dim += len(self.camera_keys) * self.visual_encoder.output_dim
        if self.tactile_keys:
            self.tactile_encoder = ImageEncoder(
                backbone=tactile_backbone,
                pretrained=tactile_pretrained,
                checkpoint_path=tactile_checkpoint,
                imagenet_normalize=(
                    tactile_pretrained and tactile_checkpoint is None
                ),
            )
            input_dim += len(self.tactile_keys) * self.tactile_encoder.output_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    @staticmethod
    def _encode_modalities(encoder, observation, keys):
        if not keys:
            return []
        batch_size = observation[keys[0]].shape[0]
        images = torch.cat([observation[key] for key in keys], dim=0)
        features = encoder(images)
        return list(features.split(batch_size, dim=0))

    def forward(self, observation: dict[str, torch.Tensor]):
        features = [observation["qpos"].float()]
        if "policy_step" in observation:
            features.append(observation["policy_step"].float())
        if self.visual_encoder is not None:
            features.extend(
                self._encode_modalities(
                    self.visual_encoder,
                    observation,
                    self.camera_keys,
                )
            )
        if self.tactile_encoder is not None:
            features.extend(
                self._encode_modalities(
                    self.tactile_encoder,
                    observation,
                    self.tactile_keys,
                )
            )
        return self.projection(torch.cat(features, dim=-1))
