from typing import Tuple

import torch

from config_loader import ModelConfig, resolve_encoder_size
from dpt import DPT


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(model_config: ModelConfig):
    model_name = "dinov3_vitb16" if model_config.dino_size == "b" else "dinov3_vits16"
    return torch.hub.load(
        model_config.dino_repo,
        model_name,
        source="local",
        weights=model_config.dino_ckpt,
    )


def build_model(model_config: ModelConfig, device: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    backbone = load_backbone(model_config)
    model = DPT(
        encoder_size=resolve_encoder_size(model_config.dino_size),
        nclass=model_config.num_classes,
        decoder_channels=model_config.decoder_dim,
        use_bn=model_config.use_bn,
        patch_size=model_config.patch_size,
        backbone=backbone,
    ).to(device)
    return model, backbone


def summarize_parameters(model, backbone) -> tuple[int, int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in backbone.parameters())
    other_params = total_params - backbone_params
    return total_params, backbone_params, other_params
