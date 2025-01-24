from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal


import timm
from timm.models.layers import Mlp, to_2tuple

import torch
import torch.nn as nn

from spatialclip.models.image_models._ctranspath import _build_ctranspath_model
from spatialclip.models.image_models._utils import freeze_batch_norm_2d

MODEL_NAMES = {
    "densenet": "timm/densenet121.ra_in1k",
    "ctranspath": "jamesdolezal/CTransPath",
    "uni": "MahmoodLab/UNI",
    "optimus": "bioptimus/H-optimus-0",
}

class ImageEncoder(nn.Module):
    """ImageEncoder that wraps timm models."""

    def __init__(
        self,
        model_name: Literal["ctranspath", "densenet", "uni", "optimus"] | str,
        embed_dim: int = -1,
        image_size: int = 224,
        proj: str = "mlp",
        **kwargs: Any,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.model_name = model_name

        if model_name == "densenet":
            self.trunk = timm.create_model(
                "densenet121",
                pretrained=False,
                num_classes=0,
                # global_pool="",
            )
            
            self.total_blocks = 4

        elif model_name == "ctranspath":
            self.trunk = _build_ctranspath_model()
            self.total_blocks = 4  # technically layers

        elif model_name == "uni":
            self.trunk = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
            )
            self.total_blocks = 24

        elif model_name == "optimus":
            self.trunk = timm.create_model(
                f"hf-hub:{MODEL_NAMES[model_name]}", pretrained=False, init_values=1e-5, dynamic_img_size=False
            )
            self.total_blocks = 4

        head_layers = OrderedDict()
        if proj == "linear":
            head_layers["proj"] = nn.Linear(self.trunk.num_features, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(
                self.trunk.num_features, 2 * embed_dim, embed_dim, drop=0.2, norm_layer=nn.LayerNorm
            )

        self.head = nn.Sequential(head_layers)

    def freeze(self, unfrozen_blocks: int, freeze_bn_stats=True):
        """Freeze model params."""
        block_patterns = {
            "uni": lambda b: f"blocks.{b}.",
            "optimus": lambda b: f"blocks.{b}.",
            "ctranspath": lambda b: f"layers.{b}.",
            "densenet": lambda b: f"features.denseblock{b+1}.",
        }
        frozen_block = max(self.total_blocks - unfrozen_blocks, 0)  # incase frozen_blocks exceed total_blocks
        block_pattern = block_patterns.get(self.model_name, lambda b: f"blocks.{b}.")  # generic fallback pattern
        for name, param in self.trunk.named_parameters():
            if block_pattern(frozen_block) in name:
                break
            param.requires_grad = False

        if freeze_bn_stats:
            freeze_batch_norm_2d(self.trunk)

    def forward(self, x):
        """Forward pass."""
        x = self.trunk(x)
        x = self.head(x)
        return x
    
def clip_models(model_name, ckpt_path, embed_dim=512):
    model = ImageEncoder(model_name=model_name, embed_dim=embed_dim)
    model_weights = torch.load(ckpt_path, weights_only=False)["state_dict"]

    filtered_weights = {} # k: v for k, v in weights.items() if 'image_encoder' in k
    # replace model.image_encoder.trunk with 0 and model.image_encoder.head with 1
    for k, v in model_weights.items():
        if 'image_encoder' in k:
            # print(k)
            if "trunk" in k:
                k = k.replace('model.image_encoder.trunk', 'trunk')
            elif "head" in k:
                k = k.replace('model.image_encoder.head', 'head')
            filtered_weights[k] = v

    model.load_state_dict(filtered_weights, strict=True)
    model.eval()

    return model