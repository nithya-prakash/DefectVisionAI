"""Patch-level feature extraction from a frozen, ImageNet-pretrained CNN backbone.

Used both by the PatchCore-style anomaly detector (models/anomaly_detector.py) and,
via global-pooled embeddings, by diversity sampling in active learning.
"""

import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import nn


class PatchFeatureExtractor(nn.Module):
    """Extracts mid-level spatial features from a frozen ResNet18.

    Mirrors PatchCore's use of mid-network layers (deep enough to be semantic, shallow
    enough to keep spatial resolution for localization): layer2 and layer3 feature maps
    are upsampled to a common resolution and concatenated channel-wise into one patch
    embedding per spatial location.
    """

    def __init__(self, layers: tuple[str, ...] = ("layer2", "layer3")):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        self.layers = layers
        self._features: dict[str, torch.Tensor] = {}
        for layer_name in layers:
            getattr(backbone, layer_name).register_forward_hook(self._make_hook(layer_name))

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output

        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a (B, C_total, H, W) patch embedding map at the first layer's resolution."""
        self._features.clear()
        self.backbone(x)
        maps = [self._features[name] for name in self.layers]
        target_size = maps[0].shape[-2:]
        aligned = [
            m if m.shape[-2:] == target_size else F.interpolate(m, size=target_size, mode="bilinear")
            for m in maps
        ]
        return torch.cat(aligned, dim=1)

    @torch.no_grad()
    def global_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled embedding per image, for diversity sampling / kNN."""
        patch_map = self.forward(x)
        return patch_map.mean(dim=(2, 3))
