import numpy as np
import torch
from torch import nn

from automotive_vision.inference.explainability import anomaly_heatmap_overlay, gradcam_overlay

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TinyCNN(nn.Module):
    """Minimal conv classifier just big enough for GradCAM to hook a target layer."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_gradcam_overlay_returns_rgb_image_matching_input_size():
    model = TinyCNN()
    model.eval()
    image_tensor = torch.randn(3, 32, 32)

    overlay = gradcam_overlay(model, image_tensor, target_layer=model.conv, mean=MEAN, std=STD)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8


def test_anomaly_heatmap_overlay_returns_rgb_image_matching_input_size():
    image_tensor = torch.randn(3, 16, 16)
    pixel_map = np.random.rand(16, 16).astype(np.float32)

    overlay = anomaly_heatmap_overlay(image_tensor, pixel_map, mean=MEAN, std=STD)

    assert overlay.shape == (16, 16, 3)
    assert overlay.dtype == np.uint8
