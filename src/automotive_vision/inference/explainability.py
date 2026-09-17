"""Visual explanations: Grad-CAM for the classifier, heatmap overlays for anomaly scores.

Wiring status: `anomaly_heatmap_overlay` is live in the default `Predictor.predict()`
path (inference/predictor.py) and is what the API's `heatmap_path` field returns.
`gradcam_overlay` is implemented and unit-testable but not currently called from the
API -- it's available for offline/notebook use, not part of the default response.

Limitation (stated here and in the README, not just implied): Grad-CAM highlights
image regions that most influenced the *logit* for the predicted class. It is a
correlational visualization of the network's gradients, not a causal proof that those
pixels are the actual defect — a well-known caveat of saliency-style methods.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from automotive_vision.data.preprocessing import denormalize


def gradcam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Returns an RGB uint8 array (H, W, 3) with the Grad-CAM heatmap overlaid on the
    denormalized input image. Explains the model's own top prediction (no explicit
    target class is passed to GradCAM, so it defaults to the highest-scoring one)."""
    cam = GradCAM(model=model, target_layers=[target_layer])
    # targets=None: the installed pytorch-grad-cam version requires this argument
    # explicitly (it has no default), but None still means "use the model's own
    # top prediction," which is the intended behavior here.
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=None)[0]

    rgb_image = denormalize(image_tensor, mean, std).permute(1, 2, 0).cpu().numpy()
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return overlay


def anomaly_heatmap_overlay(image_tensor: torch.Tensor, pixel_map: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """Overlays a PatchCore-lite pixel anomaly map on the denormalized input image."""
    rgb_image = denormalize(image_tensor, mean, std).permute(1, 2, 0).cpu().numpy()
    normalized_map = (pixel_map - pixel_map.min()) / (pixel_map.max() - pixel_map.min() + 1e-8)
    return show_cam_on_image(rgb_image, normalized_map, use_rgb=True)


def save_overlay(overlay: np.ndarray, out_path: str | Path) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)
    return str(out_path)
