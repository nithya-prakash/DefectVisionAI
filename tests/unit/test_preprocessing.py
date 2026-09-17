import numpy as np
import torch
from PIL import Image

from automotive_vision.data.preprocessing import build_eval_transform, denormalize

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def test_eval_transform_produces_expected_shape_and_is_deterministic():
    image = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype("uint8"))
    transform = build_eval_transform(224, MEAN, STD)
    tensor1 = transform(image)
    tensor2 = transform(image)
    assert tensor1.shape == (3, 224, 224)
    assert torch.allclose(tensor1, tensor2)  # no randomness


def test_denormalize_inverts_normalization():
    image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    transform = build_eval_transform(224, MEAN, STD)
    normalized = transform(image)
    recovered = denormalize(normalized, MEAN, STD)
    original = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    assert torch.allclose(recovered, original, atol=1e-4)
