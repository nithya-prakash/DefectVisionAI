"""Deterministic resize/normalization used identically at train and inference time."""

import torchvision.transforms as T


def build_eval_transform(image_size: int, mean: list[float], std: list[float]) -> T.Compose:
    """Deterministic preprocessing: resize -> tensor -> normalize. No randomness."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def denormalize(tensor, mean: list[float], std: list[float]):
    """Invert normalization for visualization (e.g. Grad-CAM overlays)."""
    mean_t = tensor.new_tensor(mean).view(-1, 1, 1)
    std_t = tensor.new_tensor(std).view(-1, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)
