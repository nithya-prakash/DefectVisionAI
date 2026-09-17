"""Training-time augmentation, kept mild and configurable since defect appearance
(scratches, dents, discoloration) can be destroyed by aggressive augmentation."""

import torchvision.transforms as T


def build_train_transform(
    image_size: int,
    mean: list[float],
    std: list[float],
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.5,
    rotation_degrees: float = 15,
    color_jitter_brightness: float = 0.1,
    color_jitter_contrast: float = 0.1,
) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=horizontal_flip_p),
            T.RandomVerticalFlip(p=vertical_flip_p),
            T.RandomRotation(degrees=rotation_degrees),
            T.ColorJitter(brightness=color_jitter_brightness, contrast=color_jitter_contrast),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
