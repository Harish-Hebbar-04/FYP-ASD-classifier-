from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InvalidImageError(ValueError):
    """Raised when the uploaded file cannot be decoded as an image."""


def preprocess_image(image_bytes: bytes, input_size: int = 224) -> torch.Tensor:
    """Decode and preprocess image bytes for PyTorch inference."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Uploaded file is not a valid image.") from exc

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transform(image).unsqueeze(0)

