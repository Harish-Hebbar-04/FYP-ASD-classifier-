from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover - depends on local environment
    timm = None

# Default assumes torchvision.datasets.ImageFolder-style ordering
# where class folders are sorted alphabetically.
DEFAULT_CLASS_NAMES = ["ASD", "Non-ASD"]
SUPPORTED_ARCHITECTURES = {
    "resnet18",
    "resnet34",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "vit_base_patch16_224",
}


@dataclass
class LoadedModel:
    model: nn.Module
    device: torch.device
    class_names: list[str]
    input_size: int
    architecture: str


def build_model(architecture: str = "resnet18", output_dim: int = 2) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    if architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        last_linear = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last_linear.in_features, output_dim)
        return model

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        last_linear = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last_linear.in_features, output_dim)
        return model

    if architecture == "vit_base_patch16_224":
        return build_vit_model(output_dim=output_dim)

    raise ValueError(
        f"Unsupported architecture '{architecture}'. "
        f"Supported values: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}."
    )


def load_trained_model(
    model_path: str | Path,
    device: torch.device | None = None,
) -> LoadedModel:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Place your trained .pth file there."
        )

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=target_device, weights_only=False)

    class_names = DEFAULT_CLASS_NAMES.copy()
    env_class_names = os.getenv("ASD_CLASS_NAMES")
    if env_class_names:
        parsed = [name.strip() for name in env_class_names.split(",") if name.strip()]
        if len(parsed) >= 2:
            class_names = parsed[:2]
    input_size = 224
    architecture = "resnet18"

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint format. Expected an nn.Module or state dict.")

        metadata_names = checkpoint.get("class_names")
        if isinstance(metadata_names, (list, tuple)) and metadata_names:
            class_names = [str(name) for name in metadata_names]

        metadata_size = checkpoint.get("input_size") or checkpoint.get("image_size")
        if isinstance(metadata_size, int) and metadata_size > 0:
            input_size = metadata_size

        embedded_model = checkpoint.get("model")
        if isinstance(embedded_model, nn.Module):
            model = embedded_model
        else:
            raw_state_dict = _extract_state_dict(checkpoint)
            state_dict = _sanitize_state_dict(raw_state_dict)
            architecture = str(
                checkpoint.get("arch")
                or checkpoint.get("architecture")
                or infer_architecture(state_dict)
            ).lower()
            input_size = infer_input_size(state_dict, fallback=input_size)
            output_dim = infer_output_dim(state_dict, architecture)

            if output_dim not in (1, 2):
                raise ValueError(
                    f"Expected a binary classifier with 1 or 2 outputs, but found {output_dim}."
                )

            head_hidden_dim = infer_head_hidden_dim(state_dict, architecture)
            model = build_model_from_state(
                architecture=architecture,
                output_dim=output_dim,
                input_size=input_size,
                head_hidden_dim=head_hidden_dim,
            )
            model.load_state_dict(state_dict, strict=True)

    model = model.to(target_device)
    model.eval()

    return LoadedModel(
        model=model,
        device=target_device,
        class_names=class_names,
        input_size=input_size,
        architecture=architecture,
    )


def predict_image(input_tensor: torch.Tensor, bundle: LoadedModel) -> tuple[str, float]:
    with torch.inference_mode():
        logits = bundle.model(input_tensor.to(bundle.device))

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.shape[1] == 1:
        asd_probability = torch.sigmoid(logits).squeeze().item()
        prediction = bundle.class_names[1] if asd_probability >= 0.5 else bundle.class_names[0]
        confidence = asd_probability if asd_probability >= 0.5 else 1 - asd_probability
        return prediction, float(confidence)

    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_index = torch.max(probabilities, dim=1)
    prediction = bundle.class_names[predicted_index.item()]
    return prediction, float(confidence.item())


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint  # Raw state_dict saved directly via torch.save(model.state_dict()).

    raise ValueError(
        "Checkpoint does not contain a recognized state dict. "
        "Expected keys like 'state_dict' or 'model_state_dict'."
    )


def _sanitize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "", 1)] = value
    return cleaned


def infer_architecture(state_dict: dict[str, torch.Tensor]) -> str:
    if "fc.weight" in state_dict:
        return "resnet18"

    if "classifier.3.weight" in state_dict:
        return "mobilenet_v3_small"

    if "classifier.1.weight" in state_dict:
        return "efficientnet_b0"

    if "patch_embed.proj.weight" in state_dict and "cls_token" in state_dict and "pos_embed" in state_dict:
        return "vit_base_patch16_224"

    return "resnet18"


def infer_output_dim(state_dict: dict[str, torch.Tensor], architecture: str) -> int:
    if architecture.startswith("resnet") and "fc.weight" in state_dict:
        return int(state_dict["fc.weight"].shape[0])

    if architecture == "mobilenet_v3_small" and "classifier.3.weight" in state_dict:
        return int(state_dict["classifier.3.weight"].shape[0])

    if architecture == "efficientnet_b0" and "classifier.1.weight" in state_dict:
        return int(state_dict["classifier.1.weight"].shape[0])

    if architecture == "vit_base_patch16_224":
        if "head.3.weight" in state_dict:
            return int(state_dict["head.3.weight"].shape[0])

        if "head.weight" in state_dict:
            return int(state_dict["head.weight"].shape[0])

    return 2


def infer_head_hidden_dim(state_dict: dict[str, torch.Tensor], architecture: str) -> int | None:
    if architecture == "vit_base_patch16_224" and "head.0.weight" in state_dict:
        return int(state_dict["head.0.weight"].shape[0])

    return None


def infer_input_size(state_dict: dict[str, torch.Tensor], fallback: int = 224) -> int:
    pos_embed = state_dict.get("pos_embed")
    patch_proj = state_dict.get("patch_embed.proj.weight")

    if pos_embed is None or patch_proj is None:
        return fallback

    if pos_embed.ndim != 3 or patch_proj.ndim != 4:
        return fallback

    patch_tokens = pos_embed.shape[1] - 1
    if patch_tokens <= 0:
        return fallback

    grid_size = int(math.sqrt(patch_tokens))
    if grid_size * grid_size != patch_tokens:
        return fallback

    patch_size = int(patch_proj.shape[-1])
    return grid_size * patch_size


def build_model_from_state(
    architecture: str,
    output_dim: int,
    input_size: int,
    head_hidden_dim: int | None = None,
) -> nn.Module:
    if architecture == "vit_base_patch16_224":
        return build_vit_model(
            output_dim=output_dim,
            input_size=input_size,
            head_hidden_dim=head_hidden_dim,
        )

    return build_model(architecture=architecture, output_dim=output_dim)


def build_vit_model(
    output_dim: int = 2,
    input_size: int = 224,
    head_hidden_dim: int | None = None,
) -> nn.Module:
    if timm is None:
        raise ImportError(
            "Loading ViT checkpoints requires the 'timm' package. "
            "Install it with: pip install timm"
        )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=output_dim,
        img_size=input_size,
    )

    if head_hidden_dim:
        in_features = int(model.head.in_features)
        activation_name = os.getenv("MODEL_HEAD_ACTIVATION", "relu").strip().lower()
        model.head = nn.Sequential(
            nn.Linear(in_features, head_hidden_dim),
            make_activation(activation_name),
            nn.Dropout(p=0.0),
            nn.Linear(head_hidden_dim, output_dim),
        )

    return model


def make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()

    if name == "relu":
        return nn.ReLU()

    raise ValueError(
        f"Unsupported MODEL_HEAD_ACTIVATION '{name}'. Use 'relu' or 'gelu'."
    )
