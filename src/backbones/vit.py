"""ViT-B backbone variants — ImageNet-21K, CLIP, and DINOv2 pretraining."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import timm


def _check_weights(path: str, variant: str, hint: str) -> Path:
    """Validate that a pretrained weights file exists.

    Raises ``FileNotFoundError`` with download instructions when missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Pretrained weights for '{variant}' not found at '{p}'.\n"
            f"Download instructions: {hint}"
        )
    return p


# ------------------------------------------------------------------
# ViT-B/16  —  ImageNet-21K supervised
# ------------------------------------------------------------------

def create_vit_b16_in21k(
    pretrained_path: str = "checkpoints/pretrained/ViT-B_16_in21k.npz",
    drop_path_rate: float = 0.1,
    num_classes: int = 0,
) -> nn.Module:
    """Create a ViT-B/16 model with ImageNet-21K supervised weights.

    Parameters
    ----------
    pretrained_path:
        Path to the ``.npz`` weight file.
    drop_path_rate:
        Stochastic depth rate.
    num_classes:
        Number of output classes.  ``0`` removes the classification head
        (useful when a separate task head is attached).
    """
    _check_weights(
        pretrained_path,
        "vit_b16_in21k",
        "Download ViT-B/16 ImageNet-21K weights from "
        "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz "
        f"and place at '{pretrained_path}'.",
    )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    # timm supports loading .npz checkpoints via its own helpers
    _load_weights_npz(model, pretrained_path)
    return model


def _load_weights_npz(model: nn.Module, path: str) -> None:
    """Load weights from a ``.npz`` file (NumPy archive).

    Google ViT checkpoints use a layout that does not match timm state-dict
    keys directly, so use timm's dedicated converter. A plain
    ``load_state_dict(..., strict=False)`` can silently load zero matching
    tensors here, leaving the backbone randomly initialised.
    """
    from timm.models.vision_transformer import _load_weights as timm_load_npz

    timm_load_npz(model, path)


# ------------------------------------------------------------------
# ViT-B/16  —  CLIP
# ------------------------------------------------------------------

def create_vit_b16_clip(
    pretrained_path: str = "checkpoints/pretrained/ViT-B_16_clip.bin",
    drop_path_rate: float = 0.1,
    num_classes: int = 0,
) -> nn.Module:
    """Create a ViT-B/16 model with CLIP pretrained weights.

    Parameters
    ----------
    pretrained_path:
        Path to the ``.bin`` (PyTorch state-dict) weight file.
    drop_path_rate:
        Stochastic depth rate.
    num_classes:
        Number of output classes (``0`` = no head).
    """
    _check_weights(
        pretrained_path,
        "vit_b16_clip",
        "Download CLIP ViT-B/16 weights from "
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt "
        f"and place at '{pretrained_path}'.",
    )

    model = timm.create_model(
        "vit_base_patch16_clip_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    # CLIP checkpoints may nest the visual encoder under a key
    if isinstance(state, dict):
        if "visual" in state:
            state = state["visual"]
        elif "state_dict" in state:
            state = state["state_dict"]
    else:
        # TorchScript model — extract state_dict
        state = state.state_dict() if hasattr(state, "state_dict") else state

    model.load_state_dict(state, strict=False)
    return model


# ------------------------------------------------------------------
# ViT-B/14  —  DINOv2
# ------------------------------------------------------------------

def create_vit_b14_dinov2(
    pretrained_path: str = "checkpoints/pretrained/ViT-B_14_dinov2.pth",
    drop_path_rate: float = 0.1,
    num_classes: int = 0,
) -> nn.Module:
    """Create a ViT-B/14 model with DINOv2 pretrained weights.

    Parameters
    ----------
    pretrained_path:
        Path to the ``.pth`` weight file.
    drop_path_rate:
        Stochastic depth rate.
    num_classes:
        Number of output classes (``0`` = no head).
    """
    _check_weights(
        pretrained_path,
        "vit_b14_dinov2",
        "Download DINOv2 ViT-B/14 weights from "
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth "
        f"and place at '{pretrained_path}'.",
    )

    model = timm.create_model(
        "vit_base_patch14_dinov2",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        img_size=518,
    )

    state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=False)
    return model
