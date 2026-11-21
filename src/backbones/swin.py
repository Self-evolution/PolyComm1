"""Swin-B backbone — ImageNet-22K pretraining."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import timm


def _check_weights(path: str, variant: str, hint: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Pretrained weights for '{variant}' not found at '{p}'.\n"
            f"Download instructions: {hint}"
        )
    return p


def create_swin_b_22k(
    pretrained_path: str = "checkpoints/pretrained/Swin-B_22k.pth",
    drop_path_rate: float = 0.1,
    num_classes: int = 0,
) -> nn.Module:
    """Create a Swin-B model with ImageNet-22K pretrained weights.

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
        "swin_b_22k",
        "Download Swin-B ImageNet-22K weights from "
        "https://github.com/SwinTransformer/storage/releases/download/"
        "v1.0.0/swin_base_patch4_window7_224_22k.pth "
        f"and place at '{pretrained_path}'.",
    )

    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Remove head weights if present and num_classes differs
    for k in list(state.keys()):
        if k.startswith("head."):
            del state[k]

    # Handle key mismatch: original Swin checkpoint has downsample at layers.0/1/2
    # but timm's Swin has downsample at layers.1/2/3 (stage 0 has no downsample).
    # Detect by checking if checkpoint has layers.0.downsample but model doesn't.
    if "layers.0.downsample.norm.weight" in state:
        remapped = {}
        for k, v in state.items():
            if k.startswith("layers."):
                parts = k.split(".", 2)
                layer_idx = int(parts[1])
                rest = parts[2]
                if "downsample" in rest:
                    # Shift downsample keys: 0->1, 1->2, 2->3
                    remapped[f"layers.{layer_idx + 1}.{rest}"] = v
                else:
                    remapped[k] = v
            else:
                remapped[k] = v
        state = remapped

    model.load_state_dict(state, strict=False)
    return model
