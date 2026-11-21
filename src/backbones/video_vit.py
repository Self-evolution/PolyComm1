"""VideoMAE ViT-B backbone — Kinetics-400 pretraining."""

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


def create_videomae_vit_b_k400(
    pretrained_path: str = "checkpoints/pretrained/VideoMAE_ViT-B_k400.pth",
    drop_path_rate: float = 0.1,
    num_classes: int = 0,
) -> nn.Module:
    """Create a VideoMAE ViT-B model with Kinetics-400 pretrained weights.

    The model reuses the standard ``vit_base_patch16_224`` architecture from
    timm.  VideoMAE pre-trains a vanilla ViT with masked autoencoding on
    video, so the spatial encoder is architecture-compatible with the image
    ViT — only the patch embedding is inflated to handle temporal frames
    at the data-loading level (frame stacking or tubelet embedding).

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
        "videomae_vit_b_k400",
        "Download VideoMAE ViT-B Kinetics-400 weights from "
        "https://github.com/MCG-NJU/VideoMAE — see the model zoo "
        f"and place at '{pretrained_path}'.",
    )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    # VideoMAE checkpoints may nest under 'model' or 'module'
    if "model" in state:
        state = state["model"]
    elif "module" in state:
        state = state["module"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Strip 'encoder.' prefix that some VideoMAE releases use
    cleaned: dict = {}
    for k, v in state.items():
        new_key = k.replace("encoder.", "") if k.startswith("encoder.") else k
        # Skip decoder / head weights
        if new_key.startswith("decoder.") or new_key.startswith("head."):
            continue
        cleaned[new_key] = v

    model.load_state_dict(cleaned, strict=False)
    return model
