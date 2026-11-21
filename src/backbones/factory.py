"""BackboneFactory — create and load pretrained backbone models by name."""

from __future__ import annotations

from typing import Dict, Callable

import torch.nn as nn


# ---------------------------------------------------------------------------
# Registry of backbone builder functions
# ---------------------------------------------------------------------------

_BACKBONE_BUILDERS: Dict[str, Callable[..., nn.Module]] = {}


def _register(name: str, builder: Callable[..., nn.Module]) -> None:
    _BACKBONE_BUILDERS[name] = builder


def _lazy_register_all() -> None:
    """Populate the registry on first use (avoids import-time heavy deps)."""
    if _BACKBONE_BUILDERS:
        return

    from src.backbones.vit import (
        create_vit_b16_in21k,
        create_vit_b16_clip,
        create_vit_b14_dinov2,
    )
    from src.backbones.swin import create_swin_b_22k
    from src.backbones.video_vit import create_videomae_vit_b_k400

    _register("vit_b16_in21k", create_vit_b16_in21k)
    _register("vit_b16_clip", create_vit_b16_clip)
    _register("vit_b14_dinov2", create_vit_b14_dinov2)
    _register("swin_b_22k", create_swin_b_22k)
    _register("videomae_vit_b_k400", create_videomae_vit_b_k400)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BackboneFactory:
    """Create backbone models from a config dict.

    Expected config structure::

        backbone:
          name: vit_b16_in21k
          pretrained_path: checkpoints/pretrained/ViT-B_16_in21k.npz
          drop_path_rate: 0.1
    """

    SUPPORTED = (
        "vit_b16_in21k",
        "vit_b16_clip",
        "vit_b14_dinov2",
        "swin_b_22k",
        "videomae_vit_b_k400",
    )

    @staticmethod
    def create(config: dict) -> nn.Module:
        """Build a backbone from *config* and load pretrained weights.

        Parameters
        ----------
        config:
            Full experiment config.  The ``backbone`` sub-dict is used.

        Returns
        -------
        nn.Module
            The backbone with pretrained weights loaded.

        Raises
        ------
        ValueError
            If the backbone name is not supported.
        FileNotFoundError
            If the pretrained weights file does not exist (raised by the
            individual builder functions).
        """
        _lazy_register_all()

        bb_cfg = config.get("backbone", {}) if "backbone" in config else config
        name = bb_cfg.get("name")
        if not name:
            raise ValueError(
                "backbone.name is required in config but was not provided."
            )

        if name not in _BACKBONE_BUILDERS:
            raise ValueError(
                f"Unknown backbone '{name}'. "
                f"Supported: {list(_BACKBONE_BUILDERS.keys())}"
            )

        builder = _BACKBONE_BUILDERS[name]

        # Collect keyword arguments understood by the builder
        kwargs: dict = {}
        if "pretrained_path" in bb_cfg:
            kwargs["pretrained_path"] = bb_cfg["pretrained_path"]
        if "drop_path_rate" in bb_cfg:
            kwargs["drop_path_rate"] = bb_cfg["drop_path_rate"]
        if "num_classes" in bb_cfg:
            kwargs["num_classes"] = bb_cfg["num_classes"]

        return builder(**kwargs)

    @staticmethod
    def list_backbones() -> list[str]:
        """Return names of all supported backbones."""
        _lazy_register_all()
        return list(_BACKBONE_BUILDERS.keys())


# Module-level constant for tests and external consumers
AVAILABLE_BACKBONES: tuple[str, ...] = (
    "vit_b16_in21k",
    "vit_b16_clip",
    "vit_b14_dinov2",
    "swin_b_22k",
    "videomae_vit_b_k400",
)
