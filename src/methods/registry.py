"""PEFT method registry — register, look-up, and instantiate methods by name."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import MethodModule

_REGISTRY: Dict[str, Type[MethodModule]] = {}


def register_method(name: str):
    """Class decorator that registers a :class:`MethodModule` subclass.

    Usage::

        @register_method("lora")
        class LoRAMethod(MethodModule):
            ...
    """

    def decorator(cls: Type[MethodModule]):
        if name in _REGISTRY:
            raise ValueError(
                f"Method '{name}' is already registered "
                f"(existing: {_REGISTRY[name].__qualname__})"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def create_method(name: str, config: dict) -> MethodModule:
    """Instantiate a registered method by *name* with the given *config*.

    Raises :class:`ValueError` with the full list of available methods
    when *name* is not found.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown method '{name}'. Available methods: {available}"
        )
    return _REGISTRY[name](config)


def list_methods() -> List[str]:
    """Return sorted names of all registered methods."""
    return sorted(_REGISTRY.keys())
