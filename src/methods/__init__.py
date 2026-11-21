from .registry import register_method, create_method, list_methods
from .base import MethodModule

from . import baselines  # noqa: F401 — trigger @register_method decorators
from . import novel  # noqa: F401

__all__ = ["register_method", "create_method", "list_methods", "MethodModule"]
