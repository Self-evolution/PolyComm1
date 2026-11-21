"""MethodModule — abstract base class for all PEFT methods."""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch.nn as nn


class MethodModule(ABC):
    """All PEFT methods must subclass this and implement the four hooks."""

    @abstractmethod
    def inject(self, backbone: nn.Module) -> nn.Module:
        """Inject PEFT parameters into *backbone* (in-place is fine).

        Returns the (possibly modified) backbone.
        """
        ...

    @abstractmethod
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return the list of parameters this method added / unfroze."""
        ...

    @abstractmethod
    def get_param_count(self) -> Dict[str, int]:
        """Return ``{"total": N, "trainable": M}``."""
        ...

    def is_compatible_with(self, other: "MethodModule") -> bool:
        """Check whether this method can be stacked with *other*.

        Defaults to ``True``; override to forbid specific combos.
        """
        return True
