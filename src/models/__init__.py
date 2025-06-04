"""Model definitions for tomato spot disease recognition."""

from .baseline import ResNetBaseline
from .attention import ResNetWithAttention

__all__ = [
    "ResNetBaseline",
    "ResNetWithAttention"
] 