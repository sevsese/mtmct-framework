"""Small compatibility helpers for NVIDIA ground-truth JSON fields."""
from __future__ import annotations


def value(obj: dict, name: str, default=None):
    """Read both documented space-separated and legacy underscore keys."""
    return obj.get(name.replace("_", " "), obj.get(name, default))
