"""LAP policy modules.

Heavy OpenPI transform imports are lazy so lightweight cover runtime modules can
be imported without pulling JAX/Flax.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CoTInputs",
    "CoTOutputs",
    "CoverPolicyWrapper",
]

_MODULE_BY_EXPORT = {
    "CoTInputs": "lap.policies.transforms",
    "CoTOutputs": "lap.policies.transforms",
    "CoverPolicyWrapper": "lap.policies.cover_policy_wrapper",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve package-level exports."""
    if name not in _MODULE_BY_EXPORT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_BY_EXPORT[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
