"""
Trading Methods Module.

Re-export from inventory for backward compatibility.
"""

from .inventory import (
    METHOD_INVENTORY,
    get_method_names,
    get_methods_by_category,
    get_methods_for_regime,
    TradingMethod,
    MethodCategory,
)

__all__ = [
    "METHOD_INVENTORY",
    "get_method_names",
    "get_methods_by_category",
    "get_methods_for_regime",
    "TradingMethod",
    "MethodCategory",
]
