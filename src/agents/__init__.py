"""
Agents module: Method selection agents and population dynamics.

Core components:
- MethodSelector: Individual agent that selects methods using Thompson Sampling
- Population: Collection of agents with knowledge transfer
- Inventory: Shared method inventory available to all agents
- Methods: Actual trading method implementations
"""

from .method_selector import MethodSelector, SelectionResult
from .population import Population, PopulationConfig
from .inventory import METHOD_INVENTORY, get_method_names

__all__ = [
    "MethodSelector",
    "SelectionResult",
    "Population",
    "PopulationConfig",
    "METHOD_INVENTORY",
    "get_method_names",
]
