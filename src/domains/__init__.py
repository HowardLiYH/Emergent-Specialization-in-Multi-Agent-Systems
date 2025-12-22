"""
Multi-Domain Validation Framework.

This module provides environments and methods for testing emergent
specialization across multiple domains beyond financial trading.

Supported Domains (Original):
1. Traffic - Traffic flow optimization with congestion regimes
2. Energy - Energy grid management with demand regimes
3. Weather - Weather prediction with meteorological regimes
4. E-commerce - Inventory management with demand regimes
5. Sports - Team strategy with game-state regimes

Tier-1 Domains (New):
1. Air Quality - EPA PM2.5/AQI prediction
2. Wikipedia - Page view prediction
3. Solar - Solar irradiance (GHI) prediction
4. Water - USGS streamflow prediction
5. Commodities - FRED commodity price prediction

Each domain has:
- Environment with regime structure
- Domain-specific methods (strategies)
- Reward function
"""

from .base import DomainEnvironment, DomainMethod, DomainConfig
from .traffic import TrafficDomain
from .energy import EnergyDomain
from .synthetic_domains import (
    create_traffic_environment,
    create_energy_environment,
    create_weather_environment,
    create_ecommerce_environment,
    create_sports_environment,
)

# Tier-1 domain imports
from .air_quality import AirQualityDomain, create_air_quality_environment
from .wikipedia import WikipediaDomain, create_wikipedia_environment
from .solar import SolarDomain, create_solar_environment
from .water import WaterDomain, create_water_environment
from .commodities import CommoditiesDomain, create_commodities_environment

__all__ = [
    "DomainEnvironment",
    "DomainMethod",
    "DomainConfig",
    "TrafficDomain",
    "EnergyDomain",
    "create_traffic_environment",
    "create_energy_environment",
    "create_weather_environment",
    "create_ecommerce_environment",
    "create_sports_environment",
    # Tier-1 domains
    "AirQualityDomain",
    "WikipediaDomain", 
    "SolarDomain",
    "WaterDomain",
    "CommoditiesDomain",
    "create_air_quality_environment",
    "create_wikipedia_environment",
    "create_solar_environment",
    "create_water_environment",
    "create_commodities_environment",
]
