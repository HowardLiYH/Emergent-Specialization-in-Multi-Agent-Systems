"""
Multi-Domain Support for Emergent Specialization.

ALL DOMAINS USE 100% VERIFIED REAL DATA:
- Crypto: Bybit exchange data (44K bars)
- Commodities: FRED (Federal Reserve) data (5.6K prices)
- Weather: Open-Meteo historical data (9K observations)
- Solar: Open-Meteo solar irradiance data (117K hourly)
- Traffic: NYC TLC Yellow Taxi data (2.9K hourly trip counts)
- Air Quality: Open-Meteo PM2.5 data (2.9K hourly readings)
"""

from . import crypto
from . import commodities
from . import weather
from . import solar
from . import traffic
from . import air_quality

# Domain registry with metadata - ALL REAL DATA
DOMAINS = {
    'crypto': {
        'module': crypto,
        'data_source': 'Bybit Exchange',
        'records': '~44K',
        'verified_real': True,
    },
    'commodities': {
        'module': commodities,
        'data_source': 'FRED (US Government)',
        'records': '~5.6K',
        'verified_real': True,
    },
    'weather': {
        'module': weather,
        'data_source': 'Open-Meteo API',
        'records': '~9K',
        'verified_real': True,
    },
    'solar': {
        'module': solar,
        'data_source': 'Open-Meteo Solar API',
        'records': '~117K',
        'verified_real': True,
    },
    'traffic': {
        'module': traffic,
        'data_source': 'NYC TLC Yellow Taxi',
        'records': '~2.9K',
        'verified_real': True,
    },
    'air_quality': {
        'module': air_quality,
        'data_source': 'Open-Meteo Air Quality API',
        'records': '~2.9K',
        'verified_real': True,
    },
}


def get_domain(name: str):
    """Get domain module by name."""
    if name not in DOMAINS:
        raise ValueError(f"Unknown domain: {name}. Available: {list(DOMAINS.keys())}")
    return DOMAINS[name]['module']


def list_domains():
    """List all available domains."""
    return list(DOMAINS.keys())


def verify_all_domains():
    """Verify all domains can load real data."""
    results = {}

    for name, info in DOMAINS.items():
        module = info['module']
        try:
            df = module.load_data() if name != 'crypto' else module.load_data('BTC')
            results[name] = {
                'status': 'OK',
                'records': len(df),
                'source': info['data_source'],
            }
        except Exception as e:
            results[name] = {
                'status': 'ERROR',
                'error': str(e),
            }

    return results
