"""Setup script for emergent_specialization package."""

from setuptools import setup, find_packages

setup(
    name="emergent_specialization",
    version="0.1.0",
    description="Emergent Specialization in Multi-Agent Trading",
    author="MAS Finance Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "dev": ["pytest>=7.3.0", "black>=23.0.0", "mypy>=1.0.0"],
    },
)
