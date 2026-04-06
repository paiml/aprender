#!/usr/bin/env python3
"""Setup script for Presentar Python utilities.

B1/F1: Reproducibility and random seed management.
"""
import os
import random

from setuptools import setup, find_packages

# F1: Set random seed for reproducible builds
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
random.seed(RANDOM_SEED)

setup(
    name="presentar",
    version="0.1.0",
    description="WASM-first visualization framework - Python utilities",
    author="PAIML",
    author_email="dev@paiml.com",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy>=1.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.92.0",
        ],
        "ml": [
            "torch>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "set-seeds=scripts.set_seeds:set_all_seeds",
        ],
    },
)
