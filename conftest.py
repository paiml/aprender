"""Pytest configuration for reproducibility.

This file ensures deterministic test execution through random seed management.
F1: Popper falsifiability criterion - random seed fixing.
"""
import os
import random

import numpy as np
import pytest

# Random seed management - CRITICAL for reproducibility
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
TEST_SEED = int(os.environ.get("TEST_SEED", str(RANDOM_SEED)))


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    F1 Criterion: Random seed management for Popperian falsifiability.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow seed (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # PyTorch seed (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def seed_everything():
    """Fixture to set random seeds at session start."""
    set_seed(TEST_SEED)
    yield


@pytest.fixture
def reproducible_seed():
    """Fixture providing the reproducible seed value."""
    return TEST_SEED


@pytest.fixture
def seeded_rng():
    """Fixture providing a seeded random generator."""
    return random.Random(TEST_SEED)
