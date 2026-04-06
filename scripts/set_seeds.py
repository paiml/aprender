#!/usr/bin/env python3
"""Random seed management for reproducibility.

F1 Criterion: Popperian falsifiability requires deterministic execution.
This script sets all random seeds across Python ML libraries.
"""
import os
import random

# F1: Random seed management
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))


def _set_numpy_seed(seed: int) -> None:
    """Set NumPy random seed."""
    try:
        import numpy as np
        np.random.seed(seed)
        print(f"NumPy seed set to {seed}")
    except ImportError:
        pass


def _set_pytorch_seed(seed: int) -> None:
    """Set PyTorch random seeds (CPU and CUDA)."""
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"PyTorch seed set to {seed}")
    except ImportError:
        pass


def _set_tensorflow_seed(seed: int) -> None:
    """Set TensorFlow random seed."""
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f"TensorFlow seed set to {seed}")
    except ImportError:
        pass


def _set_jax_seed(seed: int) -> None:
    """Document JAX PRNGKey pattern."""
    try:
        import jax
        print(f"JAX: Use jax.random.PRNGKey({seed})")
    except ImportError:
        pass


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for complete reproducibility.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _set_numpy_seed(seed)
    _set_pytorch_seed(seed)
    _set_tensorflow_seed(seed)
    _set_jax_seed(seed)
    print(f"All seeds set to {seed}")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else RANDOM_SEED
    set_all_seeds(seed)
