#!/usr/bin/env python3
"""Reproducibility utilities for Presentar.

F1/F2: ML Reproducibility - Seed management and model versioning.
"""
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# F1: Global random seed configuration
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility.

    F1 Criterion: Random seed management.

    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # Environment for hash reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow (if available)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_seed() -> int:
    """Get the current random seed.

    Returns:
        Current seed value
    """
    return RANDOM_SEED


def verify_reproducibility(func, n_runs: int = 3, seed: int = RANDOM_SEED) -> bool:
    """Verify that a function produces reproducible results.

    Args:
        func: Function to test
        n_runs: Number of runs to compare
        seed: Seed to use

    Returns:
        True if all runs produce identical results
    """
    results = []
    for _ in range(n_runs):
        set_seed(seed)
        result = func()
        results.append(result)

    return all(r == results[0] for r in results)


def create_experiment_manifest(
    name: str,
    seed: int = RANDOM_SEED,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a reproducibility manifest for an experiment.

    F2: Model versioning and experiment tracking.

    Args:
        name: Experiment name
        seed: Random seed used
        params: Additional parameters

    Returns:
        Experiment manifest dictionary
    """
    manifest = {
        "name": name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "numpy_version": np.__version__,
        "params": params or {},
        "reproducible": True,
    }

    # Add checksum for verification
    manifest_str = json.dumps(manifest, sort_keys=True)
    manifest["checksum"] = hashlib.sha256(manifest_str.encode()).hexdigest()[:16]

    return manifest


def save_experiment(manifest: dict[str, Any], output_dir: Path | str = "experiments") -> Path:
    """Save experiment manifest to disk.

    Args:
        manifest: Experiment manifest
        output_dir: Output directory

    Returns:
        Path to saved manifest
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{manifest['name']}_{manifest['timestamp'].replace(':', '-')}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return output_path


if __name__ == "__main__":
    # Set seeds on module load
    set_seed(RANDOM_SEED)

    # Verify reproducibility
    def sample_random():
        return [random.random() for _ in range(10)]

    is_reproducible = verify_reproducibility(sample_random)
    print(f"Reproducibility verified: {is_reproducible}")

    # Create sample manifest
    manifest = create_experiment_manifest(
        name="sample_experiment",
        seed=42,
        params={"learning_rate": 0.001, "batch_size": 32},
    )
    print(f"Manifest: {json.dumps(manifest, indent=2)}")
