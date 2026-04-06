#!/usr/bin/env python3
"""
NumPy Data Processing Example - Input

Real-world data analysis scenario:
- Load and preprocess sensor data
- Compute statistical metrics
- Detect anomalies using threshold-based method
- Generate summary statistics

This demonstrates common NumPy operations that Batuta can transpile to Rust/Trueno.
"""

import numpy as np


def load_sensor_data():
    """Simulate loading sensor data (normally from file)."""
    # Simulated temperature readings (in Celsius) from 100 sensors
    np.random.seed(42)
    data = np.random.normal(loc=25.0, scale=5.0, size=100)
    return data


def preprocess_data(data):
    """Clean and normalize sensor data."""
    # Remove outliers (beyond 3 standard deviations)
    mean = np.mean(data)
    std = np.std(data)

    # Create mask for valid data points
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    mask = (data >= lower_bound) & (data <= upper_bound)

    # Filter data
    cleaned_data = data[mask]

    # Normalize to [0, 1]
    min_val = np.min(cleaned_data)
    max_val = np.max(cleaned_data)
    normalized = (cleaned_data - min_val) / (max_val - min_val)

    return normalized


def compute_statistics(data):
    """Compute statistical metrics for the data."""
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'sum': np.sum(data),
        'count': len(data)
    }
    return stats


def detect_anomalies(data, threshold=0.95):
    """Detect anomalies in normalized data."""
    # Find values above threshold
    anomalies = data[data > threshold]
    anomaly_indices = np.where(data > threshold)[0]

    return anomaly_indices, anomalies


def moving_average(data, window_size=5):
    """Compute moving average using convolution."""
    kernel = np.ones(window_size) / window_size
    # Use 'valid' mode to avoid edge effects
    smoothed = np.convolve(data, kernel, mode='valid')
    return smoothed


def main():
    """Main data processing pipeline."""
    print("NumPy Data Processing Pipeline")
    print("=" * 40)

    # Load data
    print("\n1. Loading sensor data...")
    raw_data = load_sensor_data()
    print(f"   Loaded {len(raw_data)} data points")

    # Preprocess
    print("\n2. Preprocessing data...")
    processed_data = preprocess_data(raw_data)
    print(f"   Cleaned: {len(processed_data)} valid points")

    # Statistics
    print("\n3. Computing statistics...")
    stats = compute_statistics(processed_data)
    print(f"   Mean: {stats['mean']:.4f}")
    print(f"   Std:  {stats['std']:.4f}")
    print(f"   Min:  {stats['min']:.4f}")
    print(f"   Max:  {stats['max']:.4f}")

    # Anomaly detection
    print("\n4. Detecting anomalies...")
    anomaly_idx, anomaly_vals = detect_anomalies(processed_data, threshold=0.95)
    print(f"   Found {len(anomaly_idx)} anomalies")
    if len(anomaly_vals) > 0:
        print(f"   Anomaly values: {anomaly_vals}")

    # Smoothing
    print("\n5. Applying moving average (window=5)...")
    smoothed = moving_average(processed_data, window_size=5)
    print(f"   Smoothed data length: {len(smoothed)}")
    print(f"   Smoothed mean: {np.mean(smoothed):.4f}")

    print("\n✅ Pipeline complete!")

    return processed_data, stats, anomaly_idx


if __name__ == "__main__":
    main()
