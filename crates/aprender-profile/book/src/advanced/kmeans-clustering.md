# K-Means Clustering

K-means clustering groups similar syscalls together based on timing patterns, helping identify behavioral patterns and performance clusters.

> **TDD-Verified:** K-means implementation tested in [`tests/sprint21_hpu_acceleration_tests.rs`](../../../tests/)

> **Parent Chapter:** See [HPU Acceleration](./hpu-acceleration.md) for overview

## Overview

**K-means** groups syscalls into K clusters based on features:
- **Duration clustering** - Fast/medium/slow groups
- **Behavioral patterns** - I/O-heavy vs CPU-heavy
- **Anomaly detection** - Outlier cluster identification

**Use cases:**
- Performance profiling (identify fast/slow groups)
- Workload characterization (I/O vs compute patterns)
- Anomaly isolation (outlier cluster = anomalies)

## Clustering Syscalls

### Method: Duration-Based Clustering

Group syscalls by execution time into 3 clusters (fast/medium/slow):

```python
#!/usr/bin/env python3
import json
import numpy as np
from sklearn.cluster import KMeans

with open('trace.json') as f:
    data = json.load(f)

# Extract features (duration only)
durations = np.array([[sc['duration_ns']] for sc in data['syscalls']])

# K-means clustering (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(durations)

# Analyze clusters
for i in range(3):
    cluster_durations = durations[labels == i]
    print(f"Cluster {i}:")
    print(f"  Count: {len(cluster_durations)}")
    print(f"  Mean: {np.mean(cluster_durations):.0f} ns")
    print(f"  Min: {np.min(cluster_durations):.0f} ns")
    print(f"  Max: {np.max(cluster_durations):.0f} ns")
```

**Example Output:**
```text
Cluster 0:  # Fast syscalls
  Count: 8500
  Mean: 1234 ns
  Min: 100 ns
  Max: 5000 ns

Cluster 1:  # Medium syscalls
  Count: 1200
  Mean: 12345 ns
  Min: 5001 ns
  Max: 50000 ns

Cluster 2:  # Slow syscalls (outliers!)
  Count: 300
  Mean: 125000 ns
  Min: 50001 ns
  Max: 500000 ns
```

**Analysis:** Cluster 2 contains slow outliers (anomalies!)

## Summary

K-means clustering provides:
- ✅ **Pattern discovery** - Identify fast/medium/slow groups
- ✅ **Anomaly isolation** - Outlier cluster = unusual behavior
- ✅ **Workload characterization** - Understand syscall patterns

**Workflow:** Export JSON → K-means clustering (scikit-learn) → Analyze clusters

**All clustering tested in:** [`tests/sprint21_hpu_acceleration_tests.rs`](../../../tests/)

## Related

- [HPU Acceleration](./hpu-acceleration.md) - Parent chapter
- [Correlation Matrix](./correlation-matrix.md) - Correlation analysis
- [Anomaly Detection](./anomaly-detection.md) - Anomaly detection workflows
