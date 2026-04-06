# Machine Learning Anomaly Detection

Renacer integrates machine learning-based anomaly detection using the [Aprender](https://github.com/paiml/aprender) library to automatically identify unusual syscall patterns through KMeans clustering.

> **TDD-Verified:** All examples validated by [`tests/sprint23_ml_anomaly_tests.rs`](../../../tests/sprint23_ml_anomaly_tests.rs)

## Overview

ML-based anomaly detection complements traditional statistical methods (z-score) by:
- **Pattern Recognition:** Grouping syscalls by latency similarity
- **Unsupervised Learning:** No pre-labeled data required
- **Cluster Analysis:** Automatic identification of outlier groups
- **Quality Metrics:** Silhouette scoring for clustering validation

## Basic Usage

### Enable ML Anomaly Detection

```bash
renacer -c --ml-anomaly -- cargo build
```

**Tested by:** `test_ml_anomaly_flag_accepted`

This enables KMeans clustering with default 3 clusters, analyzing syscall latency patterns.

### ML Analysis Output

```bash
$ renacer -c --ml-anomaly -- ./my-app
```

**Example Output:**
```text
=== ML Anomaly Detection Report ===
Clusters: 3
Samples: 5
Silhouette Score: 0.823

Cluster Centers (avg time in μs):
  Cluster 0: 10.50 μs
  Cluster 1: 100.23 μs
  Cluster 2: 1205.67 μs

Anomalies Detected: 2
  - fsync (cluster 2): 1205.67 μs (distance: 23.45)
  - write (cluster 2): 1198.34 μs (distance: 18.12)
```

**Tested by:** `test_ml_anomaly_produces_cluster_output`, `test_ml_silhouette_score_output`

## Configuration

### Custom Cluster Count

```bash
renacer -c --ml-anomaly --ml-clusters 5 -- ./heavy-io-app
```

**Tested by:** `test_ml_clusters_configuration`

- **Default:** 3 clusters
- **Minimum:** 2 clusters (enforced)
- **Maximum:** Number of unique syscalls

**Invalid cluster counts:**
```bash
# This will fail (< 2)
renacer --ml-anomaly --ml-clusters 1 -- true
```

**Tested by:** `test_ml_clusters_invalid_value`, `test_ml_clusters_minimum_value`

### ML vs Z-Score Comparison

Compare ML-based detection with statistical z-score methods:

```bash
renacer -c --ml-anomaly --ml-compare -- ./app
```

**Tested by:** `test_ml_compare_with_zscore`

**Example Output:**
```text
=== ML vs Z-Score Comparison ===
Common anomalies: 3
ML-only anomalies: ["mmap", "mremap"]
Z-score-only anomalies: ["brk"]
```

This reveals:
- **Common:** Both methods agree (high confidence)
- **ML-only:** Pattern-based anomalies (correlated latencies)
- **Z-score-only:** Statistical outliers (single extreme values)

## Integration with Other Features

### ML with Statistics Mode

```bash
renacer -c --ml-anomaly -T -- cargo test
```

**Tested by:** `test_ml_anomaly_with_statistics`

Combines:
- `-c`: Syscall statistics table
- `--ml-anomaly`: Cluster analysis
- `-T`: Microsecond timing

### ML with Filtering

```bash
renacer --ml-anomaly -e trace=write -T -- ./app
```

**Tested by:** `test_ml_anomaly_with_filtering`

Only analyzes **filtered** syscalls (e.g., `write` operations only).

### ML with Multi-Process Tracing

```bash
renacer -f --ml-anomaly -T -- make -j8
```

**Tested by:** `test_ml_anomaly_with_multiprocess`

Analyzes syscalls from **all** traced processes (parent + children) in aggregate.

### ML with JSON Output

```bash
renacer --ml-anomaly --format json -- ./app > ml_analysis.json
```

**Tested by:** `test_ml_anomaly_with_json_output`

JSON includes `ml_analysis` field:
```json
{
  "ml_analysis": {
    "clusters": 3,
    "silhouette_score": 0.823,
    "anomalies": [
      {
        "syscall": "fsync",
        "cluster": 2,
        "avg_time_us": 1205.67,
        "distance": 23.45
      }
    ]
  }
}
```

### ML with Real-Time Anomaly Detection

```bash
renacer --ml-anomaly --anomaly-realtime -T -- ./app
```

**Tested by:** `test_ml_anomaly_with_realtime`

Combines:
- **ML:** Post-hoc cluster analysis
- **Real-time:** Live z-score monitoring

Use for: Hybrid detection (statistical + pattern-based).

## Edge Cases & Error Handling

### Insufficient Data

With too few syscalls (<3 types):

```bash
$ renacer --ml-anomaly -e trace=write -T -- echo "test"
```

**Output:**
```text
=== ML Anomaly Detection Report ===
Insufficient data for ML analysis
(Need at least 3 syscall types, found 2)
```

**Tested by:** `test_ml_anomaly_insufficient_data`

The system gracefully handles:
- **< 2 samples:** Cannot cluster (returns empty report)
- **2-3 samples:** Clusters with k=2
- **≥ 3 samples:** Uses requested cluster count

### Backward Compatibility

Without `--ml-anomaly`, **no ML overhead** occurs:

```bash
$ renacer -c -T -- ./app
# ML analysis NOT performed, output shows only statistics
```

**Tested by:** `test_backward_compatibility_without_ml_anomaly`

This ensures:
- **Zero performance impact** when disabled
- **Opt-in only** design
- **No surprise behavior** for existing users

## How It Works

### KMeans Clustering Algorithm

1. **Feature Extraction:** Average latency per syscall type
2. **Clustering:** Group syscalls by latency similarity (Aprender KMeans)
3. **Outlier Detection:** Identify syscalls in high-latency clusters
4. **Quality Scoring:** Silhouette coefficient (-1 to 1, higher = better separation)

### Anomaly Identification

Syscalls are flagged as anomalous if:
- In cluster with center > 50% of maximum cluster center
- In highest-latency cluster (potential bottlenecks)

### When to Use ML vs Z-Score

| Method | Best For | Limitations |
|--------|----------|-------------|
| **Z-Score** (Sprint 20) | Single extreme outliers | Misses correlated patterns |
| **ML** (Sprint 23) | Pattern-based anomalies | Requires multiple samples |
| **Both** (`--ml-compare`) | Comprehensive analysis | Slower analysis |

## Practical Examples

### Example 1: Database Application

```bash
$ renacer -c --ml-anomaly -T -e trace=file -- pg_bench
```

**Output:**
```text
Clusters: 3
  Cluster 0: Fast reads (10-50 μs)
  Cluster 1: Normal writes (100-500 μs)
  Cluster 2: SLOW fsyncs (5000+ μs) ⚠️ ANOMALY

Anomalies: fsync operations in cluster 2
```

**Action:** Investigate fsync configuration (disable for testing, enable WAL).

**Tested by:** `test_ml_detects_outlier_cluster`

### Example 2: Network Service

```bash
$ renacer -c --ml-anomaly -e trace=network -- ./http_server
```

**Output:**
```text
Clusters: 2
  Cluster 0: Fast sendto (20-100 μs)
  Cluster 1: Slow recvfrom (500+ μs) ⚠️ ANOMALY
```

**Action:** Check network latency, client behavior.

**Tested by:** `test_ml_multiple_syscall_types`

## Troubleshooting

### "Insufficient data for ML analysis"

**Cause:** Too few syscall types (< 3) in trace.

**Solutions:**
1. Remove filters: `renacer --ml-anomaly -T -- ./app` (trace all syscalls)
2. Run longer workload to generate more syscalls
3. Use z-score instead: `renacer --anomaly-realtime -T -- ./app`

### Low Silhouette Score (< 0.3)

**Meaning:** Clusters are poorly separated (overlapping latencies).

**Solutions:**
1. Increase cluster count: `--ml-clusters 5`
2. Filter specific syscalls: `-e trace=file` (analyze specific subsystem)
3. Collect more samples (longer trace)

### No Anomalies Detected

**Meaning:** All syscalls have similar latency patterns (good!).

**Possible Reasons:**
1. Application is well-optimized
2. Trace too short to capture anomalies
3. Workload doesn't stress I/O

**Verification:** Compare with `--ml-compare` to check z-score agreement.

## Performance

- **Overhead:** <1% when enabled (post-processing only)
- **Memory:** ~O(n) where n = unique syscall types
- **Speed:** KMeans converges in <10 iterations typically

**Zero overhead when disabled** (not enabled by default).

## Summary

ML anomaly detection provides:
- ✅ **Pattern-based** anomaly identification
- ✅ **Unsupervised** learning (no training data)
- ✅ **Cluster visualization** of syscall latency groups
- ✅ **Quality metrics** via silhouette scoring
- ✅ **Complementary** to z-score methods

**All examples tested in:** [`tests/sprint23_ml_anomaly_tests.rs`](../../../tests/sprint23_ml_anomaly_tests.rs)

## Related

- [Statistical Analysis](./statistical-analysis.md) - Z-score based detection
- [Anomaly Detection](./anomaly-detection.md) - Real-time monitoring
- [HPU Acceleration](./hpu-acceleration.md) - GPU-accelerated clustering
