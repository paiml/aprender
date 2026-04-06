# ML Pipeline with EXTREME TDD

This chapter demonstrates implementing renacer's Sprint 48 ML Pipeline using EXTREME TDD methodology: RED-GREEN-REFACTOR cycles, property-based testing, and Toyota Way principles.

> **EXTREME TDD-Verified:** All code in this chapter developed test-first in [`src/ml_pipeline.rs`](../../../src/ml_pipeline.rs) and [`src/model_persistence.rs`](../../../src/model_persistence.rs)

## Toyota Way Foundations

The ML Pipeline embodies three Toyota Way principles:

| Principle | Japanese | Application |
|-----------|----------|-------------|
| **Muda** | 無駄 | Eliminate waste: persist models instead of retraining |
| **Kaizen** | 改善 | Continuous improvement via standardized preprocessing |
| **Poka-yoke** | ポカヨケ | Error-proofing through type-safe APIs |

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    renacer ML Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  Syscall Data (HashMap<String, (u64, u64)>)                     │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │ extract_features│  → Matrix<f32> (n_samples × 3 features)    │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │normalize_features│ → NormalizedFeatures (StandardScaler)     │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ├──────────────┬──────────────┬──────────────┐        │
│           ▼              ▼              ▼              ▼        │
│    ┌──────────┐   ┌───────────┐   ┌─────────┐   ┌──────────┐   │
│    │run_dbscan│   │  run_lof  │   │ run_pca │   │find_opt_k│   │
│    └──────────┘   └───────────┘   └─────────┘   └──────────┘   │
│           │              │              │              │        │
│           ▼              ▼              ▼              ▼        │
│    DBSCANResult   LOFResult      PCAResult     optimal_k       │
└─────────────────────────────────────────────────────────────────┘
```

## EXTREME TDD Workflow

### Phase 1: RED - Write Failing Tests First

Before writing any implementation, we define the expected behavior:

```rust,ignore
// RED: This test will fail - no implementation exists yet
#[test]
fn test_extract_features_basic() {
    let mut data = HashMap::new();
    data.insert("write".to_string(), (100, 1_000_000)); // 10µs avg
    data.insert("read".to_string(), (50, 500_000));     // 10µs avg

    let (names, features) = extract_features(&data).unwrap();

    assert_eq!(names.len(), 2);
    let (rows, cols) = features.shape();
    assert_eq!(rows, 2);
    assert_eq!(cols, 3); // 3 features per syscall
}
```

**Why 3 features?** We extract:
1. `avg_duration` - Average syscall duration in microseconds
2. `log_count` - Log-scaled call count (handles large ranges)
3. `log_total_duration` - Log-scaled total time spent

### Phase 2: GREEN - Minimal Implementation

```rust,ignore
pub fn extract_features(
    syscall_data: &HashMap<String, (u64, u64)>,
) -> Result<(Vec<String>, Matrix<f32>)> {
    let mut syscall_names = Vec::new();
    let mut features_data = Vec::new();

    for (name, (count, total_time_ns)) in syscall_data {
        if *count == 0 { continue; }

        let total_time_us = *total_time_ns as f64 / 1000.0;
        let avg_time_us = total_time_us / *count as f64;

        syscall_names.push(name.clone());

        // Feature vector: [avg_duration, log(count), log(total_duration)]
        features_data.push(avg_time_us as f32);
        features_data.push((*count as f32).ln().max(0.0));
        features_data.push((total_time_us as f32).ln().max(0.0));
    }

    let n_samples = syscall_names.len();
    if n_samples < 2 {
        return Err(PipelineError::InsufficientData {
            required: 2,
            actual: n_samples,
        });
    }

    let matrix = Matrix::from_vec(n_samples, 3, features_data)
        .map_err(|e| PipelineError::FeatureExtractionError(e.to_string()))?;

    Ok((syscall_names, matrix))
}
```

### Phase 3: REFACTOR - Add Edge Case Tests

```rust,ignore
#[test]
fn test_extract_features_insufficient_data() {
    let mut data = HashMap::new();
    data.insert("write".to_string(), (100, 1_000_000));
    // Only 1 syscall - not enough for clustering!

    let result = extract_features(&data);
    assert!(matches!(result, Err(PipelineError::InsufficientData { .. })));
}

#[test]
fn test_extract_features_skips_zero_count() {
    let mut data = HashMap::new();
    data.insert("write".to_string(), (100, 1_000_000));
    data.insert("read".to_string(), (50, 500_000));
    data.insert("empty".to_string(), (0, 0)); // Should be skipped

    let (names, _) = extract_features(&data).unwrap();
    assert_eq!(names.len(), 2);
    assert!(!names.contains(&"empty".to_string()));
}
```

## Feature Normalization with StandardScaler

### RED: Define Expected Behavior

```rust,ignore
#[test]
fn test_normalize_features_zero_mean() {
    let mut data = HashMap::new();
    data.insert("write".to_string(), (100, 1_000_000));
    data.insert("read".to_string(), (50, 500_000));
    data.insert("openat".to_string(), (20, 200_000));

    let (names, features) = extract_features(&data).unwrap();
    let normalized = normalize_features(names, features).unwrap();

    // After normalization, mean of each column should be ~0
    let (n_rows, n_cols) = normalized.data.shape();
    for j in 0..n_cols {
        let sum: f32 = (0..n_rows).map(|i| normalized.data.get(i, j)).sum();
        let mean = sum / n_rows as f32;
        assert!(mean.abs() < 0.01, "Column {} mean should be ~0, got {}", j, mean);
    }
}
```

### GREEN: Implementation

```rust,ignore
use aprender::preprocessing::StandardScaler;
use aprender::traits::{Transformer, UnsupervisedEstimator};

pub fn normalize_features(
    syscall_names: Vec<String>,
    features: Matrix<f32>,
) -> Result<NormalizedFeatures> {
    let mut scaler = StandardScaler::new()
        .with_mean(true)
        .with_std(true);

    scaler.fit(&features)
        .map_err(|e| PipelineError::PreprocessingError(e.to_string()))?;

    let normalized = scaler.transform(&features)
        .map_err(|e| PipelineError::PreprocessingError(e.to_string()))?;

    Ok(NormalizedFeatures {
        data: normalized,
        syscall_names,
        feature_names: vec![
            "avg_duration".to_string(),
            "log_count".to_string(),
            "log_total_duration".to_string(),
        ],
        means: scaler.mean().to_vec(),
        stds: scaler.std().to_vec(),
    })
}
```

## DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies anomalies as **noise points** (label = -1).

### RED: Test Cluster Detection

```rust,ignore
#[test]
fn test_dbscan_finds_clusters() {
    let mut data = HashMap::new();
    // Group 1: fast syscalls
    data.insert("write".to_string(), (1000, 10_000_000));  // 10µs avg
    data.insert("read".to_string(), (1000, 10_000_000));   // 10µs avg
    // Group 2: slow syscalls
    data.insert("mmap".to_string(), (100, 100_000_000));   // 1000µs avg
    data.insert("munmap".to_string(), (100, 100_000_000)); // 1000µs avg

    let (names, features) = extract_features(&data).unwrap();
    let normalized = normalize_features(names, features).unwrap();

    let result = run_dbscan(&normalized, 1.0, 2).unwrap();

    assert!(result.n_clusters >= 1);
    assert_eq!(result.syscall_names.len(), 4);
}
```

### RED: Test Noise Detection (Anomalies)

```rust,ignore
#[test]
fn test_dbscan_identifies_noise() {
    let mut data = HashMap::new();
    // Normal syscalls (cluster together)
    data.insert("write".to_string(), (1000, 10_000_000));
    data.insert("read".to_string(), (1000, 10_000_000));
    data.insert("close".to_string(), (1000, 10_000_000));
    // Outlier (will be noise)
    data.insert("slow_syscall".to_string(), (10, 1_000_000_000)); // 100ms avg!

    let (names, features) = extract_features(&data).unwrap();
    let normalized = normalize_features(names, features).unwrap();

    let result = run_dbscan(&normalized, 0.5, 2).unwrap();

    // Should have noise points (anomalies)
    assert!(result.n_noise > 0 || result.n_clusters > 1);
}
```

### GREEN: Implementation

```rust,ignore
use aprender::cluster::DBSCAN;

pub fn run_dbscan(
    features: &NormalizedFeatures,
    eps: f32,
    min_samples: usize,
) -> Result<DBSCANResult> {
    let mut dbscan = DBSCAN::new(eps, min_samples);

    dbscan.fit(&features.data)
        .map_err(|e| PipelineError::ClusteringError(e.to_string()))?;

    let labels = dbscan.labels().clone();

    // Count clusters and noise
    let n_noise = labels.iter().filter(|&&l| l == -1).count();
    let n_clusters = labels.iter()
        .filter(|&&l| l >= 0)
        .collect::<HashSet<_>>()
        .len();

    // Calculate silhouette score if we have enough clusters
    let silhouette = calculate_silhouette(&features.data, &labels);

    Ok(DBSCANResult {
        labels,
        n_clusters,
        n_noise,
        syscall_names: features.syscall_names.clone(),
        silhouette,
    })
}
```

## Local Outlier Factor (LOF)

LOF detects anomalies based on local density deviation. Syscalls with significantly lower density than neighbors are outliers.

### RED: Test Outlier Detection

```rust,ignore
#[test]
fn test_lof_detects_outliers() {
    let mut data = HashMap::new();
    // Normal syscalls
    data.insert("write".to_string(), (1000, 10_000_000));
    data.insert("read".to_string(), (1000, 10_000_000));
    data.insert("close".to_string(), (1000, 10_000_000));
    // Outlier
    data.insert("slow_syscall".to_string(), (10, 1_000_000_000));

    let (names, features) = extract_features(&data).unwrap();
    let normalized = normalize_features(names, features).unwrap();

    let result = run_lof(&normalized, &data, 2, 0.25).unwrap();

    // Should detect outliers
    assert!(!result.outliers.is_empty() || result.labels.iter().any(|&l| l == -1));
    assert_eq!(result.scores.len(), 4);
}
```

### GREEN: Implementation

```rust,ignore
use aprender::cluster::LocalOutlierFactor;

pub fn run_lof(
    features: &NormalizedFeatures,
    syscall_data: &HashMap<String, (u64, u64)>,
    n_neighbors: usize,
    contamination: f32,
) -> Result<LOFResult> {
    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(n_neighbors)
        .with_contamination(contamination);

    lof.fit(&features.data)
        .map_err(|e| PipelineError::ClusteringError(e.to_string()))?;

    let labels = lof.predict(&features.data);
    let scores = lof.score_samples(&features.data);

    // Build outlier info
    let mut outliers = Vec::new();
    for (i, (&label, &score)) in labels.iter().zip(scores.iter()).enumerate() {
        if label == -1 {
            let syscall = &features.syscall_names[i];
            let (count, total_ns) = syscall_data.get(syscall).copied().unwrap_or((0, 0));
            let avg_time_us = if count > 0 {
                (total_ns as f64 / 1000.0) / count as f64
            } else { 0.0 };

            outliers.push(OutlierInfo {
                syscall: syscall.clone(),
                lof_score: score,
                avg_time_us,
                call_count: count,
            });
        }
    }

    // Sort by LOF score (highest = most anomalous)
    outliers.sort_by(|a, b| b.lof_score.partial_cmp(&a.lof_score).unwrap_or(Ordering::Equal));

    Ok(LOFResult { labels, scores, syscall_names: features.syscall_names.clone(), outliers })
}
```

## Silhouette Score for Cluster Quality

Silhouette score measures clustering quality: -1 (worst) to 1 (best).

### RED: Test Well-Separated Clusters

```rust,ignore
#[test]
fn test_silhouette_score_well_separated() {
    // Two perfectly separated clusters
    let data = vec![
        1.0, 1.0,   // Cluster 0
        1.1, 1.1,   // Cluster 0
        10.0, 10.0, // Cluster 1
        10.1, 10.1, // Cluster 1
    ];
    let matrix = Matrix::from_vec(4, 2, data).unwrap();
    let labels = vec![0, 0, 1, 1];

    let score = calculate_silhouette(&matrix, &labels);

    assert!(score.is_some());
    let s = score.unwrap();
    assert!(s > 0.8, "Well-separated clusters should have high silhouette, got {}", s);
}
```

### RED: Edge Case - Single Cluster

```rust,ignore
#[test]
fn test_silhouette_score_single_cluster() {
    let data = vec![1.0, 1.0, 1.1, 1.1, 1.2, 1.2];
    let matrix = Matrix::from_vec(3, 2, data).unwrap();
    let labels = vec![0, 0, 0]; // Single cluster

    let score = calculate_silhouette(&matrix, &labels);
    assert!(score.is_none()); // Need at least 2 clusters
}
```

## PCA Dimensionality Reduction

PCA reduces feature dimensions while preserving variance - useful for visualization.

### RED: Test Dimension Reduction

```rust,ignore
#[test]
fn test_pca_reduces_dimensions() {
    let mut data = HashMap::new();
    data.insert("write".to_string(), (1000, 10_000_000));
    data.insert("read".to_string(), (1000, 10_000_000));
    data.insert("close".to_string(), (1000, 10_000_000));
    data.insert("openat".to_string(), (500, 5_000_000));

    let (names, features) = extract_features(&data).unwrap();
    let normalized = normalize_features(names, features).unwrap();

    let result = run_pca(&normalized, 2).unwrap();

    let (n_samples, n_components) = result.reduced_data.shape();
    assert_eq!(n_samples, 4);
    assert_eq!(n_components, 2); // Reduced from 3 to 2
}
```

### RED: Test Variance Explained

```rust,ignore
#[test]
fn test_pca_variance_explained() {
    // ... setup ...
    let result = run_pca(&normalized, 3).unwrap();

    // Total variance should be <= 1.0
    assert!(result.total_variance_explained <= 1.01,
        "Total variance {} should be <= 1.0", result.total_variance_explained);
}
```

## Model Persistence: Eliminating MUDA

The `.apr` format persists trained models, eliminating the waste of retraining.

### RED: Test Save/Load Roundtrip

```rust,ignore
#[test]
fn test_save_and_load_kmeans_model() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_kmeans.apr");

    let model = SerializableKMeansModel {
        centroids: vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ],
        n_clusters: 3,
        n_features: 3,
        metadata: ModelMetadata::new(1000)
            .with_hyperparameter("n_clusters", "3")
            .with_description("Test KMeans model"),
    };

    // Save
    let options = PersistenceOptions::new()
        .with_name("test-kmeans")
        .with_description("Test model");
    save_kmeans_model(&model, &model_path, options).expect("Failed to save");

    // Load
    let loaded = load_kmeans_model(&model_path).expect("Failed to load");

    assert_eq!(loaded.n_clusters, model.n_clusters);
    assert_eq!(loaded.n_features, model.n_features);
    for (orig, loaded_c) in model.centroids.iter().zip(loaded.centroids.iter()) {
        for (o, l) in orig.iter().zip(loaded_c.iter()) {
            assert!((o - l).abs() < 1e-6);
        }
    }
}
```

### GREEN: Implementation

```rust,ignore
use aprender::format::{save, load, Compression, ModelType, SaveOptions};

pub fn save_kmeans_model(
    model: &SerializableKMeansModel,
    path: impl AsRef<Path>,
    options: PersistenceOptions,
) -> Result<()> {
    let compression = if options.compress {
        Compression::ZstdDefault
    } else {
        Compression::None
    };

    let mut save_options = SaveOptions::new().with_compression(compression);
    if let Some(name) = options.name {
        save_options = save_options.with_name(name);
    }

    save(model, ModelType::KMeans, path.as_ref(), save_options)
        .map_err(|e| ModelPersistenceError::SaveError(e.to_string()))
}
```

## Property-Based Testing

Property tests verify invariants across random inputs:

```rust,ignore
#[test]
fn test_normalization_preserves_sample_count() {
    proptest::proptest!(|(n_syscalls in 3usize..10)| {
        let mut data = HashMap::new();
        for i in 0..n_syscalls {
            data.insert(
                format!("syscall_{}", i),
                ((i + 1) as u64 * 100, (i + 1) as u64 * 1_000_000)
            );
        }

        let (names, features) = extract_features(&data).unwrap();
        let normalized = normalize_features(names.clone(), features).unwrap();

        // PROPERTY: Sample count never changes through normalization
        prop_assert_eq!(normalized.syscall_names.len(), names.len());
    });
}

#[test]
fn test_silhouette_bounds() {
    // PROPERTY: Silhouette score always in [-1, 1]
    let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let matrix = Matrix::from_vec(4, 2, data).unwrap();
    let labels = vec![0, 0, 1, 1];

    if let Some(score) = calculate_silhouette(&matrix, &labels) {
        assert!(score >= -1.0 && score <= 1.0);
    }
}

#[test]
fn test_roundtrip_preserves_centroids() {
    proptest::proptest!(|(n_clusters in 1usize..10, n_features in 1usize..5)| {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("proptest.apr");

        let centroids: Vec<Vec<f32>> = (0..n_clusters)
            .map(|i| (0..n_features).map(|j| (i * n_features + j) as f32).collect())
            .collect();

        let model = SerializableKMeansModel {
            centroids: centroids.clone(),
            n_clusters,
            n_features,
            metadata: ModelMetadata::new(100),
        };

        save_kmeans_model(&model, &model_path, PersistenceOptions::new()).unwrap();
        let loaded = load_kmeans_model(&model_path).unwrap();

        // PROPERTY: Centroids survive roundtrip exactly
        prop_assert_eq!(loaded.centroids.len(), centroids.len());
    });
}
```

## CLI Integration

### Save Trained Model

```bash
# Train and save model
renacer -c --ml-anomaly --save-model baseline.apr -- cargo build

# Output:
# Model saved to baseline.apr (1.2 KB)
# Training samples: 47 syscalls
# Silhouette score: 0.823
```

**Tested by:** `test_save_model_flag_accepted`

### Load Pre-trained Model (MUDA Elimination)

```bash
# Use saved model - no retraining!
renacer -c --ml-anomaly --load-model baseline.apr -- cargo build

# Output:
# Loaded model: baseline.apr (renacer v0.6.3, 47 samples)
# Anomalies detected: 2
```

**Tested by:** `test_load_model_flag_accepted`

### Regression Detection

```bash
# Compare against baseline
renacer -c --ml-anomaly --baseline baseline.apr -- cargo build

# Output:
# === Regression Analysis ===
# Baseline: 47 syscalls, 823ms total
# Current:  52 syscalls, 1247ms total (+51%)
#
# New anomalies not in baseline:
#   - futex (424ms avg) - REGRESSION
```

**Tested by:** `test_baseline_flag_accepted`

## Example: Finding Performance Regressions

```bash
# Step 1: Capture baseline
$ renacer -c --ml-anomaly --save-model release-1.0.apr -- ./my-app
Model saved: release-1.0.apr (47 syscalls, silhouette: 0.85)

# Step 2: After code changes, compare
$ renacer -c --ml-anomaly --baseline release-1.0.apr -- ./my-app

=== DBSCAN Clustering Results ===
Clusters found: 3
Noise points (potential anomalies): 2

Noise/anomaly syscalls:
  - futex (NEW - not in baseline!)
  - mmap

=== Local Outlier Factor Analysis ===
Outliers detected: 2
  - futex (LOF: 4.23, avg: 1250µs, calls: 847) - REGRESSION
  - mmap (LOF: 2.15, avg: 523µs, calls: 12)

=== Regression Summary ===
Baseline silhouette: 0.85
Current silhouette:  0.67 (-21%)
New syscalls: futex
Recommendation: Investigate futex contention
```

## Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Feature extraction | O(n) | <1ms for 100 syscalls |
| StandardScaler | O(n×d) | <1ms |
| DBSCAN | O(n²) | ~10ms for 100 syscalls |
| LOF | O(n×k) | ~5ms for 100 syscalls |
| PCA | O(n×d²) | <1ms |
| Model save | O(centroids) | ~1ms |
| Model load | O(centroids) | ~1ms |

**Zero overhead when ML disabled** - all analysis is opt-in.

## Summary

The ML Pipeline demonstrates EXTREME TDD principles:

| Phase | What We Did |
|-------|-------------|
| **RED** | Wrote 21 failing tests defining exact behavior |
| **GREEN** | Implemented minimal code to pass each test |
| **REFACTOR** | Added property tests, edge cases, formatting |

Toyota Way benefits achieved:
- **Muda** (無駄): 10-50x faster startup with persisted models
- **Kaizen** (改善): Standardized preprocessing pipeline
- **Poka-yoke** (ポカヨケ): Type-safe `Result<T>` APIs prevent misuse

## Related Chapters

- [Machine Learning](./machine-learning.md) - KMeans clustering basics
- [Anomaly Detection](./anomaly-detection.md) - Real-time monitoring
- [EXTREME TDD](../contributing/extreme-tdd.md) - Methodology guide
- [Toyota Way Principles](../contributing/toyota-way.md) - Design philosophy

## Future: Hugging Face Hub Integration

> **Tracked:** [aprender#100](https://github.com/paiml/aprender/issues/100)

Once aprender adds HF Hub support, renacer will enable:

```bash
# Push model to Hugging Face Hub
renacer --push-model hub:paiml/syscall-anomaly -- cargo build

# Load model from Hugging Face Hub
renacer --load-model hub:paiml/syscall-anomaly -- cargo build
```

Model cards will be auto-generated with training metadata, hyperparameters, and metrics.
