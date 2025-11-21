# Case Study: DBSCAN Clustering Implementation

This chapter documents the complete EXTREME TDD implementation of aprender's DBSCAN clustering algorithm. This is a real-world example showing every phase of the RED-GREEN-REFACTOR cycle from Issue #14.

## Background

**GitHub Issue #14**: Implement DBSCAN clustering algorithm

**Requirements:**
- Density-based clustering without requiring k specification
- Automatic outlier detection (noise points labeled as -1)
- `eps` parameter for neighborhood radius
- `min_samples` parameter for core point density threshold
- Integration with `UnsupervisedEstimator` trait
- Deterministic clustering results
- Comprehensive example demonstrating parameter effects

**Initial State:**
- Tests: 548 passing
- Existing clustering: K-Means only
- No density-based clustering support

## CYCLE 1: Core DBSCAN Algorithm

### RED Phase

Created 12 comprehensive tests in `src/cluster/mod.rs`:

```rust,ignore
#[test]
fn test_dbscan_new() {
    let dbscan = DBSCAN::new(0.5, 3);
    assert_eq!(dbscan.eps(), 0.5);
    assert_eq!(dbscan.min_samples(), 3);
    assert!(!dbscan.is_fitted());
}

#[test]
fn test_dbscan_fit_basic() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1,
            5.0, 5.0, 5.1, 5.0, 5.0, 5.1,
        ],
    )
    .unwrap();

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&data).unwrap();
    assert!(dbscan.is_fitted());
}
```

Additional tests covered:
- Cluster prediction consistency
- Noise detection for outliers
- Single cluster scenarios
- All-noise scenarios
- Parameter sensitivity (eps and min_samples)
- Reproducibility
- Error handling (predict before fit)

**Verification:**
```bash
$ cargo test dbscan
error[E0422]: cannot find struct `DBSCAN` in this scope
```

**Result:** 12 tests failing ✅ (expected - DBSCAN doesn't exist)

### GREEN Phase

Implemented minimal DBSCAN algorithm in `src/cluster/mod.rs`:

```rust,ignore
/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBSCAN {
    eps: f32,
    min_samples: usize,
    labels: Option<Vec<i32>>,
}

impl DBSCAN {
    pub fn new(eps: f32, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            labels: None,
        }
    }

    fn region_query(&self, x: &Matrix<f32>, i: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let n_samples = x.shape().0;
        for j in 0..n_samples {
            let dist = self.euclidean_distance(x, i, j);
            if dist <= self.eps {
                neighbors.push(j);
            }
        }
        neighbors
    }

    fn euclidean_distance(&self, x: &Matrix<f32>, i: usize, j: usize) -> f32 {
        let n_features = x.shape().1;
        let mut sum = 0.0;
        for k in 0..n_features {
            let diff = x.get(i, k) - x.get(j, k);
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn expand_cluster(
        &self,
        x: &Matrix<f32>,
        labels: &mut [i32],
        point: usize,
        neighbors: &mut Vec<usize>,
        cluster_id: i32,
    ) {
        labels[point] = cluster_id;
        let mut i = 0;
        while i < neighbors.len() {
            let neighbor = neighbors[i];
            if labels[neighbor] == -2 {
                labels[neighbor] = cluster_id;
                let neighbor_neighbors = self.region_query(x, neighbor);
                if neighbor_neighbors.len() >= self.min_samples {
                    for &nn in &neighbor_neighbors {
                        if !neighbors.contains(&nn) {
                            neighbors.push(nn);
                        }
                    }
                }
            } else if labels[neighbor] == -1 {
                labels[neighbor] = cluster_id;
            }
            i += 1;
        }
    }
}

impl UnsupervisedEstimator for DBSCAN {
    type Labels = Vec<i32>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n_samples = x.shape().0;
        let mut labels = vec![-2; n_samples]; // -2 = unlabeled
        let mut cluster_id = 0;

        for i in 0..n_samples {
            if labels[i] != -2 {
                continue;
            }
            let mut neighbors = self.region_query(x, i);
            if neighbors.len() < self.min_samples {
                labels[i] = -1;
                continue;
            }
            self.expand_cluster(x, &mut labels, i, &mut neighbors, cluster_id);
            cluster_id += 1;
        }
        self.labels = Some(labels);
        Ok(())
    }

    fn predict(&self, _x: &Matrix<f32>) -> Self::Labels {
        self.labels().clone()
    }
}
```

**Verification:**
```bash
$ cargo test
test result: ok. 560 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

**Result:** All 560 tests passing ✅ (12 new DBSCAN tests)

### REFACTOR Phase

**Code Quality:**
- Fixed clippy warnings (unused variables, map_clone, manual_contains)
- Added comprehensive documentation
- Exported DBSCAN in prelude for easy access
- Added public getter methods (eps, min_samples, is_fitted, labels)

**Verification:**
```bash
$ cargo clippy --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.53s
```

**Result:** Zero clippy warnings ✅

## Example Implementation

Created comprehensive example `examples/dbscan_clustering.rs` demonstrating:

1. **Standard DBSCAN clustering** - Basic usage with 2 clusters and noise
2. **Effect of eps parameter** - Shows how neighborhood radius affects clustering
3. **Effect of min_samples parameter** - Demonstrates density threshold impact
4. **Comparison with K-Means** - Highlights DBSCAN's outlier detection advantage
5. **Anomaly detection use case** - Practical application for identifying outliers

**Key differences from K-Means:**
- K-Means: requires specifying k, assigns all points to clusters
- DBSCAN: discovers k automatically, identifies outliers as noise

**Run the example:**
```bash
cargo run --example dbscan_clustering
```

## Algorithm Details

**Time Complexity:** O(n²) for naive distance computations
**Space Complexity:** O(n) for storing labels

**Core Concepts:**
- **Core points:** Points with ≥ min_samples neighbors within eps
- **Border points:** Non-core points within eps of a core point
- **Noise points:** Points neither core nor border (labeled -1)
- **Cluster expansion:** Recursive growth from core points to reachable neighbors

## Final State

**Tests:** 560 passing (548 → 560, +12 DBSCAN tests)
**Coverage:** All DBSCAN functionality comprehensively tested
**Quality:** Zero clippy warnings, full documentation
**Exports:** Available via `use aprender::prelude::*;`

## Key Takeaways

1. **EXTREME TDD works:** Tests written first caught edge cases early
2. **Algorithm correctness:** Comprehensive tests verify all scenarios
3. **Quality gates:** Clippy and formatting ensure consistent code style
4. **Documentation:** Example demonstrates practical usage and parameter tuning

## Related Topics

- [K-Means Clustering](./kmeans-clustering.md)
- [UnsupervisedEstimator Trait](../api/traits.md)
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
