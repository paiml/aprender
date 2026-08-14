<!-- PCU: lib-cluster | contract: contracts/apr-page-lib-cluster-v1.yaml -->

# Module: `aprender::cluster`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/cluster/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/cluster/mod.rs) or directory.

## Example

```rust
use aprender::cluster::KMeans;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::cluster` is the unsupervised-clustering grab bag: classic K-Means
plus the algorithms you reach for when K-Means breaks down — DBSCAN for
density-based clustering with noise rejection, Gaussian mixture models when
clusters are anisotropic, agglomerative hierarchical clustering, spectral
clustering for non-convex shapes, and outlier detectors (Isolation Forest,
LOF). Every type implements [`UnsupervisedEstimator`](traits.md) so the
fit/predict surface is uniform.

## Key types

| Type | Description |
|------|-------------|
| `KMeans` | Lloyd's algorithm with k-means++ init and configurable `max_iter` / `random_state`. |
| `DBSCAN` | Density-based clustering; clusters labelled in `usize` with sentinel for noise. |
| `GaussianMixture` | EM-fitted GMM with `CovarianceType` (Full, Diag, Spherical, Tied). |
| `AgglomerativeClustering` | Bottom-up hierarchical clustering with selectable `Linkage` (Single, Complete, Average, Ward). |
| `SpectralClustering` | Eigen-decomposition based clustering with selectable `Affinity`. |
| `IsolationForest`, `LocalOutlierFactor` | Unsupervised anomaly detection. |

## Usage patterns

### Pattern 1: K-Means with deterministic seeding

```rust
use aprender::prelude::*;

let data = Matrix::from_vec(6, 2, vec![
    1.0, 1.0, 1.1, 0.9, 0.9, 1.1,    // cluster 1
    5.0, 5.0, 5.1, 4.9, 4.9, 5.1,    // cluster 2
]).expect("valid 6x2 matrix");

let mut kmeans = KMeans::new(2)
    .with_max_iter(100)
    .with_random_state(42);
kmeans.fit(&data).expect("kmeans fit");

let labels = kmeans.predict(&data);
assert_eq!(labels[0], labels[1], "cluster coherence");
assert_ne!(labels[0], labels[3], "cluster separation");
```

### Pattern 2: DBSCAN with noise detection

```rust
use aprender::cluster::DBSCAN;
use aprender::primitives::Matrix;
use aprender::traits::UnsupervisedEstimator;

let data = Matrix::from_vec(7, 2, vec![
    0.0, 0.0, 0.1, 0.1, 0.2, 0.0,    // dense cluster
    5.0, 5.0, 5.1, 5.1,              // second dense cluster
    100.0, 100.0,                    // noise point
]).expect("valid 7x2 matrix");

let mut dbscan = DBSCAN::new(0.5, 2);  // eps=0.5, min_samples=2
dbscan.fit(&data).expect("dbscan fit");
let labels = dbscan.predict(&data);

// Noise points get a sentinel label distinct from real clusters.
println!("labels: {:?}", labels);
```

## See also

- [`preprocessing`](preprocessing.md) — `StandardScaler` is almost always applied before clustering
- [`metrics`](metrics.md) — `silhouette_score`, `inertia` for evaluating cluster quality
- [`decomposition`](decomposition.md) — `ICA` / PCA-style dimensionality reduction before clustering
- [`models`](models.md) — when you need parametric mixture-of-experts instead of unsupervised partitioning

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
