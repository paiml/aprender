# Case Study: Hierarchical Clustering Implementation

This chapter documents the complete EXTREME TDD implementation of aprender's Agglomerative Hierarchical Clustering algorithm. This is a real-world example showing every phase of the RED-GREEN-REFACTOR cycle from Issue #15.

## Background

**GitHub Issue #15**: Implement Hierarchical Clustering (Agglomerative)

**Requirements:**
- Bottom-up agglomerative clustering algorithm
- Four linkage methods: Single, Complete, Average, Ward
- Dendrogram construction for visualization
- Integration with `UnsupervisedEstimator` trait
- Deterministic clustering results
- Comprehensive example demonstrating linkage effects

**Initial State:**
- Tests: 560 passing
- Existing clustering: K-Means, DBSCAN
- No hierarchical clustering support

## CYCLE 1: Core Agglomerative Algorithm

### RED Phase

Created 18 comprehensive tests in `src/cluster/mod.rs`:

```rust,ignore
#[test]
fn test_agglomerative_new() {
    let hc = AgglomerativeClustering::new(3, Linkage::Average);
    assert_eq!(hc.n_clusters(), 3);
    assert_eq!(hc.linkage(), Linkage::Average);
    assert!(!hc.is_fitted());
}

#[test]
fn test_agglomerative_fit_basic() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .unwrap();

    let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
    hc.fit(&data).unwrap();
    assert!(hc.is_fitted());
}
```

Additional tests covered:
- All 4 linkage methods (Single, Complete, Average, Ward)
- n_clusters variations (1, equals samples)
- Dendrogram structure validation
- Reproducibility
- Fit-predict consistency
- Different linkages produce different results
- Well-separated clusters
- Error handling (3 panic tests for calling methods before fit)

**Verification:**
```bash
$ cargo test agglomerative
error[E0599]: no function or associated item named `new` found
```

**Result:** 18 tests failing ✅ (expected - AgglomerativeClustering doesn't exist)

### GREEN Phase

Implemented agglomerative clustering algorithm in `src/cluster/mod.rs`:

**1. Linkage Enum and Merge Structure:**
```rust,ignore
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Linkage {
    Single,   // Minimum distance
    Complete, // Maximum distance
    Average,  // Mean distance
    Ward,     // Minimize variance
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Merge {
    pub clusters: (usize, usize),
    pub distance: f32,
    pub size: usize,
}
```

**2. Main Algorithm:**
```rust,ignore
impl AgglomerativeClustering {
    pub fn new(n_clusters: usize, linkage: Linkage) -> Self {
        Self {
            n_clusters,
            linkage,
            labels: None,
            dendrogram: None,
        }
    }

    fn pairwise_distances(&self, x: &Matrix<f32>) -> Vec<Vec<f32>> {
        // Calculate all pairwise Euclidean distances
        let n_samples = x.shape().0;
        let mut distances = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = self.euclidean_distance(x, i, j);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        distances
    }

    fn find_closest_clusters(
        &self,
        distances: &[Vec<f32>],
        active: &[bool],
    ) -> (usize, usize, f32) {
        // Find minimum distance between active clusters
        // ...
    }

    fn update_distances(
        &self,
        x: &Matrix<f32>,
        distances: &mut [Vec<f32>],
        clusters: &[Vec<usize>],
        merged_idx: usize,
        other_idx: usize,
    ) {
        // Update distances based on linkage method
        let dist = match self.linkage {
            Linkage::Single => { /* minimum distance */ },
            Linkage::Complete => { /* maximum distance */ },
            Linkage::Average => { /* average distance */ },
            Linkage::Ward => { /* Ward's method */ },
        };
        // ...
    }
}
```

**3. UnsupervisedEstimator Implementation:**
```rust,ignore
impl UnsupervisedEstimator for AgglomerativeClustering {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n_samples = x.shape().0;

        // Initialize: each point is its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        let mut active = vec![true; n_samples];
        let mut dendrogram = Vec::new();

        // Calculate initial distances
        let mut distances = self.pairwise_distances(x);

        // Merge until reaching target number of clusters
        while clusters.iter().filter(|c| !c.is_empty()).count() > self.n_clusters {
            // Find closest pair
            let (i, j, dist) = self.find_closest_clusters(&distances, &active);

            // Merge clusters
            clusters[i].extend(&clusters[j]);
            clusters[j].clear();
            active[j] = false;

            // Record merge
            dendrogram.push(Merge {
                clusters: (i, j),
                distance: dist,
                size: clusters[i].len(),
            });

            // Update distances
            for k in 0..n_samples {
                if k != i && active[k] {
                    self.update_distances(x, &mut distances, &clusters, i, k);
                }
            }
        }

        // Assign final labels
        // ...
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
test result: ok. 577 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

**Result:** All 577 tests passing ✅ (18 new hierarchical clustering tests)

### REFACTOR Phase

**Code Quality:**
- Fixed clippy warnings with `#[allow(clippy::needless_range_loop)]` where index loops are clearer
- Added comprehensive documentation for all methods
- Exported `AgglomerativeClustering` and `Linkage` in prelude
- Added public getter methods (n_clusters, linkage, is_fitted, labels, dendrogram)

**Verification:**
```bash
$ cargo clippy --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.89s
```

**Result:** Zero clippy warnings ✅

## Example Implementation

Created comprehensive example `examples/hierarchical_clustering.rs` demonstrating:

1. **Average linkage clustering** - Standard usage with 3 natural clusters
2. **Dendrogram visualization** - Shows merge history with distances
3. **All four linkage methods** - Compares Single, Complete, Average, Ward
4. **Effect of n_clusters** - Shows 2, 5, and 9 clusters
5. **Practical use cases** - Taxonomy building, customer segmentation
6. **Reproducibility** - Demonstrates deterministic results

**Linkage Method Characteristics:**
- **Single**: Minimum distance between clusters (chain-like clusters)
- **Complete**: Maximum distance between clusters (compact clusters)
- **Average**: Mean distance between all pairs (balanced)
- **Ward**: Minimize within-cluster variance (variance-based)

**Run the example:**
```bash
cargo run --example hierarchical_clustering
```

## Algorithm Details

**Time Complexity:** O(n³) for naive implementation
**Space Complexity:** O(n²) for distance matrix

**Core Algorithm Steps:**
1. Initialize each point as its own cluster
2. Calculate pairwise distances
3. Repeat until reaching n_clusters:
   - Find closest pair of clusters
   - Merge them
   - Update distance matrix using linkage method
   - Record merge in dendrogram
4. Assign final cluster labels

**Linkage Distance Calculations:**
- **Single:** d(A,B) = min{d(a,b) : a ∈ A, b ∈ B}
- **Complete:** d(A,B) = max{d(a,b) : a ∈ A, b ∈ B}
- **Average:** d(A,B) = mean{d(a,b) : a ∈ A, b ∈ B}
- **Ward:** d(A,B) = sqrt((|A||B|)/(|A|+|B|)) * ||centroid(A) - centroid(B)||

## Final State

**Tests:** 577 passing (560 → 577, +17 hierarchical clustering tests)
**Coverage:** All AgglomerativeClustering functionality comprehensively tested
**Quality:** Zero clippy warnings, full documentation
**Exports:** Available via `use aprender::prelude::*;`

## Key Takeaways

1. **Hierarchical clustering advantages:**
   - No need to pre-specify exact number of clusters
   - Dendrogram provides hierarchy of relationships
   - Can examine merge history to choose optimal cut point
   - Deterministic results

2. **Linkage method selection:**
   - Single: best for irregular cluster shapes (chain effect)
   - Complete: best for compact, spherical clusters
   - Average: balanced general-purpose choice
   - Ward: best when minimizing variance is important

3. **EXTREME TDD benefits:**
   - Tests for all 4 linkage methods caught edge cases
   - Dendrogram structure tests ensured correct merge tracking
   - Comprehensive testing verified algorithm correctness

## Related Topics

- [DBSCAN Clustering](./dbscan-clustering.md)
- [K-Means Clustering](./kmeans-clustering.md)
- [UnsupervisedEstimator Trait](../api/traits.md)
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
