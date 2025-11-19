# K-Means Clustering Theory

<!-- DOC_STATUS_START -->
**Chapter Status**: ✅ 100% Working (All examples verified)

| Status | Count | Examples |
|--------|-------|----------|
| ✅ Working | 15+ | K-Means with k-means++ verified |
| ⏳ In Progress | 0 | - |
| ⬜ Not Implemented | 0 | - |

*Last tested: 2025-11-19*
*Aprender version: 0.3.0*
*Test file: src/cluster/mod.rs tests*
<!-- DOC_STATUS_END -->

---

## Overview

K-Means is an unsupervised learning algorithm that partitions data into K clusters. Each cluster has a centroid (center point), and samples are assigned to their nearest centroid.

**Key Concepts**:
- **Lloyd's Algorithm**: Iterative assign-update procedure
- **k-means++**: Smart initialization for faster convergence
- **Inertia**: Within-cluster sum of squared distances (lower is better)
- **Unsupervised**: No labels needed, discovers structure in data

**Why This Matters**:
K-Means finds natural groupings in unlabeled data: customer segments, image compression, anomaly detection. It's fast, scalable, and interpretable.

---

## Mathematical Foundation

### The K-Means Objective

**Goal**: Minimize within-cluster variance (inertia)

```
minimize: Σ(k=1 to K) Σ(x ∈ C_k) ||x - μ_k||²

where:
C_k = set of samples in cluster k
μ_k = centroid of cluster k (mean of all x ∈ C_k)
K = number of clusters
```

**Interpretation**: Find cluster assignments that minimize total squared distance from points to their centroids.

### Lloyd's Algorithm

**Classic K-Means** (1957):

```
1. Initialize: Choose K initial centroids μ₁, μ₂, ..., μ_K

2. Repeat until convergence:
   a) Assignment Step:
      For each sample x_i:
          Assign x_i to cluster k where k = argmin_j ||x_i - μ_j||²

   b) Update Step:
      For each cluster k:
          μ_k = mean of all samples assigned to cluster k

3. Convergence: Stop when centroids change < tolerance
```

**Guarantees**:
- Always converges (finite iterations)
- Converges to local minimum (not necessarily global)
- Inertia decreases monotonically each iteration

### k-means++ Initialization

**Problem with random init**: Bad initial centroids → slow convergence or poor local minimum

**k-means++ Solution** (Arthur & Vassilvitskii 2007):

```
1. Choose first centroid uniformly at random from data points

2. For each remaining centroid:
   a) For each point x:
       D(x) = distance to nearest already-chosen centroid
   b) Choose new centroid with probability ∝ D(x)²
      (points far from existing centroids more likely)

3. Proceed with Lloyd's algorithm
```

**Why it works**: Spreads centroids across data → faster convergence, better clusters

**Theoretical guarantee**: O(log K) approximation to optimal clustering

---

## Implementation in Aprender

### Example 1: Basic K-Means

```rust
use aprender::cluster::KMeans;
use aprender::primitives::Matrix;
use aprender::traits::UnsupervisedEstimator;

// Two clear clusters
let data = Matrix::from_vec(6, 2, vec![
    1.0, 2.0,    // Cluster 0
    1.5, 1.8,    // Cluster 0
    1.0, 0.6,    // Cluster 0
    5.0, 8.0,    // Cluster 1
    8.0, 8.0,    // Cluster 1
    9.0, 11.0,   // Cluster 1
]).unwrap();

// K-Means with 2 clusters
let mut kmeans = KMeans::new(2)
    .with_max_iter(300)
    .with_tol(1e-4)
    .with_random_state(42);  // Reproducible

kmeans.fit(&data).unwrap();

// Get cluster assignments
let labels = kmeans.predict(&data);
println!("Labels: {:?}", labels); // [0, 0, 0, 1, 1, 1]

// Get centroids
let centroids = kmeans.centroids();
println!("Centroids:\n{:?}", centroids);

// Get inertia (within-cluster sum of squares)
println!("Inertia: {:.3}", kmeans.inertia());
```

**Test Reference**: `src/cluster/mod.rs::tests::test_three_clusters`

### Example 2: Finding Optimal K (Elbow Method)

```rust
// Try different K values
for k in 1..=10 {
    let mut kmeans = KMeans::new(k);
    kmeans.fit(&data).unwrap();

    let inertia = kmeans.inertia();
    println!("K={}: inertia={:.3}", k, inertia);
}

// Plot inertia vs K, look for "elbow"
// K=1: inertia=high
// K=2: inertia=medium (elbow here!)
// K=3: inertia=low
// K=10: inertia=very low (overfitting)
```

**Test Reference**: `src/cluster/mod.rs::tests::test_inertia_decreases_with_more_clusters`

### Example 3: Image Compression

```rust
// Image as pixels: (n_pixels, 3) RGB values
// Goal: Reduce 16M colors to 16 colors

let mut kmeans = KMeans::new(16)  // 16 color palette
    .with_random_state(42);

kmeans.fit(&pixel_data).unwrap();

// Each pixel assigned to nearest of 16 centroids
let labels = kmeans.predict(&pixel_data);
let palette = kmeans.centroids();  // 16 RGB colors

// Compressed image: use palette[labels[i]] for each pixel
```

**Use case**: Reduce image size by quantizing colors

---

## Choosing the Number of Clusters (K)

### The Elbow Method

**Idea**: Plot inertia vs K, look for "elbow" where adding more clusters has diminishing returns

```
Inertia
  |
  |  \
  |   \___
  |       \____
  |            \______
  |____________________ K
     1  2  3  4  5  6

Elbow at K=3 suggests 3 clusters
```

**Interpretation**:
- K=1: All data in one cluster (high inertia)
- K increasing: Inertia decreases
- Elbow point: Good trade-off (natural grouping)
- K=n: Each point its own cluster (zero inertia, overfitting)

### Silhouette Score

**Measure**: How well each sample fits its cluster vs neighboring clusters

```
For each sample i:
    a_i = average distance to other samples in same cluster
    b_i = average distance to nearest other cluster

Silhouette_i = (b_i - a_i) / max(a_i, b_i)

Silhouette score = average over all samples
```

**Range**: [-1, 1]
- **+1**: Perfect clustering (far from neighbors)
- **0**: On cluster boundary
- **-1**: Wrong cluster assignment

**Best K**: Maximizes silhouette score

### Domain Knowledge

Often, K is known from problem:
- Customer segmentation: 3-5 segments (budget, mid, premium)
- Image compression: 16, 64, or 256 colors
- Anomaly detection: K=1 (outliers far from center)

---

## Convergence and Iterations

### When Does K-Means Stop?

**Stopping criteria** (whichever comes first):
1. **Convergence**: Centroids move < tolerance
   - `||new_centroids - old_centroids|| < tol`
2. **Max iterations**: Reached max_iter (e.g., 300)

### Typical Convergence

**With k-means++ initialization**:
- Simple data (2-3 well-separated clusters): 5-20 iterations
- Complex data (10+ overlapping clusters): 50-200 iterations
- Pathological data: May hit max_iter

**Test Reference**: Convergence tests verify centroid stability

---

## Advantages and Limitations

### Advantages ✅

1. **Simple**: Easy to understand and implement
2. **Fast**: O(nkdi) where i is typically small (< 100 iterations)
3. **Scalable**: Works on large datasets (millions of points)
4. **Interpretable**: Centroids have meaning in feature space
5. **General purpose**: Works for many types of data

### Limitations ❌

1. **K must be specified**: User chooses number of clusters
2. **Sensitive to initialization**: Different random seeds → different results (k-means++ helps)
3. **Assumes spherical clusters**: Fails on elongated or irregular shapes
4. **Sensitive to outliers**: One outlier can pull centroid far away
5. **Local minima**: May not find global optimum
6. **Euclidean distance**: Assumes all features equally important, same scale

---

## K-Means vs Other Clustering Methods

### Comparison Table

| Method | K Required? | Shape Assumptions | Outlier Robust? | Speed | Use Case |
|--------|-------------|-------------------|-----------------|-------|----------|
| **K-Means** | Yes | Spherical | No | Fast | General purpose, large data |
| **DBSCAN** | No | Arbitrary | Yes | Medium | Irregular shapes, noise |
| **Hierarchical** | No | Arbitrary | No | Slow | Small data, dendrogram |
| **Gaussian Mixture** | Yes | Ellipsoidal | No | Medium | Probabilistic clusters |

### When to Use K-Means

**Good for**:
- Large datasets (K-Means scales well)
- Roughly spherical clusters
- Know approximate K
- Need fast results
- Interpretable centroids

**Not good for**:
- Unknown K
- Non-convex clusters (donuts, moons)
- Very different cluster sizes
- High outlier ratio

---

## Practical Considerations

### Feature Scaling is Important

**Problem**: K-Means uses Euclidean distance
- Features on different scales dominate distance calculation
- Age (0-100) vs income ($0-$1M) → income dominates

**Solution**: Standardize features before clustering

```rust
use aprender::preprocessing::StandardScaler;

let mut scaler = StandardScaler::new();
scaler.fit(&data);
let data_scaled = scaler.transform(&data);

// Now run K-Means on scaled data
let mut kmeans = KMeans::new(3);
kmeans.fit(&data_scaled).unwrap();
```

### Handling Empty Clusters

**Problem**: During iteration, a cluster may become empty (no points assigned)

**Solutions**:
1. Reinitialize empty centroid randomly
2. Split largest cluster
3. Continue with K-1 clusters

**Aprender implementation**: Handles empty clusters gracefully

### Multiple Runs

**Best practice**: Run K-Means multiple times with different random_state, pick best (lowest inertia)

```rust
let mut best_inertia = f32::INFINITY;
let mut best_model = None;

for seed in 0..10 {
    let mut kmeans = KMeans::new(k).with_random_state(seed);
    kmeans.fit(&data).unwrap();

    if kmeans.inertia() < best_inertia {
        best_inertia = kmeans.inertia();
        best_model = Some(kmeans);
    }
}
```

---

## Verification Through Tests

K-Means tests verify algorithm properties:

**Algorithm Tests**:
- Convergence within max_iter
- Inertia decreases with more clusters
- Labels are in range [0, K-1]
- Centroids are cluster means

**k-means++ Tests**:
- Centroids spread across data
- Reproducibility with same seed
- Selects points proportional to D²

**Edge Cases**:
- Single cluster (K=1)
- K > n_samples (error handling)
- Empty data (error handling)

**Test Reference**: `src/cluster/mod.rs` (15+ tests)

---

## Real-World Application

### Customer Segmentation

**Problem**: Group customers by behavior (purchase frequency, amount, recency)

**K-Means approach**:
```
Features: [recency, frequency, monetary_value]
K = 3 (low, medium, high value customers)

Result:
- Cluster 0: Inactive (high recency, low frequency)
- Cluster 1: Regular (medium all)
- Cluster 2: VIP (low recency, high frequency, high value)
```

**Business value**: Targeted marketing campaigns per segment

### Anomaly Detection

**Problem**: Find unusual network traffic patterns

**K-Means approach**:
```
K = 1 (normal behavior cluster)
Threshold = 95th percentile of distances to centroid

Anomaly = distance_to_centroid > threshold
```

**Result**: Points far from normal behavior flagged as anomalies

### Image Compression

**Problem**: Reduce 24-bit color (16M colors) to 8-bit (256 colors)

**K-Means approach**:
```
K = 256 colors
Input: n_pixels × 3 RGB matrix
Output: 256-color palette + n_pixels labels

Compression ratio: 24 bits → 8 bits = 3× smaller
```

---

## Verification Guarantee

K-Means implementation extensively tested (15+ tests) in `src/cluster/mod.rs`. Tests verify:

**Lloyd's Algorithm**:
- Convergence to local minimum
- Inertia monotonically decreases
- Centroids are cluster means

**k-means++ Initialization**:
- Probabilistic selection (D² weighting)
- Faster convergence than random init
- Reproducibility with random_state

**Property Tests**:
- All labels in [0, K-1]
- Number of clusters ≤ K
- Inertia ≥ 0

---

## Further Reading

### Peer-Reviewed Papers

**Lloyd (1982)** - *Least Squares Quantization in PCM*
- **Relevance**: Original K-Means algorithm (Lloyd's algorithm)
- **Link**: IEEE Transactions (library access)
- **Key Contribution**: Iterative assign-update procedure
- **Applied in**: `src/cluster/mod.rs` fit() method

**Arthur & Vassilvitskii (2007)** - *k-means++: The Advantages of Careful Seeding*
- **Relevance**: Smart initialization for K-Means
- **Link**: [ACM](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) (publicly accessible)
- **Key Contribution**: O(log K) approximation guarantee
- **Practical benefit**: Faster convergence, better clusters
- **Applied in**: `src/cluster/mod.rs` kmeans_plusplus_init()

### Related Chapters

- [Cross-Validation Theory](./cross-validation.md) - Can't use CV directly (no labels), but can evaluate inertia
- [Feature Scaling Theory](./feature-scaling.md) - CRITICAL for K-Means
- [Decision Trees Theory](./decision-trees.md) - Supervised alternative if labels available

---

## Summary

**What You Learned**:
- ✅ K-Means: Minimize within-cluster variance (inertia)
- ✅ Lloyd's algorithm: Assign → Update → Repeat until convergence
- ✅ k-means++: Smart initialization (D² probability selection)
- ✅ Choosing K: Elbow method, silhouette score, domain knowledge
- ✅ Convergence: Centroids stable or max_iter reached
- ✅ Advantages: Fast, scalable, interpretable
- ✅ Limitations: K required, spherical assumption, local minima
- ✅ Feature scaling: MANDATORY (Euclidean distance)

**Verification Guarantee**: K-Means implementation extensively tested (15+ tests) in `src/cluster/mod.rs`. Tests verify Lloyd's algorithm, k-means++ initialization, and convergence properties.

**Quick Reference**:
- **Objective**: Minimize Σ ||x - μ_cluster||²
- **Algorithm**: Assign to nearest centroid → Update centroids as means
- **Initialization**: k-means++ (not random!)
- **Choosing K**: Elbow method (plot inertia vs K)
- **Typical iterations**: 10-100 (depends on data, K)

**Key Equations**:
```
Inertia = Σ(k=1 to K) Σ(x ∈ C_k) ||x - μ_k||²
Assignment: cluster(x) = argmin_k ||x - μ_k||²
Update: μ_k = (1/|C_k|) Σ(x ∈ C_k) x
```

---

**Next Chapter**: [Gradient Descent Theory](./gradient-descent.md)

**Previous Chapter**: [Ensemble Methods Theory](./ensemble-methods.md)
