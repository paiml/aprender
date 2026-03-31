# Graph and Traditional Descriptive Statistics Specification

**Version**: 1.1.0 (Toyota Way Review Incorporated)
**Status**: Draft - Revised
**Last Updated**: 2025-11-19
**Authors**: Pragmatic AI Labs Team, Claude (AI pair programmer)
**Review Status**: Toyota Way code review incorporated (Muda elimination, Poka-Yoke, Kaizen)

## Abstract

This specification defines the architecture, implementation requirements, and quality standards for graph construction and traditional descriptive statistics in the **Aprender** library. Aprender provides high-level statistical analysis and graph-theoretic operations built on top of **Trueno**'s SIMD-optimized compute primitives. The design follows EXTREME TDD methodology [1] with >90% test coverage, mutation testing ≥80%, and PMAT quality gates.

## 1. Introduction

### 1.1 Motivation

Modern data science workflows require both traditional descriptive statistics and graph-based analysis [2]. While libraries like NumPy and NetworkX provide these capabilities in Python, Rust lacks a unified, high-performance solution that combines:

1. **Traditional descriptive statistics** (quantiles, percentiles, histograms)
2. **Graph construction and analysis** (centrality measures, community detection)
3. **SIMD-optimized performance** via low-level primitives
4. **Type safety** and zero-cost abstractions

### 1.2 Scope

**In Scope**:
- Traditional descriptive statistics (quantiles, percentiles, histograms, box plots)
- Graph construction (directed, undirected, weighted, bipartite)
- Graph centrality measures (degree, betweenness, closeness, eigenvector, PageRank)
- Community detection algorithms (Louvain, label propagation)
- Graph visualization exports (GraphML, DOT, JSON)

**Out of Scope** (delegated to Trueno):
- SIMD-optimized vector operations (mean, variance, stddev, correlation)
- Matrix operations (matmul, transpose)
- Activation functions and ML primitives

### 1.3 Architecture Principle

**Separation of Concerns**:
```
┌──────────────────────────────────────────┐
│          Aprender (High-Level)           │
│  - Graph construction & analysis         │
│  - Quantiles, percentiles, histograms    │
│  - Visualization exports                 │
│  - Orchestration & workflows             │
└──────────────────────────────────────────┘
                    ▼ uses
┌──────────────────────────────────────────┐
│         Trueno (Low-Level SIMD)          │
│  - mean(), variance(), stddev()          │
│  - min(), max(), sum(), dot()            │
│  - SIMD backends (AVX2, AVX-512)         │
└──────────────────────────────────────────┘
```

This follows the **FFmpeg model** [3]: low-level SIMD operations in library (Trueno), high-level tools consume it (Aprender).

## 2. Traditional Descriptive Statistics

### 2.1 Quantiles and Percentiles

#### 2.1.1 API Design

```rust
use trueno::Vector;

pub struct DescriptiveStats<'a> {
    data: &'a Vector<f32>,
}

impl<'a> DescriptiveStats<'a> {
    /// Compute quantile using linear interpolation (R-7 method)
    ///
    /// Uses the method from Hyndman & Fan (1996) [4] commonly used
    /// in statistical packages (R, NumPy, Pandas).
    ///
    /// # Arguments
    /// * `q` - Quantile value in [0, 1]
    ///
    /// # Returns
    /// Interpolated quantile value
    ///
    /// # Examples
    /// ```
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&v);
    /// assert_eq!(stats.quantile(0.5).unwrap(), 3.0); // median
    /// ```
    pub fn quantile(&self, q: f64) -> Result<f32, AprenderError>;

    /// Compute multiple percentiles efficiently (single sort)
    ///
    /// # Arguments
    /// * `percentiles` - Slice of percentile values (0-100)
    ///
    /// # Examples
    /// ```
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&v);
    /// let p = stats.percentiles(&[25.0, 50.0, 75.0]).unwrap();
    /// assert_eq!(p, vec![2.0, 3.0, 4.0]);
    /// ```
    pub fn percentiles(&self, percentiles: &[f64]) -> Result<Vec<f32>, AprenderError>;

    /// Five-number summary: min, Q1, median, Q3, max
    pub fn five_number_summary(&self) -> Result<FiveNumberSummary, AprenderError>;

    /// Interquartile range (IQR = Q3 - Q1)
    pub fn iqr(&self) -> Result<f32, AprenderError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct FiveNumberSummary {
    pub min: f32,
    pub q1: f32,
    pub median: f32,
    pub q3: f32,
    pub max: f32,
}
```

#### 2.1.2 Algorithm Selection

**Quantile Estimation Methods** [4]:
- **R-1 to R-9**: Nine methods defined by Hyndman & Fan (1996)
- **Default: R-7** (linear interpolation between closest ranks)
  - Used by R, NumPy (default), Pandas
  - Formula: `Q(p) = x[floor(h)] + (h - floor(h)) * (x[ceil(h)] - x[floor(h)])` where `h = (n-1)*p + 1`

**Performance Optimization (Muda Elimination)** [11]:
- **Single quantile**: Use **QuickSelect** (Floyd-Rivest SELECT algorithm) - O(n) average case
  - Rust provides `select_nth_unstable()` which implements this
  - Avoids O(n log n) full sort + O(n) memory copy
- **Multiple quantiles**: Full sort O(n log n) is optimal (amortized over k quantiles)
- **Caching strategy**: Cache sorted data if `percentiles()` called multiple times
  - Use `Cow<'a, [f32]>` to avoid cloning unless mutation needed

**Implementation**:
```rust
pub struct DescriptiveStats<'a> {
    data: &'a Vector<f32>,
    cached_sorted: Option<Vec<f32>>,  // Lazy sort caching
}

impl<'a> DescriptiveStats<'a> {
    pub fn quantile(&self, q: f64) -> Result<f32, AprenderError> {
        // Use QuickSelect for single quantile (O(n))
        let mut working_copy = self.data.as_slice().to_vec();
        let k = ((working_copy.len() - 1) as f64 * q) as usize;
        working_copy.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
        // Interpolation logic here
    }

    pub fn percentiles(&self, percentiles: &[f64]) -> Result<Vec<f32>, AprenderError> {
        // Full sort is optimal for multiple quantiles
        // Use cached sort if available
    }
}
```

#### 2.1.3 Testing Requirements (EXTREME TDD)

**Unit Tests**:
- Empty vector (error case)
- Single element
- Two elements
- Odd length (exact median)
- Even length (interpolated median)
- Edge quantiles (q=0.0, q=1.0)
- Invalid quantiles (q < 0, q > 1)

**Property-Based Tests** (proptest):
- Monotonicity: `quantile(p1) ≤ quantile(p2)` for `p1 ≤ p2`
- Boundary: `quantile(0.0) == min()`, `quantile(1.0) == max()`
- Percentile equivalence: `percentile(50) == quantile(0.5)`

**Benchmark Tests**:
- Compare against NumPy's `np.quantile()` for correctness
- Measure performance for sizes: 100, 1K, 10K, 100K, 1M

### 2.2 Histograms

#### 2.2.1 API Design

```rust
pub struct Histogram {
    pub bins: Vec<f32>,      // Bin edges (length = n_bins + 1)
    pub counts: Vec<usize>,  // Bin counts (length = n_bins)
    pub density: Vec<f64>,   // Normalized density (optional)
}

pub enum BinMethod {
    FreedmanDiaconis,  // Default for unimodal distributions
    Sturges,           // For small datasets (n < 200)
    Scott,             // For smooth, unimodal data
    SquareRoot,        // Simple rule of thumb
    Bayesian,          // For multimodal/heavy-tailed distributions [17]
}

impl<'a> DescriptiveStats<'a> {
    /// Compute histogram with automatic bin selection
    ///
    /// Uses Freedman-Diaconis rule [5] by default:
    /// bin_width = 2 * IQR / n^(1/3)
    ///
    /// For multimodal or heavy-tailed distributions, use `BinMethod::Bayesian` [17].
    ///
    /// # Examples
    /// ```
    /// let v = Vector::from_slice(&[1.0, 2.0, 2.0, 3.0, 5.0]);
    /// let stats = DescriptiveStats::new(&v);
    /// let hist = stats.histogram_auto().unwrap();
    /// assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    /// ```
    pub fn histogram_auto(&self) -> Result<Histogram, AprenderError> {
        self.histogram_method(BinMethod::FreedmanDiaconis)
    }

    /// Compute histogram with specified bin selection method
    pub fn histogram_method(&self, method: BinMethod) -> Result<Histogram, AprenderError>;

    /// Compute histogram with fixed number of bins
    pub fn histogram(&self, n_bins: usize) -> Result<Histogram, AprenderError>;

    /// Compute histogram with custom bin edges
    pub fn histogram_edges(&self, edges: &[f32]) -> Result<Histogram, AprenderError>;
}
```

#### 2.2.2 Bin Selection Methods [5, 17, 18]

1. **Freedman-Diaconis** (default): `bin_width = 2 * IQR * n^(-1/3)`
   - **Best for**: Unimodal, symmetric distributions
   - **Fails on**: Heavy-tailed, multimodal data
2. **Sturges**: `n_bins = ceil(log2(n)) + 1`
   - **Best for**: Small datasets (n < 200)
3. **Scott** [18]: `bin_width = 3.5 * σ * n^(-1/3)`
   - **Best for**: Smooth, normal-like data
   - Minimizes integrated mean squared error (IMSE)
4. **Square root**: `n_bins = ceil(sqrt(n))`
   - Simple rule of thumb
5. **Bayesian Blocks** [17] (Kaizen improvement):
   - **Best for**: Multimodal, heavy-tailed, non-parametric distributions
   - Variable-width bins optimized via dynamic programming
   - More robust than Freedman-Diaconis for complex data

**Performance**:
- Freedman-Diaconis/Scott/Sturges: O(n) single-pass binning
- Bayesian Blocks: O(n²) dynamic programming (use for n < 10K)

#### 2.2.3 Testing Requirements

**Unit Tests**:
- Uniform distribution (equal counts)
- Normal distribution (bell curve)
- Edge values (exactly on bin boundaries)
- Out-of-range values (below min, above max)

**Property-Based Tests**:
- Total counts: `sum(counts) == data.len()`
- Density integration: `sum(density * bin_width) ≈ 1.0`
- Bin coverage: `bins[0] ≤ min(data)`, `bins[last] ≥ max(data)`

## 3. Graph Construction and Analysis

### 3.1 Graph Representation

#### 3.1.1 Core Types (Cache-Optimized Representation)

**Design Rationale** [11, 12]: Use Compressed Sparse Row (CSR) format to maximize cache locality and eliminate pointer chasing inherent in `HashMap<NodeId, Vec<NodeId>>`.

```rust
use std::collections::HashMap;

/// Graph node identifier (contiguous integers for cache efficiency)
pub type NodeId = usize;

/// Graph edge with optional weight
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: Option<f64>,
}

/// Graph structure using CSR (Compressed Sparse Row) for cache efficiency
///
/// Memory layout inspired by Combinatorial BLAS [12]:
/// - Adjacency stored as two flat vectors (CSR format)
/// - Node labels stored separately (accessed rarely)
/// - String→NodeId mapping via HashMap (build-time only)
pub struct Graph {
    // CSR adjacency representation (cache-friendly)
    row_ptr: Vec<usize>,        // Offset into col_indices (length = n_nodes + 1)
    col_indices: Vec<NodeId>,   // Flattened neighbor lists (length = n_edges)
    edge_weights: Vec<f64>,     // Parallel to col_indices (empty if unweighted)

    // Metadata (accessed less frequently)
    node_labels: Vec<Option<String>>,  // Indexed by NodeId
    label_to_id: HashMap<String, NodeId>,  // For label lookups

    is_directed: bool,
    n_nodes: usize,
    n_edges: usize,
}

impl Graph {
    /// Get neighbors of node v in O(degree(v)) time with perfect cache locality
    pub fn neighbors(&self, v: NodeId) -> &[NodeId] {
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        &self.col_indices[start..end]
    }
}
```

**Performance Impact**:
- **Memory**: 50-70% reduction vs HashMap (no pointer overhead)
- **Cache misses**: 3-5x fewer (sequential access pattern)
- **Iteration**: SIMD-friendly (can use Trueno for parallel neighbor processing)

#### 3.1.2 Construction API

```rust
impl Graph {
    /// Create empty graph
    pub fn new(is_directed: bool) -> Self;

    /// Add node with optional label
    pub fn add_node(&mut self, id: NodeId, label: Option<String>) -> Result<(), AprenderError>;

    /// Add edge between nodes
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, weight: Option<f64>) -> Result<(), AprenderError>;

    /// Build from edge list
    pub fn from_edges(edges: &[(NodeId, NodeId)], is_directed: bool) -> Self;

    /// Build from adjacency matrix (uses Trueno Matrix)
    pub fn from_adjacency_matrix(matrix: &trueno::Matrix<f32>, is_directed: bool) -> Self;
}
```

### 3.2 Centrality Measures

#### 3.2.1 Degree Centrality

**Definition** [6]: For node `v`, degree centrality = `degree(v) / (n - 1)` where `n` is number of nodes.

```rust
impl Graph {
    /// Compute degree centrality for all nodes
    ///
    /// Uses Freeman's normalization [6]: C_D(v) = deg(v) / (n - 1)
    ///
    /// # Returns
    /// HashMap mapping NodeId to centrality score in [0, 1]
    ///
    /// # Performance
    /// O(n + m) where n = nodes, m = edges
    pub fn degree_centrality(&self) -> HashMap<NodeId, f64>;
}
```

**Testing**:
- Star graph: center = 1.0, leaves = 1/(n-1)
- Complete graph: all nodes = 1.0
- Path graph: endpoints = 1/(n-1), middle ≈ 2/(n-1)

#### 3.2.2 Betweenness Centrality (Parallelized)

**Definition** [7]: `C_B(v) = Σ(σ_st(v) / σ_st)` for all pairs s,t where σ_st is total shortest paths from s to t, and σ_st(v) is those passing through v.

```rust
use rayon::prelude::*;

impl Graph {
    /// Compute betweenness centrality using parallel Brandes' algorithm [7, 13]
    ///
    /// # Performance
    /// - Serial: O(nm) for unweighted graphs, O(nm + n² log n) for weighted
    /// - Parallel: O(nm/p) where p = number of CPU cores
    ///
    /// # Implementation
    /// Uses Rayon parallel iterator for the outer loop (BFS from each source).
    /// Each source's BFS is independent, making this "embarrassingly parallel" [13].
    ///
    /// # Examples
    /// ```
    /// let mut g = Graph::new(false);
    /// g.add_edge(0, 1, None).unwrap();
    /// g.add_edge(1, 2, None).unwrap();
    /// let bc = g.betweenness_centrality();
    /// assert!(bc[&1] > bc[&0]); // middle node has higher betweenness
    /// ```
    pub fn betweenness_centrality(&self) -> Vec<f64> {
        // Parallel outer loop (one BFS per source node)
        let partial_scores: Vec<Vec<f64>> = (0..self.n_nodes)
            .into_par_iter()
            .map(|source| {
                // Each thread computes BFS from its assigned source
                self.brandes_bfs_from_source(source)
            })
            .collect();

        // Reduce partial scores (single-threaded, fast)
        let mut centrality = vec![0.0; self.n_nodes];
        for partial in partial_scores {
            for (i, &score) in partial.iter().enumerate() {
                centrality[i] += score;
            }
        }
        centrality
    }
}
```

**Algorithm**: Parallel Brandes' algorithm [7, 13] (2001)
- Uses BFS from each node to compute shortest path counts
- Backward accumulation to compute dependencies
- **Parallelization**: Outer loop over source nodes (embarrassingly parallel)
- **Expected speedup**: ~8x on 8-core CPU for graphs with >1K nodes

**Testing**:
- Path graph: middle nodes have higher betweenness
- Star graph: center = (n-1)(n-2)/2, leaves = 0
- Bridge graph: bridge node has maximum betweenness
- **Parallel correctness**: serial result == parallel result (deterministic reduction)

#### 3.2.3 PageRank (Numerically Stable)

**Definition** [8]: Iterative algorithm computing stationary distribution of random walk on graph.

```rust
impl Graph {
    /// Compute PageRank using power iteration with Kahan summation [8, 14]
    ///
    /// # Arguments
    /// * `damping` - Damping factor (default 0.85)
    /// * `max_iter` - Maximum iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    ///
    /// # Numerical Stability
    /// Uses Kahan summation [14] for rank accumulation to prevent floating-point
    /// drift in large graphs (>10K nodes). Naive summation can accumulate O(n·ε)
    /// error where ε is machine epsilon.
    ///
    /// # Performance
    /// O(k * m) where k = iterations, m = edges
    ///
    /// # Examples
    /// ```
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    /// let pr = g.pagerank(0.85, 100, 1e-6).unwrap();
    /// assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-10); // Kahan ensures precision
    /// ```
    pub fn pagerank(&self, damping: f64, max_iter: usize, tol: f64) -> Result<Vec<f64>, AprenderError> {
        let n = self.n_nodes;
        let mut ranks = vec![1.0 / n as f64; n];
        let mut new_ranks = vec![0.0; n];

        for _iter in 0..max_iter {
            // Kahan summation for each node's new rank
            for v in 0..n {
                let neighbors = self.neighbors(v);
                let mut sum = 0.0;
                let mut c = 0.0;  // Kahan compensation term

                for &u in neighbors {
                    let out_degree = self.neighbors(u).len() as f64;
                    let y = (ranks[u] / out_degree) - c;
                    let t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }

                new_ranks[v] = (1.0 - damping) / n as f64 + damping * sum;
            }

            // Convergence check (also use Kahan for diff calculation)
            let diff = kahan_diff(&ranks, &new_ranks);
            if diff < tol {
                return Ok(new_ranks);
            }

            std::mem::swap(&mut ranks, &mut new_ranks);
        }

        Ok(ranks)
    }
}

/// Kahan summation for computing L1 distance between two vectors
fn kahan_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for i in 0..a.len() {
        let y = (a[i] - b[i]).abs() - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}
```

**Testing**:
- Sum of ranks = 1.0 (within 1e-10, not 1e-5 due to Kahan)
- Convergence within max_iter
- Compare against NetworkX for known graphs
- **Numerical stability test**: 100K node graph should converge (naive summation fails)

### 3.3 Community Detection

#### 3.3.1 Leiden Method (Improved Louvain)

**Algorithm** [15]: Leiden algorithm - fixes disconnected communities and resolution limit issues in Louvain.

**Rationale (Kaizen)**: While Louvain [9] is widely used, it suffers from:
1. **Non-deterministic results** (order-dependent)
2. **Resolution limit** [16]: Cannot detect communities smaller than `sqrt(m)` where m = edges
3. **Disconnected communities**: Can produce communities with disconnected components

**Leiden** [15] (2019) addresses all three issues while being faster than Louvain.

```rust
pub struct Community {
    pub id: usize,
    pub nodes: Vec<NodeId>,
    pub modularity: f64,
}

impl Graph {
    /// Detect communities using Leiden algorithm [15]
    ///
    /// Optimizes modularity Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
    /// with guarantee of well-connected communities.
    ///
    /// # Advantages over Louvain
    /// - Guarantees communities are connected
    /// - Faster convergence (fewer iterations)
    /// - More stable results (less order-dependent)
    ///
    /// # Returns
    /// Vector of communities with modularity score
    ///
    /// # Performance
    /// O(n log n) average case (same as Louvain, but fewer iterations)
    ///
    /// # Examples
    /// ```
    /// let g = Graph::karate_club();  // Zachary's karate club
    /// let communities = g.detect_communities_leiden().unwrap();
    /// assert_eq!(communities.len(), 2);  // Known to split into 2 groups
    /// assert!(communities[0].modularity > 0.4);  // High modularity
    /// ```
    pub fn detect_communities_leiden(&self) -> Result<Vec<Community>, AprenderError>;

    /// Legacy Louvain implementation (deprecated, use Leiden instead)
    ///
    /// # Warning
    /// Louvain suffers from resolution limit [16] and may produce
    /// disconnected communities. Use `detect_communities_leiden()` instead.
    #[deprecated(since = "1.1.0", note = "Use detect_communities_leiden() instead")]
    pub fn detect_communities_louvain(&self) -> Result<Vec<Community>, AprenderError>;
}
```

**Testing**:
- **Karate club graph** [10]: Should find 2 communities (instructor vs. administrator factions)
- **Resolution limit test**: Small cliques within large graph should be detected
- **Connectedness test**: All communities must be connected subgraphs
- Modularity Q ∈ [0, 1] (higher is better, typically >0.3 for real networks)
- Compare against NetworkX + `leidenalg` library

## 4. Visualization Exports

### 4.1 Export Formats

```rust
impl Graph {
    /// Export to GraphML (XML format)
    pub fn to_graphml(&self) -> Result<String, AprenderError>;

    /// Export to DOT (Graphviz format)
    pub fn to_dot(&self) -> Result<String, AprenderError>;

    /// Export to JSON (node-link format)
    pub fn to_json(&self) -> Result<String, AprenderError>;
}
```

**Testing**:
- Round-trip: export → parse → compare
- Schema validation for GraphML
- Valid DOT syntax (can be rendered by Graphviz)

## 5. Quality Standards (EXTREME TDD)

### 5.1 Test Coverage Requirements

| Component | Minimum Coverage | Target Coverage |
|-----------|-----------------|-----------------|
| Descriptive stats | 95% | 100% |
| Graph construction | 90% | 95% |
| Centrality measures | 85% | 90% |
| Community detection | 85% | 90% |
| Visualization exports | 90% | 95% |
| **Overall** | **90%** | **95%+** |

### 5.2 PMAT Quality Gates

Every commit must pass:
```bash
pmat analyze complexity --max-complexity 10
pmat analyze satd --max-violations 5
pmat tdg --min-grade B+  # ≥87/100
```

### 5.3 Mutation Testing

```bash
cargo mutants --timeout 120 --minimum-pass-rate 80
```

**Target**: ≥80% mutation kill rate

### 5.4 Property-Based Testing (Algebraic Oracles)

Use `proptest` [19] for all statistical operations with **algebraic graph theory oracles** to verify correctness:

**Statistical Properties**:
- Quantile monotonicity: `quantile(p1) ≤ quantile(p2)` for `p1 ≤ p2`
- Histogram normalization: `sum(density * bin_width) ≈ 1.0`
- Five-number summary ordering: `min ≤ Q1 ≤ median ≤ Q3 ≤ max`

**Graph Invariants** (using Trueno's linear algebra):
1. **Handshaking lemma**: `sum(degrees) = 2 * n_edges` for undirected graphs
2. **Spectral property**: `trace(adjacency_matrix) = 0` for simple graphs (no self-loops)
3. **Laplacian eigenvalues**: `λ₁ = 0` (connected graph has exactly one zero eigenvalue)
4. **PageRank sum**: `sum(ranks) = 1.0` (within numerical precision)
5. **Modularity bounds**: `-0.5 ≤ Q ≤ 1.0` for any community partition

**Implementation Pattern**:
```rust
use proptest::prelude::*;
use trueno::Matrix;

proptest! {
    #[test]
    fn test_graph_spectral_invariant(edges in prop::collection::vec((0usize..100, 0usize..100), 10..1000)) {
        let g = Graph::from_edges(&edges, false);
        let adj_matrix = g.to_adjacency_matrix();

        // Algebraic oracle: trace(A) = 0 for simple graphs
        let trace = adj_matrix.trace();
        prop_assert!((trace.abs() < 1e-10), "Trace must be 0 for simple graph, got {}", trace);
    }
}
```

### 5.5 Benchmark Testing

Compare against established libraries:
- **NumPy** (Python): quantile, percentile, histogram
- **NetworkX** (Python): centrality measures, community detection
- **Trueno** (Rust): Ensure we use Trueno for primitives, not reimplementing

## 6. Performance Targets

### 6.1 Descriptive Statistics

| Operation | Size | Target Latency | vs NumPy |
|-----------|------|---------------|----------|
| Quantile | 1M | <10ms | 2-3x faster |
| Percentiles (5) | 1M | <15ms | 2-3x faster |
| Histogram (auto) | 1M | <5ms | 2-3x faster |

### 6.2 Graph Operations

| Operation | Graph Size | Target Latency | vs NetworkX |
|-----------|-----------|----------------|-------------|
| Degree centrality | 10K nodes | <100ms | 5-10x faster |
| Betweenness | 1K nodes | <1s | 3-5x faster |
| PageRank (100 iter) | 10K nodes | <500ms | 5-10x faster |
| Louvain | 10K nodes | <2s | 2-4x faster |

## 7. Dependencies

```toml
[dependencies]
trueno = { path = "../trueno", version = "0.2" }
rayon = "1.10"  # Parallel betweenness centrality
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"

[dev-dependencies]
proptest = "1.8"
criterion = "0.5"
approx = "0.5"  # For floating-point assertions in tests
```

## 8. References

[1] Beck, K. (2003). *Test-Driven Development: By Example*. Addison-Wesley. https://doi.org/10.1109/MS.2003.1231162

[2] Newman, M. E. J. (2018). *Networks* (2nd ed.). Oxford University Press. https://doi.org/10.1093/oso/9780198805090.001.0001

[3] Tomar, S. (2006). Converting video formats with FFmpeg. *Linux Journal*, 2006(146), 10. https://www.linuxjournal.com/article/8517

[4] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages. *The American Statistician*, 50(4), 361-365. https://doi.org/10.1080/00031305.1996.10473566

[5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: L2 theory. *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete*, 57(4), 453-476. https://doi.org/10.1007/BF01025868

[6] Freeman, L. C. (1978). Centrality in social networks: Conceptual clarification. *Social Networks*, 1(3), 215-239. https://doi.org/10.1016/0378-8733(78)90021-7

[7] Brandes, U. (2001). A faster algorithm for betweenness centrality. *Journal of Mathematical Sociology*, 25(2), 163-177. https://doi.org/10.1080/0022250X.2001.9990249

[8] Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank citation ranking: Bringing order to the web*. Stanford InfoLab Technical Report. http://ilpubs.stanford.edu:8090/422/

[9] Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008

[10] Zachary, W. W. (1977). An information flow model for conflict and fission in small groups. *Journal of Anthropological Research*, 33(4), 452-473. https://doi.org/10.1086/jar.33.4.3629752

[11] Floyd, R. W., & Rivest, R. L. (1975). Algorithm 489: The algorithm SELECT—for finding the ith smallest of n elements. *Communications of the ACM*, 18(3), 173. https://doi.org/10.1145/360680.360694
*Toyota Way: Muda elimination via QuickSelect for single quantile (O(n) vs O(n log n))*

[12] Buluc, A., Fineman, J. T., Frigo, M., Gilbert, J. R., & Leiserson, C. E. (2009). Parallel sparse matrix-vector and matrix-transpose-vector multiplication using compressed sparse blocks. *Proceedings of the 21st ACM Symposium on Parallelism in Algorithms and Architectures*, 233-244. https://doi.org/10.1145/1583991.1584053
*Toyota Way: Cache-optimized CSR format eliminates HashMap pointer chasing*

[13] Bader, D. A., & Madduri, K. (2006). Parallel algorithms for evaluating centrality indices in real-world networks. *35th International Conference on Parallel Processing (ICPP'06)*, 539-550. https://doi.org/10.1109/ICPP.2006.57
*Toyota Way: Heijunka (load balancing) via parallel Brandes' algorithm*

[14] Higham, N. J. (1993). The accuracy of floating point summation. *SIAM Journal on Scientific Computing*, 14(4), 783-799. https://doi.org/10.1137/0914050
*Toyota Way: Poka-Yoke (error prevention) via Kahan summation for PageRank*

[15] Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9, 5233. https://doi.org/10.1038/s41598-019-41695-z
*Toyota Way: Kaizen (continuous improvement) - Leiden fixes Louvain's disconnected community bug*

[16] Fortunato, S., & Barthelemy, M. (2007). Resolution limit in community detection. *Proceedings of the National Academy of Sciences*, 104(1), 36-41. https://doi.org/10.1073/pnas.0605965104
*Toyota Way: Genchi Genbutsu (go and see) - empirical evidence of Louvain's resolution limit*

[17] Scargle, J. D., Norris, J. P., Jackson, B., & Chiang, J. (2013). Studies in astronomical time series analysis. VI. Bayesian block representations. *The Astrophysical Journal*, 764(2), 167. https://doi.org/10.1088/0004-637X/764/2/167
*Toyota Way: Kaizen - Bayesian Blocks superior to Freedman-Diaconis for multimodal data*

[18] Scott, D. W. (1979). On optimal and data-based histograms. *Biometrika*, 66(3), 605-610. https://doi.org/10.1093/biomet/66.3.605
*Toyota Way: Jidoka (built-in quality) - IMSE minimization ensures scientific validity*

[19] Claessen, K., & Hughes, J. (2000). QuickCheck: a lightweight tool for random testing of Haskell programs. *ACM SIGPLAN Notices*, 35(9), 268-279. https://doi.org/10.1145/357766.351266
*Toyota Way: Jidoka - property-based testing catches algebraic invariant violations*

## 9. Appendix A: EXTREME TDD Workflow

**RED-GREEN-REFACTOR Cycle**:

1. **RED**: Write failing test first
   ```rust
   #[test]
   fn test_quantile_median() {
       let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
       let stats = DescriptiveStats::new(&v);
       assert_eq!(stats.quantile(0.5).unwrap(), 3.0);
   }
   ```

2. **GREEN**: Implement minimal code to pass
   ```rust
   pub fn quantile(&self, q: f64) -> Result<f32, AprenderError> {
       if !(0.0..=1.0).contains(&q) {
           return Err(AprenderError::InvalidQuantile(q));
       }
       let mut sorted = self.data.as_slice().to_vec();
       sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
       let h = (sorted.len() as f64 - 1.0) * q;
       let h_floor = h.floor() as usize;
       let h_ceil = h.ceil() as usize;
       let value = sorted[h_floor] + (h - h_floor as f64) as f32 * (sorted[h_ceil] - sorted[h_floor]);
       Ok(value)
   }
   ```

3. **REFACTOR**: Optimize while keeping tests green
   - Extract sorting logic
   - Add caching for repeated calls
   - Verify performance benchmarks

**Commit only when all gates pass**:
```bash
cargo test --all-features
cargo clippy -- -D warnings
pmat tdg --min-grade B+
```

## 10. Appendix B: Example Usage

```rust
use aprender::{DescriptiveStats, Graph};
use trueno::Vector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Descriptive statistics
    let data = Vector::from_slice(&[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 100.0]);
    let stats = DescriptiveStats::new(&data);

    println!("Five-number summary: {:?}", stats.five_number_summary()?);
    println!("IQR: {}", stats.iqr()?);

    let hist = stats.histogram_auto()?;
    println!("Histogram bins: {:?}", hist.bins);
    println!("Histogram counts: {:?}", hist.counts);

    // Graph analysis
    let mut g = Graph::new(false);
    for i in 0..5 {
        g.add_node(i, Some(format!("Node {}", i)))?;
    }
    g.add_edge(0, 1, Some(1.0))?;
    g.add_edge(1, 2, Some(1.0))?;
    g.add_edge(2, 3, Some(1.0))?;
    g.add_edge(3, 4, Some(1.0))?;
    g.add_edge(4, 0, Some(1.0))?;

    let dc = g.degree_centrality();
    println!("Degree centrality: {:?}", dc);

    let pr = g.pagerank(0.85, 100, 1e-6)?;
    println!("PageRank: {:?}", pr);

    // Export for visualization
    std::fs::write("graph.dot", g.to_dot()?)?;

    Ok(())
}
```

---

**Document Status**: Draft v1.0.0 - Ready for implementation using EXTREME TDD methodology with PMAT quality gates.
