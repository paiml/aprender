# Graph and Traditional Descriptive Statistics Specification

**Version**: 1.0.0
**Status**: Draft
**Last Updated**: 2025-11-19
**Authors**: Pragmatic AI Labs Team, Claude (AI pair programmer)

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

**Performance**:
- Sorting: O(n log n) using Rust's `sort_by` (introsort)
- Quantile lookup: O(1) after sort
- Optimization: Single sort for multiple percentiles

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

impl<'a> DescriptiveStats<'a> {
    /// Compute histogram with automatic bin selection
    ///
    /// Uses Freedman-Diaconis rule [5] by default:
    /// bin_width = 2 * IQR / n^(1/3)
    ///
    /// # Examples
    /// ```
    /// let v = Vector::from_slice(&[1.0, 2.0, 2.0, 3.0, 5.0]);
    /// let stats = DescriptiveStats::new(&v);
    /// let hist = stats.histogram_auto().unwrap();
    /// assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    /// ```
    pub fn histogram_auto(&self) -> Result<Histogram, AprenderError>;

    /// Compute histogram with fixed number of bins
    pub fn histogram(&self, n_bins: usize) -> Result<Histogram, AprenderError>;

    /// Compute histogram with custom bin edges
    pub fn histogram_edges(&self, edges: &[f32]) -> Result<Histogram, AprenderError>;
}
```

#### 2.2.2 Bin Selection Methods [5]

1. **Freedman-Diaconis** (default): `bin_width = 2 * IQR * n^(-1/3)`
2. **Sturges**: `n_bins = ceil(log2(n)) + 1`
3. **Scott**: `bin_width = 3.5 * σ * n^(-1/3)`
4. **Square root**: `n_bins = ceil(sqrt(n))`

**Performance**: O(n) single-pass binning after computing IQR/stddev

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

#### 3.1.1 Core Types

```rust
use std::collections::HashMap;

/// Graph node identifier
pub type NodeId = usize;

/// Graph edge with optional weight
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub weight: Option<f64>,
}

/// Graph structure supporting directed/undirected, weighted/unweighted
pub struct Graph {
    nodes: HashMap<NodeId, NodeData>,
    adjacency: HashMap<NodeId, Vec<NodeId>>,
    edges: Vec<Edge>,
    is_directed: bool,
}

#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: NodeId,
    pub label: Option<String>,
    pub attributes: HashMap<String, AttributeValue>,
}
```

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

#### 3.2.2 Betweenness Centrality

**Definition** [7]: `C_B(v) = Σ(σ_st(v) / σ_st)` for all pairs s,t where σ_st is total shortest paths from s to t, and σ_st(v) is those passing through v.

```rust
impl Graph {
    /// Compute betweenness centrality using Brandes' algorithm [7]
    ///
    /// # Performance
    /// O(nm) for unweighted graphs, O(nm + n² log n) for weighted
    ///
    /// # Examples
    /// ```
    /// let mut g = Graph::new(false);
    /// g.add_edge(0, 1, None).unwrap();
    /// g.add_edge(1, 2, None).unwrap();
    /// let bc = g.betweenness_centrality();
    /// assert!(bc[&1] > bc[&0]); // middle node has higher betweenness
    /// ```
    pub fn betweenness_centrality(&self) -> HashMap<NodeId, f64>;
}
```

**Algorithm**: Brandes' algorithm [7] (2001) - O(nm) for unweighted graphs
- Uses BFS from each node to compute shortest path counts
- Backward accumulation to compute dependencies

**Testing**:
- Path graph: middle nodes have higher betweenness
- Star graph: center = (n-1)(n-2)/2, leaves = 0
- Bridge graph: bridge node has maximum betweenness

#### 3.2.3 PageRank

**Definition** [8]: Iterative algorithm computing stationary distribution of random walk on graph.

```rust
impl Graph {
    /// Compute PageRank using power iteration [8]
    ///
    /// # Arguments
    /// * `damping` - Damping factor (default 0.85)
    /// * `max_iter` - Maximum iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    ///
    /// # Performance
    /// O(k * m) where k = iterations, m = edges
    ///
    /// # Examples
    /// ```
    /// let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
    /// let pr = g.pagerank(0.85, 100, 1e-6).unwrap();
    /// assert!((pr[&0] + pr[&1] + pr[&2] - 1.0).abs() < 1e-5); // sum = 1
    /// ```
    pub fn pagerank(&self, damping: f64, max_iter: usize, tol: f64) -> Result<HashMap<NodeId, f64>, AprenderError>;
}
```

**Testing**:
- Sum of ranks = 1.0
- Convergence within max_iter
- Compare against NetworkX for known graphs

### 3.3 Community Detection

#### 3.3.1 Louvain Method

**Algorithm** [9]: Fast modularity optimization for community detection.

```rust
pub struct Community {
    pub id: usize,
    pub nodes: Vec<NodeId>,
    pub modularity: f64,
}

impl Graph {
    /// Detect communities using Louvain method [9]
    ///
    /// Optimizes modularity Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
    ///
    /// # Returns
    /// Vector of communities with modularity score
    ///
    /// # Performance
    /// O(n log n) average case
    pub fn detect_communities_louvain(&self) -> Result<Vec<Community>, AprenderError>;
}
```

**Testing**:
- Karate club graph [10]: Should find 2-4 communities
- Modularity Q ∈ [0, 1] (higher is better)
- Compare against NetworkX implementation

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

### 5.4 Property-Based Testing

Use `proptest` for all statistical operations:
- Quantile monotonicity
- Histogram normalization
- Graph invariants (e.g., sum of degrees = 2 * edges for undirected)

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
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"

[dev-dependencies]
proptest = "1.8"
criterion = "0.5"
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
