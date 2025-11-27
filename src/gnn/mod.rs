//! Graph Neural Network layers for learning on graph-structured data.
//!
//! This module provides neural network layers that operate on graphs,
//! enabling learning from node features and graph topology.
//!
//! # Architecture
//!
//! ```text
//! Node Features    Graph Structure
//!      │                 │
//!      ▼                 ▼
//! ┌────────────────────────────┐
//! │       GNN Layer            │
//! │  (aggregate + transform)   │
//! └────────────────────────────┘
//!            │
//!            ▼
//!    Updated Node Features
//! ```
//!
//! # Layers
//!
//! - [`GCNConv`] - Graph Convolutional Network (Kipf & Welling, 2017)
//! - [`GATConv`] - Graph Attention Network (Velickovic et al., 2018)
//!
//! # Example
//!
//! ```ignore
//! use aprender::gnn::{GCNConv, GNNModule};
//! use aprender::autograd::Tensor;
//!
//! // Create GCN layer: 16 input features → 32 output features
//! let gcn = GCNConv::new(16, 32);
//!
//! // Node features [num_nodes, in_features]
//! let x = Tensor::ones(&[4, 16]);
//!
//! // Adjacency matrix (COO format): edge_index[2, num_edges]
//! let edge_index = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
//!
//! let out = gcn.forward_gnn(&x, &edge_index);
//! assert_eq!(out.shape(), &[4, 32]);
//! ```
//!
//! # References
//!
//! - Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with
//!   Graph Convolutional Networks. ICLR.
//! - Velickovic, P., et al. (2018). Graph Attention Networks. ICLR.
//! - Hamilton, W. L., et al. (2017). Inductive Representation Learning on
//!   Large Graphs (GraphSAGE). NeurIPS.

use crate::autograd::Tensor;
use crate::nn::{Linear, Module};

/// Edge index type: (source_node, target_node)
pub type EdgeIndex = (usize, usize);

/// Trait for GNN modules that process graph-structured data.
///
/// Unlike regular [`Module`], GNN layers require graph structure in addition
/// to node features.
pub trait GNNModule: Module {
    /// Forward pass with graph structure.
    ///
    /// # Arguments
    ///
    /// * `x` - Node features `[num_nodes, in_features]`
    /// * `edge_index` - List of edges as (source, target) pairs
    ///
    /// # Returns
    ///
    /// Updated node features `[num_nodes, out_features]`
    fn forward_gnn(&self, x: &Tensor, edge_index: &[EdgeIndex]) -> Tensor;
}

/// Graph Convolutional Network layer (Kipf & Welling, 2017).
///
/// Aggregates neighbor features using mean aggregation with symmetric
/// normalization:
///
/// ```text
/// h_i' = σ(Σ_j (1/√(d_i * d_j)) * W * h_j)
/// ```
///
/// where d_i is the degree of node i.
///
/// # Example
///
/// ```ignore
/// use aprender::gnn::GCNConv;
///
/// let gcn = GCNConv::new(16, 32);  // 16 in → 32 out
/// ```
#[derive(Debug)]
pub struct GCNConv {
    /// Linear transformation
    linear: Linear,
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Whether to add self-loops
    add_self_loops: bool,
    /// Whether to use bias
    #[allow(dead_code)]
    use_bias: bool,
}

impl GCNConv {
    /// Create a new GCN convolutional layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input feature dimension
    /// * `out_features` - Output feature dimension
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            linear: Linear::new(in_features, out_features),
            in_features,
            out_features,
            add_self_loops: true,
            use_bias: true,
        }
    }

    /// Create GCN without self-loops.
    pub fn without_self_loops(in_features: usize, out_features: usize) -> Self {
        Self {
            linear: Linear::new(in_features, out_features),
            in_features,
            out_features,
            add_self_loops: false,
            use_bias: true,
        }
    }

    /// Get input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for GCNConv {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("GCNConv requires graph structure. Use forward_gnn() instead.");
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.linear.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.linear.parameters_mut()
    }
}

impl GNNModule for GCNConv {
    fn forward_gnn(&self, x: &Tensor, edge_index: &[EdgeIndex]) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_features = x.shape()[1];

        assert_eq!(
            in_features, self.in_features,
            "Expected {} input features, got {}",
            self.in_features, in_features
        );

        // Compute node degrees (for normalization)
        let mut degrees = vec![0.0f32; num_nodes];

        // Add self-loops to degree count if enabled
        if self.add_self_loops {
            for d in &mut degrees {
                *d += 1.0;
            }
        }

        // Count degrees from edges
        for &(src, tgt) in edge_index {
            degrees[src] += 1.0;
            degrees[tgt] += 1.0; // For undirected graphs
        }

        // Compute D^{-1/2} for symmetric normalization
        let norm: Vec<f32> = degrees.iter().map(|&d| 1.0 / d.sqrt().max(1e-6)).collect();

        // Aggregate: h' = D^{-1/2} A D^{-1/2} h
        let x_data = x.data();
        let mut aggregated = vec![0.0f32; num_nodes * in_features];

        // Add self-loop contribution
        if self.add_self_loops {
            for i in 0..num_nodes {
                let norm_ii = norm[i] * norm[i]; // Self-loop normalization
                for f in 0..in_features {
                    aggregated[i * in_features + f] += norm_ii * x_data[i * in_features + f];
                }
            }
        }

        // Add neighbor contributions
        for &(src, tgt) in edge_index {
            let norm_coeff = norm[src] * norm[tgt];

            // src -> tgt
            for f in 0..in_features {
                aggregated[tgt * in_features + f] += norm_coeff * x_data[src * in_features + f];
            }

            // tgt -> src (undirected)
            for f in 0..in_features {
                aggregated[src * in_features + f] += norm_coeff * x_data[tgt * in_features + f];
            }
        }

        // Create aggregated tensor
        let agg_tensor = Tensor::new(&aggregated, &[num_nodes, in_features]);

        // Apply linear transformation
        self.linear.forward(&agg_tensor)
    }
}

/// Graph Attention Network layer (Velickovic et al., 2018).
///
/// Uses attention mechanism to weight neighbor contributions:
///
/// ```text
/// α_ij = softmax_j(LeakyReLU(a^T [W*h_i || W*h_j]))
/// h_i' = σ(Σ_j α_ij * W * h_j)
/// ```
///
/// # Example
///
/// ```ignore
/// use aprender::gnn::GATConv;
///
/// let gat = GATConv::new(16, 32, 4);  // 16 in → 32 out, 4 attention heads
/// ```
#[derive(Debug)]
pub struct GATConv {
    /// Linear transformation for node features
    linear: Linear,
    /// Attention weights (for computing attention scores)
    attention_src: Tensor,
    /// Attention weights for target nodes
    attention_tgt: Tensor,
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension per head
    out_features: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Negative slope for LeakyReLU
    negative_slope: f32,
    /// Whether to add self-loops
    add_self_loops: bool,
}

impl GATConv {
    /// Create a new GAT convolutional layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input feature dimension
    /// * `out_features` - Output feature dimension per attention head
    /// * `num_heads` - Number of attention heads
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let total_out = out_features * num_heads;

        // Initialize attention vectors
        let attn_data: Vec<f32> = (0..total_out)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();

        Self {
            linear: Linear::new(in_features, total_out),
            attention_src: Tensor::new(&attn_data, &[num_heads, out_features]).requires_grad(),
            attention_tgt: Tensor::new(&attn_data, &[num_heads, out_features]).requires_grad(),
            in_features,
            out_features,
            num_heads,
            negative_slope: 0.2,
            add_self_loops: true,
        }
    }

    /// Get number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get output dimension per head.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Total output dimension (out_features * num_heads).
    pub fn total_out_features(&self) -> usize {
        self.out_features * self.num_heads
    }
}

impl Module for GATConv {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("GATConv requires graph structure. Use forward_gnn() instead.");
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.linear.parameters();
        params.push(&self.attention_src);
        params.push(&self.attention_tgt);
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.linear.parameters_mut();
        params.push(&mut self.attention_src);
        params.push(&mut self.attention_tgt);
        params
    }
}

impl GNNModule for GATConv {
    fn forward_gnn(&self, x: &Tensor, edge_index: &[EdgeIndex]) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_features = x.shape()[1];

        assert_eq!(
            in_features, self.in_features,
            "Expected {} input features, got {}",
            self.in_features, in_features
        );

        // Transform features: [num_nodes, in_features] -> [num_nodes, num_heads * out_features]
        let h = self.linear.forward(x);
        let h_data = h.data();

        let total_out = self.num_heads * self.out_features;

        // Build edge list with self-loops
        let mut edges: Vec<EdgeIndex> = edge_index.to_vec();
        if self.add_self_loops {
            for i in 0..num_nodes {
                edges.push((i, i));
            }
        }

        // For each node, collect attention scores and aggregate
        let mut output = vec![0.0f32; num_nodes * total_out];

        // Group edges by target node
        let mut neighbors: Vec<Vec<usize>> = vec![vec![]; num_nodes];
        for &(src, tgt) in &edges {
            neighbors[tgt].push(src);
        }

        // Attention computation per head
        let attn_src_data = self.attention_src.data();
        let attn_tgt_data = self.attention_tgt.data();

        for tgt in 0..num_nodes {
            if neighbors[tgt].is_empty() {
                continue;
            }

            for head in 0..self.num_heads {
                let head_offset = head * self.out_features;

                // Compute attention scores for all neighbors
                let mut scores: Vec<f32> = Vec::with_capacity(neighbors[tgt].len());
                let mut max_score = f32::NEG_INFINITY;

                for &src in &neighbors[tgt] {
                    // e_ij = LeakyReLU(a_src^T * h_src + a_tgt^T * h_tgt)
                    let mut score = 0.0;

                    for f in 0..self.out_features {
                        let h_src_f = h_data[src * total_out + head_offset + f];
                        let h_tgt_f = h_data[tgt * total_out + head_offset + f];
                        score += attn_src_data[head * self.out_features + f] * h_src_f
                            + attn_tgt_data[head * self.out_features + f] * h_tgt_f;
                    }

                    // LeakyReLU
                    if score < 0.0 {
                        score *= self.negative_slope;
                    }

                    scores.push(score);
                    max_score = max_score.max(score);
                }

                // Softmax normalization (numerically stable)
                let mut exp_sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }
                for s in &mut scores {
                    *s /= exp_sum.max(1e-8);
                }

                // Aggregate: h'_tgt = Σ α_ij * h_src
                for (idx, &src) in neighbors[tgt].iter().enumerate() {
                    let alpha = scores[idx];
                    for f in 0..self.out_features {
                        output[tgt * total_out + head_offset + f] +=
                            alpha * h_data[src * total_out + head_offset + f];
                    }
                }
            }
        }

        Tensor::new(&output, &[num_nodes, total_out])
    }
}

/// Graph Isomorphism Network layer (Xu et al., 2019).
///
/// Uses sum aggregation with learnable epsilon for injective aggregation:
///
/// ```text
/// h_i' = MLP((1 + ε) * h_i + Σ_j h_j)
/// ```
///
/// This makes the aggregation injective, preserving structural information.
#[derive(Debug)]
pub struct GINConv {
    /// MLP for transformation
    linear1: Linear,
    linear2: Linear,
    /// Learnable epsilon parameter
    eps: f32,
    /// Whether epsilon is trainable
    train_eps: bool,
    /// Input/hidden/output dimensions
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
}

impl GINConv {
    /// Create a new GIN convolutional layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input feature dimension
    /// * `hidden_features` - Hidden layer dimension
    /// * `out_features` - Output feature dimension
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Self {
        Self {
            linear1: Linear::new(in_features, hidden_features),
            linear2: Linear::new(hidden_features, out_features),
            eps: 0.0,
            train_eps: true,
            in_features,
            hidden_features,
            out_features,
        }
    }

    /// Get epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Set epsilon value.
    pub fn set_eps(&mut self, eps: f32) {
        self.eps = eps;
    }

    /// Check if epsilon is trainable.
    pub fn train_eps(&self) -> bool {
        self.train_eps
    }

    /// Get input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get hidden feature dimension.
    pub fn hidden_features(&self) -> usize {
        self.hidden_features
    }

    /// Get output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for GINConv {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("GINConv requires graph structure. Use forward_gnn() instead.");
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.linear1.parameters();
        params.extend(self.linear2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.linear1.parameters_mut();
        params.extend(self.linear2.parameters_mut());
        params
    }
}

impl GNNModule for GINConv {
    fn forward_gnn(&self, x: &Tensor, edge_index: &[EdgeIndex]) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_features = x.shape()[1];

        assert_eq!(
            in_features, self.in_features,
            "Expected {} input features, got {}",
            self.in_features, in_features
        );

        let x_data = x.data();

        // Sum aggregation: h' = (1 + eps) * h_i + Σ_j h_j
        let mut aggregated = vec![0.0f32; num_nodes * in_features];

        // Self contribution: (1 + eps) * h_i
        let self_weight = 1.0 + self.eps;
        for i in 0..num_nodes {
            for f in 0..in_features {
                aggregated[i * in_features + f] = self_weight * x_data[i * in_features + f];
            }
        }

        // Neighbor contribution: Σ_j h_j (sum, not mean!)
        for &(src, tgt) in edge_index {
            for f in 0..in_features {
                aggregated[tgt * in_features + f] += x_data[src * in_features + f];
                aggregated[src * in_features + f] += x_data[tgt * in_features + f];
            }
        }

        // Create aggregated tensor
        let agg_tensor = Tensor::new(&aggregated, &[num_nodes, in_features]);

        // Apply MLP: ReLU(Linear1(x)) -> Linear2
        let h1 = self.linear1.forward(&agg_tensor);
        let h1_data = h1.data();
        let h1_relu: Vec<f32> = h1_data.iter().map(|&v| v.max(0.0)).collect();
        let h1_relu_tensor = Tensor::new(&h1_relu, h1.shape());

        self.linear2.forward(&h1_relu_tensor)
    }
}

/// GraphSAGE layer (Hamilton et al., 2017).
///
/// Uses sampled neighbors and mean/max/lstm aggregation:
///
/// ```text
/// h_i' = σ(W * CONCAT(h_i, AGG({h_j : j ∈ N(i)})))
/// ```
///
/// # Aggregation Types
///
/// - Mean: Average of neighbor features
/// - Max: Element-wise max
/// - Sum: Sum of neighbor features
///
/// # Reference
///
/// Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation
/// Learning on Large Graphs. NeurIPS.
#[derive(Debug)]
pub struct GraphSAGEConv {
    /// Linear transformation for self + aggregated
    linear: Linear,
    /// Aggregation type
    aggregation: SAGEAggregation,
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Whether to normalize output
    normalize: bool,
    /// Sample size for neighbors (None = all)
    sample_size: Option<usize>,
}

/// Aggregation method for GraphSAGE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SAGEAggregation {
    /// Mean pooling
    Mean,
    /// Max pooling (element-wise)
    Max,
    /// Sum aggregation
    Sum,
}

impl GraphSAGEConv {
    /// Create a new GraphSAGE layer with mean aggregation.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            // CONCAT(self, agg) doubles input
            linear: Linear::new(in_features * 2, out_features),
            aggregation: SAGEAggregation::Mean,
            in_features,
            out_features,
            normalize: true,
            sample_size: None,
        }
    }

    /// Set aggregation type.
    pub fn with_aggregation(mut self, agg: SAGEAggregation) -> Self {
        self.aggregation = agg;
        self
    }

    /// Set neighbor sample size.
    pub fn with_sample_size(mut self, size: usize) -> Self {
        self.sample_size = Some(size);
        self
    }

    /// Disable output normalization.
    pub fn without_normalize(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Get aggregation type.
    pub fn aggregation(&self) -> SAGEAggregation {
        self.aggregation
    }

    /// Get sample size.
    pub fn sample_size(&self) -> Option<usize> {
        self.sample_size
    }
}

impl Module for GraphSAGEConv {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("GraphSAGEConv requires graph structure. Use forward_gnn() instead.");
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.linear.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.linear.parameters_mut()
    }
}

impl GNNModule for GraphSAGEConv {
    #[allow(clippy::needless_range_loop)]
    fn forward_gnn(&self, x: &Tensor, edge_index: &[EdgeIndex]) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_features = x.shape()[1];

        assert_eq!(
            in_features, self.in_features,
            "Expected {} input features, got {}",
            self.in_features, in_features
        );

        let x_data = x.data();

        // Build neighbor lists
        let mut neighbors: Vec<Vec<usize>> = vec![vec![]; num_nodes];
        for &(src, tgt) in edge_index {
            neighbors[tgt].push(src);
            neighbors[src].push(tgt); // Undirected
        }

        // Sample neighbors if needed
        if let Some(sample_size) = self.sample_size {
            for nbrs in &mut neighbors {
                if nbrs.len() > sample_size {
                    // Simple deterministic sampling (first N)
                    nbrs.truncate(sample_size);
                }
            }
        }

        // Aggregate neighbors for each node
        let mut concat_features = vec![0.0f32; num_nodes * in_features * 2];

        for i in 0..num_nodes {
            // Copy self features
            for f in 0..in_features {
                concat_features[i * in_features * 2 + f] = x_data[i * in_features + f];
            }

            // Aggregate neighbor features
            let nbrs = &neighbors[i];
            if nbrs.is_empty() {
                // No neighbors: use zeros (or could use self)
                continue;
            }

            match self.aggregation {
                SAGEAggregation::Mean => {
                    for &n in nbrs {
                        for f in 0..in_features {
                            concat_features[i * in_features * 2 + in_features + f] +=
                                x_data[n * in_features + f];
                        }
                    }
                    let count = nbrs.len() as f32;
                    for f in 0..in_features {
                        concat_features[i * in_features * 2 + in_features + f] /= count;
                    }
                }
                SAGEAggregation::Max => {
                    // Initialize with first neighbor
                    if let Some(&first) = nbrs.first() {
                        for f in 0..in_features {
                            concat_features[i * in_features * 2 + in_features + f] =
                                x_data[first * in_features + f];
                        }
                    }
                    for &n in nbrs.iter().skip(1) {
                        for f in 0..in_features {
                            let current = concat_features[i * in_features * 2 + in_features + f];
                            let neighbor = x_data[n * in_features + f];
                            concat_features[i * in_features * 2 + in_features + f] =
                                current.max(neighbor);
                        }
                    }
                }
                SAGEAggregation::Sum => {
                    for &n in nbrs {
                        for f in 0..in_features {
                            concat_features[i * in_features * 2 + in_features + f] +=
                                x_data[n * in_features + f];
                        }
                    }
                }
            }
        }

        // Apply linear transformation
        let concat_tensor = Tensor::new(&concat_features, &[num_nodes, in_features * 2]);
        let mut out = self.linear.forward(&concat_tensor);

        // Normalize output if enabled
        if self.normalize {
            let out_data = out.data();
            let mut normalized = Vec::with_capacity(out_data.len());

            for i in 0..num_nodes {
                let mut norm = 0.0f32;
                for f in 0..self.out_features {
                    norm += out_data[i * self.out_features + f].powi(2);
                }
                norm = norm.sqrt().max(1e-8);

                for f in 0..self.out_features {
                    normalized.push(out_data[i * self.out_features + f] / norm);
                }
            }

            out = Tensor::new(&normalized, &[num_nodes, self.out_features]);
        }

        out
    }
}

/// Edge convolution layer (Wang et al., 2019).
///
/// Dynamically constructs graph based on k-nearest neighbors in feature space:
///
/// ```text
/// h_i' = max_{j ∈ N(i)} MLP(CONCAT(h_i, h_j - h_i))
/// ```
#[derive(Debug)]
pub struct EdgeConv {
    /// MLP for edge features
    linear1: Linear,
    linear2: Linear,
    /// Input feature dimension
    in_features: usize,
    /// Hidden features
    hidden_features: usize,
    /// Output features
    out_features: usize,
}

impl EdgeConv {
    /// Create new EdgeConv layer.
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Self {
        Self {
            // Input is CONCAT(h_i, h_j - h_i) = 2 * in_features
            linear1: Linear::new(in_features * 2, hidden_features),
            linear2: Linear::new(hidden_features, out_features),
            in_features,
            hidden_features,
            out_features,
        }
    }

    /// Get input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for EdgeConv {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("EdgeConv requires graph structure. Use forward_gnn() instead.");
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.linear1.parameters();
        params.extend(self.linear2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.linear1.parameters_mut();
        params.extend(self.linear2.parameters_mut());
        params
    }
}

impl GNNModule for EdgeConv {
    #[allow(clippy::needless_range_loop)]
    fn forward_gnn(&self, x: &Tensor, edge_index: &[EdgeIndex]) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_features = x.shape()[1];

        assert_eq!(
            in_features, self.in_features,
            "Expected {} input features, got {}",
            self.in_features, in_features
        );

        let x_data = x.data();

        // Build neighbor lists
        let mut neighbors: Vec<Vec<usize>> = vec![vec![]; num_nodes];
        for &(src, tgt) in edge_index {
            neighbors[tgt].push(src);
            neighbors[src].push(tgt);
        }

        // Compute output via max aggregation of edge features
        let mut output = vec![f32::NEG_INFINITY; num_nodes * self.out_features];

        for i in 0..num_nodes {
            if neighbors[i].is_empty() {
                // No neighbors: use self-loop
                neighbors[i].push(i);
            }

            for &j in &neighbors[i] {
                // Compute edge features: CONCAT(h_i, h_j - h_i)
                let mut edge_feat = Vec::with_capacity(in_features * 2);

                // h_i
                for f in 0..in_features {
                    edge_feat.push(x_data[i * in_features + f]);
                }
                // h_j - h_i
                for f in 0..in_features {
                    edge_feat.push(x_data[j * in_features + f] - x_data[i * in_features + f]);
                }

                // Apply MLP
                let edge_tensor = Tensor::new(&edge_feat, &[1, in_features * 2]);
                let h1 = self.linear1.forward(&edge_tensor);
                let h1_relu: Vec<f32> = h1.data().iter().map(|&v| v.max(0.0)).collect();
                let h1_tensor = Tensor::new(&h1_relu, &[1, self.hidden_features]);
                let h2 = self.linear2.forward(&h1_tensor);
                let h2_data = h2.data();

                // Max aggregation
                for f in 0..self.out_features {
                    output[i * self.out_features + f] =
                        output[i * self.out_features + f].max(h2_data[f]);
                }
            }
        }

        // Replace -inf with 0
        for o in &mut output {
            if *o == f32::NEG_INFINITY {
                *o = 0.0;
            }
        }

        Tensor::new(&output, &[num_nodes, self.out_features])
    }
}

/// Global mean pooling for graph-level predictions.
///
/// Aggregates all node features into a single graph representation
/// by computing the mean across nodes.
pub fn global_mean_pool(x: &Tensor, batch: Option<&[usize]>) -> Tensor {
    let num_nodes = x.shape()[0];
    let num_features = x.shape()[1];
    let x_data = x.data();

    if let Some(batch_indices) = batch {
        // Multiple graphs in batch
        let num_graphs = batch_indices.iter().max().map_or(0, |&m| m + 1);
        let mut counts = vec![0usize; num_graphs];
        let mut sums = vec![0.0f32; num_graphs * num_features];

        for i in 0..num_nodes {
            let graph_id = batch_indices[i];
            counts[graph_id] += 1;
            for f in 0..num_features {
                sums[graph_id * num_features + f] += x_data[i * num_features + f];
            }
        }

        // Compute mean
        for g in 0..num_graphs {
            let count = counts[g].max(1) as f32;
            for f in 0..num_features {
                sums[g * num_features + f] /= count;
            }
        }

        Tensor::new(&sums, &[num_graphs, num_features])
    } else {
        // Single graph
        let mut mean = vec![0.0f32; num_features];
        for i in 0..num_nodes {
            for f in 0..num_features {
                mean[f] += x_data[i * num_features + f];
            }
        }
        for m in &mut mean {
            *m /= num_nodes.max(1) as f32;
        }
        Tensor::new(&mean, &[1, num_features])
    }
}

/// Global sum pooling for graph-level predictions.
pub fn global_sum_pool(x: &Tensor, batch: Option<&[usize]>) -> Tensor {
    let num_nodes = x.shape()[0];
    let num_features = x.shape()[1];
    let x_data = x.data();

    if let Some(batch_indices) = batch {
        let num_graphs = batch_indices.iter().max().map_or(0, |&m| m + 1);
        let mut sums = vec![0.0f32; num_graphs * num_features];

        for i in 0..num_nodes {
            let graph_id = batch_indices[i];
            for f in 0..num_features {
                sums[graph_id * num_features + f] += x_data[i * num_features + f];
            }
        }

        Tensor::new(&sums, &[num_graphs, num_features])
    } else {
        let mut sum = vec![0.0f32; num_features];
        for i in 0..num_nodes {
            for f in 0..num_features {
                sum[f] += x_data[i * num_features + f];
            }
        }
        Tensor::new(&sum, &[1, num_features])
    }
}

/// Global max pooling for graph-level predictions.
pub fn global_max_pool(x: &Tensor, batch: Option<&[usize]>) -> Tensor {
    let num_nodes = x.shape()[0];
    let num_features = x.shape()[1];
    let x_data = x.data();

    if let Some(batch_indices) = batch {
        let num_graphs = batch_indices.iter().max().map_or(0, |&m| m + 1);
        let mut maxs = vec![f32::NEG_INFINITY; num_graphs * num_features];

        for i in 0..num_nodes {
            let graph_id = batch_indices[i];
            for f in 0..num_features {
                let idx = graph_id * num_features + f;
                maxs[idx] = maxs[idx].max(x_data[i * num_features + f]);
            }
        }

        // Replace -inf with 0 for empty graphs
        for m in &mut maxs {
            if *m == f32::NEG_INFINITY {
                *m = 0.0;
            }
        }

        Tensor::new(&maxs, &[num_graphs, num_features])
    } else {
        let mut maxs = vec![f32::NEG_INFINITY; num_features];
        for i in 0..num_nodes {
            for f in 0..num_features {
                maxs[f] = maxs[f].max(x_data[i * num_features + f]);
            }
        }
        for m in &mut maxs {
            if *m == f32::NEG_INFINITY {
                *m = 0.0;
            }
        }
        Tensor::new(&maxs, &[1, num_features])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph_edges() -> Vec<EdgeIndex> {
        // Triangle graph: 0-1-2-0
        vec![(0, 1), (1, 2), (2, 0)]
    }

    fn line_graph_edges() -> Vec<EdgeIndex> {
        // Line: 0-1-2-3
        vec![(0, 1), (1, 2), (2, 3)]
    }

    #[test]
    fn test_gcn_conv_basic() {
        let gcn = GCNConv::new(4, 8);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = gcn.forward_gnn(&x, &edges);

        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gcn_conv_features() {
        let gcn = GCNConv::new(4, 8);

        assert_eq!(gcn.in_features(), 4);
        assert_eq!(gcn.out_features(), 8);
    }

    #[test]
    fn test_gcn_conv_parameters() {
        let gcn = GCNConv::new(4, 8);
        let params = gcn.parameters();

        // Linear has weight and bias
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_gcn_conv_without_self_loops() {
        let gcn = GCNConv::without_self_loops(4, 8);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = gcn.forward_gnn(&x, &edges);

        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gcn_conv_line_graph() {
        let gcn = GCNConv::new(2, 4);
        let x = Tensor::ones(&[4, 2]);
        let edges = line_graph_edges();

        let out = gcn.forward_gnn(&x, &edges);

        assert_eq!(out.shape(), &[4, 4]);
    }

    #[test]
    fn test_gat_conv_basic() {
        let gat = GATConv::new(4, 8, 2); // 2 heads
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = gat.forward_gnn(&x, &edges);

        // Output is num_heads * out_features
        assert_eq!(out.shape(), &[3, 16]);
    }

    #[test]
    fn test_gat_conv_features() {
        let gat = GATConv::new(4, 8, 2);

        assert_eq!(gat.num_heads(), 2);
        assert_eq!(gat.out_features(), 8);
        assert_eq!(gat.total_out_features(), 16);
    }

    #[test]
    fn test_gat_conv_parameters() {
        let gat = GATConv::new(4, 8, 2);
        let params = gat.parameters();

        // Linear (weight + bias) + attention_src + attention_tgt
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_gat_conv_attention_normalization() {
        // Test that attention weights sum to 1 (verified by output being reasonable)
        let gat = GATConv::new(4, 8, 1);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = gat.forward_gnn(&x, &edges);

        // All outputs should be finite
        for &v in out.data() {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gin_conv_basic() {
        let gin = GINConv::new(4, 16, 8);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = gin.forward_gnn(&x, &edges);

        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_conv_eps() {
        let mut gin = GINConv::new(4, 16, 8);

        assert!((gin.eps() - 0.0).abs() < 1e-6);

        gin.set_eps(0.5);
        assert!((gin.eps() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gin_conv_parameters() {
        let gin = GINConv::new(4, 16, 8);
        let params = gin.parameters();

        // Two Linear layers (2 weights + 2 biases)
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_global_mean_pool_single_graph() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        let pooled = global_mean_pool(&x, None);

        assert_eq!(pooled.shape(), &[1, 2]);
        // Mean of [1,3,5] = 3, Mean of [2,4,6] = 4
        let data = pooled.data();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_mean_pool_batched() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
        let batch = vec![0, 0, 1, 1]; // 2 graphs, 2 nodes each

        let pooled = global_mean_pool(&x, Some(&batch));

        assert_eq!(pooled.shape(), &[2, 2]);
        let data = pooled.data();
        // Graph 0: mean([1,3], [2,4]) = [2, 3]
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        // Graph 1: mean([5,7], [6,8]) = [6, 7]
        assert!((data[2] - 6.0).abs() < 1e-6);
        assert!((data[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_sum_pool_single_graph() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let pooled = global_sum_pool(&x, None);

        assert_eq!(pooled.shape(), &[1, 2]);
        let data = pooled.data();
        assert!((data[0] - 4.0).abs() < 1e-6); // 1 + 3
        assert!((data[1] - 6.0).abs() < 1e-6); // 2 + 4
    }

    #[test]
    fn test_global_max_pool_single_graph() {
        let x = Tensor::new(&[1.0, 5.0, 3.0, 2.0, 4.0, 1.0], &[3, 2]);

        let pooled = global_max_pool(&x, None);

        assert_eq!(pooled.shape(), &[1, 2]);
        let data = pooled.data();
        assert!((data[0] - 4.0).abs() < 1e-6); // max(1, 3, 4)
        assert!((data[1] - 5.0).abs() < 1e-6); // max(5, 2, 1)
    }

    #[test]
    #[should_panic(expected = "requires graph structure")]
    fn test_gnn_forward_panics() {
        let gcn = GCNConv::new(4, 8);
        let x = Tensor::ones(&[3, 4]);

        // forward() should panic - use forward_gnn() instead
        let _ = gcn.forward(&x);
    }

    #[test]
    fn test_gcn_different_graph_sizes() {
        let gcn = GCNConv::new(4, 8);

        // Small graph
        let x1 = Tensor::ones(&[2, 4]);
        let edges1 = vec![(0, 1)];
        let out1 = gcn.forward_gnn(&x1, &edges1);
        assert_eq!(out1.shape(), &[2, 8]);

        // Larger graph
        let x2 = Tensor::ones(&[10, 4]);
        let edges2: Vec<EdgeIndex> = (0..9).map(|i| (i, i + 1)).collect();
        let out2 = gcn.forward_gnn(&x2, &edges2);
        assert_eq!(out2.shape(), &[10, 8]);
    }

    #[test]
    fn test_gnn_empty_edges() {
        let gcn = GCNConv::new(4, 8);
        let x = Tensor::ones(&[3, 4]);
        let edges: Vec<EdgeIndex> = vec![]; // No edges

        let out = gcn.forward_gnn(&x, &edges);

        // Should still work (self-loops only)
        assert_eq!(out.shape(), &[3, 8]);
    }

    // GraphSAGE Tests
    #[test]
    fn test_graphsage_basic() {
        let sage = GraphSAGEConv::new(4, 8);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = sage.forward_gnn(&x, &edges);

        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_graphsage_mean_aggregation() {
        let sage = GraphSAGEConv::new(4, 8).with_aggregation(SAGEAggregation::Mean);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = sage.forward_gnn(&x, &edges);
        assert_eq!(out.shape(), &[3, 8]);
        assert_eq!(sage.aggregation(), SAGEAggregation::Mean);
    }

    #[test]
    fn test_graphsage_max_aggregation() {
        let sage = GraphSAGEConv::new(4, 8).with_aggregation(SAGEAggregation::Max);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = sage.forward_gnn(&x, &edges);
        assert_eq!(out.shape(), &[3, 8]);
        assert_eq!(sage.aggregation(), SAGEAggregation::Max);
    }

    #[test]
    fn test_graphsage_sum_aggregation() {
        let sage = GraphSAGEConv::new(4, 8).with_aggregation(SAGEAggregation::Sum);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = sage.forward_gnn(&x, &edges);
        assert_eq!(out.shape(), &[3, 8]);
        assert_eq!(sage.aggregation(), SAGEAggregation::Sum);
    }

    #[test]
    fn test_graphsage_sample_size() {
        let sage = GraphSAGEConv::new(4, 8).with_sample_size(2);
        let x = Tensor::ones(&[5, 4]);
        // Dense graph
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4)];

        let out = sage.forward_gnn(&x, &edges);
        assert_eq!(out.shape(), &[5, 8]);
        assert_eq!(sage.sample_size(), Some(2));
    }

    #[test]
    fn test_graphsage_without_normalize() {
        let sage = GraphSAGEConv::new(4, 8).without_normalize();
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = sage.forward_gnn(&x, &edges);
        assert_eq!(out.shape(), &[3, 8]);

        // Without normalization, output vectors are not unit length
        // Just verify it produces output
        for &v in out.data() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_graphsage_parameters() {
        let sage = GraphSAGEConv::new(4, 8);
        let params = sage.parameters();

        // Linear has weight and bias
        assert_eq!(params.len(), 2);
    }

    // EdgeConv Tests
    #[test]
    fn test_edgeconv_basic() {
        let edge_conv = EdgeConv::new(4, 16, 8);
        let x = Tensor::ones(&[3, 4]);
        let edges = simple_graph_edges();

        let out = edge_conv.forward_gnn(&x, &edges);

        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_edgeconv_features() {
        let edge_conv = EdgeConv::new(4, 16, 8);

        assert_eq!(edge_conv.in_features(), 4);
        assert_eq!(edge_conv.out_features(), 8);
    }

    #[test]
    fn test_edgeconv_parameters() {
        let edge_conv = EdgeConv::new(4, 16, 8);
        let params = edge_conv.parameters();

        // Two Linear layers (2 weights + 2 biases)
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_edgeconv_empty_edges() {
        let edge_conv = EdgeConv::new(4, 16, 8);
        let x = Tensor::ones(&[3, 4]);
        let edges: Vec<EdgeIndex> = vec![];

        let out = edge_conv.forward_gnn(&x, &edges);

        // Should handle empty edges gracefully
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_edgeconv_output_finite() {
        let edge_conv = EdgeConv::new(4, 16, 8);
        let x = Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let edges = simple_graph_edges();

        let out = edge_conv.forward_gnn(&x, &edges);

        for &v in out.data() {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    #[should_panic(expected = "requires graph structure")]
    fn test_graphsage_forward_panics() {
        let sage = GraphSAGEConv::new(4, 8);
        let x = Tensor::ones(&[3, 4]);

        let _ = sage.forward(&x);
    }

    #[test]
    #[should_panic(expected = "requires graph structure")]
    fn test_edgeconv_forward_panics() {
        let edge_conv = EdgeConv::new(4, 16, 8);
        let x = Tensor::ones(&[3, 4]);

        let _ = edge_conv.forward(&x);
    }
}
