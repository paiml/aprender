//! Graph Neural Network layers for learning on graph-structured data.
//!
//! This module provides GNN layers commonly used for:
//! - AST/CFG analysis in transpilers (depyler, ruchy, bashrs)
//! - Code structure understanding
//! - Error pattern detection in CITL
//!
//! # Implemented Layers
//!
//! - [`GCNConv`] - Graph Convolutional Network (Kipf & Welling, 2017)
//! - [`GATConv`] - Graph Attention Network (Veličković et al., 2018)
//! - [`SAGEConv`] - GraphSAGE (Hamilton et al., 2017)
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::gnn::{GCNConv, GATConv, SAGEConv, AdjacencyMatrix};
//!
//! // Create adjacency matrix for a simple graph
//! let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 0]], 3);
//!
//! // GCN layer
//! let gcn = GCNConv::new(64, 32);
//! let x = Tensor::new(&vec![0.0; 3 * 64], &[3, 64]);  // 3 nodes, 64 features
//! let out = gcn.forward(&x, &adj);  // [3, 32]
//! ```
//!
//! # References
//!
//! - Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with
//!   Graph Convolutional Networks. ICLR.
//! - Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
//! - Hamilton, W. L., et al. (2017). Inductive Representation Learning on
//!   Large Graphs. NeurIPS.

use crate::autograd::Tensor;
use crate::primitives::Matrix;

/// Adjacency matrix representation for GNN operations.
///
/// Supports both dense and sparse (COO) formats for flexibility.
#[derive(Debug, Clone)]
pub struct AdjacencyMatrix {
    /// Number of nodes
    num_nodes: usize,
    /// Edge sources (COO format)
    edge_src: Vec<usize>,
    /// Edge targets (COO format)
    edge_tgt: Vec<usize>,
    /// Edge weights (optional, defaults to 1.0)
    edge_weights: Option<Vec<f32>>,
    /// Whether graph has self-loops
    has_self_loops: bool,
}

impl AdjacencyMatrix {
    /// Create adjacency matrix from edge index pairs.
    ///
    /// # Arguments
    /// * `edges` - Slice of [source, target] pairs
    /// * `num_nodes` - Total number of nodes in the graph
    ///
    /// # Example
    /// ```
    /// use aprender::nn::gnn::AdjacencyMatrix;
    ///
    /// let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 0]], 3);
    /// assert_eq!(adj.num_nodes(), 3);
    /// assert_eq!(adj.num_edges(), 3);
    /// ```
    #[must_use]
    pub fn from_edge_index(edges: &[[usize; 2]], num_nodes: usize) -> Self {
        let edge_src: Vec<usize> = edges.iter().map(|e| e[0]).collect();
        let edge_tgt: Vec<usize> = edges.iter().map(|e| e[1]).collect();

        Self {
            num_nodes,
            edge_src,
            edge_tgt,
            edge_weights: None,
            has_self_loops: false,
        }
    }

    /// Create adjacency matrix from separate source and target vectors.
    #[must_use]
    pub fn from_coo(src: Vec<usize>, tgt: Vec<usize>, num_nodes: usize) -> Self {
        Self {
            num_nodes,
            edge_src: src,
            edge_tgt: tgt,
            edge_weights: None,
            has_self_loops: false,
        }
    }

    /// Add self-loops (edges from each node to itself).
    ///
    /// Required for GCN normalization: A_hat = A + I
    #[must_use]
    pub fn add_self_loops(mut self) -> Self {
        if self.has_self_loops {
            return self;
        }

        for i in 0..self.num_nodes {
            self.edge_src.push(i);
            self.edge_tgt.push(i);
        }

        if let Some(ref mut weights) = self.edge_weights {
            weights.extend(vec![1.0; self.num_nodes]);
        }

        self.has_self_loops = true;
        self
    }

    /// Set edge weights.
    #[must_use]
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.edge_weights = Some(weights);
        self
    }

    /// Get number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get number of edges.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edge_src.len()
    }

    /// Get edge sources.
    #[must_use]
    pub fn edge_src(&self) -> &[usize] {
        &self.edge_src
    }

    /// Get edge targets.
    #[must_use]
    pub fn edge_tgt(&self) -> &[usize] {
        &self.edge_tgt
    }

    /// Check if graph has self-loops.
    #[must_use]
    pub fn has_self_loops(&self) -> bool {
        self.has_self_loops
    }

    /// Compute degree of each node (number of incoming edges).
    #[must_use]
    pub fn in_degrees(&self) -> Vec<f32> {
        let mut degrees = vec![0.0f32; self.num_nodes];
        for &tgt in &self.edge_tgt {
            if tgt < self.num_nodes {
                degrees[tgt] += 1.0;
            }
        }
        degrees
    }

    /// Compute out-degree of each node.
    #[must_use]
    pub fn out_degrees(&self) -> Vec<f32> {
        let mut degrees = vec![0.0f32; self.num_nodes];
        for &src in &self.edge_src {
            if src < self.num_nodes {
                degrees[src] += 1.0;
            }
        }
        degrees
    }

    /// Get neighbors of a specific node.
    #[must_use]
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edge_src
            .iter()
            .zip(self.edge_tgt.iter())
            .filter(|(&src, _)| src == node)
            .map(|(_, &tgt)| tgt)
            .collect()
    }

    /// Convert to dense matrix representation.
    #[must_use]
    pub fn to_dense(&self) -> Matrix<f32> {
        let n = self.num_nodes;
        let mut data = vec![0.0f32; n * n];

        for (i, (&src, &tgt)) in self.edge_src.iter().zip(self.edge_tgt.iter()).enumerate() {
            if src < n && tgt < n {
                let weight = self
                    .edge_weights
                    .as_ref()
                    .map_or(1.0, |w| w.get(i).copied().unwrap_or(1.0));
                data[src * n + tgt] = weight;
            }
        }

        Matrix::from_vec(n, n, data).expect("Valid matrix dimensions")
    }
}

/// Graph Convolutional Network layer (Kipf & Welling, 2017).
///
/// Implements the propagation rule:
/// H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
///
/// Where:
/// - Ã = A + I (adjacency with self-loops)
/// - D̃ = degree matrix of Ã
/// - W = learnable weight matrix
/// - σ = activation function (ReLU by default)
///
/// # Example
/// ```
/// use aprender::nn::gnn::{GCNConv, AdjacencyMatrix};
/// use aprender::autograd::Tensor;
///
/// let gcn = GCNConv::new(64, 32);
/// let x = Tensor::new(&vec![0.1; 3 * 64], &[3, 64]);
/// let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3).add_self_loops();
/// let out = gcn.forward(&x, &adj);
/// assert_eq!(out.shape(), &[3, 32]);
/// ```
#[derive(Debug, Clone)]
pub struct GCNConv {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Weight matrix [in_features, out_features]
    weight: Tensor,
    /// Bias vector [out_features]
    bias: Option<Tensor>,
    /// Whether to use bias
    use_bias: bool,
    /// Whether to add self-loops automatically
    add_self_loops: bool,
    /// Normalization type
    normalize: bool,
}

impl GCNConv {
    /// Create a new GCN layer.
    ///
    /// # Arguments
    /// * `in_features` - Input feature dimension per node
    /// * `out_features` - Output feature dimension per node
    #[must_use]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier initialization
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| (i as f32 * 0.1).sin() * std)
            .collect();

        let bias_data: Vec<f32> = vec![0.0; out_features];

        Self {
            in_features,
            out_features,
            weight: Tensor::new(&weight_data, &[in_features, out_features]),
            bias: Some(Tensor::new(&bias_data, &[out_features])),
            use_bias: true,
            add_self_loops: true,
            normalize: true,
        }
    }

    /// Disable bias.
    #[must_use]
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self.bias = None;
        self
    }

    /// Disable automatic self-loop addition.
    #[must_use]
    pub fn without_self_loops(mut self) -> Self {
        self.add_self_loops = false;
        self
    }

    /// Disable normalization.
    #[must_use]
    pub fn without_normalize(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Get input feature dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Forward pass: X' = D^(-1/2) A D^(-1/2) X W + b
    ///
    /// # Arguments
    /// * `x` - Node features [num_nodes, in_features]
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Output features [num_nodes, out_features]
    pub fn forward(&self, x: &Tensor, adj: &AdjacencyMatrix) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_feat = x.shape()[1];

        assert_eq!(
            in_feat, self.in_features,
            "Input features mismatch: expected {}, got {}",
            self.in_features, in_feat
        );

        // Prepare adjacency with self-loops if needed
        let adj_normalized = if self.add_self_loops && !adj.has_self_loops() {
            adj.clone().add_self_loops()
        } else {
            adj.clone()
        };

        // Compute symmetric normalization: D^(-1/2) A D^(-1/2)
        let degrees = adj_normalized.in_degrees();
        let norm_coeffs: Vec<f32> = degrees
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // Step 1: Transform features: H = X * W
        let x_data = x.data();
        let w_data = self.weight.data();
        let mut h_data = vec![0.0f32; num_nodes * self.out_features];

        for node in 0..num_nodes {
            for out_f in 0..self.out_features {
                let mut sum = 0.0f32;
                for in_f in 0..self.in_features {
                    sum += x_data[node * in_feat + in_f] * w_data[in_f * self.out_features + out_f];
                }
                h_data[node * self.out_features + out_f] = sum;
            }
        }

        // Step 2: Message passing with normalization
        let mut output = vec![0.0f32; num_nodes * self.out_features];

        if self.normalize {
            // Normalized adjacency: D^(-1/2) A D^(-1/2)
            for (i, (&src, &tgt)) in adj_normalized
                .edge_src()
                .iter()
                .zip(adj_normalized.edge_tgt().iter())
                .enumerate()
            {
                if src < num_nodes && tgt < num_nodes {
                    let edge_weight = adj_normalized
                        .edge_weights
                        .as_ref()
                        .map_or(1.0, |w| w.get(i).copied().unwrap_or(1.0));

                    let norm = norm_coeffs[src] * norm_coeffs[tgt] * edge_weight;

                    for f in 0..self.out_features {
                        output[tgt * self.out_features + f] +=
                            norm * h_data[src * self.out_features + f];
                    }
                }
            }
        } else {
            // Simple sum aggregation without normalization
            for (&src, &tgt) in adj_normalized
                .edge_src()
                .iter()
                .zip(adj_normalized.edge_tgt().iter())
            {
                if src < num_nodes && tgt < num_nodes {
                    for f in 0..self.out_features {
                        output[tgt * self.out_features + f] += h_data[src * self.out_features + f];
                    }
                }
            }
        }

        // Step 3: Add bias
        if self.use_bias {
            if let Some(ref bias) = self.bias {
                let bias_data = bias.data();
                for node in 0..num_nodes {
                    for f in 0..self.out_features {
                        output[node * self.out_features + f] += bias_data[f];
                    }
                }
            }
        }

        Tensor::new(&output, &[num_nodes, self.out_features])
    }

    /// Get weight tensor (for inspection/serialization).
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get bias tensor (for inspection/serialization).
    #[must_use]
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

/// Aggregation method for GraphSAGE.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SAGEAggregation {
    /// Mean aggregation (default)
    #[default]
    Mean,
    /// Max pooling aggregation
    Max,
    /// Sum aggregation
    Sum,
    /// LSTM aggregation (sequential)
    Lstm,
}

/// GraphSAGE convolutional layer (Hamilton et al., 2017).
///
/// Implements the aggregation rule:
/// h_v^(l+1) = σ(W · CONCAT(h_v^(l), AGG({h_u^(l) : u ∈ N(v)})))
///
/// Where AGG can be mean, max, sum, or LSTM aggregation.
///
/// # Example
/// ```
/// use aprender::nn::gnn::{SAGEConv, AdjacencyMatrix, SAGEAggregation};
/// use aprender::autograd::Tensor;
///
/// let sage = SAGEConv::new(64, 32).with_aggregation(SAGEAggregation::Mean);
/// let x = Tensor::new(&vec![0.1; 5 * 64], &[5, 64]);
/// let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);
/// let out = sage.forward(&x, &adj);
/// assert_eq!(out.shape(), &[5, 32]);
/// ```
#[derive(Debug, Clone)]
pub struct SAGEConv {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Weight for self features [in_features, out_features]
    weight_self: Tensor,
    /// Weight for neighbor aggregation [in_features, out_features]
    weight_neigh: Tensor,
    /// Bias vector [out_features]
    bias: Option<Tensor>,
    /// Aggregation method
    aggregation: SAGEAggregation,
    /// Whether to normalize output
    normalize: bool,
    /// Root weight (whether to include self in aggregation)
    root_weight: bool,
}

impl SAGEConv {
    /// Create a new GraphSAGE layer.
    ///
    /// # Arguments
    /// * `in_features` - Input feature dimension per node
    /// * `out_features` - Output feature dimension per node
    #[must_use]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let std = (2.0 / (in_features + out_features) as f32).sqrt();

        let weight_self_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| (i as f32 * 0.13).sin() * std)
            .collect();

        let weight_neigh_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| (i as f32 * 0.17).sin() * std)
            .collect();

        let bias_data = vec![0.0f32; out_features];

        Self {
            in_features,
            out_features,
            weight_self: Tensor::new(&weight_self_data, &[in_features, out_features]),
            weight_neigh: Tensor::new(&weight_neigh_data, &[in_features, out_features]),
            bias: Some(Tensor::new(&bias_data, &[out_features])),
            aggregation: SAGEAggregation::Mean,
            normalize: false,
            root_weight: true,
        }
    }

    /// Set aggregation method.
    #[must_use]
    pub fn with_aggregation(mut self, agg: SAGEAggregation) -> Self {
        self.aggregation = agg;
        self
    }

    /// Enable L2 normalization of output.
    #[must_use]
    pub fn with_normalize(mut self) -> Self {
        self.normalize = true;
        self
    }

    /// Disable root weight (self features).
    #[must_use]
    pub fn without_root(mut self) -> Self {
        self.root_weight = false;
        self
    }

    /// Disable bias.
    #[must_use]
    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// Get input feature dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get aggregation method.
    #[must_use]
    pub fn aggregation(&self) -> SAGEAggregation {
        self.aggregation
    }

    /// Forward pass with neighbor aggregation.
    ///
    /// # Arguments
    /// * `x` - Node features [num_nodes, in_features]
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Output features [num_nodes, out_features]
    pub fn forward(&self, x: &Tensor, adj: &AdjacencyMatrix) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_feat = x.shape()[1];

        assert_eq!(in_feat, self.in_features);

        let x_data = x.data();
        let ws_data = self.weight_self.data();
        let wn_data = self.weight_neigh.data();

        // Build neighbor lists for aggregation
        let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for (&src, &tgt) in adj.edge_src().iter().zip(adj.edge_tgt().iter()) {
            if tgt < num_nodes && src < num_nodes {
                neighbor_lists[tgt].push(src);
            }
        }

        let mut output = vec![0.0f32; num_nodes * self.out_features];

        for node in 0..num_nodes {
            let neighbors = &neighbor_lists[node];

            // Aggregate neighbor features
            let mut agg_features = vec![0.0f32; self.in_features];

            if !neighbors.is_empty() {
                match self.aggregation {
                    SAGEAggregation::Mean => {
                        for &neigh in neighbors {
                            for f in 0..self.in_features {
                                agg_features[f] += x_data[neigh * in_feat + f];
                            }
                        }
                        let count = neighbors.len() as f32;
                        for f in &mut agg_features {
                            *f /= count;
                        }
                    }
                    SAGEAggregation::Sum => {
                        for &neigh in neighbors {
                            for f in 0..self.in_features {
                                agg_features[f] += x_data[neigh * in_feat + f];
                            }
                        }
                    }
                    SAGEAggregation::Max => {
                        agg_features = vec![f32::NEG_INFINITY; self.in_features];
                        for &neigh in neighbors {
                            for f in 0..self.in_features {
                                agg_features[f] = agg_features[f].max(x_data[neigh * in_feat + f]);
                            }
                        }
                        // Replace -inf with 0 for nodes without that feature
                        for f in &mut agg_features {
                            if f.is_infinite() {
                                *f = 0.0;
                            }
                        }
                    }
                    SAGEAggregation::Lstm => {
                        // Simplified: just use mean for now (full LSTM would need state)
                        for &neigh in neighbors {
                            for f in 0..self.in_features {
                                agg_features[f] += x_data[neigh * in_feat + f];
                            }
                        }
                        let count = neighbors.len() as f32;
                        for f in &mut agg_features {
                            *f /= count;
                        }
                    }
                }
            }

            // Transform: out = W_self * x + W_neigh * agg_neigh + bias
            for out_f in 0..self.out_features {
                let mut val = 0.0f32;

                // Self contribution
                if self.root_weight {
                    for in_f in 0..self.in_features {
                        val += x_data[node * in_feat + in_f]
                            * ws_data[in_f * self.out_features + out_f];
                    }
                }

                // Neighbor contribution
                for in_f in 0..self.in_features {
                    val += agg_features[in_f] * wn_data[in_f * self.out_features + out_f];
                }

                output[node * self.out_features + out_f] = val;
            }
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data();
            for node in 0..num_nodes {
                for f in 0..self.out_features {
                    output[node * self.out_features + f] += bias_data[f];
                }
            }
        }

        // L2 normalize if requested
        if self.normalize {
            for node in 0..num_nodes {
                let start = node * self.out_features;
                let end = start + self.out_features;
                let norm: f32 = output[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for val in &mut output[start..end] {
                        *val /= norm;
                    }
                }
            }
        }

        Tensor::new(&output, &[num_nodes, self.out_features])
    }
}

/// Graph Attention Network layer (Veličković et al., 2018).
///
/// Implements multi-head attention over graph neighbors:
/// α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
/// h'_i = σ(Σ_j α_ij W h_j)
///
/// # Example
/// ```
/// use aprender::nn::gnn::{GATConv, AdjacencyMatrix};
/// use aprender::autograd::Tensor;
///
/// let gat = GATConv::new(64, 32, 4);  // 4 attention heads
/// let x = Tensor::new(&vec![0.1; 5 * 64], &[5, 64]);
/// let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3]], 5);
/// let out = gat.forward(&x, &adj);
/// assert_eq!(out.shape(), &[5, 128]);  // 32 * 4 heads
/// ```
#[derive(Debug, Clone)]
pub struct GATConv {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension (per head)
    out_features: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Linear transformation weight [in_features, out_features * num_heads]
    weight: Tensor,
    /// Attention weight for source nodes [num_heads, out_features]
    att_src: Tensor,
    /// Attention weight for target nodes [num_heads, out_features]
    att_tgt: Tensor,
    /// Bias [out_features * num_heads]
    bias: Option<Tensor>,
    /// Negative slope for LeakyReLU
    negative_slope: f32,
    /// Dropout probability
    dropout: f32,
    /// Whether to concatenate heads (true) or average them (false)
    concat: bool,
    /// Add self-loops
    add_self_loops: bool,
}

impl GATConv {
    /// Create a new GAT layer.
    ///
    /// # Arguments
    /// * `in_features` - Input feature dimension per node
    /// * `out_features` - Output feature dimension per head
    /// * `num_heads` - Number of attention heads
    #[must_use]
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let total_out = out_features * num_heads;
        let std = (2.0 / (in_features + out_features) as f32).sqrt();

        let weight_data: Vec<f32> = (0..in_features * total_out)
            .map(|i| (i as f32 * 0.11).sin() * std)
            .collect();

        let att_std = (1.0 / out_features as f32).sqrt();
        let att_src_data: Vec<f32> = (0..num_heads * out_features)
            .map(|i| (i as f32 * 0.19).sin() * att_std)
            .collect();

        let att_tgt_data: Vec<f32> = (0..num_heads * out_features)
            .map(|i| (i as f32 * 0.23).sin() * att_std)
            .collect();

        let bias_data = vec![0.0f32; total_out];

        Self {
            in_features,
            out_features,
            num_heads,
            weight: Tensor::new(&weight_data, &[in_features, total_out]),
            att_src: Tensor::new(&att_src_data, &[num_heads, out_features]),
            att_tgt: Tensor::new(&att_tgt_data, &[num_heads, out_features]),
            bias: Some(Tensor::new(&bias_data, &[total_out])),
            negative_slope: 0.2,
            dropout: 0.0,
            concat: true,
            add_self_loops: true,
        }
    }

    /// Set negative slope for LeakyReLU.
    #[must_use]
    pub fn with_negative_slope(mut self, slope: f32) -> Self {
        self.negative_slope = slope;
        self
    }

    /// Set dropout probability.
    #[must_use]
    pub fn with_dropout(mut self, p: f32) -> Self {
        self.dropout = p;
        self
    }

    /// Average heads instead of concatenating.
    #[must_use]
    pub fn without_concat(mut self) -> Self {
        self.concat = false;
        self
    }

    /// Disable automatic self-loop addition.
    #[must_use]
    pub fn without_self_loops(mut self) -> Self {
        self.add_self_loops = false;
        self
    }

    /// Disable bias.
    #[must_use]
    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// Get input feature dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension per head.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get total output dimension (out_features * num_heads if concat, else out_features).
    #[must_use]
    pub fn total_out_features(&self) -> usize {
        if self.concat {
            self.out_features * self.num_heads
        } else {
            self.out_features
        }
    }

    /// LeakyReLU activation.
    fn leaky_relu(&self, x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            self.negative_slope * x
        }
    }

    /// Forward pass with multi-head attention.
    ///
    /// # Arguments
    /// * `x` - Node features [num_nodes, in_features]
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Output features [num_nodes, out_features * num_heads] if concat
    /// or [num_nodes, out_features] if averaging heads
    #[allow(clippy::too_many_lines)]
    pub fn forward(&self, x: &Tensor, adj: &AdjacencyMatrix) -> Tensor {
        let num_nodes = x.shape()[0];
        let in_feat = x.shape()[1];

        assert_eq!(in_feat, self.in_features);

        // Add self-loops if needed
        let adj_with_loops = if self.add_self_loops && !adj.has_self_loops() {
            adj.clone().add_self_loops()
        } else {
            adj.clone()
        };

        let x_data = x.data();
        let w_data = self.weight.data();
        let att_src_data = self.att_src.data();
        let att_tgt_data = self.att_tgt.data();

        let total_out = self.out_features * self.num_heads;

        // Step 1: Linear transformation: H = X * W [num_nodes, out_features * num_heads]
        let mut h_data = vec![0.0f32; num_nodes * total_out];
        for node in 0..num_nodes {
            for out_f in 0..total_out {
                let mut sum = 0.0f32;
                for in_f in 0..self.in_features {
                    sum += x_data[node * in_feat + in_f] * w_data[in_f * total_out + out_f];
                }
                h_data[node * total_out + out_f] = sum;
            }
        }

        // Step 2: Compute attention scores for each head
        // Build neighbor list
        let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for (&src, &tgt) in adj_with_loops
            .edge_src()
            .iter()
            .zip(adj_with_loops.edge_tgt().iter())
        {
            if tgt < num_nodes && src < num_nodes {
                neighbor_lists[tgt].push(src);
            }
        }

        // Output: [num_nodes, total_out] if concat, [num_nodes, out_features] if avg
        let final_out = if self.concat {
            total_out
        } else {
            self.out_features
        };
        let mut output = vec![0.0f32; num_nodes * final_out];

        // For each node, compute attention-weighted sum of neighbor features
        for node in 0..num_nodes {
            let neighbors = &neighbor_lists[node];

            if neighbors.is_empty() {
                continue;
            }

            // For each head
            for head in 0..self.num_heads {
                let head_offset = head * self.out_features;

                // Compute attention scores for all neighbors
                let mut attn_scores: Vec<f32> = Vec::with_capacity(neighbors.len());

                for &neigh in neighbors {
                    // e_ij = LeakyReLU(a_src^T * h_i + a_tgt^T * h_j)
                    let mut score = 0.0f32;

                    // Source (current node) contribution
                    for f in 0..self.out_features {
                        score += att_src_data[head * self.out_features + f]
                            * h_data[node * total_out + head_offset + f];
                    }

                    // Target (neighbor) contribution
                    for f in 0..self.out_features {
                        score += att_tgt_data[head * self.out_features + f]
                            * h_data[neigh * total_out + head_offset + f];
                    }

                    attn_scores.push(self.leaky_relu(score));
                }

                // Softmax over attention scores
                let max_score = attn_scores
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> =
                    attn_scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> =
                    exp_scores.iter().map(|&e| e / (sum_exp + 1e-8)).collect();

                // Compute attention-weighted sum
                for (i, &neigh) in neighbors.iter().enumerate() {
                    let alpha = attn_weights[i];

                    if self.concat {
                        for f in 0..self.out_features {
                            output[node * final_out + head_offset + f] +=
                                alpha * h_data[neigh * total_out + head_offset + f];
                        }
                    } else {
                        // Average across heads
                        for f in 0..self.out_features {
                            output[node * final_out + f] += alpha
                                * h_data[neigh * total_out + head_offset + f]
                                / self.num_heads as f32;
                        }
                    }
                }
            }
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data();
            if self.concat {
                for node in 0..num_nodes {
                    for f in 0..final_out {
                        output[node * final_out + f] += bias_data[f];
                    }
                }
            } else {
                // When averaging, we also average the bias
                for node in 0..num_nodes {
                    for f in 0..self.out_features {
                        let mut avg_bias = 0.0f32;
                        for head in 0..self.num_heads {
                            avg_bias += bias_data[head * self.out_features + f];
                        }
                        output[node * final_out + f] += avg_bias / self.num_heads as f32;
                    }
                }
            }
        }

        Tensor::new(&output, &[num_nodes, final_out])
    }
}

/// Message Passing Neural Network base trait.
///
/// Defines the generic message passing framework that underlies all GNN layers.
pub trait MessagePassing {
    /// Compute messages from source to target nodes.
    fn message(&self, x_src: &Tensor, x_tgt: &Tensor, edge_index: &AdjacencyMatrix) -> Tensor;

    /// Aggregate messages for each node.
    fn aggregate(
        &self,
        messages: &Tensor,
        edge_index: &AdjacencyMatrix,
        num_nodes: usize,
    ) -> Tensor;

    /// Update node representations based on aggregated messages.
    fn update(&self, x: &Tensor, aggregated: &Tensor) -> Tensor;

    /// Full message passing forward.
    fn propagate(&self, x: &Tensor, edge_index: &AdjacencyMatrix) -> Tensor {
        let messages = self.message(x, x, edge_index);
        let aggregated = self.aggregate(&messages, edge_index, x.shape()[0]);
        self.update(x, &aggregated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create deterministic test data.
    fn create_test_tensor(shape: &[usize], seed: u32) -> Tensor {
        let len: usize = shape.iter().product();
        let data: Vec<f32> = (0..len)
            .map(|i| ((i as f32 + seed as f32) * 0.1).sin())
            .collect();
        Tensor::new(&data, shape)
    }

    // ==================== AdjacencyMatrix Tests ====================

    #[test]
    fn test_adjacency_matrix_creation() {
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 0]], 3);
        assert_eq!(adj.num_nodes(), 3);
        assert_eq!(adj.num_edges(), 3);
    }

    #[test]
    fn test_adjacency_matrix_from_coo() {
        let adj = AdjacencyMatrix::from_coo(vec![0, 1, 2], vec![1, 2, 0], 3);
        assert_eq!(adj.num_nodes(), 3);
        assert_eq!(adj.num_edges(), 3);
    }

    #[test]
    fn test_adjacency_matrix_add_self_loops() {
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);
        assert!(!adj.has_self_loops());

        let adj_with_loops = adj.add_self_loops();
        assert!(adj_with_loops.has_self_loops());
        assert_eq!(adj_with_loops.num_edges(), 5); // 2 original + 3 self-loops
    }

    #[test]
    fn test_adjacency_matrix_degrees() {
        // Graph: 0 -> 1 -> 2
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let in_deg = adj.in_degrees();
        assert_eq!(in_deg, vec![0.0, 1.0, 1.0]); // 0 has no incoming, 1 and 2 have 1 each

        let out_deg = adj.out_degrees();
        assert_eq!(out_deg, vec![1.0, 1.0, 0.0]); // 0 and 1 have 1 outgoing, 2 has none
    }

    #[test]
    fn test_adjacency_matrix_neighbors() {
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [0, 2], [1, 2]], 3);
        let neighbors = adj.neighbors(0);
        assert_eq!(neighbors, vec![1, 2]);
    }

    #[test]
    fn test_adjacency_matrix_to_dense() {
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);
        let dense = adj.to_dense();

        assert_eq!(dense.n_rows(), 3);
        assert_eq!(dense.n_cols(), 3);
        // Check edge (0,1) exists
        assert!((dense.get(0, 1) - 1.0).abs() < 0.01);
        // Check edge (1,2) exists
        assert!((dense.get(1, 2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_adjacency_matrix_with_weights() {
        let adj =
            AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3).with_weights(vec![0.5, 2.0]);
        let dense = adj.to_dense();

        assert!((dense.get(0, 1) - 0.5).abs() < 0.01);
        assert!((dense.get(1, 2) - 2.0).abs() < 0.01);
    }

    // ==================== GCNConv Tests ====================

    #[test]
    fn test_gcn_creation() {
        let gcn = GCNConv::new(64, 32);
        assert_eq!(gcn.in_features(), 64);
        assert_eq!(gcn.out_features(), 32);
    }

    #[test]
    fn test_gcn_without_bias() {
        let gcn = GCNConv::new(64, 32).without_bias();
        assert!(gcn.bias().is_none());
    }

    #[test]
    fn test_gcn_forward_shape() {
        let gcn = GCNConv::new(8, 4);
        let x = create_test_tensor(&[5, 8], 1); // 5 nodes, 8 features
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

        let out = gcn.forward(&x, &adj);
        assert_eq!(out.shape(), &[5, 4]);
    }

    #[test]
    fn test_gcn_forward_values() {
        let gcn = GCNConv::new(4, 2);

        // Simple graph: 0 <-> 1 <-> 2
        let x = Tensor::new(
            &[
                1.0, 0.0, 0.0, 0.0, // Node 0
                0.0, 1.0, 0.0, 0.0, // Node 1
                0.0, 0.0, 1.0, 0.0, // Node 2
            ],
            &[3, 4],
        );
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0], [1, 2], [2, 1]], 3);

        let out = gcn.forward(&x, &adj);

        // Output should be non-zero (features are propagated)
        let out_data = out.data();
        let sum: f32 = out_data.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Output should have non-zero values");
    }

    #[test]
    fn test_gcn_normalized_aggregation() {
        let gcn = GCNConv::new(2, 2);

        // Complete graph K3 (fully connected)
        let x = Tensor::new(
            &[
                1.0, 1.0, // Node 0
                1.0, 1.0, // Node 1
                1.0, 1.0, // Node 2
            ],
            &[3, 2],
        );
        let adj =
            AdjacencyMatrix::from_edge_index(&[[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]], 3);

        let out = gcn.forward(&x, &adj);

        // All nodes should have similar output (symmetric graph, same features)
        let out_data = out.data();
        let diff_01 = (out_data[0] - out_data[2]).abs() + (out_data[1] - out_data[3]).abs();
        let diff_12 = (out_data[2] - out_data[4]).abs() + (out_data[3] - out_data[5]).abs();

        assert!(diff_01 < 0.1, "Symmetric nodes should have similar outputs");
        assert!(diff_12 < 0.1, "Symmetric nodes should have similar outputs");
    }

    #[test]
    fn test_gcn_without_self_loops() {
        let gcn = GCNConv::new(4, 2).without_self_loops();
        let x = create_test_tensor(&[3, 4], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = gcn.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 2]);
    }

    #[test]
    fn test_gcn_without_normalize() {
        let gcn = GCNConv::new(4, 2).without_normalize();
        let x = create_test_tensor(&[3, 4], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = gcn.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 2]);
    }

    // ==================== SAGEConv Tests ====================

    #[test]
    fn test_sage_creation() {
        let sage = SAGEConv::new(64, 32);
        assert_eq!(sage.in_features(), 64);
        assert_eq!(sage.out_features(), 32);
        assert_eq!(sage.aggregation(), SAGEAggregation::Mean);
    }

    #[test]
    fn test_sage_with_aggregation() {
        let sage_max = SAGEConv::new(64, 32).with_aggregation(SAGEAggregation::Max);
        assert_eq!(sage_max.aggregation(), SAGEAggregation::Max);

        let sage_sum = SAGEConv::new(64, 32).with_aggregation(SAGEAggregation::Sum);
        assert_eq!(sage_sum.aggregation(), SAGEAggregation::Sum);
    }

    #[test]
    fn test_sage_forward_shape() {
        let sage = SAGEConv::new(8, 4);
        let x = create_test_tensor(&[5, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

        let out = sage.forward(&x, &adj);
        assert_eq!(out.shape(), &[5, 4]);
    }

    #[test]
    fn test_sage_mean_aggregation() {
        let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Mean);
        let x = Tensor::new(
            &[
                1.0, 0.0, 0.0, 0.0, // Node 0
                0.0, 1.0, 0.0, 0.0, // Node 1
                0.0, 0.0, 1.0, 0.0, // Node 2
            ],
            &[3, 4],
        );
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = sage.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 2]);
    }

    #[test]
    fn test_sage_max_aggregation() {
        let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Max);
        let x = Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, // Node 0
                5.0, 6.0, 7.0, 8.0, // Node 1
            ],
            &[2, 4],
        );
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0]], 2);

        let out = sage.forward(&x, &adj);
        assert_eq!(out.shape(), &[2, 2]);
    }

    #[test]
    fn test_sage_sum_aggregation() {
        let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Sum);
        let x = create_test_tensor(&[3, 4], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = sage.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 2]);
    }

    #[test]
    fn test_sage_with_normalize() {
        let sage = SAGEConv::new(4, 2).with_normalize();
        let x = create_test_tensor(&[3, 4], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = sage.forward(&x, &adj);

        // Check that outputs are normalized (L2 norm ≈ 1)
        let out_data = out.data();
        for node in 0..3 {
            let norm: f32 = (0..2)
                .map(|f| out_data[node * 2 + f].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01 || norm < 0.01,
                "Normalized output should have unit norm, got {}",
                norm
            );
        }
    }

    #[test]
    fn test_sage_without_root() {
        let sage = SAGEConv::new(4, 2).without_root();
        let x = create_test_tensor(&[3, 4], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = sage.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 2]);
    }

    // ==================== GATConv Tests ====================

    #[test]
    fn test_gat_creation() {
        let gat = GATConv::new(64, 32, 4);
        assert_eq!(gat.in_features(), 64);
        assert_eq!(gat.out_features(), 32);
        assert_eq!(gat.num_heads(), 4);
        assert_eq!(gat.total_out_features(), 128); // 32 * 4
    }

    #[test]
    fn test_gat_without_concat() {
        let gat = GATConv::new(64, 32, 4).without_concat();
        assert_eq!(gat.total_out_features(), 32); // Averaged, not concatenated
    }

    #[test]
    fn test_gat_forward_shape_concat() {
        let gat = GATConv::new(8, 4, 2); // 2 heads, 4 features each
        let x = create_test_tensor(&[5, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

        let out = gat.forward(&x, &adj);
        assert_eq!(out.shape(), &[5, 8]); // 4 * 2 = 8
    }

    #[test]
    fn test_gat_forward_shape_avg() {
        let gat = GATConv::new(8, 4, 2).without_concat();
        let x = create_test_tensor(&[5, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

        let out = gat.forward(&x, &adj);
        assert_eq!(out.shape(), &[5, 4]); // Averaged heads
    }

    #[test]
    fn test_gat_attention_different_neighbors() {
        let gat = GATConv::new(4, 2, 1);

        // Graph where node 2 has two different neighbors
        let x = Tensor::new(
            &[
                1.0, 0.0, 0.0, 0.0, // Node 0 (distinct feature)
                0.0, 1.0, 0.0, 0.0, // Node 1 (distinct feature)
                0.0, 0.0, 0.0, 0.0, // Node 2 (target)
            ],
            &[3, 4],
        );
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 2], [1, 2]], 3);

        let out = gat.forward(&x, &adj);

        // Node 2 should have output that is a weighted combination
        let out_data = out.data();
        let node2_out = &out_data[4..6];
        let has_nonzero = node2_out.iter().any(|&x| x.abs() > 1e-6);
        assert!(
            has_nonzero,
            "Node 2 should have non-zero output from attention"
        );
    }

    #[test]
    fn test_gat_with_negative_slope() {
        let gat = GATConv::new(8, 4, 2).with_negative_slope(0.1);
        let x = create_test_tensor(&[3, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = gat.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_without_self_loops() {
        let gat = GATConv::new(8, 4, 2).without_self_loops();
        let x = create_test_tensor(&[3, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = gat.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_without_bias() {
        let gat = GATConv::new(8, 4, 2).without_bias();
        let x = create_test_tensor(&[3, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

        let out = gat.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_single_head() {
        let gat = GATConv::new(8, 4, 1);
        let x = create_test_tensor(&[5, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3]], 5);

        let out = gat.forward(&x, &adj);
        assert_eq!(out.shape(), &[5, 4]);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_gnn_stack() {
        // Test stacking multiple GNN layers
        let gcn1 = GCNConv::new(8, 16);
        let gcn2 = GCNConv::new(16, 4);

        let x = create_test_tensor(&[5, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], 5);

        let h1 = gcn1.forward(&x, &adj);
        assert_eq!(h1.shape(), &[5, 16]);

        let h2 = gcn2.forward(&h1, &adj);
        assert_eq!(h2.shape(), &[5, 4]);
    }

    #[test]
    fn test_gnn_heterogeneous_layers() {
        // Test mixing different GNN layers
        let gcn = GCNConv::new(8, 16);
        let gat = GATConv::new(16, 8, 2).without_concat();
        let sage = SAGEConv::new(8, 4);

        let x = create_test_tensor(&[5, 8], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

        let h1 = gcn.forward(&x, &adj);
        assert_eq!(h1.shape(), &[5, 16]);

        let h2 = gat.forward(&h1, &adj);
        assert_eq!(h2.shape(), &[5, 8]);

        let h3 = sage.forward(&h2, &adj);
        assert_eq!(h3.shape(), &[5, 4]);
    }

    #[test]
    fn test_gnn_empty_graph() {
        let gcn = GCNConv::new(4, 2);
        let x = create_test_tensor(&[3, 4], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[], 3); // No edges

        let out = gcn.forward(&x, &adj);
        assert_eq!(out.shape(), &[3, 2]);
    }

    #[test]
    fn test_gnn_single_node() {
        let gcn = GCNConv::new(4, 2);
        let x = create_test_tensor(&[1, 4], 1);
        let adj = AdjacencyMatrix::from_edge_index(&[], 1);

        let out = gcn.forward(&x, &adj);
        assert_eq!(out.shape(), &[1, 2]);
    }

    #[test]
    fn test_gnn_disconnected_graph() {
        // Graph with two disconnected components
        let sage = SAGEConv::new(4, 2);
        let x = create_test_tensor(&[4, 4], 1);
        // Component 1: 0-1, Component 2: 2-3
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0], [2, 3], [3, 2]], 4);

        let out = sage.forward(&x, &adj);
        assert_eq!(out.shape(), &[4, 2]);
    }

    // ==================== Default Trait Tests ====================

    #[test]
    fn test_sage_aggregation_default() {
        assert_eq!(SAGEAggregation::default(), SAGEAggregation::Mean);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_gnn_large_graph() {
        let gcn = GCNConv::new(16, 8);

        // Create a larger graph (100 nodes, ~200 edges)
        let mut edges = Vec::new();
        for i in 0..100 {
            edges.push([i, (i + 1) % 100]); // Ring
            if i < 50 {
                edges.push([i, i + 50]); // Cross connections
            }
        }

        let x = create_test_tensor(&[100, 16], 1);
        let adj = AdjacencyMatrix::from_edge_index(&edges, 100);

        let out = gcn.forward(&x, &adj);
        assert_eq!(out.shape(), &[100, 8]);
    }

    #[test]
    fn test_gat_multiple_heads_attention() {
        // Test that different heads learn different patterns
        let gat = GATConv::new(4, 2, 4); // 4 heads

        let x = Tensor::new(
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &[4, 4],
        );
        let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 0]], 4);

        let out = gat.forward(&x, &adj);
        assert_eq!(out.shape(), &[4, 8]); // 2 * 4 heads = 8

        // Each head should contribute to the output
        let out_data = out.data();
        let has_variance = out_data.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6);
        assert!(has_variance, "Multi-head output should have variance");
    }
}
