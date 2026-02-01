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
//! - [`SAGEConv`] - `GraphSAGE` (Hamilton et al., 2017)
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
//!   Large Graphs. `NeurIPS`.

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
    /// Required for GCN normalization: `A_hat` = A + I
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
/// - σ = activation function (`ReLU` by default)
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
    /// Weight matrix [`in_features`, `out_features`]
    weight: Tensor,
    /// Bias vector [`out_features`]
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
    /// * `x` - Node features [`num_nodes`, `in_features`]
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Output features [`num_nodes`, `out_features`]
    #[must_use]
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

/// Aggregation method for `GraphSAGE`.
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

/// `GraphSAGE` convolutional layer (Hamilton et al., 2017).
///
/// Implements the aggregation rule:
/// `h_v^(l+1)` = σ(W · `CONCAT(h_v^(l)`, `AGG({h_u^(l)` : u ∈ N(v)})))
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
    /// Weight for self features [`in_features`, `out_features`]
    weight_self: Tensor,
    /// Weight for neighbor aggregation [`in_features`, `out_features`]
    weight_neigh: Tensor,
    /// Bias vector [`out_features`]
    bias: Option<Tensor>,
    /// Aggregation method
    aggregation: SAGEAggregation,
    /// Whether to normalize output
    normalize: bool,
    /// Root weight (whether to include self in aggregation)
    root_weight: bool,
}

impl SAGEConv {
    /// Create a new `GraphSAGE` layer.
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
    /// * `x` - Node features [`num_nodes`, `in_features`]
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Output features [`num_nodes`, `out_features`]
    #[must_use]
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
/// `α_ij` = `softmax_j(LeakyReLU(a^T` [`Wh_i` || `Wh_j`]))
/// h'_i = `σ(Σ_j` `α_ij` W `h_j`)
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
    /// Linear transformation weight [`in_features`, `out_features` * `num_heads`]
    weight: Tensor,
    /// Attention weight for source nodes [`num_heads`, `out_features`]
    att_src: Tensor,
    /// Attention weight for target nodes [`num_heads`, `out_features`]
    att_tgt: Tensor,
    /// Bias [`out_features` * `num_heads`]
    bias: Option<Tensor>,
    /// Negative slope for `LeakyReLU`
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

    /// Set negative slope for `LeakyReLU`.
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

    /// Get total output dimension (`out_features` * `num_heads` if concat, else `out_features`).
    #[must_use]
    pub fn total_out_features(&self) -> usize {
        if self.concat {
            self.out_features * self.num_heads
        } else {
            self.out_features
        }
    }

    /// `LeakyReLU` activation.
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
    /// * `x` - Node features [`num_nodes`, `in_features`]
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Output features [`num_nodes`, `out_features` * `num_heads`] if concat
    /// or [`num_nodes`, `out_features`] if averaging heads
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
mod tests;
