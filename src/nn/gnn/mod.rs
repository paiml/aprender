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
    // Contract: gnn-v1, equation = "gcn_aggregate"
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

#[path = "sage_gat.rs"]
mod sage_gat;
pub use sage_gat::*;

#[path = "message_passing.rs"]
mod message_passing;
pub use message_passing::*;
