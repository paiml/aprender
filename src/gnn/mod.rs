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
//!   Large Graphs (`GraphSAGE`). `NeurIPS`.

use crate::autograd::Tensor;
use crate::nn::{Linear, Module};

/// Edge index type: (`source_node`, `target_node`)
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
/// where `d_i` is the degree of node i.
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension.
    #[must_use]
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
    /// Negative slope for `LeakyReLU`
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
    #[must_use]
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
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get output dimension per head.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Total output dimension (`out_features` * `num_heads`).
    #[must_use]
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

mod gin_conv;
pub use gin_conv::*;

mod accumulate;
pub use accumulate::*;

#[cfg(test)]
mod tests;
