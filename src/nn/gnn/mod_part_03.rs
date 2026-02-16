#[allow(clippy::wildcard_imports)]
use super::*;

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

    /// Linear transformation: H = X * W [num_nodes, total_out].
    fn linear_transform(
        x_data: &[f32],
        w_data: &[f32],
        num_nodes: usize,
        in_features: usize,
        total_out: usize,
    ) -> Vec<f32> {
        let mut h_data = vec![0.0f32; num_nodes * total_out];
        for node in 0..num_nodes {
            for out_f in 0..total_out {
                let mut sum = 0.0f32;
                for in_f in 0..in_features {
                    sum += x_data[node * in_features + in_f] * w_data[in_f * total_out + out_f];
                }
                h_data[node * total_out + out_f] = sum;
            }
        }
        h_data
    }

    /// Build adjacency neighbor lists (target -> list of sources).
    fn build_neighbor_lists(adj: &AdjacencyMatrix, num_nodes: usize) -> Vec<Vec<usize>> {
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for (&src, &tgt) in adj.edge_src().iter().zip(adj.edge_tgt().iter()) {
            if tgt < num_nodes && src < num_nodes {
                lists[tgt].push(src);
            }
        }
        lists
    }

    /// Compute attention score for a single edge (node → neighbor) for one head.
    fn edge_attention_score(
        &self,
        att_src_data: &[f32],
        att_tgt_data: &[f32],
        h_data: &[f32],
        node: usize,
        neigh: usize,
        head: usize,
        total_out: usize,
    ) -> f32 {
        let head_off = head * self.out_features;
        let mut score = 0.0f32;
        for f in 0..self.out_features {
            score += att_src_data[head * self.out_features + f]
                * h_data[node * total_out + head_off + f];
            score += att_tgt_data[head * self.out_features + f]
                * h_data[neigh * total_out + head_off + f];
        }
        self.leaky_relu(score)
    }

    /// Softmax over raw attention scores, returning normalized weights.
    fn softmax_attention(scores: &[f32]) -> Vec<f32> {
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|&e| e / (sum + 1e-8)).collect()
    }

    /// Scatter attention-weighted neighbor features into output for one head.
    fn scatter_attention(
        &self,
        h_data: &[f32],
        neighbors: &[usize],
        attn_weights: &[f32],
        node: usize,
        head: usize,
        total_out: usize,
        final_out: usize,
        output: &mut [f32],
    ) {
        let head_off = head * self.out_features;
        for (i, &neigh) in neighbors.iter().enumerate() {
            let alpha = attn_weights[i];
            let scale = if self.concat {
                1.0
            } else {
                1.0 / self.num_heads as f32
            };
            let out_off = if self.concat { head_off } else { 0 };
            for f in 0..self.out_features {
                output[node * final_out + out_off + f] +=
                    alpha * scale * h_data[neigh * total_out + head_off + f];
            }
        }
    }

    /// Add bias to GAT output, handling concat vs average mode.
    fn add_gat_bias(
        bias_data: &[f32],
        num_nodes: usize,
        out_features: usize,
        num_heads: usize,
        final_out: usize,
        concat: bool,
        output: &mut [f32],
    ) {
        for node in 0..num_nodes {
            if concat {
                for f in 0..final_out {
                    output[node * final_out + f] += bias_data[f];
                }
            } else {
                for f in 0..out_features {
                    let avg_bias: f32 = (0..num_heads)
                        .map(|h| bias_data[h * out_features + f])
                        .sum::<f32>()
                        / num_heads as f32;
                    output[node * final_out + f] += avg_bias;
                }
            }
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
    #[must_use]
    pub fn forward(&self, x: &Tensor, adj: &AdjacencyMatrix) -> Tensor {
        let num_nodes = x.shape()[0];
        assert_eq!(x.shape()[1], self.in_features);

        let adj_with_loops = if self.add_self_loops && !adj.has_self_loops() {
            adj.clone().add_self_loops()
        } else {
            adj.clone()
        };

        let total_out = self.out_features * self.num_heads;
        let h_data = Self::linear_transform(
            x.data(),
            self.weight.data(),
            num_nodes,
            self.in_features,
            total_out,
        );
        let neighbor_lists = Self::build_neighbor_lists(&adj_with_loops, num_nodes);
        let att_src_data = self.att_src.data();
        let att_tgt_data = self.att_tgt.data();

        let final_out = if self.concat {
            total_out
        } else {
            self.out_features
        };
        let mut output = vec![0.0f32; num_nodes * final_out];

        for node in 0..num_nodes {
            let neighbors = &neighbor_lists[node];
            if neighbors.is_empty() {
                continue;
            }
            for head in 0..self.num_heads {
                let scores: Vec<f32> = neighbors
                    .iter()
                    .map(|&n| {
                        self.edge_attention_score(
                            att_src_data,
                            att_tgt_data,
                            &h_data,
                            node,
                            n,
                            head,
                            total_out,
                        )
                    })
                    .collect();
                let weights = Self::softmax_attention(&scores);
                self.scatter_attention(
                    &h_data,
                    neighbors,
                    &weights,
                    node,
                    head,
                    total_out,
                    final_out,
                    &mut output,
                );
            }
        }

        if let Some(ref bias) = self.bias {
            Self::add_gat_bias(
                bias.data(),
                num_nodes,
                self.out_features,
                self.num_heads,
                final_out,
                self.concat,
                &mut output,
            );
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
