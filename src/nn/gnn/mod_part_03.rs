
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
