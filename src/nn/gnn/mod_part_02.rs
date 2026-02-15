
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
