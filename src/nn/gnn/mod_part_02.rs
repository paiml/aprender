use super::*;

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

    /// Accumulate neighbor features into `agg` by summing.
    fn accumulate_neighbors(
        x_data: &[f32],
        neighbors: &[usize],
        in_features: usize,
        agg: &mut [f32],
    ) {
        for &neigh in neighbors {
            for f in 0..in_features {
                agg[f] += x_data[neigh * in_features + f];
            }
        }
    }

    /// Scale aggregated features in-place by `1 / count`.
    fn scale_by_mean(agg: &mut [f32], count: usize) {
        let divisor = count as f32;
        for f in agg.iter_mut() {
            *f /= divisor;
        }
    }

    /// Aggregate neighbor features for a single node according to `self.aggregation`.
    fn aggregate_neighbors(
        &self,
        x_data: &[f32],
        neighbors: &[usize],
    ) -> Vec<f32> {
        if neighbors.is_empty() {
            return vec![0.0f32; self.in_features];
        }

        match self.aggregation {
            SAGEAggregation::Mean | SAGEAggregation::Lstm => {
                // Lstm uses mean as simplified fallback (full LSTM would need state)
                let mut agg = vec![0.0f32; self.in_features];
                Self::accumulate_neighbors(x_data, neighbors, self.in_features, &mut agg);
                Self::scale_by_mean(&mut agg, neighbors.len());
                agg
            }
            SAGEAggregation::Sum => {
                let mut agg = vec![0.0f32; self.in_features];
                Self::accumulate_neighbors(x_data, neighbors, self.in_features, &mut agg);
                agg
            }
            SAGEAggregation::Max => {
                let mut agg = vec![f32::NEG_INFINITY; self.in_features];
                for &neigh in neighbors {
                    for f in 0..self.in_features {
                        agg[f] = agg[f].max(x_data[neigh * self.in_features + f]);
                    }
                }
                // Replace -inf with 0 for features that had no finite value
                for f in agg.iter_mut() {
                    if f.is_infinite() {
                        *f = 0.0;
                    }
                }
                agg
            }
        }
    }

    /// Linear transform for a single node: out = W_self * x_node + W_neigh * agg_neigh.
    fn transform_node(
        &self,
        node: usize,
        x_data: &[f32],
        in_feat: usize,
        ws_data: &[f32],
        wn_data: &[f32],
        agg_features: &[f32],
        output: &mut [f32],
    ) {
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

    /// Add bias to every node row in `output`.
    fn add_bias_to_output(bias_data: &[f32], num_nodes: usize, out_features: usize, output: &mut [f32]) {
        for node in 0..num_nodes {
            for f in 0..out_features {
                output[node * out_features + f] += bias_data[f];
            }
        }
    }

    /// L2-normalize each node row in `output`.
    fn l2_normalize_rows(num_nodes: usize, out_features: usize, output: &mut [f32]) {
        for node in 0..num_nodes {
            let start = node * out_features;
            let end = start + out_features;
            let norm: f32 = output[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for val in &mut output[start..end] {
                    *val /= norm;
                }
            }
        }
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
        let neighbor_lists = Self::build_neighbor_lists(adj, num_nodes);

        let mut output = vec![0.0f32; num_nodes * self.out_features];

        for node in 0..num_nodes {
            let agg_features = self.aggregate_neighbors(&x_data, &neighbor_lists[node]);
            self.transform_node(
                node, &x_data, in_feat, &ws_data, &wn_data, &agg_features, &mut output,
            );
        }

        if let Some(ref bias) = self.bias {
            Self::add_bias_to_output(&bias.data(), num_nodes, self.out_features, &mut output);
        }

        if self.normalize {
            Self::l2_normalize_rows(num_nodes, self.out_features, &mut output);
        }

        Tensor::new(&output, &[num_nodes, self.out_features])
    }

    /// Build adjacency neighbor lists (target -> list of sources).
    fn build_neighbor_lists(adj: &AdjacencyMatrix, num_nodes: usize) -> Vec<Vec<usize>> {
        let mut neighbor_lists: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for (&src, &tgt) in adj.edge_src().iter().zip(adj.edge_tgt().iter()) {
            if tgt < num_nodes && src < num_nodes {
                neighbor_lists[tgt].push(src);
            }
        }
        neighbor_lists
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
    pub(crate) in_features: usize,
    /// Output feature dimension (per head)
    pub(crate) out_features: usize,
    /// Number of attention heads
    pub(crate) num_heads: usize,
    /// Linear transformation weight [`in_features`, `out_features` * `num_heads`]
    pub(crate) weight: Tensor,
    /// Attention weight for source nodes [`num_heads`, `out_features`]
    pub(crate) att_src: Tensor,
    /// Attention weight for target nodes [`num_heads`, `out_features`]
    pub(crate) att_tgt: Tensor,
    /// Bias [`out_features` * `num_heads`]
    pub(crate) bias: Option<Tensor>,
    /// Negative slope for `LeakyReLU`
    pub(crate) negative_slope: f32,
    /// Dropout probability
    pub(crate) dropout: f32,
    /// Whether to concatenate heads (true) or average them (false)
    pub(crate) concat: bool,
    /// Add self-loops
    pub(crate) add_self_loops: bool,
}
