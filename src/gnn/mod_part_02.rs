
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
    #[must_use]
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
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Set epsilon value.
    pub fn set_eps(&mut self, eps: f32) {
        self.eps = eps;
    }

    /// Check if epsilon is trainable.
    #[must_use]
    pub fn train_eps(&self) -> bool {
        self.train_eps
    }

    /// Get input feature dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get hidden feature dimension.
    #[must_use]
    pub fn hidden_features(&self) -> usize {
        self.hidden_features
    }

    /// Get output feature dimension.
    #[must_use]
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

/// `GraphSAGE` layer (Hamilton et al., 2017).
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
/// Learning on Large Graphs. `NeurIPS`.
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

/// Aggregation method for `GraphSAGE`.
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
    /// Create a new `GraphSAGE` layer with mean aggregation.
    #[must_use]
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
    #[must_use]
    pub fn with_aggregation(mut self, agg: SAGEAggregation) -> Self {
        self.aggregation = agg;
        self
    }

    /// Set neighbor sample size.
    #[must_use]
    pub fn with_sample_size(mut self, size: usize) -> Self {
        self.sample_size = Some(size);
        self
    }

    /// Disable output normalization.
    #[must_use]
    pub fn without_normalize(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Get aggregation type.
    #[must_use]
    pub fn aggregation(&self) -> SAGEAggregation {
        self.aggregation
    }

    /// Get sample size.
    #[must_use]
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
    /// Create new `EdgeConv` layer.
    #[must_use]
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
