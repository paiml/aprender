
/// Additional model-specific data
#[derive(Debug, Clone, Default)]
pub struct ModelExtra {
    /// Tree structure for decision trees
    pub tree_data: Option<TreeData>,

    /// Layer information for neural networks
    pub layer_data: Option<Vec<LayerData>>,

    /// Cluster centroids for K-Means
    pub centroids: Option<AlignedVec<f32>>,

    /// Custom metadata
    pub metadata: std::collections::HashMap<String, Vec<u8>>,
}

impl ModelExtra {
    /// Create empty extra data
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tree data
    #[must_use]
    pub fn with_tree(mut self, tree: TreeData) -> Self {
        self.tree_data = Some(tree);
        self
    }

    /// Set layer data
    #[must_use]
    pub fn with_layers(mut self, layers: Vec<LayerData>) -> Self {
        self.layer_data = Some(layers);
        self
    }

    /// Set centroids
    #[must_use]
    pub fn with_centroids(mut self, centroids: AlignedVec<f32>) -> Self {
        self.centroids = Some(centroids);
        self
    }

    /// Add custom metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Vec<u8>) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let tree_size = self.tree_data.as_ref().map_or(0, TreeData::size_bytes);
        let layer_size: usize = self
            .layer_data
            .as_ref()
            .map_or(0, |layers| layers.iter().map(LayerData::size_bytes).sum());
        let centroid_size = self.centroids.as_ref().map_or(0, AlignedVec::size_bytes);
        let metadata_size: usize = self.metadata.values().map(Vec::len).sum();
        tree_size + layer_size + centroid_size + metadata_size
    }
}

/// Decision tree structure data
#[derive(Debug, Clone)]
pub struct TreeData {
    /// Feature indices for each node
    pub feature_indices: Vec<u16>,
    /// Thresholds for each node
    pub thresholds: Vec<f32>,
    /// Left child indices (-1 for leaf)
    pub left_children: Vec<i32>,
    /// Right child indices (-1 for leaf)
    pub right_children: Vec<i32>,
    /// Leaf values (predictions)
    pub leaf_values: Vec<f32>,
}

impl TreeData {
    /// Create empty tree
    #[must_use]
    pub fn new() -> Self {
        Self {
            feature_indices: Vec::new(),
            thresholds: Vec::new(),
            left_children: Vec::new(),
            right_children: Vec::new(),
            leaf_values: Vec::new(),
        }
    }

    /// Number of nodes
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.thresholds.len()
    }

    /// Size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.feature_indices.len() * 2
            + self.thresholds.len() * 4
            + self.left_children.len() * 4
            + self.right_children.len() * 4
            + self.leaf_values.len() * 4
    }
}

impl Default for TreeData {
    fn default() -> Self {
        Self::new()
    }
}

/// Neural network layer data
#[derive(Debug, Clone)]
pub struct LayerData {
    /// Layer type
    pub layer_type: LayerType,
    /// Input dimension
    pub input_dim: u32,
    /// Output dimension
    pub output_dim: u32,
    /// Weights (row-major)
    pub weights: Option<AlignedVec<f32>>,
    /// Biases
    pub biases: Option<AlignedVec<f32>>,
}

impl LayerData {
    /// Create a dense layer
    #[must_use]
    pub fn dense(input_dim: u32, output_dim: u32) -> Self {
        Self {
            layer_type: LayerType::Dense,
            input_dim,
            output_dim,
            weights: None,
            biases: None,
        }
    }

    /// Set weights
    #[must_use]
    pub fn with_weights(mut self, weights: AlignedVec<f32>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set biases
    #[must_use]
    pub fn with_biases(mut self, biases: AlignedVec<f32>) -> Self {
        self.biases = Some(biases);
        self
    }

    /// Size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let weights_size = self.weights.as_ref().map_or(0, AlignedVec::size_bytes);
        let biases_size = self.biases.as_ref().map_or(0, AlignedVec::size_bytes);
        weights_size + biases_size + 12 // type + input + output
    }
}

/// Neural network layer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Fully connected layer
    Dense,
    /// `ReLU` activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax activation
    Softmax,
    /// Dropout (inference mode = identity)
    Dropout,
    /// Batch normalization
    BatchNorm,
}

/// Errors for native model operations
#[derive(Debug, Clone)]
pub enum NativeModelError {
    /// Parameter count mismatch
    ParamCountMismatch { declared: usize, actual: usize },
    /// Invalid parameter value (NaN/Inf)
    InvalidParameter { index: usize, value: f32 },
    /// Invalid bias value (NaN/Inf)
    InvalidBias { index: usize, value: f32 },
    /// Feature count mismatch
    FeatureMismatch { expected: usize, got: usize },
    /// Missing required parameters
    MissingParams,
    /// Alignment error
    AlignmentError { ptr: usize, required: usize },
}

impl std::fmt::Display for NativeModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParamCountMismatch { declared, actual } => {
                write!(
                    f,
                    "Parameter count mismatch: declared {declared}, actual {actual}"
                )
            }
            Self::InvalidParameter { index, value } => {
                write!(f, "Invalid parameter at index {index}: {value}")
            }
            Self::InvalidBias { index, value } => {
                write!(f, "Invalid bias at index {index}: {value}")
            }
            Self::FeatureMismatch { expected, got } => {
                write!(f, "Feature mismatch: expected {expected}, got {got}")
            }
            Self::MissingParams => write!(f, "Missing model parameters"),
            Self::AlignmentError { ptr, required } => {
                write!(
                    f,
                    "Alignment error: ptr 0x{ptr:x} not aligned to {required}"
                )
            }
        }
    }
}

impl std::error::Error for NativeModelError {}

#[cfg(test)]
mod tests;
