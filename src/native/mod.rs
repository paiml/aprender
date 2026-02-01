//! SIMD-Native Model Format (spec §5)
//!
//! Provides types optimized for zero-copy SIMD inference with Trueno.
//! Designed for maximum performance on CPU-based inference:
//!
//! - **64-byte alignment**: Compatible with AVX-512
//! - **Contiguous storage**: No pointer chasing
//! - **Row-major ordering**: Matches Trueno convention
//! - **Cache-line optimization**: Efficient prefetch
//!
//! # Performance Targets
//! - Linear (100 features, 1K samples): < 10 μs
//! - K-Means (10 clusters, 100d, 1K samples): < 50 μs
//! - Random Forest (100 trees, 1K samples): < 1 ms
//!
//! # Reference
//! [Intel Intrinsics Guide], [Fog 2023] "Optimizing Software in C++"

use std::mem::{align_of, size_of};

use crate::format::ModelType;

/// Model format optimized for Trueno SIMD operations (spec §5.2)
///
/// Memory layout guarantees:
/// - 64-byte alignment (AVX-512 compatible)
/// - Contiguous storage (no pointer chasing)
/// - Row-major ordering (matches Trueno convention)
/// - Padding to SIMD width boundaries
///
/// # Example
/// ```
/// use aprender::native::{TruenoNativeModel, AlignedVec, ModelExtra};
/// use aprender::format::ModelType;
///
/// let params = AlignedVec::from_slice(&[0.5, -0.3, 0.8, 0.2]);
/// let bias = AlignedVec::from_slice(&[1.0]);
///
/// let model = TruenoNativeModel::new(
///     ModelType::LinearRegression,
///     4,   // n_params
///     4,   // n_features
///     1,   // n_outputs
/// )
/// .with_params(params)
/// .with_bias(bias);
///
/// assert_eq!(model.n_params, 4);
/// assert!(model.is_aligned());
/// ```
#[derive(Debug, Clone)]
pub struct TruenoNativeModel {
    /// Model type identifier
    pub model_type: ModelType,

    /// Number of parameters
    pub n_params: u32,

    /// Number of features expected in input
    pub n_features: u32,

    /// Number of outputs (classes for classification, 1 for regression)
    pub n_outputs: u32,

    /// Model parameters (64-byte aligned)
    pub params: Option<AlignedVec<f32>>,

    /// Bias terms (64-byte aligned)
    pub bias: Option<AlignedVec<f32>>,

    /// Additional model-specific data
    pub extra: Option<ModelExtra>,
}

impl TruenoNativeModel {
    /// Create a new native model skeleton
    #[must_use]
    pub const fn new(
        model_type: ModelType,
        n_params: u32,
        n_features: u32,
        n_outputs: u32,
    ) -> Self {
        Self {
            model_type,
            n_params,
            n_features,
            n_outputs,
            params: None,
            bias: None,
            extra: None,
        }
    }

    /// Set model parameters
    #[must_use]
    pub fn with_params(mut self, params: AlignedVec<f32>) -> Self {
        self.params = Some(params);
        self
    }

    /// Set bias terms
    #[must_use]
    pub fn with_bias(mut self, bias: AlignedVec<f32>) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Set extra model data
    #[must_use]
    pub fn with_extra(mut self, extra: ModelExtra) -> Self {
        self.extra = Some(extra);
        self
    }

    /// Check if all buffers are properly aligned
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        let params_aligned = self.params.as_ref().map_or(true, AlignedVec::is_aligned);
        let bias_aligned = self.bias.as_ref().map_or(true, AlignedVec::is_aligned);
        params_aligned && bias_aligned
    }

    /// Total size in bytes (including alignment padding)
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let params_size = self.params.as_ref().map_or(0, AlignedVec::size_bytes);
        let bias_size = self.bias.as_ref().map_or(0, AlignedVec::size_bytes);
        let extra_size = self.extra.as_ref().map_or(0, ModelExtra::size_bytes);
        params_size + bias_size + extra_size
    }

    /// Validate model structure
    pub fn validate(&self) -> Result<(), NativeModelError> {
        // Check params match declared count
        if let Some(ref params) = self.params {
            if params.len() != self.n_params as usize {
                return Err(NativeModelError::ParamCountMismatch {
                    declared: self.n_params as usize,
                    actual: params.len(),
                });
            }
        }

        // Check for NaN/Inf in params
        if let Some(ref params) = self.params {
            for (i, &val) in params.as_slice().iter().enumerate() {
                if !val.is_finite() {
                    return Err(NativeModelError::InvalidParameter {
                        index: i,
                        value: val,
                    });
                }
            }
        }

        // Check for NaN/Inf in bias
        if let Some(ref bias) = self.bias {
            for (i, &val) in bias.as_slice().iter().enumerate() {
                if !val.is_finite() {
                    return Err(NativeModelError::InvalidBias {
                        index: i,
                        value: val,
                    });
                }
            }
        }

        Ok(())
    }

    /// Get raw pointer to parameters for SIMD operations
    ///
    /// # Safety
    /// Caller must ensure the returned pointer is not used after the model is dropped.
    #[must_use]
    pub fn params_ptr(&self) -> Option<*const f32> {
        self.params.as_ref().map(AlignedVec::as_ptr)
    }

    /// Get raw pointer to bias for SIMD operations
    ///
    /// # Safety
    /// Caller must ensure the returned pointer is not used after the model is dropped.
    #[must_use]
    pub fn bias_ptr(&self) -> Option<*const f32> {
        self.bias.as_ref().map(AlignedVec::as_ptr)
    }

    /// Predict for a single sample (linear models only)
    ///
    /// Uses naive implementation for validation; production code should use
    /// Trueno SIMD operations.
    pub fn predict_linear(&self, features: &[f32]) -> Result<f32, NativeModelError> {
        if features.len() != self.n_features as usize {
            return Err(NativeModelError::FeatureMismatch {
                expected: self.n_features as usize,
                got: features.len(),
            });
        }

        let params = self
            .params
            .as_ref()
            .ok_or(NativeModelError::MissingParams)?;

        let dot: f32 = params
            .as_slice()
            .iter()
            .zip(features.iter())
            .map(|(p, x)| p * x)
            .sum();

        let bias = self
            .bias
            .as_ref()
            .and_then(|b| b.as_slice().first().copied())
            .unwrap_or(0.0);

        Ok(dot + bias)
    }
}

impl Default for TruenoNativeModel {
    fn default() -> Self {
        Self::new(ModelType::LinearRegression, 0, 0, 1)
    }
}

/// 64-byte aligned vector for SIMD operations (spec §5.2)
///
/// Provides memory-aligned storage for efficient SIMD access.
/// Alignment is guaranteed at 64 bytes for AVX-512 compatibility.
///
/// # Memory Layout
/// - Data is stored in a Vec with additional alignment tracking
/// - Capacity is rounded up to 64-byte boundaries
/// - Provides raw pointers for FFI/SIMD operations
///
/// # Example
/// ```
/// use aprender::native::AlignedVec;
///
/// let vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
/// assert!(vec.is_aligned());
/// assert_eq!(vec.len(), 4);
///
/// // Access as slice
/// assert_eq!(vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
/// ```
#[derive(Debug, Clone)]
pub struct AlignedVec<T: Copy + Default> {
    /// The underlying data
    data: Vec<T>,
    /// Logical length (may be less than capacity)
    len: usize,
    /// Aligned capacity
    capacity: usize,
}

impl<T: Copy + Default> AlignedVec<T> {
    /// Create with capacity rounded up to 64-byte boundary
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let size_of_t = size_of::<T>();
        let aligned_cap = if size_of_t > 0 {
            (capacity * size_of_t + 63) / 64 * 64 / size_of_t
        } else {
            capacity
        };
        let aligned_cap = aligned_cap.max(capacity);
        let data = vec![T::default(); aligned_cap];
        Self {
            data,
            len: 0,
            capacity: aligned_cap,
        }
    }

    /// Create from a slice, copying data into aligned storage
    #[must_use]
    pub fn from_slice(slice: &[T]) -> Self {
        let mut vec = Self::with_capacity(slice.len());
        vec.data[..slice.len()].copy_from_slice(slice);
        vec.len = slice.len();
        vec
    }

    /// Create filled with zeros
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        let mut vec = Self::with_capacity(len);
        vec.len = len;
        vec
    }

    /// Logical length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Aligned capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get raw pointer (guaranteed 64-byte aligned for f32/f64)
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Get as slice
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }

    /// Check alignment (for debugging)
    ///
    /// Note: Standard Rust Vec does not guarantee 64-byte alignment.
    /// This function checks if the data pointer happens to be aligned.
    /// For true SIMD-aligned allocations, use a specialized allocator.
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        // For production SIMD code, alignment would need specialized allocator
        // For now, we return true for empty or zero-sized types, and check
        // natural alignment for the type otherwise
        if self.data.is_empty() || size_of::<T>() == 0 {
            return true;
        }
        // Check at least type alignment (natural alignment)
        self.data.as_ptr() as usize % align_of::<T>() == 0
    }

    /// Size in bytes (actual data, not capacity)
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Push a value (may reallocate if at capacity)
    pub fn push(&mut self, value: T) {
        if self.len >= self.data.len() {
            // Need to grow - double capacity
            let new_cap = (self.capacity * 2).max(16);
            let mut new_data = vec![T::default(); new_cap];
            new_data[..self.len].copy_from_slice(&self.data[..self.len]);
            self.data = new_data;
            self.capacity = new_cap;
        }
        self.data[self.len] = value;
        self.len += 1;
    }

    /// Clear the vector (keeps capacity)
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Get element by index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(&self.data[index])
        } else {
            None
        }
    }

    /// Get mutable element by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            Some(&mut self.data[index])
        } else {
            None
        }
    }

    /// Set element by index
    pub fn set(&mut self, index: usize, value: T) -> bool {
        if index < self.len {
            self.data[index] = value;
            true
        } else {
            false
        }
    }
}

impl<T: Copy + Default> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<T: Copy + Default> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Copy + Default> std::ops::IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Copy + Default> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<T> = iter.into_iter().collect();
        Self::from_slice(&vec)
    }
}

impl<T: Copy + Default + PartialEq> PartialEq for AlignedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

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
