//! Tensor with automatic differentiation support.
//!
//! This module provides the core `Tensor` type that tracks gradients
//! through computational operations.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::primitives::Vector;

use super::grad_fn::GradFn;
use super::with_graph;

/// Unique identifier for tensors in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    /// Generate a new unique tensor ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        TensorId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

/// A tensor with optional gradient tracking for automatic differentiation.
///
/// # Design
///
/// The tensor stores:
/// - `data`: The actual numerical values (backed by aprender's Vector)
/// - `shape`: Dimensions of the tensor
/// - `grad`: Accumulated gradient (populated after `backward()`)
/// - `requires_grad`: Whether this tensor participates in gradient computation
/// - `grad_fn`: The operation that created this tensor (for backprop)
/// - `id`: Unique identifier for graph tracking
///
/// # Thread Safety
///
/// Tensors use `Arc` internally for shared ownership of gradient functions,
/// making them safe to share across threads for inference (but not training).
#[derive(Clone)]
pub struct Tensor {
    /// Underlying data storage
    data: Vector<f32>,

    /// Shape of the tensor
    shape: Vec<usize>,

    /// Gradient (populated after `backward()`)
    grad: Option<Box<Tensor>>,

    /// Whether this tensor requires gradient computation
    requires_grad: bool,

    /// Whether this is a leaf tensor (created by user, not by operation)
    is_leaf: bool,

    /// Function that computes gradients during backward pass
    grad_fn: Option<Arc<dyn GradFn>>,

    /// Unique identifier for graph construction
    id: TensorId,
}

impl Tensor {
    /// Create a new tensor from a slice with the given shape.
    ///
    /// By default, gradient tracking is disabled.
    ///
    /// # Panics
    ///
    /// Panics if the data length doesn't match the product of shape dimensions.
    #[must_use]
    pub fn new(data: &[f32], shape: &[usize]) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        Self {
            data: Vector::from_slice(data),
            shape: shape.to_vec(),
            grad: None,
            requires_grad: false,
            is_leaf: true,
            grad_fn: None,
            id: TensorId::new(),
        }
    }

    /// Create a tensor from a 1D slice (vector).
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Self {
        Self::new(data, &[data.len()])
    }

    /// Create a tensor filled with zeros.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self::new(&vec![0.0; len], shape)
    }

    /// Create a tensor filled with ones.
    #[must_use]
    pub fn ones(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self::new(&vec![1.0; len], shape)
    }

    /// Create a tensor with the same shape as another, filled with zeros.
    #[must_use]
    pub fn zeros_like(other: &Tensor) -> Self {
        Self::zeros(&other.shape)
    }

    /// Create a tensor with the same shape as another, filled with ones.
    #[must_use]
    pub fn ones_like(other: &Tensor) -> Self {
        Self::ones(&other.shape)
    }

    /// Enable gradient tracking for this tensor.
    ///
    /// Returns self for method chaining.
    #[must_use]
    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Enable or disable gradient tracking (in-place).
    pub fn requires_grad_(&mut self, requires: bool) -> &mut Self {
        self.requires_grad = requires;
        self
    }

    /// Check if this tensor requires gradient computation.
    #[must_use]
    pub fn requires_grad_enabled(&self) -> bool {
        self.requires_grad
    }

    /// Check if this is a leaf tensor (not created by an operation).
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// Get the tensor's unique identifier.
    #[must_use]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Get the shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get a reference to the underlying data.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        self.data.as_slice()
    }

    /// Get a mutable reference to the underlying data.
    ///
    /// # Warning
    ///
    /// Modifying data directly may invalidate gradients.
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data.as_mut_slice()
    }

    /// Get the gradient tensor (if computed).
    #[must_use]
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_deref()
    }

    /// Zero out the gradient.
    pub fn zero_grad_(&mut self) {
        self.grad = None;
    }

    /// Clear the gradient (alias for `zero_grad`_).
    pub fn clear_grad(&mut self) {
        self.grad = None;
    }

    /// Accumulate gradient (used during backward pass).
    pub(crate) fn accumulate_grad(&mut self, grad: Tensor) {
        match &mut self.grad {
            Some(existing) => {
                // Accumulate gradients
                let new_data: Vec<f32> = existing
                    .data()
                    .iter()
                    .zip(grad.data().iter())
                    .map(|(a, b)| a + b)
                    .collect();
                **existing = Tensor::new(&new_data, &self.shape);
            }
            None => {
                self.grad = Some(Box::new(grad));
            }
        }
    }

    /// Set the gradient function (used internally by operations).
    pub(crate) fn set_grad_fn(&mut self, grad_fn: Arc<dyn GradFn>) {
        self.grad_fn = Some(grad_fn);
        self.is_leaf = false;
    }

    /// Get the gradient function.
    #[allow(dead_code)]
    pub(crate) fn grad_fn(&self) -> Option<&Arc<dyn GradFn>> {
        self.grad_fn.as_ref()
    }

    /// Detach tensor from computation graph.
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    #[must_use]
    pub fn detach(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            grad: None,
            requires_grad: false,
            is_leaf: true,
            grad_fn: None,
            id: TensorId::new(),
        }
    }

    /// Get a scalar value (for 0-d or 1-element tensors).
    ///
    /// # Panics
    ///
    /// Panics if the tensor has more than one element.
    #[must_use]
    pub fn item(&self) -> f32 {
        assert_eq!(
            self.numel(),
            1,
            "item() only works on tensors with exactly 1 element, got {}",
            self.numel()
        );
        self.data[0]
    }

    /// Compute gradients via backpropagation.
    ///
    /// This implements the reverse-mode automatic differentiation algorithm
    /// described in Rumelhart et al. (1986).
    ///
    /// # Panics
    ///
    /// Panics if called on a tensor with more than one element
    /// (use `backward_with_grad` for non-scalar outputs).
    pub fn backward(&self) {
        assert_eq!(
            self.numel(),
            1,
            "backward() requires scalar output, got shape {:?}. Use backward_with_grad() instead.",
            self.shape
        );

        self.backward_with_grad(Tensor::ones(&self.shape));
    }

    /// Compute gradients with a specified output gradient.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the loss with respect to this tensor
    pub fn backward_with_grad(&self, grad_output: Tensor) {
        with_graph(|graph| {
            graph.backward(self.id, grad_output);
        });
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field("has_grad", &self.grad.is_some())
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
