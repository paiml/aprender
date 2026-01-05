//! Reverse-mode automatic differentiation engine for neural network training.
//!
//! This module implements tape-based automatic differentiation following the
//! methodology described in Baydin et al. (2018) and Griewank & Walther (2008).
//!
//! # Architecture
//!
//! The autograd engine uses a define-by-run (dynamic) computational graph:
//! - Operations are recorded to a tape during forward pass
//! - Gradients are computed in reverse order during backward pass
//! - Supports gradient accumulation for multi-use tensors
//!
//! # Example
//!
//! ```ignore
//! use aprender::autograd::{Tensor, no_grad};
//!
//! // Create tensors with gradient tracking
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
//! let w = Tensor::from_slice(&[0.5, 0.5, 0.5]).requires_grad();
//!
//! // Forward pass (operations recorded to tape)
//! let y = x.mul(&w).sum();
//!
//! // Backward pass (compute gradients)
//! y.backward();
//!
//! // Access gradients
//! println!("dL/dx = {:?}", x.grad());
//! println!("dL/dw = {:?}", w.grad());
//! ```
//!
//! # References
//!
//! - Baydin, A. G., et al. (2018). Automatic differentiation in machine learning: a survey. JMLR.
//! - Rumelhart, D. E., et al. (1986). Learning representations by back-propagating errors. Nature.
//! - Griewank, A., & Walther, A. (2008). Evaluating derivatives. SIAM.

pub(crate) mod grad_fn;
mod graph;
mod ops;
mod tensor;

pub use grad_fn::GradFn;
pub use graph::ComputationGraph;
pub use tensor::{Tensor, TensorId};

use std::cell::RefCell;

thread_local! {
    /// Global computation graph for the current thread.
    static GRAPH: RefCell<ComputationGraph> = RefCell::new(ComputationGraph::new());

    /// Flag to disable gradient tracking (for inference).
    static GRAD_ENABLED: RefCell<bool> = const { RefCell::new(true) };
}

/// Execute a closure without gradient tracking.
///
/// Useful for inference or when gradients are not needed.
///
/// # Example
///
/// ```ignore
/// use aprender::autograd::{Tensor, no_grad};
///
/// let x = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
///
/// // No gradients computed inside this block
/// let y = no_grad(|| {
///     x.mul(&x).sum()
/// });
///
/// assert!(y.grad().is_none());
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    GRAD_ENABLED.with(|enabled| {
        let prev = *enabled.borrow();
        *enabled.borrow_mut() = false;
        let result = f();
        *enabled.borrow_mut() = prev;
        result
    })
}

/// Check if gradient tracking is currently enabled.
#[must_use] 
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|enabled| *enabled.borrow())
}

/// Get a reference to the thread-local computation graph.
pub(crate) fn with_graph<F, R>(f: F) -> R
where
    F: FnOnce(&mut ComputationGraph) -> R,
{
    GRAPH.with(|graph| f(&mut graph.borrow_mut()))
}

/// Clear the computation graph (called after backward).
pub fn clear_graph() {
    GRAPH.with(|graph| graph.borrow_mut().clear());
}

/// Get gradient for a tensor by ID from the graph.
#[must_use] 
pub fn get_grad(id: TensorId) -> Option<Tensor> {
    with_graph(|graph| graph.get_grad(id))
}

/// Clear gradient for a specific tensor by ID.
pub fn clear_grad(id: TensorId) {
    with_graph(|graph| graph.clear_grad(id));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_grad_context() {
        assert!(is_grad_enabled());

        no_grad(|| {
            assert!(!is_grad_enabled());
        });

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_nested_no_grad() {
        assert!(is_grad_enabled());

        no_grad(|| {
            assert!(!is_grad_enabled());
            no_grad(|| {
                assert!(!is_grad_enabled());
            });
            assert!(!is_grad_enabled());
        });

        assert!(is_grad_enabled());
    }
}
