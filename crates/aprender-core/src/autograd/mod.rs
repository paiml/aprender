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

/// SetFit differentiable primitives (phase 01, contract
/// `setfit-encoder-conformance-v1`).
///
/// Re-exported here rather than under `autograd::ops` because `ops` is private —
/// every other operation in it is an inherent `Tensor` method, and these are the
/// first free functions. They are deliberately NOT behind the `setfit` feature
/// (D-03): the severed-graph debt they retire belongs to every consumer of
/// `autograd`, not just the SetFit path.
pub use ops::{
    additive_attention_mask, cosine_similarity_rows, embedding_gather, l2_normalize_rows,
    masked_mean_pool, mse_loss, OpError, NEG_MASK,
};

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

/// Number of operations currently recorded on this thread's tape.
///
/// # A determinism/no-grad OBSERVATION accessor, not an API for computing anything
///
/// [`ComputationGraph::len`] has always been `pub`, but the graph itself is a private
/// thread-local reachable only through the crate-private `with_graph`, so until now there
/// was no way for a caller outside this crate to observe tape growth at all. Two properties
/// this workspace needs to gate on are otherwise unobservable:
///
/// 1. **"The tape was cleared this step."** Nothing prunes the tape; every recorded op
///    appends. A training loop that forgets [`clear_graph`] still produces correct gradients
///    — `register_tensor` overwrites the leaf entry each forward, and a previous step's
///    sub-tape has no seeded output gradient, so it is skipped — but the tape grows without
///    bound across an epoch and every backward re-walks it. That is a resource defect with
///    no wrong-number symptom, which is exactly the kind a test has to be able to see.
/// 2. **"No graph was built."** Asserting that a block ran under [`no_grad`] by checking
///    that some tensor has no gradient is weaker than checking that the block recorded no
///    operations at all.
///
/// It returns a COUNT, not a handle: no caller can reach a tensor or a `grad_fn` through it.
#[must_use]
pub fn graph_tape_len() -> usize {
    with_graph(|graph| graph.len())
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
#[path = "tests_tensor_contract.rs"]
mod tests_tensor_contract;

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

    /// The accessor observes the tape, and the tape actually moves.
    ///
    /// Three assertions rather than one, because "returns 0 after `clear_graph`" is
    /// satisfied by a function that always returns 0. The middle assertion is what makes
    /// the first and third non-vacuous.
    #[test]
    fn autograd_graph_tape_len_observes_growth_and_clearing() {
        clear_graph();
        assert_eq!(graph_tape_len(), 0, "a cleared tape is empty");

        let a = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
        let b = Tensor::from_slice(&[3.0, 4.0]).requires_grad();
        let _ = a.add(&b);
        let grew = graph_tape_len();
        assert!(grew > 0, "a recorded op must be visible on the tape");

        let _ = a.add(&b);
        assert!(
            graph_tape_len() > grew,
            "a second op must append rather than replace",
        );

        clear_graph();
        assert_eq!(graph_tape_len(), 0, "clear_graph must empty the tape");
    }

    /// `no_grad` records NOTHING — a stronger statement than "the output has no gradient".
    #[test]
    fn autograd_graph_tape_len_stays_zero_under_no_grad() {
        clear_graph();
        let a = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
        let b = Tensor::from_slice(&[3.0, 4.0]).requires_grad();

        no_grad(|| {
            let _ = a.add(&b);
        });
        assert_eq!(
            graph_tape_len(),
            0,
            "no_grad must record no operation at all",
        );

        // Control: the same call OUTSIDE no_grad does record, so the assertion above is
        // measuring the guard rather than an op that never records.
        let _ = a.add(&b);
        assert!(graph_tape_len() > 0, "the control must record");
        clear_graph();
    }
}
