//! Computation graph for automatic differentiation.
//!
//! This module implements the tape-based recording of operations
//! and the backward pass algorithm.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::grad_fn::GradFn;
use super::tensor::{Tensor, TensorId};

/// Entry in the computation tape.
#[derive(Clone)]
pub(crate) struct TapeEntry {
    /// ID of the output tensor
    pub output_id: TensorId,

    /// Function to compute gradients
    pub grad_fn: Arc<dyn GradFn>,

    /// IDs of input tensors
    pub input_ids: Vec<TensorId>,
}

/// Computation graph that records operations for backward pass.
///
/// The graph uses a tape-based approach where operations are recorded
/// in order during the forward pass, then gradients are computed in
/// reverse order during the backward pass.
///
/// # Thread Safety
///
/// Each thread has its own computation graph (via `thread_local` storage
/// in the parent module). This avoids synchronization overhead during
/// single-threaded training.
#[allow(missing_debug_implementations)]
pub struct ComputationGraph {
    /// Recorded operations (tape)
    tape: Vec<TapeEntry>,

    /// Map from tensor ID to tensor (for leaf tensors that need gradients)
    tensors: HashMap<TensorId, Tensor>,

    /// Set of tensor IDs that require gradients
    requires_grad: HashSet<TensorId>,
}

impl ComputationGraph {
    /// Create a new empty computation graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tape: Vec::new(),
            tensors: HashMap::new(),
            requires_grad: HashSet::new(),
        }
    }

    /// Clear all recorded operations.
    pub fn clear(&mut self) {
        self.tape.clear();
        self.tensors.clear();
        self.requires_grad.clear();
    }

    /// Register a tensor that requires gradients.
    pub fn register_tensor(&mut self, tensor: Tensor) {
        if tensor.requires_grad_enabled() {
            self.requires_grad.insert(tensor.id());
        }
        self.tensors.insert(tensor.id(), tensor);
    }

    /// Record an operation to the tape.
    pub fn record(
        &mut self,
        output_id: TensorId,
        grad_fn: Arc<dyn GradFn>,
        input_ids: Vec<TensorId>,
    ) {
        self.tape.push(TapeEntry {
            output_id,
            grad_fn,
            input_ids,
        });
    }

    /// Get a tensor by ID.
    #[must_use]
    pub fn get_tensor(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(&id)
    }

    /// Get a mutable tensor by ID.
    pub fn get_tensor_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        self.tensors.get_mut(&id)
    }

    /// Compute gradients via backpropagation.
    ///
    /// This implements the reverse-mode automatic differentiation algorithm:
    /// 1. Start with `grad_output` for the output tensor
    /// 2. Iterate through operations in reverse order
    /// 3. For each operation, compute gradients w.r.t. inputs
    /// 4. Accumulate gradients for tensors used multiple times
    ///
    /// # Arguments
    ///
    /// * `output_id` - ID of the tensor to differentiate
    /// * `grad_output` - Initial gradient (typically ones for scalar loss)
    pub fn backward(&mut self, output_id: TensorId, grad_output: Tensor) {
        // Map from tensor ID to accumulated gradient
        let mut grads: HashMap<TensorId, Tensor> = HashMap::new();
        grads.insert(output_id, grad_output);

        // Process tape in reverse order
        for entry in self.tape.iter().rev() {
            // Skip if we don't have a gradient for this output
            let grad_out = match grads.get(&entry.output_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            // Compute gradients w.r.t. inputs
            let input_grads = entry.grad_fn.backward(&grad_out);

            // Accumulate gradients for each input
            for (input_id, input_grad) in entry.input_ids.iter().zip(input_grads) {
                grads
                    .entry(*input_id)
                    .and_modify(|existing| {
                        // Accumulate: existing += input_grad
                        let new_data: Vec<f32> = existing
                            .data()
                            .iter()
                            .zip(input_grad.data().iter())
                            .map(|(a, b)| a + b)
                            .collect();
                        *existing = Tensor::new(&new_data, existing.shape());
                    })
                    .or_insert(input_grad);
            }
        }

        // Store gradients in leaf tensors
        for (id, grad) in grads {
            if let Some(tensor) = self.tensors.get_mut(&id) {
                if tensor.requires_grad_enabled() && tensor.is_leaf() {
                    tensor.accumulate_grad(grad);
                }
            }
        }
    }

    /// Get the number of recorded operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tape.len()
    }

    /// Check if the tape is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }

    /// Get gradient for a tensor by ID (after backward).
    #[must_use]
    pub fn get_grad(&self, id: TensorId) -> Option<Tensor> {
        self.tensors.get(&id).and_then(|t| t.grad().cloned())
    }

    /// Clear gradient for a specific tensor.
    pub fn clear_grad(&mut self, id: TensorId) {
        if let Some(tensor) = self.tensors.get_mut(&id) {
            tensor.clear_grad();
        }
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = ComputationGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_graph_clear() {
        let mut graph = ComputationGraph::new();
        let t = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
        graph.register_tensor(t);

        assert!(!graph.tensors.is_empty());

        graph.clear();
        assert!(graph.is_empty());
        assert!(graph.tensors.is_empty());
    }

    #[test]
    fn test_tensor_registration() {
        let mut graph = ComputationGraph::new();

        let t1 = Tensor::from_slice(&[1.0]).requires_grad();
        let t2 = Tensor::from_slice(&[2.0]); // no grad

        let id1 = t1.id();
        let id2 = t2.id();

        graph.register_tensor(t1);
        graph.register_tensor(t2);

        assert!(graph.get_tensor(id1).is_some());
        assert!(graph.get_tensor(id2).is_some());
        assert!(graph.requires_grad.contains(&id1));
        assert!(!graph.requires_grad.contains(&id2));
    }

    #[test]
    fn test_graph_default() {
        let graph = ComputationGraph::default();
        assert!(graph.is_empty());
    }

    #[test]
    fn test_get_tensor_mut() {
        let mut graph = ComputationGraph::new();
        let t = Tensor::from_slice(&[1.0, 2.0]);
        let id = t.id();
        graph.register_tensor(t);

        // Modify tensor through mutable reference
        if let Some(tensor) = graph.get_tensor_mut(id) {
            // Just verify we can get mutable access
            assert_eq!(tensor.data(), &[1.0, 2.0]);
        }

        // Non-existent tensor - use ID from a tensor not in graph
        let other = Tensor::from_slice(&[3.0]);
        assert!(graph.get_tensor_mut(other.id()).is_none());
    }

    #[test]
    fn test_record_operation() {
        use crate::autograd::grad_fn::NegBackward;

        let mut graph = ComputationGraph::new();
        let t1 = Tensor::from_slice(&[1.0, 2.0]);
        let output = Tensor::from_slice(&[-1.0, -2.0]);
        let output_id = output.id();

        graph.record(output_id, Arc::new(NegBackward), vec![t1.id()]);

        assert_eq!(graph.len(), 1);
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_get_grad_and_clear_grad() {
        let mut graph = ComputationGraph::new();
        let t = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
        let id = t.id();
        graph.register_tensor(t);

        // Initially no gradient
        assert!(graph.get_grad(id).is_none());

        // Non-existent tensor
        let other = Tensor::from_slice(&[3.0]);
        assert!(graph.get_grad(other.id()).is_none());

        // Clear grad on non-existent tensor (should not panic)
        graph.clear_grad(other.id());
    }

    // ========================================================================
    // Additional Coverage Tests for graph.rs
    // ========================================================================

    #[test]
    fn test_graph_len_empty() {
        let graph = ComputationGraph::new();
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_graph_multiple_register() {
        let mut graph = ComputationGraph::new();
        let t1 = Tensor::from_slice(&[1.0]).requires_grad();
        let t2 = Tensor::from_slice(&[2.0]).requires_grad();
        let t3 = Tensor::from_slice(&[3.0]).requires_grad();

        graph.register_tensor(t1);
        graph.register_tensor(t2);
        graph.register_tensor(t3);

        assert_eq!(graph.tensors.len(), 3);
    }

    #[test]
    fn test_graph_register_same_tensor_twice() {
        let mut graph = ComputationGraph::new();
        let t = Tensor::from_slice(&[1.0]).requires_grad();
        let id = t.id();

        // Clone to register same ID twice
        graph.register_tensor(t.clone());
        graph.register_tensor(t);

        // Should still have only one entry for that ID
        assert!(graph.get_tensor(id).is_some());
    }

    #[test]
    fn test_backward_simple() {
        use crate::autograd::grad_fn::NegBackward;

        let mut graph = ComputationGraph::new();

        // Create input tensor that requires gradients
        let input = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
        let input_id = input.id();
        graph.register_tensor(input);

        // Create output tensor
        let output = Tensor::from_slice(&[-1.0, -2.0]);
        let output_id = output.id();
        graph.register_tensor(output);

        // Record the negation operation
        graph.record(output_id, Arc::new(NegBackward), vec![input_id]);

        // Backward pass with ones gradient
        let grad_output = Tensor::from_slice(&[1.0, 1.0]);
        graph.backward(output_id, grad_output);

        // Check that gradient was computed (negation backward passes gradient as-is negated)
        let grad = graph.get_grad(input_id);
        assert!(grad.is_some());
    }

    #[test]
    fn test_backward_no_matching_output() {
        let mut graph = ComputationGraph::new();

        // Backward with non-existent output_id - should not panic
        let output_id = Tensor::from_slice(&[1.0]).id();
        let grad_output = Tensor::from_slice(&[1.0]);
        graph.backward(output_id, grad_output);

        // Should complete without error
        assert!(graph.is_empty());
    }

    #[test]
    fn test_backward_empty_tape() {
        let mut graph = ComputationGraph::new();

        let t = Tensor::from_slice(&[1.0]).requires_grad();
        let id = t.id();
        graph.register_tensor(t);

        // Backward with empty tape
        let grad_output = Tensor::from_slice(&[1.0]);
        graph.backward(id, grad_output);

        // Should complete without error
        assert!(graph.is_empty());
    }

    #[test]
    fn test_clear_grad_existing_tensor() {
        let mut graph = ComputationGraph::new();
        let t = Tensor::from_slice(&[1.0, 2.0]).requires_grad();
        let id = t.id();
        graph.register_tensor(t);

        // Clear grad on existing tensor
        graph.clear_grad(id);

        // Should complete without error
        assert!(graph.get_grad(id).is_none());
    }

    #[test]
    fn test_tape_entry_clone() {
        use crate::autograd::grad_fn::NegBackward;

        let entry = TapeEntry {
            output_id: TensorId::new(),
            grad_fn: Arc::new(NegBackward),
            input_ids: vec![TensorId::new(), TensorId::new()],
        };

        let cloned = entry.clone();
        assert_eq!(cloned.input_ids.len(), 2);
    }

    #[test]
    fn test_graph_record_multiple_operations() {
        use crate::autograd::grad_fn::NegBackward;

        let mut graph = ComputationGraph::new();

        let t1 = Tensor::from_slice(&[1.0]);
        let t2 = Tensor::from_slice(&[-1.0]);
        let t3 = Tensor::from_slice(&[1.0]);

        // Record two operations
        graph.record(t2.id(), Arc::new(NegBackward), vec![t1.id()]);
        graph.record(t3.id(), Arc::new(NegBackward), vec![t2.id()]);

        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_backward_skips_unrelated_operations() {
        use crate::autograd::grad_fn::NegBackward;

        let mut graph = ComputationGraph::new();

        // Create tensors
        let t1 = Tensor::from_slice(&[1.0]).requires_grad();
        let t1_id = t1.id();
        let t2 = Tensor::from_slice(&[-1.0]);
        let t2_id = t2.id();
        let t3 = Tensor::from_slice(&[5.0]); // unrelated
        let t3_id = t3.id();

        graph.register_tensor(t1);
        graph.register_tensor(t2);

        // Record operation for t1 -> t2
        graph.record(t2_id, Arc::new(NegBackward), vec![t1_id]);

        // Record unrelated operation (t3 not connected to backward path)
        graph.record(TensorId::new(), Arc::new(NegBackward), vec![t3_id]);

        // Backward from t2
        let grad_output = Tensor::from_slice(&[1.0]);
        graph.backward(t2_id, grad_output);

        // Should still work
        assert!(graph.get_grad(t1_id).is_some());
    }

    #[test]
    fn test_graph_get_tensor_nonexistent() {
        let graph = ComputationGraph::new();
        let fake_id = TensorId::new();
        assert!(graph.get_tensor(fake_id).is_none());
    }

    #[test]
    fn test_requires_grad_set_tracking() {
        let mut graph = ComputationGraph::new();

        let t1 = Tensor::from_slice(&[1.0]).requires_grad();
        let t2 = Tensor::from_slice(&[2.0]); // no requires_grad
        let t3 = Tensor::from_slice(&[3.0]).requires_grad();

        let id1 = t1.id();
        let id2 = t2.id();
        let id3 = t3.id();

        graph.register_tensor(t1);
        graph.register_tensor(t2);
        graph.register_tensor(t3);

        // Only t1 and t3 should be in requires_grad set
        assert!(graph.requires_grad.contains(&id1));
        assert!(!graph.requires_grad.contains(&id2));
        assert!(graph.requires_grad.contains(&id3));
    }
}
