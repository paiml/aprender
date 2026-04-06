//! Tensor type with gradient tracking

use super::BackwardOp;
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// Tensor with automatic differentiation support
///
/// Data is stored behind `Rc` for O(1) clone.  Backward ops clone input
/// tensors to hold references for gradient computation — with `Rc`, this
/// is a reference-count bump instead of a full memcpy.  For Qwen3-4B with
/// batch_size=8, this eliminates ~16 GB of redundant frozen-weight copies
/// per training step (KAIZEN-019).
///
/// `data_mut()` uses `Rc::make_mut` for copy-on-write: mutation only copies
/// when there are multiple Rc references to the same data.
#[derive(Clone)]
pub struct Tensor {
    data: Rc<Array1<f32>>,
    grad: Rc<RefCell<Option<Array1<f32>>>>,
    backward_op: Option<Rc<dyn BackwardOp>>,
    requires_grad: bool,
}

impl Tensor {
    /// Create a new tensor with data
    pub fn new(data: Array1<f32>, requires_grad: bool) -> Self {
        Self {
            data: Rc::new(data),
            grad: Rc::new(RefCell::new(None)),
            backward_op: None,
            requires_grad,
        }
    }

    /// Create a tensor from a vector
    pub fn from_vec(data: Vec<f32>, requires_grad: bool) -> Self {
        Self::new(Array1::from(data), requires_grad)
    }

    /// Create a tensor filled with zeros
    pub fn zeros(size: usize, requires_grad: bool) -> Self {
        Self::new(Array1::zeros(size), requires_grad)
    }

    /// Create a tensor filled with ones
    pub fn ones(size: usize, requires_grad: bool) -> Self {
        Self::new(Array1::ones(size), requires_grad)
    }

    /// Get reference to data
    pub fn data(&self) -> &Array1<f32> {
        contract_pre_data_read!();
        &self.data
    }

    /// Get mutable reference to data (copy-on-write via `Rc::make_mut`)
    ///
    /// If this is the only `Rc` reference, returns a mutable reference to
    /// the existing data with no copy.  If there are other references
    /// (e.g. backward ops holding clones), clones the data first so
    /// mutations don't affect other holders.
    pub fn data_mut(&mut self) -> &mut Array1<f32> {
        contract_pre_data_mut!();
        Rc::make_mut(&mut self.data)
    }

    /// Get gradient (if computed)
    pub fn grad(&self) -> Option<Array1<f32>> {
        self.grad.borrow().clone()
    }

    /// Set gradient
    pub fn set_grad(&self, grad: Array1<f32>) {
        *self.grad.borrow_mut() = Some(grad);
    }

    /// Accumulate gradient (for when tensor is used multiple times)
    pub fn accumulate_grad(&self, grad: Array1<f32>) {
        let mut grad_ref = self.grad.borrow_mut();
        if let Some(existing) = grad_ref.as_mut() {
            *existing = &*existing + &grad;
        } else {
            *grad_ref = Some(grad);
        }
    }

    /// Zero out gradient
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Scale gradient in-place (KAIZEN-037: zero-allocation gradient scaling)
    ///
    /// # Contract (C-SCALE-GRAD-001)
    ///
    /// - **Precondition**: factor is finite
    /// - **Postcondition**: grad[i] *= factor for all i
    /// - **Invariant**: No heap allocation (mutates existing Array1 in-place)
    pub fn scale_grad(&self, factor: f32) {
        let mut grad_ref = self.grad.borrow_mut();
        if let Some(existing) = grad_ref.as_mut() {
            existing.mapv_inplace(|v| v * factor);
        }
    }

    /// Check if requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set requires gradient flag
    ///
    /// Use this to enable gradient tracking on loaded tensors before training.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Get reference to gradient cell (for backward operations)
    pub fn grad_cell(&self) -> Rc<RefCell<Option<Array1<f32>>>> {
        self.grad.clone()
    }

    /// Set backward operation
    pub fn set_backward_op(&mut self, op: Rc<dyn BackwardOp>) {
        self.backward_op = Some(op);
    }

    /// Get backward operation
    pub fn backward_op(&self) -> Option<Rc<dyn BackwardOp>> {
        self.backward_op.clone()
    }

    /// Get size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data)
            .field("grad", &self.grad.borrow())
            .field("requires_grad", &self.requires_grad)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let data = Array1::from(vec![1.0, 2.0, 3.0]);
        let t = Tensor::new(data.clone(), true);
        assert_eq!(t.data(), &data);
        assert!(t.requires_grad());
    }

    #[test]
    fn test_tensor_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0], false);
        assert_eq!(t.len(), 2);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(5, true);
        assert_eq!(t.len(), 5);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::ones(3, false);
        assert_eq!(t.len(), 3);
        assert!(t.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_data_mut() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0], true);
        t.data_mut()[0] = 5.0;
        assert_eq!(t.data()[0], 5.0);
    }

    #[test]
    fn test_tensor_grad_operations() {
        let t = Tensor::from_vec(vec![1.0, 2.0], true);
        assert!(t.grad().is_none());

        t.set_grad(Array1::from(vec![0.1, 0.2]));
        assert!(t.grad().is_some());
        assert_eq!(t.grad().expect("gradient should be available")[0], 0.1);

        t.zero_grad();
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_tensor_accumulate_grad() {
        let t = Tensor::from_vec(vec![1.0, 2.0], true);

        // First accumulation - should set grad
        t.accumulate_grad(Array1::from(vec![0.1, 0.2]));
        assert_eq!(t.grad().expect("gradient should be available")[0], 0.1);

        // Second accumulation - should add
        t.accumulate_grad(Array1::from(vec![0.3, 0.4]));
        let grad = t.grad().expect("gradient should be available");
        assert!((grad[0] - 0.4).abs() < 1e-6);
        assert!((grad[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_grad_cell() {
        let t = Tensor::from_vec(vec![1.0], true);
        let cell = t.grad_cell();
        assert!(cell.borrow().is_none());
    }

    #[test]
    fn test_tensor_backward_op() {
        let t = Tensor::from_vec(vec![1.0], true);
        assert!(t.backward_op().is_none());
        // Note: Setting backward op requires an actual BackwardOp implementation
    }

    #[test]
    fn test_tensor_is_empty() {
        let t = Tensor::from_vec(vec![], false);
        assert!(t.is_empty());

        let t2 = Tensor::from_vec(vec![1.0], false);
        assert!(!t2.is_empty());
    }

    #[test]
    fn test_tensor_set_requires_grad() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0], false);
        assert!(!t.requires_grad());

        t.set_requires_grad(true);
        assert!(t.requires_grad());

        t.set_requires_grad(false);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_tensor_debug() {
        let t = Tensor::from_vec(vec![1.0, 2.0], true);
        let debug_str = format!("{t:?}");
        assert!(debug_str.contains("Tensor"));
        assert!(debug_str.contains("data"));
    }

    #[test]
    fn test_tensor_clone() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], true);
        t1.set_grad(Array1::from(vec![0.1, 0.2]));
        let t2 = t1.clone();

        assert_eq!(t2.data(), t1.data());
        assert_eq!(t2.requires_grad(), t1.requires_grad());
    }
}
