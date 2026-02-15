//! Differentiable operations for tensors.
//!
//! Each operation:
//! 1. Computes the forward result
//! 2. Records a `GradFn` to the computation graph (if gradient tracking is enabled)
//!
//! Operations use trueno for SIMD-accelerated computation where available.

use std::sync::Arc;

use super::grad_fn::{
    AddBackward, BroadcastAddBackward, DivBackward, ExpBackward, GeluBackward, LeakyReluBackward,
    LogBackward, MatmulBackward, MeanBackward, MulBackward, NegBackward, PowBackward, ReluBackward,
    SigmoidBackward, SoftmaxBackward, SqrtBackward, SubBackward, SumBackward, TanhBackward,
    TransposeBackward, ViewBackward,
};
use super::tensor::Tensor;
use super::{is_grad_enabled, with_graph};

// ============================================================================
// Element-wise Operations
// ============================================================================

impl Tensor {
    /// Element-wise addition: z = self + other
    #[must_use]
    pub fn add(&self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let mut result = Tensor::new(&data, self.shape());

        // Record to graph if needed
        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(AddBackward {
                x_shape: self.shape().to_vec(),
                y_shape: other.shape().to_vec(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.register_tensor(other.clone());
                graph.record(result.id(), grad_fn, vec![self.id(), other.id()]);
            });
        }

        result
    }

    /// Element-wise subtraction: z = self - other
    #[must_use]
    pub fn sub(&self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a - b)
            .collect();

        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(SubBackward {
                x_shape: self.shape().to_vec(),
                y_shape: other.shape().to_vec(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.register_tensor(other.clone());
                graph.record(result.id(), grad_fn, vec![self.id(), other.id()]);
            });
        }

        result
    }

    /// Element-wise multiplication: z = self * other
    #[must_use]
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a * b)
            .collect();

        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(MulBackward {
                x: self.clone(),
                y: other.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.register_tensor(other.clone());
                graph.record(result.id(), grad_fn, vec![self.id(), other.id()]);
            });
        }

        result
    }

    /// Element-wise division: z = self / other
    #[must_use]
    pub fn div(&self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(&a, &b)| a / b)
            .collect();

        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(DivBackward {
                x: self.clone(),
                y: other.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.register_tensor(other.clone());
                graph.record(result.id(), grad_fn, vec![self.id(), other.id()]);
            });
        }

        result
    }

    /// Element-wise negation: z = -self
    #[must_use]
    pub fn neg(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&a| -a).collect();

        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(NegBackward);
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Scalar multiplication: z = self * scalar
    #[must_use]
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        // Broadcast scalar to match self shape
        let broadcast: Vec<f32> = self.data().iter().map(|&a| a * scalar).collect();
        let mut result = Tensor::new(&broadcast, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            // Use MulBackward with broadcast handling
            let grad_fn = Arc::new(MulBackward {
                x: self.clone(),
                y: Tensor::new(&vec![scalar; self.numel()], self.shape()),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }
}

// ============================================================================
// Transcendental Operations
// ============================================================================

impl Tensor {
    /// Element-wise exponential: z = exp(self)
    #[must_use]
    pub fn exp(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&a| a.exp()).collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(ExpBackward {
                output: result.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Element-wise natural logarithm: z = log(self)
    #[must_use]
    pub fn log(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&a| a.ln()).collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(LogBackward { x: self.clone() });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Element-wise power: z = self^n
    #[must_use]
    pub fn pow(&self, n: f32) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&a| a.powf(n)).collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(PowBackward { x: self.clone(), n });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Element-wise square root: z = sqrt(self)
    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&a| a.sqrt()).collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(SqrtBackward {
                output: result.clone(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

impl Tensor {
    /// Sum all elements: z = sum(self)
    #[must_use]
    pub fn sum(&self) -> Tensor {
        let sum: f32 = self.data().iter().sum();
        let mut result = Tensor::new(&[sum], &[1]);

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(SumBackward {
                input_shape: self.shape().to_vec(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Mean of all elements: z = mean(self)
    #[must_use]
    pub fn mean(&self) -> Tensor {
        let sum: f32 = self.data().iter().sum();
        let mean = sum / self.numel() as f32;
        let mut result = Tensor::new(&[mean], &[1]);

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(MeanBackward {
                input_shape: self.shape().to_vec(),
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }
}

include!("mod_part_02.rs");
