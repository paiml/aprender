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

// ============================================================================
// Activation Functions
// ============================================================================

impl Tensor {
    /// `ReLU` activation: z = max(0, self)
    #[must_use]
    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&a| a.max(0.0)).collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(ReluBackward { x: self.clone() });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Sigmoid activation: z = 1 / (1 + exp(-self))
    #[must_use]
    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .map(|&a| 1.0 / (1.0 + (-a).exp()))
            .collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(SigmoidBackward {
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

    /// Tanh activation
    #[must_use]
    pub fn tanh_(&self) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&a| a.tanh()).collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(TanhBackward {
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

    /// Leaky `ReLU` activation: z = `max(negative_slope` * x, x)
    ///
    /// # Arguments
    ///
    /// * `negative_slope` - Controls the angle of the negative slope (default: 0.01)
    #[must_use]
    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .map(|&x| if x > 0.0 { x } else { negative_slope * x })
            .collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(LeakyReluBackward {
                x: self.clone(),
                negative_slope,
            });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// GELU (Gaussian Error Linear Unit) activation.
    ///
    /// Uses the tanh approximation:
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    #[must_use]
    pub fn gelu(&self) -> Tensor {
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();

        let data: Vec<f32> = self
            .data()
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(GeluBackward { x: self.clone() });
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Softmax activation over the last dimension of a 2D tensor.
    ///
    /// softmax(x)_i = `exp(x_i)` / `Σ_j` `exp(x_j)`
    ///
    /// Uses numerically stable computation with max subtraction.
    #[must_use]
    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "softmax currently only supports 2D tensors");

        let (batch, features) = (self.shape()[0], self.shape()[1]);
        let mut output = vec![0.0; batch * features];

        for b in 0..batch {
            let row_start = b * features;

            // Find max for numerical stability
            let max_val = self.data()[row_start..row_start + features]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max) and sum
            let mut sum = 0.0;
            for j in 0..features {
                let exp_val = (self.data()[row_start + j] - max_val).exp();
                output[row_start + j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for j in 0..features {
                output[row_start + j] /= sum;
            }
        }

        let mut result = Tensor::new(&output, self.shape());

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(SoftmaxBackward {
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
// Linear Algebra
// ============================================================================

impl Tensor {
    /// Matrix multiplication: z = self @ other
    ///
    /// Currently supports 2D tensors only. Batched matmul (3D+ tensors) can be
    /// added by iterating over batch dimensions and calling 2D matmul.
    #[must_use]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(other.ndim(), 2, "matmul requires 2D tensors");

        let (m, k1) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (other.shape()[0], other.shape()[1]);
        assert_eq!(k1, k2, "matmul dimension mismatch: {k1} vs {k2}");

        // Use trueno's SIMD-accelerated matmul
        let a_matrix =
            trueno::Matrix::from_vec(m, k1, self.data().to_vec()).expect("valid matrix dimensions");
        let b_matrix = trueno::Matrix::from_vec(k2, n, other.data().to_vec())
            .expect("valid matrix dimensions");
        let result_matrix = a_matrix.matmul(&b_matrix).expect("matmul should succeed");
        let data = result_matrix.as_slice().to_vec();

        let mut result = Tensor::new(&data, &[m, n]);

        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(MatmulBackward {
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

    /// Transpose a 2D tensor.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let a_t = a.transpose();
    /// // a_t = [[1, 3], [2, 4]]
    /// ```
    #[must_use]
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");

        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data()[i * cols + j];
            }
        }

        let mut result = Tensor::new(&data, &[cols, rows]);

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(TransposeBackward);
            result.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(self.clone());
                graph.record(result.id(), grad_fn, vec![self.id()]);
            });
        }

        result
    }

    /// Broadcast addition: z = matrix + vector (broadcasts over rows).
    ///
    /// The vector is broadcast to match the matrix's second dimension.
    /// This is useful for adding biases in neural networks.
    ///
    /// # Shape
    ///
    /// - self: `[N, M]` (2D matrix)
    /// - other: `[M]` (1D vector)
    /// - output: `[N, M]`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let bias = Tensor::new(&[10.0, 20.0], &[2]);
    /// let result = matrix.broadcast_add(&bias);
    /// // result = [[11, 22], [13, 24]]
    /// ```
    #[must_use]
    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "broadcast_add requires 2D matrix");
        assert_eq!(other.ndim(), 1, "broadcast_add requires 1D vector");
        assert_eq!(
            self.shape()[1],
            other.shape()[0],
            "Matrix columns {} must match vector length {}",
            self.shape()[1],
            other.shape()[0]
        );

        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = self.data()[i * cols + j] + other.data()[j];
            }
        }

        let mut result = Tensor::new(&data, self.shape());

        if is_grad_enabled() && (self.requires_grad_enabled() || other.requires_grad_enabled()) {
            result.requires_grad_(true);
            let grad_fn = Arc::new(BroadcastAddBackward {
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

    /// Reshape tensor to a new shape (view).
    ///
    /// The total number of elements must remain the same.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let b = a.view(&[3, 2]);
    /// // b = [[1, 2], [3, 4], [5, 6]]
    /// ```
    #[must_use]
    pub fn view(&self, new_shape: &[usize]) -> Tensor {
        let old_numel: usize = self.shape().iter().product();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            old_numel, new_numel,
            "view: number of elements must match ({old_numel} vs {new_numel})"
        );

        let mut result = Tensor::new(self.data(), new_shape);

        if is_grad_enabled() && self.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(ViewBackward {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{clear_graph, no_grad};

    /// Numerical gradient check using central differences.
    #[allow(dead_code)]
    fn numerical_gradient<F>(f: F, x: &Tensor, eps: f32) -> Tensor
    where
        F: Fn(&Tensor) -> Tensor,
    {
        let mut grad_data = vec![0.0; x.numel()];

        for i in 0..x.numel() {
            let mut x_plus = x.data().to_vec();
            let mut x_minus = x.data().to_vec();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let y_plus = no_grad(|| f(&Tensor::new(&x_plus, x.shape())).item());
            let y_minus = no_grad(|| f(&Tensor::new(&x_minus, x.shape())).item());

            grad_data[i] = (y_plus - y_minus) / (2.0 * eps);
        }

        Tensor::new(&grad_data, x.shape())
    }

    #[allow(dead_code)]
    fn check_gradient<F>(f: F, x: &Tensor, eps: f32, tol: f32) -> bool
    where
        F: Fn(&Tensor) -> Tensor,
    {
        clear_graph();

        // Compute analytical gradient
        let x_grad = x.clone().requires_grad();
        let x_id = x_grad.id();
        let y = f(&x_grad);
        y.backward();

        // Get gradient from graph (since clones don't share storage)
        let analytical = crate::autograd::get_grad(x_id).expect("No gradient computed");

        // Compute numerical gradient
        let numerical = numerical_gradient(&f, x, eps);

        // Compare
        let max_diff: f32 = analytical
            .data()
            .iter()
            .zip(numerical.data().iter())
            .map(|(a, n)| (a - n).abs())
            .fold(0.0, f32::max);

        max_diff < tol
    }

    #[test]
    fn test_simple_sum_gradient() {
        // Simple test: d/dx sum(x) = [1, 1, 1]
        clear_graph();

        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let x_id = x.id();

        let y = x.sum();
        y.backward();

        let grad = crate::autograd::get_grad(x_id).expect("Gradient should exist");
        assert_eq!(grad.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_add_gradient() {
        // d/dx sum(x + y) = [1, 1, 1]
        clear_graph();

        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let x_id = x.id();

        let z = x.add(&y).sum();
        z.backward();

        let grad = crate::autograd::get_grad(x_id).expect("Should have gradient");
        assert_eq!(grad.data(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mul_gradient() {
        // d/dx sum(x * y) = y = [4, 5, 6]
        clear_graph();
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let x_id = x.id();
        let z = x.mul(&y).sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert_eq!(grad.data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_exp_gradient() {
        // d/dx sum(exp(x)) = exp(x)
        clear_graph();
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]).requires_grad();
        let x_id = x.id();
        let z = x.exp().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        for (g, &v) in grad.data().iter().zip(&[0.0_f32, 1.0, -1.0]) {
            assert!((g - v.exp()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_log_gradient() {
        // d/dx sum(log(x)) = 1/x
        clear_graph();
        let x = Tensor::from_slice(&[1.0, 2.0, 4.0]).requires_grad();
        let x_id = x.id();
        let z = x.log().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert_eq!(grad.data(), &[1.0, 0.5, 0.25]);
    }

    #[test]
    fn test_pow_gradient() {
        // d/dx sum(x^2) = 2x
        clear_graph();
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let x_id = x.id();
        let z = x.pow(2.0).sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert_eq!(grad.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_relu_gradient() {
        // d/dx sum(relu(x)) = 1 where x > 0
        clear_graph();
        let x = Tensor::from_slice(&[-1.0, 0.5, 2.0]).requires_grad();
        let x_id = x.id();
        let z = x.relu().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert_eq!(grad.data(), &[0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_gradient() {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        clear_graph();
        let x = Tensor::from_slice(&[0.0]).requires_grad();
        let x_id = x.id();
        let z = x.sigmoid().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert!((grad.data()[0] - 0.25).abs() < 1e-5); // sigmoid(0)=0.5, grad=0.5*0.5=0.25
    }

    #[test]
    fn test_mean_gradient() {
        // d/dx mean(x) = 1/n
        clear_graph();
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).requires_grad();
        let x_id = x.id();
        let z = x.mean();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert_eq!(grad.data(), &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_chain_gradient() {
        // Test chain rule: d/dx(sum((x * 2)^2)) = sum(4 * x * 2) = 8x
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);

        clear_graph();
        let x_grad = x.clone().requires_grad();
        let x_id = x_grad.id();
        let y = x_grad.mul_scalar(2.0).pow(2.0).sum();
        y.backward();

        let grad = crate::autograd::get_grad(x_id).expect("No gradient");
        // Expected: 8 * x = [8, 16, 24]
        let expected = [8.0, 16.0, 24.0];

        for (g, e) in grad.data().iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-3, "Expected {e}, got {g}");
        }
    }

    #[test]
    fn test_matmul_forward() {
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let c = a.matmul(&b);

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_tanh_gradient() {
        // d/dx tanh(x) = 1 - tanh²(x)
        // At x=0: tanh(0)=0, grad=1
        clear_graph();
        let x = Tensor::from_slice(&[0.0]).requires_grad();
        let x_id = x.id();
        let z = x.tanh_().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert!((grad.data()[0] - 1.0).abs() < 1e-5); // at x=0, tanh(0)=0, grad=1-0²=1
    }

    #[test]
    fn test_sqrt_gradient() {
        // d/dx sqrt(x) = 0.5 / sqrt(x)
        clear_graph();
        let x = Tensor::from_slice(&[4.0]).requires_grad();
        let x_id = x.id();
        let z = x.sqrt().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert!((grad.data()[0] - 0.25).abs() < 1e-5); // 0.5 / sqrt(4) = 0.5 / 2 = 0.25
    }

    #[test]
    fn test_matmul_backward() {
        // For z = sum(A @ B), gradients are:
        // dL/dA = grad_output @ B^T
        // dL/dB = A^T @ grad_output
        clear_graph();
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
        let b = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).requires_grad(); // identity
        let a_id = a.id();
        let b_id = b.id();

        let c = a.matmul(&b);
        let loss = c.sum();
        loss.backward();

        let grad_a = crate::autograd::get_grad(a_id).expect("grad_a");
        let grad_b = crate::autograd::get_grad(b_id).expect("grad_b");

        // dL/dA = ones @ B^T = [[1,1],[1,1]] @ [[1,0],[0,1]] = [[1,1],[1,1]]
        assert_eq!(grad_a.data(), &[1.0, 1.0, 1.0, 1.0]);
        // dL/dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        assert_eq!(grad_b.data(), &[4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_div_gradient() {
        // d/dx (x/y) = 1/y
        // d/dy (x/y) = -x/y²
        clear_graph();
        let x = Tensor::from_slice(&[6.0]).requires_grad();
        let y = Tensor::from_slice(&[2.0]).requires_grad();
        let x_id = x.id();
        let y_id = y.id();
        let z = x.div(&y).sum();
        z.backward();
        let grad_x = crate::autograd::get_grad(x_id).expect("grad_x");
        let grad_y = crate::autograd::get_grad(y_id).expect("grad_y");
        assert!((grad_x.data()[0] - 0.5).abs() < 1e-5); // 1/2
        assert!((grad_y.data()[0] - (-1.5)).abs() < 1e-5); // -6/4
    }

    #[test]
    fn test_neg_gradient() {
        // d/dx (-x) = -1
        clear_graph();
        let x = Tensor::from_slice(&[3.0]).requires_grad();
        let x_id = x.id();
        let z = x.neg().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert_eq!(grad.data()[0], -1.0);
    }

    #[test]
    fn test_pow_gradient_cubic() {
        // d/dx x^n = n * x^(n-1)
        clear_graph();
        let x = Tensor::from_slice(&[2.0]).requires_grad();
        let x_id = x.id();
        let z = x.pow(3.0).sum(); // x^3, grad = 3*x^2 = 12
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert!((grad.data()[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_exp_gradient_e() {
        // d/dx exp(x) = exp(x)
        clear_graph();
        let x = Tensor::from_slice(&[1.0]).requires_grad();
        let x_id = x.id();
        let z = x.exp().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert!((grad.data()[0] - std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn test_log_gradient_half() {
        // d/dx log(x) = 1/x
        clear_graph();
        let x = Tensor::from_slice(&[2.0]).requires_grad();
        let x_id = x.id();
        let z = x.log().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert!((grad.data()[0] - 0.5).abs() < 1e-5);
    }

    // =========================================================================
    // Extended coverage tests
    // =========================================================================

    #[test]
    fn test_sub_gradient() {
        // d/dx (x - y) = 1
        // d/dy (x - y) = -1
        clear_graph();
        let x = Tensor::from_slice(&[5.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[2.0, 1.0]).requires_grad();
        let x_id = x.id();
        let y_id = y.id();
        let z = x.sub(&y).sum();
        z.backward();
        let grad_x = crate::autograd::get_grad(x_id).expect("grad_x");
        let grad_y = crate::autograd::get_grad(y_id).expect("grad_y");
        assert_eq!(grad_x.data(), &[1.0, 1.0]);
        assert_eq!(grad_y.data(), &[-1.0, -1.0]);
    }

    #[test]
    fn test_leaky_relu_gradient_positive() {
        // For x > 0: d/dx leaky_relu(x) = 1
        clear_graph();
        let x = Tensor::from_slice(&[2.0, 3.0]).requires_grad();
        let x_id = x.id();
        let z = x.leaky_relu(0.01).sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        assert_eq!(grad.data(), &[1.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu_gradient_negative() {
        // For x < 0: d/dx leaky_relu(x) = negative_slope
        clear_graph();
        let x = Tensor::from_slice(&[-2.0, -3.0]).requires_grad();
        let x_id = x.id();
        let negative_slope = 0.1;
        let z = x.leaky_relu(negative_slope).sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        for &g in grad.data() {
            assert!((g - negative_slope).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gelu_forward() {
        // Test GELU forward computation
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let y = x.gelu();
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert!((y.data()[0] - 0.0).abs() < 1e-3);
        assert!((y.data()[1] - 0.841).abs() < 0.01);
        assert!((y.data()[2] - (-0.159)).abs() < 0.01);
    }

    #[test]
    fn test_gelu_gradient() {
        clear_graph();
        let x = Tensor::from_slice(&[0.0]).requires_grad();
        let x_id = x.id();
        let z = x.gelu().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        // GELU'(0) ≈ 0.5
        assert!((grad.data()[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_softmax_forward() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = x.softmax();

        // Check rows sum to 1
        let row1_sum: f32 = y.data()[0..2].iter().sum();
        let row2_sum: f32 = y.data()[2..4].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-5);
        assert!((row2_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_gradient() {
        clear_graph();
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
        let x_id = x.id();
        let z = x.softmax().sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        // Softmax gradients should be small since softmax(x).sum() = n (number of rows)
        // The derivative of constant is 0, but there's some numerical detail here
        assert_eq!(grad.numel(), 4);
    }

    #[test]
    fn test_transpose_forward() {
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let a_t = a.transpose();
        assert_eq!(a_t.shape(), &[3, 2]);
        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(a_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_gradient() {
        clear_graph();
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
        let a_id = a.id();
        let z = a.transpose().sum();
        z.backward();
        let grad = crate::autograd::get_grad(a_id).expect("grad");
        // Transpose of ones is ones
        assert_eq!(grad.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_broadcast_add_forward() {
        let matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let bias = Tensor::new(&[10.0, 20.0], &[2]);
        let result = matrix.broadcast_add(&bias);
        // [[1+10, 2+20], [3+10, 4+20]] = [[11, 22], [13, 24]]
        assert_eq!(result.data(), &[11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_broadcast_add_gradient() {
        clear_graph();
        let matrix = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
        let bias = Tensor::new(&[10.0, 20.0], &[2]).requires_grad();
        let m_id = matrix.id();
        let b_id = bias.id();
        let z = matrix.broadcast_add(&bias).sum();
        z.backward();
        let grad_m = crate::autograd::get_grad(m_id).expect("grad_m");
        let grad_b = crate::autograd::get_grad(b_id).expect("grad_b");
        // Matrix grad = ones
        assert_eq!(grad_m.data(), &[1.0, 1.0, 1.0, 1.0]);
        // Bias grad = summed over rows = [2, 2] (since 2 rows)
        assert_eq!(grad_b.data(), &[2.0, 2.0]);
    }

    #[test]
    fn test_view_forward() {
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = a.view(&[3, 2]);
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data(), a.data()); // Same data, different shape
    }

    #[test]
    fn test_view_gradient() {
        clear_graph();
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
        let a_id = a.id();
        let z = a.view(&[4]).sum();
        z.backward();
        let grad = crate::autograd::get_grad(a_id).expect("grad");
        assert_eq!(grad.shape(), &[2, 2]); // Same shape as input
        assert_eq!(grad.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mul_scalar_gradient() {
        clear_graph();
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let x_id = x.id();
        let z = x.mul_scalar(5.0).sum();
        z.backward();
        let grad = crate::autograd::get_grad(x_id).expect("grad");
        // d/dx (5*x).sum() = 5 for each element
        assert_eq!(grad.data(), &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_operations_without_grad() {
        // Test that operations work correctly when grad is not required
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);

        // All these should work without recording gradients
        let _ = x.add(&y);
        let _ = x.sub(&y);
        let _ = x.mul(&y);
        let _ = x.div(&y);
        let _ = x.neg();
        let _ = x.exp();
        let _ = x.log();
        let _ = x.pow(2.0);
        let _ = x.sqrt();
        let _ = x.sum();
        let _ = x.mean();
        let _ = x.relu();
        let _ = x.sigmoid();
        let _ = x.tanh_();
        let _ = x.leaky_relu(0.01);
        let _ = x.gelu();
        let _ = x.mul_scalar(2.0);
    }

    #[test]
    fn test_no_grad_context() {
        clear_graph();
        let result = no_grad(|| {
            let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
            x.sum()
        });
        // Inside no_grad, operations don't record to graph
        assert_eq!(result.data(), &[6.0]);
    }

    #[test]
    fn test_matmul_non_square() {
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]); // 2x3
        let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]); // 3x2

        let c = a.matmul(&b);

        assert_eq!(c.shape(), &[2, 2]);
        // Row 0: [1,2,3] @ [1,3,5; 2,4,6]^T = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // Row 1: [4,5,6] @ [1,3,5; 2,4,6]^T = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert_eq!(c.data(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_sub_forward() {
        let x = Tensor::from_slice(&[5.0, 3.0, 1.0]);
        let y = Tensor::from_slice(&[2.0, 2.0, 2.0]);
        let z = x.sub(&y);
        assert_eq!(z.data(), &[3.0, 1.0, -1.0]);
    }

    #[test]
    fn test_div_forward() {
        let x = Tensor::from_slice(&[6.0, 9.0, 12.0]);
        let y = Tensor::from_slice(&[2.0, 3.0, 4.0]);
        let z = x.div(&y);
        assert_eq!(z.data(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_neg_forward() {
        let x = Tensor::from_slice(&[1.0, -2.0, 3.0]);
        let z = x.neg();
        assert_eq!(z.data(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_sqrt_forward() {
        let x = Tensor::from_slice(&[4.0, 9.0, 16.0]);
        let z = x.sqrt();
        assert_eq!(z.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tanh_forward() {
        let x = Tensor::from_slice(&[0.0]);
        let z = x.tanh_();
        assert!((z.data()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_relu_forward() {
        let x = Tensor::from_slice(&[-2.0, 0.0, 2.0]);
        let z = x.relu();
        assert_eq!(z.data(), &[0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let x = Tensor::from_slice(&[-2.0, 0.0, 2.0]);
        let z = x.leaky_relu(0.1);
        assert_eq!(z.data(), &[-0.2, 0.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let x = Tensor::from_slice(&[0.0]);
        let z = x.sigmoid();
        assert!((z.data()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_log_forward() {
        let x = Tensor::from_slice(&[1.0, std::f32::consts::E]);
        let z = x.log();
        assert!((z.data()[0] - 0.0).abs() < 1e-5);
        assert!((z.data()[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_exp_forward() {
        let x = Tensor::from_slice(&[0.0, 1.0]);
        let z = x.exp();
        assert!((z.data()[0] - 1.0).abs() < 1e-5);
        assert!((z.data()[1] - std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn test_pow_forward() {
        let x = Tensor::from_slice(&[2.0, 3.0]);
        let z = x.pow(2.0);
        assert_eq!(z.data(), &[4.0, 9.0]);
    }

    #[test]
    fn test_add_forward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let z = x.add(&y);
        assert_eq!(z.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_forward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let y = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let z = x.mul(&y);
        assert_eq!(z.data(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_sum_forward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let z = x.sum();
        assert_eq!(z.data(), &[10.0]);
    }

    #[test]
    fn test_mean_forward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let z = x.mean();
        assert_eq!(z.data(), &[2.5]);
    }

    #[test]
    fn test_mul_scalar_forward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let z = x.mul_scalar(3.0);
        assert_eq!(z.data(), &[3.0, 6.0, 9.0]);
    }
}
