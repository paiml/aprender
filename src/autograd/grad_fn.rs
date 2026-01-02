//! Gradient function trait and implementations.
//!
//! Each differentiable operation implements `GradFn` to define
//! how gradients flow backward through the operation.
//!
//! Uses trueno for SIMD-accelerated backward passes to achieve
//! Ollama-parity performance.

use super::tensor::Tensor;

/// Trait for functions that compute gradients during backward pass.
///
/// Each differentiable operation creates a `GradFn` implementation
/// that captures the necessary context for gradient computation.
///
/// # Example Implementation
///
/// For element-wise addition z = x + y:
/// - ∂z/∂x = 1
/// - ∂z/∂y = 1
///
/// So backward(grad_output) returns [grad_output, grad_output].
pub trait GradFn: Send + Sync {
    /// Compute gradients with respect to inputs.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient flowing back from downstream operations
    ///
    /// # Returns
    ///
    /// Vector of gradients, one for each input tensor.
    /// The order must match the input order used during forward pass.
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;

    /// Human-readable name for debugging.
    fn name(&self) -> &'static str;
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Gradient function for addition: z = x + y
pub(crate) struct AddBackward {
    pub(crate) x_shape: Vec<usize>,
    pub(crate) y_shape: Vec<usize>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
        // Handle broadcasting by summing over broadcast dimensions
        let grad_x = maybe_reduce_grad(grad_output, &self.x_shape);
        let grad_y = maybe_reduce_grad(grad_output, &self.y_shape);
        vec![grad_x, grad_y]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient function for subtraction: z = x - y
pub(crate) struct SubBackward {
    pub(crate) x_shape: Vec<usize>,
    pub(crate) y_shape: Vec<usize>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1
        let grad_x = maybe_reduce_grad(grad_output, &self.x_shape);
        let grad_y_data: Vec<f32> = grad_output.data().iter().map(|&g| -g).collect();
        let grad_y_full = Tensor::new(&grad_y_data, grad_output.shape());
        let grad_y = maybe_reduce_grad(&grad_y_full, &self.y_shape);
        vec![grad_x, grad_y]
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Gradient function for multiplication: z = x * y
pub(crate) struct MulBackward {
    pub(crate) x: Tensor,
    pub(crate) y: Tensor,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂(x*y)/∂x = y, ∂(x*y)/∂y = x
        let grad_x_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.y.data().iter())
            .map(|(&g, &y)| g * y)
            .collect();
        let grad_y_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .map(|(&g, &x)| g * x)
            .collect();

        let grad_x = maybe_reduce_grad(
            &Tensor::new(&grad_x_data, grad_output.shape()),
            self.x.shape(),
        );
        let grad_y = maybe_reduce_grad(
            &Tensor::new(&grad_y_data, grad_output.shape()),
            self.y.shape(),
        );
        vec![grad_x, grad_y]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient function for division: z = x / y
pub(crate) struct DivBackward {
    pub(crate) x: Tensor,
    pub(crate) y: Tensor,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y²
        let grad_x_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.y.data().iter())
            .map(|(&g, &y)| g / y)
            .collect();
        let grad_y_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .zip(self.y.data().iter())
            .map(|((&g, &x), &y)| -g * x / (y * y))
            .collect();

        let grad_x = maybe_reduce_grad(
            &Tensor::new(&grad_x_data, grad_output.shape()),
            self.x.shape(),
        );
        let grad_y = maybe_reduce_grad(
            &Tensor::new(&grad_y_data, grad_output.shape()),
            self.y.shape(),
        );
        vec![grad_x, grad_y]
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

/// Gradient function for negation: z = -x
pub(crate) struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂(-x)/∂x = -1
        let grad_data: Vec<f32> = grad_output.data().iter().map(|&g| -g).collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

// ============================================================================
// Transcendental Operations
// ============================================================================

/// Gradient function for exp: z = exp(x)
pub(crate) struct ExpBackward {
    pub(crate) output: Tensor, // exp(x) - we save the output, not input
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂exp(x)/∂x = exp(x)
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.output.data().iter())
            .map(|(&g, &exp_x)| g * exp_x)
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

/// Gradient function for log: z = log(x)
pub(crate) struct LogBackward {
    pub(crate) x: Tensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂log(x)/∂x = 1/x
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .map(|(&g, &x)| g / x)
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

/// Gradient function for pow: z = x^n
pub(crate) struct PowBackward {
    pub(crate) x: Tensor,
    pub(crate) n: f32,
}

impl GradFn for PowBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂(x^n)/∂x = n * x^(n-1)
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .map(|(&g, &x)| g * self.n * x.powf(self.n - 1.0))
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }
}

/// Gradient function for sqrt: z = sqrt(x)
pub(crate) struct SqrtBackward {
    pub(crate) output: Tensor, // sqrt(x)
}

impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂sqrt(x)/∂x = 0.5 / sqrt(x)
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.output.data().iter())
            .map(|(&g, &sqrt_x)| g * 0.5 / sqrt_x)
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Gradient function for sum: z = sum(x)
pub(crate) struct SumBackward {
    pub(crate) input_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂sum(x)/∂x_i = 1 for all i
        // Broadcast scalar gradient to input shape
        let g = grad_output.item();
        let numel: usize = self.input_shape.iter().product();
        vec![Tensor::new(&vec![g; numel], &self.input_shape)]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Gradient function for mean: z = mean(x)
pub(crate) struct MeanBackward {
    pub(crate) input_shape: Vec<usize>,
}

impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂mean(x)/∂x_i = 1/n for all i
        let g = grad_output.item();
        let numel: usize = self.input_shape.iter().product();
        let grad_val = g / numel as f32;
        vec![Tensor::new(&vec![grad_val; numel], &self.input_shape)]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/// Gradient function for ReLU: z = max(0, x)
pub(crate) struct ReluBackward {
    pub(crate) x: Tensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂relu(x)/∂x = 1 if x > 0, else 0
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

/// Gradient function for LeakyReLU: z = max(negative_slope * x, x)
pub(crate) struct LeakyReluBackward {
    pub(crate) x: Tensor,
    pub(crate) negative_slope: f32,
}

impl GradFn for LeakyReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂leaky_relu(x)/∂x = 1 if x > 0, else negative_slope
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .map(|(&g, &x)| if x > 0.0 { g } else { g * self.negative_slope })
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "LeakyReluBackward"
    }
}

/// Gradient function for GELU (Gaussian Error Linear Unit)
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub(crate) struct GeluBackward {
    pub(crate) x: Tensor,
}

impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // GELU gradient approximation (same as PyTorch's tanh approximation)
        // d/dx[GELU(x)] ≈ 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh²(inner)) * inner'
        // where inner = sqrt(2/π) * (x + 0.044715 * x³)
        // and inner' = sqrt(2/π) * (1 + 3 * 0.044715 * x²)
        let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();

        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .map(|(&g, &x)| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                let tanh_inner = inner.tanh();
                let inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x.powi(2));
                let gelu_deriv =
                    0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner.powi(2)) * inner_deriv;
                g * gelu_deriv
            })
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "GeluBackward"
    }
}

/// Gradient function for Softmax over last dimension of 2D tensor
/// For y = softmax(x), the gradient is:
/// ∂L/∂x_i = y_i * (g_i - Σ_j g_j * y_j)
pub(crate) struct SoftmaxBackward {
    pub(crate) output: Tensor, // softmax output (needed for gradient)
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        assert_eq!(self.output.ndim(), 2, "SoftmaxBackward expects 2D tensor");

        let (batch, features) = (self.output.shape()[0], self.output.shape()[1]);
        let mut grad_input = vec![0.0; batch * features];

        let out_data = self.output.data();
        let grad_data = grad_output.data();

        for b in 0..batch {
            let row_start = b * features;

            // Compute dot(grad_output, output) for this row
            let mut dot_product = 0.0;
            for j in 0..features {
                dot_product += grad_data[row_start + j] * out_data[row_start + j];
            }

            // grad_input = output * (grad_output - dot_product)
            for j in 0..features {
                let idx = row_start + j;
                grad_input[idx] = out_data[idx] * (grad_data[idx] - dot_product);
            }
        }

        vec![Tensor::new(&grad_input, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

/// Gradient function for Cross-Entropy Loss (combined softmax + NLL)
/// For L = -log(softmax(x)[target]), the gradient is:
/// ∂L/∂x_i = softmax(x)_i - 1 if i == target else softmax(x)_i
/// This is simply: grad = softmax(logits) - one_hot(targets)
pub(crate) struct CrossEntropyBackward {
    pub(crate) softmax_output: Tensor, // softmax(logits)
    pub(crate) targets: Vec<usize>,    // target class indices
}

impl GradFn for CrossEntropyBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (batch, num_classes) = (
            self.softmax_output.shape()[0],
            self.softmax_output.shape()[1],
        );
        let mut grad_input = self.softmax_output.data().to_vec();

        // grad = softmax - one_hot(targets)
        // Then multiply by upstream gradient (for reduction)
        let grad_scale = grad_output.data()[0]; // scalar after mean reduction

        for b in 0..batch {
            let target = self.targets[b];
            let idx = b * num_classes + target;
            grad_input[idx] -= 1.0;
        }

        // Scale by upstream gradient and divide by batch size (for mean reduction)
        for g in &mut grad_input {
            *g *= grad_scale / batch as f32;
        }

        vec![Tensor::new(&grad_input, self.softmax_output.shape())]
    }

    fn name(&self) -> &'static str {
        "CrossEntropyBackward"
    }
}

/// Gradient function for sigmoid: z = 1 / (1 + exp(-x))
pub(crate) struct SigmoidBackward {
    pub(crate) output: Tensor, // sigmoid(x)
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.output.data().iter())
            .map(|(&g, &s)| g * s * (1.0 - s))
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

/// Gradient function for tanh
pub(crate) struct TanhBackward {
    pub(crate) output: Tensor, // tanh(x)
}

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂tanh(x)/∂x = 1 - tanh²(x)
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.output.data().iter())
            .map(|(&g, &t)| g * (1.0 - t * t))
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}

// ============================================================================
// Linear Algebra
// ============================================================================

/// Gradient function for matrix multiplication: z = x @ y
pub(crate) struct MatmulBackward {
    pub(crate) x: Tensor,
    pub(crate) y: Tensor,
}

/// Gradient function for transpose: z = x^T
pub(crate) struct TransposeBackward;

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂(x^T)/∂x is also transpose: grad_x = grad_output^T
        vec![transpose_2d(grad_output)]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

/// Gradient function for broadcast add: z = x + y (with broadcasting)
pub(crate) struct BroadcastAddBackward {
    pub(crate) x_shape: Vec<usize>,
    pub(crate) y_shape: Vec<usize>,
}

impl GradFn for BroadcastAddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // For broadcast add, we need to sum over broadcast dimensions
        let grad_x = maybe_reduce_grad(grad_output, &self.x_shape);
        let grad_y = maybe_reduce_grad(grad_output, &self.y_shape);
        vec![grad_x, grad_y]
    }

    fn name(&self) -> &'static str {
        "BroadcastAddBackward"
    }
}

/// Gradient function for view/reshape: z = x.view(new_shape)
pub(crate) struct ViewBackward {
    pub(crate) input_shape: Vec<usize>,
}

impl GradFn for ViewBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Gradient of reshape is just reshaping back to original shape
        vec![Tensor::new(grad_output.data(), &self.input_shape)]
    }

    fn name(&self) -> &'static str {
        "ViewBackward"
    }
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // For z = x @ y:
        // ∂L/∂x = ∂L/∂z @ y^T
        // ∂L/∂y = x^T @ ∂L/∂z
        //
        // This implementation handles 2D matrices. For batched matmul (3D+),
        // compute gradients per-batch by iterating over the batch dimension.

        let grad_x = matmul_2d(grad_output, &transpose_2d(&self.y));
        let grad_y = matmul_2d(&transpose_2d(&self.x), grad_output);

        vec![grad_x, grad_y]
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Reduce gradient to scalar by summing all elements.
fn reduce_to_scalar(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let sum: f32 = grad.data().iter().sum();
    Tensor::new(&[sum], target_shape)
}

/// Reduce 2D gradient to 1D by summing over batch dimension.
fn reduce_batch_to_features(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let (rows, cols) = (grad.shape()[0], grad.shape()[1]);
    let mut reduced = vec![0.0; cols];
    let grad_data = grad.data();
    for i in 0..rows {
        for (j, r) in reduced.iter_mut().enumerate() {
            *r += grad_data[i * cols + j];
        }
    }
    Tensor::new(&reduced, target_shape)
}

/// Check if gradient needs 2D -> 1D reduction.
fn needs_batch_reduction(grad: &Tensor, target_shape: &[usize]) -> bool {
    grad.ndim() == 2 && target_shape.len() == 1 && grad.shape()[1] == target_shape[0]
}

/// Reduce gradient if shapes don't match (for broadcasting).
fn maybe_reduce_grad(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    if grad.shape() == target_shape {
        return grad.clone();
    }

    // Simple case: target is scalar
    if target_shape.is_empty() || target_shape == [1] {
        return reduce_to_scalar(grad, target_shape);
    }

    // Handle 2D -> 1D case: sum over batch dimension (for bias gradients)
    if needs_batch_reduction(grad, target_shape) {
        return reduce_batch_to_features(grad, target_shape);
    }

    // If shapes match in size, just reshape
    if grad.numel() == target_shape.iter().product::<usize>() {
        return Tensor::new(grad.data(), target_shape);
    }

    grad.clone()
}

/// SIMD-friendly 2D matrix transpose using trueno.
///
/// Uses trueno's cache-optimized transpose for better memory access patterns.
fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.ndim(), 2, "transpose_2d requires 2D tensor");
    let (rows, cols) = (t.shape()[0], t.shape()[1]);

    // Use trueno's Matrix transpose for optimized memory access
    let matrix =
        trueno::Matrix::from_vec(rows, cols, t.data().to_vec()).expect("valid matrix dimensions");
    let transposed = matrix.transpose();

    Tensor::new(transposed.as_slice(), &[cols, rows])
}

/// SIMD-accelerated 2D matrix multiplication using trueno.
///
/// Uses trueno's SIMD-optimized matmul for Ollama-parity performance.
/// Performance: ~10-50x faster than naive triple loop on large matrices.
fn matmul_2d(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "matmul_2d requires 2D tensors");
    assert_eq!(b.ndim(), 2, "matmul_2d requires 2D tensors");

    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    assert_eq!(k1, k2, "matmul dimension mismatch: {k1} vs {k2}");

    // Use trueno's SIMD-accelerated matmul for performance parity with Ollama
    let a_matrix =
        trueno::Matrix::from_vec(m, k1, a.data().to_vec()).expect("valid matrix dimensions");
    let b_matrix =
        trueno::Matrix::from_vec(k2, n, b.data().to_vec()).expect("valid matrix dimensions");
    let result_matrix = a_matrix.matmul(&b_matrix).expect("matmul should succeed");

    Tensor::new(result_matrix.as_slice(), &[m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward() {
        let grad_fn = AddBackward {
            x_shape: vec![3],
            y_shape: vec![3],
        };
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(grads[1].data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mul_backward() {
        let x = Tensor::from_slice(&[2.0, 3.0]);
        let y = Tensor::from_slice(&[4.0, 5.0]);
        let grad_fn = MulBackward {
            x: x.clone(),
            y: y.clone(),
        };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad_x = grad_out * y = [1*4, 1*5] = [4, 5]
        assert_eq!(grads[0].data(), &[4.0, 5.0]);
        // grad_y = grad_out * x = [1*2, 1*3] = [2, 3]
        assert_eq!(grads[1].data(), &[2.0, 3.0]);
    }

    #[test]
    fn test_relu_backward() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0]);
        let grad_fn = ReluBackward { x };

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = grad_out where x > 0, else 0
        assert_eq!(grads[0].data(), &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_backward() {
        let grad_fn = SumBackward {
            input_shape: vec![3],
        };
        let grad_out = Tensor::new(&[2.0], &[1]);
        let grads = grad_fn.backward(&grad_out);

        // Gradient is broadcast to all elements
        assert_eq!(grads[0].data(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_mean_backward() {
        let grad_fn = MeanBackward {
            input_shape: vec![4],
        };
        let grad_out = Tensor::new(&[1.0], &[1]);
        let grads = grad_fn.backward(&grad_out);

        // Gradient is 1/n for each element
        assert_eq!(grads[0].data(), &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_transpose_2d() {
        let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t_t = transpose_2d(&t);

        assert_eq!(t_t.shape(), &[3, 2]);
        assert_eq!(t_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul_2d() {
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        let c = matmul_2d(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        // Row 0: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // Row 1: [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert_eq!(c.data(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_sub_backward() {
        let grad_fn = SubBackward {
            x_shape: vec![3],
            y_shape: vec![3],
        };
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(grads[1].data(), &[-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_div_backward() {
        let x = Tensor::from_slice(&[6.0, 8.0]);
        let y = Tensor::from_slice(&[2.0, 4.0]);
        let grad_fn = DivBackward {
            x: x.clone(),
            y: y.clone(),
        };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad_x = grad_out / y = [1/2, 1/4] = [0.5, 0.25]
        assert_eq!(grads[0].data(), &[0.5, 0.25]);
        // grad_y = -grad_out * x / y^2 = [-1*6/4, -1*8/16] = [-1.5, -0.5]
        assert_eq!(grads[1].data(), &[-1.5, -0.5]);
    }

    #[test]
    fn test_pow_backward() {
        let x = Tensor::from_slice(&[2.0, 3.0]);
        let grad_fn = PowBackward {
            x: x.clone(),
            n: 2.0,
        };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = n * x^(n-1) * grad_out = 2 * [2, 3] = [4, 6]
        assert_eq!(grads[0].data(), &[4.0, 6.0]);
    }

    #[test]
    fn test_exp_backward() {
        let output = Tensor::from_slice(&[2.718281828, 7.389056099]); // e^1, e^2
        let grad_fn = ExpBackward { output };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = exp(x) * grad_out = output * grad_out
        assert!((grads[0].data()[0] - 2.718281828).abs() < 1e-5);
        assert!((grads[0].data()[1] - 7.389056099).abs() < 1e-5);
    }

    #[test]
    fn test_log_backward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 4.0]);
        let grad_fn = LogBackward { x };

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = 1/x * grad_out
        assert_eq!(grads[0].data(), &[1.0, 0.5, 0.25]);
    }

    #[test]
    fn test_sigmoid_backward() {
        let output = Tensor::from_slice(&[0.5, 0.731]); // sigmoid values
        let grad_fn = SigmoidBackward { output };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = sigmoid(x) * (1 - sigmoid(x)) * grad_out
        assert!((grads[0].data()[0] - 0.25).abs() < 1e-5); // 0.5 * 0.5
    }

    #[test]
    fn test_tanh_backward() {
        let output = Tensor::from_slice(&[0.0, 0.7616]); // tanh values
        let grad_fn = TanhBackward { output };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = (1 - tanh(x)^2) * grad_out
        assert!((grads[0].data()[0] - 1.0).abs() < 1e-5); // 1 - 0^2
    }

    #[test]
    fn test_backward_names() {
        assert_eq!(
            AddBackward {
                x_shape: vec![],
                y_shape: vec![]
            }
            .name(),
            "AddBackward"
        );
        assert_eq!(
            SubBackward {
                x_shape: vec![],
                y_shape: vec![]
            }
            .name(),
            "SubBackward"
        );
        assert_eq!(
            MulBackward {
                x: Tensor::from_slice(&[1.0]),
                y: Tensor::from_slice(&[1.0])
            }
            .name(),
            "MulBackward"
        );
        assert_eq!(
            DivBackward {
                x: Tensor::from_slice(&[1.0]),
                y: Tensor::from_slice(&[1.0])
            }
            .name(),
            "DivBackward"
        );
    }

    #[test]
    fn test_neg_backward() {
        let grad_fn = NegBackward;
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].data(), &[-1.0, -2.0, -3.0]);
        assert_eq!(grad_fn.name(), "NegBackward");
    }

    #[test]
    fn test_sqrt_backward() {
        // sqrt(4) = 2, sqrt(9) = 3, sqrt(16) = 4
        let output = Tensor::from_slice(&[2.0, 3.0, 4.0]);
        let grad_fn = SqrtBackward { output };
        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = grad_out / (2 * sqrt(x)) = 1 / (2 * output)
        assert_eq!(grads.len(), 1);
        assert!((grads[0].data()[0] - 0.25).abs() < 1e-5); // 1/(2*2)
        assert!((grads[0].data()[1] - (1.0 / 6.0)).abs() < 1e-5); // 1/(2*3)
        assert!((grads[0].data()[2] - 0.125).abs() < 1e-5); // 1/(2*4)
        assert_eq!(grad_fn.name(), "SqrtBackward");
    }

    #[test]
    fn test_leaky_relu_backward() {
        let x = Tensor::from_slice(&[1.0, -1.0, 0.0, 2.0]);
        let grad_fn = LeakyReluBackward {
            x,
            negative_slope: 0.01,
        };
        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        // For x > 0: grad = 1.0, for x <= 0: grad = negative_slope
        assert!((grads[0].data()[0] - 1.0).abs() < 1e-5); // x = 1.0 > 0
        assert!((grads[0].data()[1] - 0.01).abs() < 1e-5); // x = -1.0 <= 0
        assert!((grads[0].data()[2] - 0.01).abs() < 1e-5); // x = 0.0 <= 0
        assert!((grads[0].data()[3] - 1.0).abs() < 1e-5); // x = 2.0 > 0
        assert_eq!(grad_fn.name(), "LeakyReluBackward");
    }

    #[test]
    fn test_gelu_backward() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let grad_fn = GeluBackward { x };
        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        // GELU'(0) ≈ 0.5
        assert!((grads[0].data()[0] - 0.5).abs() < 0.01);
        assert_eq!(grad_fn.name(), "GeluBackward");
    }

    #[test]
    fn test_softmax_backward() {
        // SoftmaxBackward expects 2D tensor (batch, features)
        let output = Tensor::new(&[0.5, 0.5], &[1, 2]);
        let grad_fn = SoftmaxBackward { output };
        let grad_out = Tensor::new(&[1.0, 0.0], &[1, 2]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[1, 2]);
        assert_eq!(grad_fn.name(), "SoftmaxBackward");
    }

    #[test]
    fn test_cross_entropy_backward() {
        // CrossEntropyBackward expects 2D tensor (batch, num_classes)
        let softmax_output = Tensor::new(&[0.7, 0.2, 0.1], &[1, 3]);
        let targets = vec![0_usize]; // target class index (one per batch item)
        let grad_fn = CrossEntropyBackward {
            softmax_output,
            targets,
        };
        let grad_out = Tensor::from_slice(&[1.0]); // scalar after mean reduction
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[1, 3]);
        assert_eq!(grad_fn.name(), "CrossEntropyBackward");
    }

    #[test]
    fn test_broadcast_add_backward() {
        let grad_fn = BroadcastAddBackward {
            x_shape: vec![2, 3],
            y_shape: vec![3],
        };
        let grad_out = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]); // x grad unchanged
        assert_eq!(grads[1].shape(), &[3]); // y grad reduced
        assert_eq!(grad_fn.name(), "BroadcastAddBackward");
    }

    #[test]
    fn test_view_backward() {
        let grad_fn = ViewBackward {
            input_shape: vec![2, 3],
        };
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[2, 3]); // Reshaped back
        assert_eq!(grad_fn.name(), "ViewBackward");
    }

    #[test]
    fn test_transpose_backward_fn() {
        let grad_fn = TransposeBackward;
        let grad_out = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[2, 3]); // Transposed back
        assert_eq!(grad_fn.name(), "TransposeBackward");
    }

    #[test]
    fn test_all_backward_names() {
        // Ensure all backward functions have unique, descriptive names
        let names = vec![
            NegBackward.name(),
            ExpBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            LogBackward {
                x: Tensor::from_slice(&[1.0]),
            }
            .name(),
            SqrtBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            SumBackward {
                input_shape: vec![1],
            }
            .name(),
            MeanBackward {
                input_shape: vec![1],
            }
            .name(),
            ReluBackward {
                x: Tensor::from_slice(&[1.0]),
            }
            .name(),
            LeakyReluBackward {
                x: Tensor::from_slice(&[1.0]),
                negative_slope: 0.01,
            }
            .name(),
            GeluBackward {
                x: Tensor::from_slice(&[1.0]),
            }
            .name(),
            SoftmaxBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            CrossEntropyBackward {
                softmax_output: Tensor::from_slice(&[1.0]),
                targets: vec![0],
            }
            .name(),
            SigmoidBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            TanhBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            TransposeBackward.name(),
            ViewBackward {
                input_shape: vec![1],
            }
            .name(),
            BroadcastAddBackward {
                x_shape: vec![1],
                y_shape: vec![1],
            }
            .name(),
        ];

        // All names should be unique
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "All backward names should be unique"
        );

        // All names should end with "Backward"
        for name in &names {
            assert!(
                name.ends_with("Backward"),
                "Name {} should end with 'Backward'",
                name
            );
        }
    }
}
