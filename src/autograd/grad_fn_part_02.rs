
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

/// Gradient function for view/reshape: z = `x.view(new_shape)`
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
