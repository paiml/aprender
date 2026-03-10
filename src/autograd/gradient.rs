impl GradFn for CrossEntropyBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let output_contig = self.softmax_output.contiguous();
        let (batch, num_classes) = (output_contig.shape()[0], output_contig.shape()[1]);
        let mut grad_input = output_contig.data().to_vec();

        // grad = softmax - one_hot(targets)
        // Then multiply by upstream gradient (for reduction)
        let input_grad = grad_output.contiguous();
        let grad_scale = input_grad.data()[0]; // scalar after mean reduction

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
        let input_grad = grad_output.contiguous();
        let output_contig = self.output.contiguous();
        // ∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))
        let grad_data: Vec<f32> = input_grad
            .data()
            .iter()
            .zip(output_contig.data().iter())
            .map(|(&g, &s)| g * s * (1.0 - s))
            .collect();
        vec![Tensor::new(&grad_data, input_grad.shape())]
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
        let input_grad = grad_output.contiguous();
        let output_contig = self.output.contiguous();
        // ∂tanh(x)/∂x = 1 - tanh²(x)
        let grad_data: Vec<f32> = input_grad
            .data()
            .iter()
            .zip(output_contig.data().iter())
            .map(|(&g, &t)| g * (1.0 - t * t))
            .collect();
        vec![Tensor::new(&grad_data, input_grad.shape())]
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
        let input_grad = grad_output.contiguous();
        vec![Tensor::new(input_grad.data(), &self.input_shape)]
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
    let input = grad.contiguous();
    let sum: f32 = input.data().iter().sum();
    Tensor::new(&[sum], target_shape)
}

/// Reduce 2D gradient to 1D by summing over batch dimension.
fn reduce_batch_to_features(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let input = grad.contiguous();
    let (rows, cols) = (input.shape()[0], input.shape()[1]);
    let mut reduced = vec![0.0; cols];
    let grad_data = input.data();
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
        let input = grad.contiguous();
        return Tensor::new(input.data(), target_shape);
    }

    grad.clone()
}

/// SIMD-friendly 2D matrix transpose using trueno.
///
/// Refactored to use lazy transpose for performance.
fn transpose_2d(t: &Tensor) -> Tensor {
    t.transpose()
}

/// SIMD-accelerated 2D matrix multiplication using trueno.
///
/// Uses trueno's SIMD-optimized matmul for Ollama-parity performance.
/// Performance: ~10-50x faster than naive triple loop on large matrices.
fn matmul_2d(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "matmul_2d requires 2D tensors");
    assert_eq!(b.ndim(), 2, "matmul_2d requires 2D tensors");

    let input_a = a.contiguous();
    let input_b = b.contiguous();

    let (m, k1) = (input_a.shape()[0], input_a.shape()[1]);
    let (k2, n) = (input_b.shape()[0], input_b.shape()[1]);
    assert_eq!(k1, k2, "matmul dimension mismatch: {k1} vs {k2}");

    // Use trueno's SIMD-accelerated matmul for performance parity with Ollama
    let a_matrix =
        trueno::Matrix::from_vec(m, k1, input_a.data().to_vec()).expect("valid matrix dimensions");
    let b_matrix =
        trueno::Matrix::from_vec(k2, n, input_b.data().to_vec()).expect("valid matrix dimensions");
    let result_matrix = a_matrix.matmul(&b_matrix).expect("matmul should succeed");

    Tensor::new(result_matrix.as_slice(), &[m, n])
}
