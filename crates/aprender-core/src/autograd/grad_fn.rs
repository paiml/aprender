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
/// So `backward(grad_output)` returns [`grad_output`, `grad_output`].
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

/// Gradient function for abs: z = |x|
///
/// PMAT-896: without this grad_fn, `abs()` severed the autograd graph and
/// `L1Loss` (which is `mean(|pred - target|)`) produced no input gradient,
/// silently training to zero. ∂|x|/∂x = sign(x), with sign(0) = 0 to match
/// PyTorch's subgradient convention at the non-differentiable point.
pub(crate) struct AbsBackward {
    pub(crate) x: Tensor,
}

impl GradFn for AbsBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // ∂|x|/∂x = sign(x); sign(0) = 0 (PyTorch subgradient convention)
        let grad_data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data().iter())
            .map(|(&g, &x)| {
                let sign = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                g * sign
            })
            .collect();
        vec![Tensor::new(&grad_data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "AbsBackward"
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

/// Gradient function for `ReLU`: z = max(0, x)
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

/// Gradient function for `LeakyReLU`: z = `max(negative_slope` * x, x)
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
/// ∂`L/∂x_i` = `y_i` * (`g_i` - `Σ_j` `g_j` * `y_j`)
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
/// For L = -log(softmax(x)[target]), the per-sample gradient is:
/// ∂`L/∂x_i` = softmax(x)_i - 1 if i == target else softmax(x)_i
/// i.e. grad = softmax(logits) - `one_hot(targets)`.
///
/// The final input gradient depends on the loss `reduction` mode (PyTorch parity,
/// F-AUTOGRAD-CE-REDUCTION-001):
/// - `Mean`: grad = (softmax - onehot) / batch   (upstream is a scalar)
/// - `Sum`:  grad = (softmax - onehot)            (upstream is a scalar; NO /batch)
/// - `None`: grad = (softmax - onehot) * upstream\[b]  (per-sample upstream row-scale)
pub(crate) struct CrossEntropyBackward {
    pub(crate) softmax_output: Tensor,                // softmax(logits)
    pub(crate) targets: Vec<usize>,                   // target class indices
    pub(crate) reduction: crate::nn::loss::Reduction, // loss reduction mode
}

// ============================================================================
// Normalization Operations (PMAT-907)
// ============================================================================

/// Gradient function for affine LayerNorm: y = x_hat * gamma + beta,
/// where x_hat = (x - mean) / sqrt(var + eps) per normalized row.
///
/// Obligation: OBLIG-LAYERNORM-BACKWARD-GRAD-FLOW.
///
/// Before PMAT-907 the canonical functional `layer_norm` built its output via
/// `Tensor::from_vec`, severing the autograd graph: gamma/beta/x received no
/// gradient, making every LayerNorm transformer non-fine-tunable. This grad_fn
/// flows gradient to ALL THREE inputs (x, gamma, beta), in that input order.
///
/// Math (per row of width `norm_dim`, with std_inv = 1/sqrt(var+eps)):
/// - dL/dgamma_i = sum_batch g_i * x_hat_i
/// - dL/dbeta_i  = sum_batch g_i
/// - dL/dx_i     = std_inv * (g'_i - mean_j(g'_j) - x_hat_i * mean_j(g'_j * x_hat_j))
///   where g'_i = g_i * gamma_i  (the gradient scaled by the affine weight).
pub(crate) struct LayerNormBackward {
    pub(crate) x: Tensor,
    pub(crate) gamma: Tensor,
    pub(crate) eps: f32,
}

impl GradFn for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let norm_dim = self.gamma.data().len();
        let x_data = self.x.data();
        let gamma_data = self.gamma.data();
        let g_data = grad_output.data();
        let batch = x_data.len() / norm_dim;
        let n = norm_dim as f32;

        let mut grad_x = vec![0.0f32; x_data.len()];
        let mut grad_gamma = vec![0.0f32; norm_dim];
        let mut grad_beta = vec![0.0f32; norm_dim];

        for b in 0..batch {
            let off = b * norm_dim;
            let xs = &x_data[off..off + norm_dim];
            let gs = &g_data[off..off + norm_dim];

            let mean: f32 = xs.iter().sum::<f32>() / n;
            let var: f32 = xs.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let std_inv = 1.0 / (var + self.eps).sqrt();

            // x_hat and g' = g * gamma; accumulate affine-param grads.
            let mut x_hat = vec![0.0f32; norm_dim];
            let mut g_prime = vec![0.0f32; norm_dim];
            for i in 0..norm_dim {
                let xh = (xs[i] - mean) * std_inv;
                x_hat[i] = xh;
                g_prime[i] = gs[i] * gamma_data[i];
                grad_gamma[i] += gs[i] * xh;
                grad_beta[i] += gs[i];
            }

            let mean_gp: f32 = g_prime.iter().sum::<f32>() / n;
            let mean_gp_xhat: f32 = g_prime
                .iter()
                .zip(x_hat.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                / n;

            for i in 0..norm_dim {
                grad_x[off + i] = std_inv * (g_prime[i] - mean_gp - x_hat[i] * mean_gp_xhat);
            }
        }

        vec![
            Tensor::new(&grad_x, self.x.shape()),
            Tensor::new(&grad_gamma, self.gamma.shape()),
            Tensor::new(&grad_beta, self.gamma.shape()),
        ]
    }

    fn name(&self) -> &'static str {
        "LayerNormBackward"
    }
}

/// Gradient function for affine RMSNorm: y = x_hat * gamma,
/// where x_hat = x / rms and rms = sqrt(mean(x^2) + eps) per normalized row.
///
/// Obligation: OBLIG-RMSNORM-BACKWARD-GRAD-FLOW.
///
/// Flows gradient to BOTH inputs (x, gamma), in that input order. RMSNorm has
/// no mean-subtraction term (unlike LayerNorm).
///
/// Math (per row, with r = rms, x_hat_i = x_i / r):
/// - dL/dgamma_i = sum_batch g_i * x_hat_i
/// - dL/dx_j     = (1/r) * (g'_j - x_hat_j * mean_i(g'_i * x_hat_i))
///   where g'_i = g_i * gamma_i.
pub(crate) struct RmsNormBackward {
    pub(crate) x: Tensor,
    pub(crate) gamma: Tensor,
    pub(crate) eps: f32,
}

impl GradFn for RmsNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let norm_dim = self.gamma.data().len();
        let x_data = self.x.data();
        let gamma_data = self.gamma.data();
        let g_data = grad_output.data();
        let batch = x_data.len() / norm_dim;
        let n = norm_dim as f32;

        let mut grad_x = vec![0.0f32; x_data.len()];
        let mut grad_gamma = vec![0.0f32; norm_dim];

        for b in 0..batch {
            let off = b * norm_dim;
            let xs = &x_data[off..off + norm_dim];
            let gs = &g_data[off..off + norm_dim];

            let ms: f32 = xs.iter().map(|&v| v * v).sum::<f32>() / n;
            let r = (ms + self.eps).sqrt();
            let r_inv = 1.0 / r;

            let mut x_hat = vec![0.0f32; norm_dim];
            let mut g_prime = vec![0.0f32; norm_dim];
            for i in 0..norm_dim {
                let xh = xs[i] * r_inv;
                x_hat[i] = xh;
                g_prime[i] = gs[i] * gamma_data[i];
                grad_gamma[i] += gs[i] * xh;
            }

            let mean_gp_xhat: f32 = g_prime
                .iter()
                .zip(x_hat.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                / n;

            for i in 0..norm_dim {
                grad_x[off + i] = r_inv * (g_prime[i] - x_hat[i] * mean_gp_xhat);
            }
        }

        vec![
            Tensor::new(&grad_x, self.x.shape()),
            Tensor::new(&grad_gamma, self.gamma.shape()),
        ]
    }

    fn name(&self) -> &'static str {
        "RmsNormBackward"
    }
}

include!("gradient.rs");
include!("grad_fn_tests.rs");
