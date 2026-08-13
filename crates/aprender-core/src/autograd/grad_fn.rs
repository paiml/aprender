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

/// Gradient function for affine BatchNorm1d in TRAIN mode: y = x_hat * gamma + beta,
/// where x_hat = (x - mu_batch) / sqrt(var_batch + eps), with mu/var computed over the
/// BATCH (and spatial) dimension per feature using the BIASED (÷N) variance — matching
/// the forward used for normalization (Ioffe & Szegedy).
///
/// Obligation: OBLIG-BATCHNORM1D-BACKWARD-GRAD-FLOW.
///
/// Before PMAT-911 the `BatchNorm1d::forward` built its output via `Tensor::new`,
/// severing the autograd graph: gamma/beta/x received no gradient, making every
/// BatchNorm layer non-fine-tunable. This grad_fn flows gradient to ALL THREE inputs
/// (x, gamma, beta), in that input order.
///
/// Layout: input is `[N, C]` (2D) or `[N, C, L]` (3D); feature axis is dim 1. For
/// each feature `j`, the reduction set is every (n, l) index with channel `j`
/// (size `m = N * L`). With std_inv = 1/sqrt(var+eps):
/// - dL/dgamma_j = sum_set g * x_hat
/// - dL/dbeta_j  = sum_set g
/// - dL/dx_k     = (gamma_j * std_inv / m) * (m*g_k - sum_set(g) - x_hat_k * sum_set(g*x_hat))
///   (the standard batchnorm-backward; mean over the BATCH set, not the feature dim).
pub(crate) struct BatchNorm1dBackward {
    pub(crate) x: Tensor,
    pub(crate) gamma: Tensor,
    pub(crate) eps: f32,
}

impl BatchNorm1dBackward {
    /// Indices for one feature across batch (and spatial) dims — mirrors
    /// `BatchNorm1d::feature_indices`.
    fn feature_indices(shape: &[usize], feature: usize) -> Vec<usize> {
        let (batch_size, features) = (shape[0], shape[1]);
        if shape.len() == 2 {
            (0..batch_size).map(|b| b * features + feature).collect()
        } else {
            let length = shape[2];
            let mut indices = Vec::with_capacity(batch_size * length);
            for b in 0..batch_size {
                for l in 0..length {
                    indices.push(b * features * length + feature * length + l);
                }
            }
            indices
        }
    }
}

impl GradFn for BatchNorm1dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let shape = self.x.shape();
        let features = shape[1];
        let x_data = self.x.data();
        let gamma_data = self.gamma.data();
        let g_data = grad_output.data();

        let mut grad_x = vec![0.0f32; x_data.len()];
        let mut grad_gamma = vec![0.0f32; features];
        let mut grad_beta = vec![0.0f32; features];

        for f in 0..features {
            let indices = Self::feature_indices(shape, f);
            let m = indices.len() as f32;

            // Batch statistics (train mode), biased variance (÷N) as in forward.
            let mean: f32 = indices.iter().map(|&i| x_data[i]).sum::<f32>() / m;
            let var: f32 = indices
                .iter()
                .map(|&i| (x_data[i] - mean).powi(2))
                .sum::<f32>()
                / m;
            let std_inv = 1.0 / (var + self.eps).sqrt();

            // x_hat and the two reductions over the batch set.
            let mut sum_g = 0.0f32;
            let mut sum_g_xhat = 0.0f32;
            for &i in &indices {
                let xh = (x_data[i] - mean) * std_inv;
                let g = g_data[i];
                sum_g += g;
                sum_g_xhat += g * xh;
                grad_gamma[f] += g * xh;
                grad_beta[f] += g;
            }

            let scale = gamma_data[f] * std_inv / m;
            for &i in &indices {
                let xh = (x_data[i] - mean) * std_inv;
                grad_x[i] = scale * (m * g_data[i] - sum_g - xh * sum_g_xhat);
            }
        }

        vec![
            Tensor::new(&grad_x, self.x.shape()),
            Tensor::new(&grad_gamma, self.gamma.shape()),
            Tensor::new(&grad_beta, self.gamma.shape()),
        ]
    }

    fn name(&self) -> &'static str {
        "BatchNorm1dBackward"
    }
}

/// Gradient function for affine GroupNorm: y = x_hat * gamma_c + beta_c,
/// where x_hat normalizes within each (sample, group) over the
/// `channels_per_group * spatial` elements (biased ÷group_size variance).
///
/// Obligation: OBLIG-GROUPNORM-BACKWARD-GRAD-FLOW.
///
/// Before PMAT-911 `GroupNorm::forward` built its output via `Tensor::new`,
/// severing the autograd graph. This grad_fn flows gradient to ALL THREE inputs
/// (x, gamma, beta), in that input order. gamma/beta are PER-CHANNEL.
///
/// Math (per (n, group), group of size `gs`, std_inv = 1/sqrt(var+eps)):
/// - dL/dgamma_c = sum_{n, spatial} g * x_hat   (accumulated over the channel c)
/// - dL/dbeta_c  = sum_{n, spatial} g
/// - dL/dx_k     = std_inv * (g'_k - mean_grp(g') - x_hat_k * mean_grp(g'*x_hat))
///   where g'_k = g_k * gamma_{c(k)}, and the means are over the whole group
///   (channels_per_group * spatial), like LayerNorm but within each group.
pub(crate) struct GroupNormBackward {
    pub(crate) x: Tensor,
    pub(crate) gamma: Tensor,
    pub(crate) num_groups: usize,
    pub(crate) eps: f32,
}

impl GradFn for GroupNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let shape = self.x.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let channels_per_group = channels / self.num_groups;
        let spatial_size: usize = shape[2..].iter().product();
        let group_size = channels_per_group * spatial_size;
        let gs = group_size as f32;

        let x_data = self.x.data();
        let gamma_data = self.gamma.data();
        let g_data = grad_output.data();

        let mut grad_x = vec![0.0f32; x_data.len()];
        let mut grad_gamma = vec![0.0f32; channels];
        let mut grad_beta = vec![0.0f32; channels];

        for n in 0..batch_size {
            for grp in 0..self.num_groups {
                // Group mean / biased variance.
                let mut sum = 0.0f32;
                for c in 0..channels_per_group {
                    let channel_idx = grp * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = n * channels * spatial_size + channel_idx * spatial_size + s;
                        sum += x_data[idx];
                    }
                }
                let mean = sum / gs;
                let mut var_sum = 0.0f32;
                for c in 0..channels_per_group {
                    let channel_idx = grp * channels_per_group + c;
                    for s in 0..spatial_size {
                        let idx = n * channels * spatial_size + channel_idx * spatial_size + s;
                        var_sum += (x_data[idx] - mean).powi(2);
                    }
                }
                let var = var_sum / gs;
                let std_inv = 1.0 / (var + self.eps).sqrt();

                // First pass: x_hat, g' = g*gamma, group reductions, affine grads.
                let mut sum_gp = 0.0f32;
                let mut sum_gp_xhat = 0.0f32;
                for c in 0..channels_per_group {
                    let channel_idx = grp * channels_per_group + c;
                    let gamma_c = gamma_data[channel_idx];
                    for s in 0..spatial_size {
                        let idx = n * channels * spatial_size + channel_idx * spatial_size + s;
                        let xh = (x_data[idx] - mean) * std_inv;
                        let g = g_data[idx];
                        let gp = g * gamma_c;
                        sum_gp += gp;
                        sum_gp_xhat += gp * xh;
                        grad_gamma[channel_idx] += g * xh;
                        grad_beta[channel_idx] += g;
                    }
                }
                let mean_gp = sum_gp / gs;
                let mean_gp_xhat = sum_gp_xhat / gs;

                // Second pass: dx within the group.
                for c in 0..channels_per_group {
                    let channel_idx = grp * channels_per_group + c;
                    let gamma_c = gamma_data[channel_idx];
                    for s in 0..spatial_size {
                        let idx = n * channels * spatial_size + channel_idx * spatial_size + s;
                        let xh = (x_data[idx] - mean) * std_inv;
                        let gp = g_data[idx] * gamma_c;
                        grad_x[idx] = std_inv * (gp - mean_gp - xh * mean_gp_xhat);
                    }
                }
            }
        }

        vec![
            Tensor::new(&grad_x, self.x.shape()),
            Tensor::new(&grad_gamma, self.gamma.shape()),
            Tensor::new(&grad_beta, self.gamma.shape()),
        ]
    }

    fn name(&self) -> &'static str {
        "GroupNormBackward"
    }
}

// ============================================================================
// Shape / Pooling / Embedding Operations (PMAT-913)
//
// Before PMAT-913 the canonical nn Flatten / MaxPool1d / MaxPool2d / AvgPool2d /
// GlobalAvgPool2d forwards built their output via `Tensor::new`, severing the
// autograd graph: `input.grad` was `None` after `backward()`, so any model with
// a pooling/flatten layer in the middle could not propagate gradient to the
// upstream conv/linear weights. Likewise the token `Embedding` forward built its
// lookup output via `Tensor::new`, so the embedding TABLE never received
// gradient (non-trainable token embeddings). These grad_fns wire the missing
// backward edges.
// ============================================================================

/// Gradient function for `Flatten` (and any pure reshape view).
///
/// Obligation: OBLIG-FLATTEN-BACKWARD-GRAD-FLOW.
///
/// Flatten is a pure view: it only changes the shape, not the values. The
/// backward is therefore the identity on values, reshaped back to the input
/// shape: `dL/dx = reshape(grad_output, input_shape)`.
pub(crate) struct FlattenBackward {
    pub(crate) input_shape: Vec<usize>,
}

impl GradFn for FlattenBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Pure reshape: same data, original shape.
        vec![Tensor::new(grad_output.data(), &self.input_shape)]
    }

    fn name(&self) -> &'static str {
        "FlattenBackward"
    }
}

/// Gradient function for `MaxPool1d`: route grad_output to the argmax position
/// within each pooling window (subgradient of max). Ties go to the FIRST max
/// (matching the forward `max()` left-to-right scan).
///
/// Obligation: OBLIG-MAXPOOL1D-BACKWARD-GRAD-FLOW.
///
/// Input `[N, C, L]`, output `[N, C, L_out]`, `L_out = (L - k)/s + 1`.
pub(crate) struct MaxPool1dBackward {
    pub(crate) input: Tensor,
    pub(crate) kernel_size: usize,
    pub(crate) stride: usize,
}

impl GradFn for MaxPool1dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let shape = self.input.shape();
        let (batch, channels, in_len) = (shape[0], shape[1], shape[2]);
        let out_len = (in_len - self.kernel_size) / self.stride + 1;
        let x = self.input.data();
        let g = grad_output.data();
        let mut grad_x = vec![0.0f32; x.len()];

        for n in 0..batch {
            for c in 0..channels {
                let base = n * channels * in_len + c * in_len;
                for ol in 0..out_len {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut argmax = 0usize;
                    for k in 0..self.kernel_size {
                        let il = ol * self.stride + k;
                        let v = x[base + il];
                        if v > max_val {
                            max_val = v;
                            argmax = il;
                        }
                    }
                    let out_idx = n * channels * out_len + c * out_len + ol;
                    grad_x[base + argmax] += g[out_idx];
                }
            }
        }

        vec![Tensor::new(&grad_x, shape)]
    }

    fn name(&self) -> &'static str {
        "MaxPool1dBackward"
    }
}

/// Gradient function for `MaxPool2d`: route grad_output to the argmax position
/// within each 2D window. Ties go to the FIRST max in row-major (kh, kw) scan.
///
/// Obligation: OBLIG-MAXPOOL2D-BACKWARD-GRAD-FLOW.
///
/// Input `[N, C, H, W]`, output `[N, C, H_out, W_out]`.
pub(crate) struct MaxPool2dBackward {
    pub(crate) input: Tensor,
    pub(crate) kernel_h: usize,
    pub(crate) kernel_w: usize,
    pub(crate) stride_h: usize,
    pub(crate) stride_w: usize,
}

impl GradFn for MaxPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let shape = self.input.shape();
        let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_h = (in_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w - self.kernel_w) / self.stride_w + 1;
        let x = self.input.data();
        let g = grad_output.data();
        let mut grad_x = vec![0.0f32; x.len()];

        for n in 0..batch {
            for c in 0..channels {
                let plane = n * channels * in_h * in_w + c * in_h * in_w;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut argmax = 0usize;
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = oh * self.stride_h + kh;
                                let iw = ow * self.stride_w + kw;
                                let idx = plane + ih * in_w + iw;
                                if x[idx] > max_val {
                                    max_val = x[idx];
                                    argmax = idx;
                                }
                            }
                        }
                        let out_idx =
                            n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        grad_x[argmax] += g[out_idx];
                    }
                }
            }
        }

        vec![Tensor::new(&grad_x, shape)]
    }

    fn name(&self) -> &'static str {
        "MaxPool2dBackward"
    }
}

/// Gradient function for `AvgPool2d`: distribute `grad_output / kernel_area`
/// evenly to every input element in the pooling window.
///
/// Obligation: OBLIG-AVGPOOL2D-BACKWARD-GRAD-FLOW.
pub(crate) struct AvgPool2dBackward {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) kernel_h: usize,
    pub(crate) kernel_w: usize,
    pub(crate) stride_h: usize,
    pub(crate) stride_w: usize,
}

impl GradFn for AvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (batch, channels, in_h, in_w) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        );
        let out_h = (in_h - self.kernel_h) / self.stride_h + 1;
        let out_w = (in_w - self.kernel_w) / self.stride_w + 1;
        let g = grad_output.data();
        let area = (self.kernel_h * self.kernel_w) as f32;
        let mut grad_x = vec![0.0f32; batch * channels * in_h * in_w];

        for n in 0..batch {
            for c in 0..channels {
                let plane = n * channels * in_h * in_w + c * in_h * in_w;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let out_idx =
                            n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        let share = g[out_idx] / area;
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let ih = oh * self.stride_h + kh;
                                let iw = ow * self.stride_w + kw;
                                grad_x[plane + ih * in_w + iw] += share;
                            }
                        }
                    }
                }
            }
        }

        vec![Tensor::new(&grad_x, &self.input_shape)]
    }

    fn name(&self) -> &'static str {
        "AvgPool2dBackward"
    }
}

/// Gradient function for `GlobalAvgPool2d`: each output `[n, c]` is the mean of
/// the whole `H x W` plane, so every input element in that plane receives
/// `grad_output[n, c] / (H * W)`.
///
/// Obligation: OBLIG-GLOBALAVGPOOL2D-BACKWARD-GRAD-FLOW.
pub(crate) struct GlobalAvgPool2dBackward {
    pub(crate) input_shape: Vec<usize>,
}

impl GradFn for GlobalAvgPool2dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (batch, channels, in_h, in_w) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        );
        let spatial = (in_h * in_w) as f32;
        let g = grad_output.data();
        let mut grad_x = vec![0.0f32; batch * channels * in_h * in_w];

        for n in 0..batch {
            for c in 0..channels {
                let share = g[n * channels + c] / spatial;
                let plane = n * channels * in_h * in_w + c * in_h * in_w;
                for k in 0..(in_h * in_w) {
                    grad_x[plane + k] = share;
                }
            }
        }

        vec![Tensor::new(&grad_x, &self.input_shape)]
    }

    fn name(&self) -> &'static str {
        "GlobalAvgPool2dBackward"
    }
}

/// Gradient function for the token `Embedding` lookup table.
///
/// Obligation: OBLIG-EMBEDDING-BACKWARD-GRAD-FLOW.
///
/// Forward gathers row `idx[i]` of the `[vocab, hidden]` weight into output row
/// `i`. The backward SCATTER-ADDs each upstream gradient row back into the
/// corresponding weight row: `dW[idx[i]] += grad_output[i]`. Because multiple
/// positions can reference the SAME token id, the accumulation MUST be additive
/// (not an overwrite) — otherwise repeated tokens would lose gradient. The token
/// indices themselves are integers and carry no gradient.
pub(crate) struct EmbeddingBackward {
    pub(crate) indices: Vec<u32>,
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
}

impl GradFn for EmbeddingBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let g = grad_output.data();
        let h = self.hidden_size;
        let mut grad_w = vec![0.0f32; self.vocab_size * h];

        for (i, &tok) in self.indices.iter().enumerate() {
            let row = tok as usize;
            if row >= self.vocab_size {
                continue; // OOB token contributes no gradient (matches forward N-09 escape).
            }
            let g_off = i * h;
            let w_off = row * h;
            for j in 0..h {
                grad_w[w_off + j] += g[g_off + j];
            }
        }

        vec![Tensor::new(&grad_w, &[self.vocab_size, h])]
    }

    fn name(&self) -> &'static str {
        "EmbeddingBackward"
    }
}

// ============================================================================
// Attention Operations (PMAT-914) — OBLIG-ATTENTION-BACKWARD-GRAD-FLOW
//
// The scaled-dot-product attention core built its intermediates via
// `Tensor::from_vec` / `Tensor::new`, severing the autograd graph so the
// Q/K/V/out projection weights received no gradient (transformer attention
// non-fine-tunable). These grad_fns flow gradient through each helper so the
// chain loss -> out_proj -> sdpa -> reshape -> {q,k,v}_proj stays unbroken.
// ============================================================================

/// Backward for softmax over the LAST dimension of an N-D tensor.
///
/// Treats the tensor as (rows = product of all leading dims) × (features = last
/// dim). For each row: dL/dx = y * (g - <g, y>), the standard softmax Jacobian.
pub(crate) struct SoftmaxLastDimBackward {
    pub(crate) output: Tensor, // softmax output y (same shape as input)
}

impl GradFn for SoftmaxLastDimBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let shape = self.output.shape();
        let ndim = shape.len();
        let features = if ndim == 0 { 1 } else { shape[ndim - 1] };
        let total = self.output.numel();
        let rows = total.checked_div(features).unwrap_or(0);

        let out_data = self.output.data();
        let grad_data = grad_output.data();
        let mut grad_input = vec![0.0; total];

        for r in 0..rows {
            let base = r * features;
            let mut dot = 0.0;
            for j in 0..features {
                dot += grad_data[base + j] * out_data[base + j];
            }
            for j in 0..features {
                let idx = base + j;
                grad_input[idx] = out_data[idx] * (grad_data[idx] - dot);
            }
        }

        vec![Tensor::new(&grad_input, shape)]
    }

    fn name(&self) -> &'static str {
        "SoftmaxLastDimBackward"
    }
}

/// Backward for `transpose_last_two`: the operation is its own inverse, so the
/// incoming gradient is transposed back along the last two dims.
pub(crate) struct TransposeLastTwoBackward {
    pub(crate) input_shape: Vec<usize>, // shape of the ORIGINAL (pre-transpose) input
}

impl GradFn for TransposeLastTwoBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let g = transpose_last_two_raw(grad_output);
        // g now has the original input's shape.
        vec![Tensor::new(g.data(), &self.input_shape)]
    }

    fn name(&self) -> &'static str {
        "TransposeLastTwoBackward"
    }
}

/// Backward for 4D batched matmul C = A @ B where
/// A:[B,H,M,K], B:[B,H,K,N], C:[B,H,M,N].
/// dA = grad @ B^T  (per batch,head);  dB = A^T @ grad  (per batch,head).
pub(crate) struct BatchedMatmul4dBackward {
    pub(crate) a: Tensor,
    pub(crate) b: Tensor,
}

impl GradFn for BatchedMatmul4dBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let a_shape = self.a.shape();
        let b_shape = self.b.shape();
        let (batch, heads, m, k) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
        let n = b_shape[3];

        let a = self.a.data();
        let b = self.b.data();
        let g = grad_output.data();

        let mut grad_a = vec![0.0f32; batch * heads * m * k];
        let mut grad_b = vec![0.0f32; batch * heads * k * n];

        for bh in 0..(batch * heads) {
            let a_off = bh * m * k;
            let b_off = bh * k * n;
            let g_off = bh * m * n;

            // dA[m,k] = sum_n grad[m,n] * B[k,n]
            for i in 0..m {
                for kk in 0..k {
                    let mut acc = 0.0;
                    for j in 0..n {
                        acc += g[g_off + i * n + j] * b[b_off + kk * n + j];
                    }
                    grad_a[a_off + i * k + kk] = acc;
                }
            }

            // dB[k,n] = sum_m A[m,k] * grad[m,n]
            for kk in 0..k {
                for j in 0..n {
                    let mut acc = 0.0;
                    for i in 0..m {
                        acc += a[a_off + i * k + kk] * g[g_off + i * n + j];
                    }
                    grad_b[b_off + kk * n + j] = acc;
                }
            }
        }

        vec![Tensor::new(&grad_a, a_shape), Tensor::new(&grad_b, b_shape)]
    }

    fn name(&self) -> &'static str {
        "BatchedMatmul4dBackward"
    }
}

/// Backward for `reshape_for_attention`: [b,s,embed] -> [b,heads,s,head_dim].
/// The inverse permutation is exactly `reshape_from_attention`.
pub(crate) struct ReshapeForAttentionBackward {
    pub(crate) batch: usize,
    pub(crate) seq_len: usize,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
}

impl GradFn for ReshapeForAttentionBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad_output: [b, heads, s, head_dim]; produce [b, s, embed].
        let embed = self.num_heads * self.head_dim;
        let g = grad_output.data();
        let mut out = vec![0.0f32; self.batch * self.seq_len * embed];
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        let in_idx = b * self.num_heads * self.seq_len * self.head_dim
                            + h * self.seq_len * self.head_dim
                            + s * self.head_dim
                            + d;
                        let out_idx = b * self.seq_len * embed + s * embed + h * self.head_dim + d;
                        out[out_idx] = g[in_idx];
                    }
                }
            }
        }
        vec![Tensor::new(&out, &[self.batch, self.seq_len, embed])]
    }

    fn name(&self) -> &'static str {
        "ReshapeForAttentionBackward"
    }
}

/// Backward for `reshape_from_attention`: [b,heads,s,head_dim] -> [b,s,embed].
/// The inverse permutation is exactly `reshape_for_attention`.
pub(crate) struct ReshapeFromAttentionBackward {
    pub(crate) batch: usize,
    pub(crate) seq_len: usize,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
}

impl GradFn for ReshapeFromAttentionBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // grad_output: [b, s, embed]; produce [b, heads, s, head_dim].
        let embed = self.num_heads * self.head_dim;
        let g = grad_output.data();
        let mut out = vec![0.0f32; self.batch * self.num_heads * self.seq_len * self.head_dim];
        for b in 0..self.batch {
            for s in 0..self.seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        let out_idx = b * self.num_heads * self.seq_len * self.head_dim
                            + h * self.seq_len * self.head_dim
                            + s * self.head_dim
                            + d;
                        let in_idx = b * self.seq_len * embed + s * embed + h * self.head_dim + d;
                        out[out_idx] = g[in_idx];
                    }
                }
            }
        }
        vec![Tensor::new(
            &out,
            &[self.batch, self.num_heads, self.seq_len, self.head_dim],
        )]
    }

    fn name(&self) -> &'static str {
        "ReshapeFromAttentionBackward"
    }
}

/// Pure (non-autograd) transpose of the last two dims of an N-D tensor.
/// Shared by the forward helper and `TransposeLastTwoBackward`.
pub(crate) fn transpose_last_two_raw(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return Tensor::new(x.data(), shape);
    }
    let last = shape[ndim - 1];
    let second_last = shape[ndim - 2];
    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = last;
    new_shape[ndim - 1] = second_last;

    let batch_size: usize = shape[..ndim - 2].iter().product();
    let matrix_size = last * second_last;
    let src = x.data();
    let mut output = vec![0.0; src.len()];
    for b in 0..batch_size {
        let offset = b * matrix_size;
        for i in 0..second_last {
            let src_base = offset + i * last;
            for j in 0..last {
                output[offset + j * second_last + i] = src[src_base + j];
            }
        }
    }
    Tensor::from_vec(output, &new_shape)
}

include!("gradient.rs");
include!("grad_fn_tests.rs");
