# Deep Learning Specification: PyTorch-Compatible Pure Rust Implementation

**Version:** 1.0.0
**Status:** Draft
**Authors:** Pragmatic AI Labs
**Date:** 2025-11-25

## Executive Summary

This specification defines the architecture and implementation plan for PyTorch-compatible deep learning capabilities in aprender, using **pure Rust only** (zero C/C++ dependencies, no FFI). The implementation leverages the paiml sovereign AI stack (trueno, alimentar, trueno-db) to provide a fully independent, FLOSS alternative to PyTorch.

### Design Constraints

| Constraint | Rationale |
|------------|-----------|
| Pure Rust | Sovereignty, WASM compatibility, memory safety |
| No C/FFI | Auditability, no hidden dependencies |
| trueno backend | SIMD/GPU acceleration via sovereign stack |
| FLOSS (MIT) | Maximum adoption, no vendor lock-in |

## Scientific Foundation

This specification is grounded in peer-reviewed research on automatic differentiation, neural network architectures, and optimization theory.

### Citations

1. **Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018).** Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18(153), 1-43.
   - *Foundation for reverse-mode automatic differentiation (backpropagation)*

2. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
   - *Original backpropagation algorithm*

3. **Kingma, D. P., & Ba, J. (2015).** Adam: A method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.
   - *Adam optimizer implementation*

4. **Ioffe, S., & Szegedy, C. (2015).** Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448-456.
   - *BatchNorm layer specification*

5. **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 1026-1034.
   - *Kaiming initialization, PReLU activation*

6. **Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 249-256.
   - *Xavier/Glorot initialization*

7. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.
   - *Dropout regularization*

8. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998-6008.
   - *Transformer architecture, multi-head attention*

9. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 8026-8037.
   - *Reference architecture for API design*

10. **Griewank, A., & Walther, A. (2008).** Evaluating derivatives: Principles and techniques of algorithmic differentiation (2nd ed.). *Society for Industrial and Applied Mathematics (SIAM)*.
    - *Theoretical foundation for computational graph implementation*

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      aprender Deep Learning Stack                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         User API Layer                            │ │
│  │  • nn::Module trait                                               │ │
│  │  • Sequential, ModuleList containers                              │ │
│  │  • Functional API (F::relu, F::softmax)                           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                       Autograd Engine                             │ │
│  │  • Tensor with gradient tracking                                  │ │
│  │  • Computational graph (tape-based)                               │ │
│  │  • Backward pass (reverse-mode AD)                                │ │
│  │  • Gradient accumulation                                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      Differentiable Ops                           │ │
│  │  • Element-wise: add, mul, div, neg, exp, log, pow                │ │
│  │  • Reductions: sum, mean, max, min                                │ │
│  │  • Linear algebra: matmul, transpose, reshape                     │ │
│  │  • Activations: relu, sigmoid, tanh, softmax, gelu                │ │
│  │  • Loss functions: cross_entropy, mse, nll                        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    trueno Compute Backend                         │ │
│  │  • SIMD: AVX-512, AVX2, SSE2, NEON, WASM SIMD128                  │ │
│  │  • GPU: wgpu (Vulkan/Metal/DX12/WebGPU)                           │ │
│  │  • Zero-copy Arrow interop (alimentar)                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module 1: Autograd Engine

### 1.1 Design Philosophy

Following Baydin et al. (2018), we implement **reverse-mode automatic differentiation** (backpropagation) using a tape-based computational graph. This approach:

- Records operations during forward pass
- Computes gradients in reverse order during backward pass
- Supports dynamic computational graphs (define-by-run, like PyTorch)

### 1.2 Core Types

```rust
/// Tensor with optional gradient tracking.
///
/// Implements reverse-mode automatic differentiation per Griewank & Walther (2008).
pub struct Tensor {
    /// Underlying data storage (backed by trueno)
    data: trueno::Tensor<f32>,

    /// Gradient (populated after backward())
    grad: Option<Box<Tensor>>,

    /// Whether this tensor requires gradient computation
    requires_grad: bool,

    /// Reference to the operation that created this tensor
    grad_fn: Option<Arc<dyn GradFn>>,

    /// Unique identifier for graph construction
    id: TensorId,
}

/// Function that computes gradients during backward pass.
///
/// Each differentiable operation implements this trait.
pub trait GradFn: Send + Sync {
    /// Compute gradients with respect to inputs.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient flowing back from downstream operations
    ///
    /// # Returns
    /// Vector of gradients for each input tensor
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;

    /// Input tensors that require gradients
    fn inputs(&self) -> Vec<TensorId>;
}

/// Computational graph for automatic differentiation.
///
/// Implements tape-based recording per Paszke et al. (2019).
pub struct ComputationGraph {
    /// Recorded operations in topological order
    tape: Vec<TapeEntry>,

    /// Tensor storage for gradient accumulation
    tensors: HashMap<TensorId, Tensor>,
}

struct TapeEntry {
    output_id: TensorId,
    grad_fn: Arc<dyn GradFn>,
    input_ids: Vec<TensorId>,
}
```

### 1.3 Tensor API

```rust
impl Tensor {
    /// Create tensor from data (no gradient tracking by default)
    pub fn new(data: &[f32], shape: &[usize]) -> Self;

    /// Create tensor with gradient tracking enabled
    pub fn from_data(data: &[f32], shape: &[usize]) -> Self;

    /// Enable gradient tracking (in-place)
    pub fn requires_grad_(&mut self, requires: bool) -> &mut Self;

    /// Check if gradient tracking is enabled
    pub fn requires_grad(&self) -> bool;

    /// Get gradient (None if backward() not called or no gradient required)
    pub fn grad(&self) -> Option<&Tensor>;

    /// Zero out gradient (for optimizer step)
    pub fn zero_grad_(&mut self);

    /// Compute gradients via backpropagation
    ///
    /// Implements Algorithm 1 from Rumelhart et al. (1986)
    pub fn backward(&self);

    /// Detach from computation graph (no gradient flow)
    pub fn detach(&self) -> Tensor;

    /// Execute operations without gradient tracking
    pub fn no_grad<F, R>(f: F) -> R
    where
        F: FnOnce() -> R;
}
```

### 1.4 Differentiable Operations

Each operation records itself to the tape and implements backward:

```rust
// Element-wise operations (Griewank & Walther, 2008, Chapter 3)
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor;      // ∂/∂x = 1, ∂/∂y = 1
    pub fn sub(&self, other: &Tensor) -> Tensor;      // ∂/∂x = 1, ∂/∂y = -1
    pub fn mul(&self, other: &Tensor) -> Tensor;      // ∂/∂x = y, ∂/∂y = x
    pub fn div(&self, other: &Tensor) -> Tensor;      // ∂/∂x = 1/y, ∂/∂y = -x/y²
    pub fn neg(&self) -> Tensor;                       // ∂/∂x = -1
    pub fn exp(&self) -> Tensor;                       // ∂/∂x = exp(x)
    pub fn log(&self) -> Tensor;                       // ∂/∂x = 1/x
    pub fn pow(&self, n: f32) -> Tensor;               // ∂/∂x = n * x^(n-1)
    pub fn sqrt(&self) -> Tensor;                      // ∂/∂x = 0.5 / sqrt(x)
}

// Reduction operations
impl Tensor {
    pub fn sum(&self) -> Tensor;                       // ∂/∂x = 1 (broadcast)
    pub fn mean(&self) -> Tensor;                      // ∂/∂x = 1/n (broadcast)
    pub fn sum_dim(&self, dim: usize) -> Tensor;
    pub fn mean_dim(&self, dim: usize) -> Tensor;
}

// Linear algebra (trueno backend)
impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor;   // ∂/∂A = grad @ B.T, ∂/∂B = A.T @ grad
    pub fn transpose(&self) -> Tensor;
    pub fn reshape(&self, shape: &[usize]) -> Tensor;
    pub fn view(&self, shape: &[usize]) -> Tensor;
}

// Activation functions (He et al., 2015; Glorot & Bengio, 2010)
impl Tensor {
    pub fn relu(&self) -> Tensor;                      // ∂/∂x = 1 if x > 0 else 0
    pub fn sigmoid(&self) -> Tensor;                   // ∂/∂x = σ(x)(1 - σ(x))
    pub fn tanh(&self) -> Tensor;                      // ∂/∂x = 1 - tanh²(x)
    pub fn softmax(&self, dim: usize) -> Tensor;       // Jacobian computation
    pub fn gelu(&self) -> Tensor;                      // Gaussian Error Linear Unit
    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor;
}
```

### 1.5 Backward Pass Algorithm

```rust
impl Tensor {
    /// Backpropagation algorithm (Rumelhart et al., 1986)
    ///
    /// Complexity: O(n) where n is number of operations in graph
    pub fn backward(&self) {
        // 1. Topological sort of computation graph
        let sorted_ops = self.topological_sort();

        // 2. Initialize output gradient to 1.0
        let mut grads: HashMap<TensorId, Tensor> = HashMap::new();
        grads.insert(self.id, Tensor::ones_like(self));

        // 3. Reverse iteration (reverse-mode AD)
        for op in sorted_ops.iter().rev() {
            if let Some(grad_fn) = &op.grad_fn {
                let grad_output = grads.get(&op.output_id).unwrap();
                let input_grads = grad_fn.backward(grad_output);

                // 4. Accumulate gradients (for tensors used multiple times)
                for (input_id, input_grad) in op.input_ids.iter().zip(input_grads) {
                    grads.entry(*input_id)
                        .and_modify(|g| *g = g.add(&input_grad))
                        .or_insert(input_grad);
                }
            }
        }

        // 5. Store gradients in leaf tensors
        for (id, grad) in grads {
            if let Some(tensor) = self.get_tensor_mut(id) {
                if tensor.requires_grad && tensor.is_leaf() {
                    tensor.grad = Some(Box::new(grad));
                }
            }
        }
    }
}
```

## Module 2: Neural Network Layers (nn)

### 2.1 Module Trait

```rust
/// Base trait for all neural network modules.
///
/// Follows PyTorch's Module design (Paszke et al., 2019).
pub trait Module: Send + Sync {
    /// Forward pass computation
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Iterator over learnable parameters
    fn parameters(&self) -> Vec<&Tensor>;

    /// Mutable iterator for optimizer updates
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Set training mode (affects Dropout, BatchNorm)
    fn train(&mut self);

    /// Set evaluation mode
    fn eval(&mut self);

    /// Check if in training mode
    fn training(&self) -> bool;
}

/// Automatic Module implementation via derive macro
///
/// ```rust
/// #[derive(Module)]
/// struct MyModel {
///     linear1: Linear,
///     linear2: Linear,
/// }
/// ```
pub use aprender_derive::Module;
```

### 2.2 Linear Layer

```rust
/// Fully connected layer: y = xW^T + b
///
/// Weight initialization follows Glorot & Bengio (2010).
pub struct Linear {
    weight: Tensor,  // Shape: [out_features, in_features]
    bias: Option<Tensor>,  // Shape: [out_features]
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create new linear layer with Xavier initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(&[out_features, in_features]) * std;
        let bias = Tensor::zeros(&[out_features]);

        Self {
            weight: weight.requires_grad_(true),
            bias: Some(bias.requires_grad_(true)),
            in_features,
            out_features,
        }
    }

    /// Create without bias
    pub fn without_bias(in_features: usize, out_features: usize) -> Self;

    /// Kaiming initialization for ReLU networks (He et al., 2015)
    pub fn with_kaiming_init(in_features: usize, out_features: usize) -> Self {
        let std = (2.0 / in_features as f32).sqrt();
        // ...
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let output = input.matmul(&self.weight.transpose());
        match &self.bias {
            Some(b) => output.add(b),
            None => output,
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        match &self.bias {
            Some(b) => vec![&self.weight, b],
            None => vec![&self.weight],
        }
    }
}
```

### 2.3 Convolutional Layers

```rust
/// 2D Convolution layer
///
/// Implements cross-correlation (standard in deep learning frameworks).
pub struct Conv2d {
    weight: Tensor,  // Shape: [out_channels, in_channels, kernel_h, kernel_w]
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Self;

    pub fn with_stride(self, stride: (usize, usize)) -> Self;
    pub fn with_padding(self, padding: (usize, usize)) -> Self;
    pub fn with_dilation(self, dilation: (usize, usize)) -> Self;
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        // im2col + matmul implementation (efficient on trueno backend)
        conv2d_forward(input, &self.weight, self.bias.as_ref(),
                       self.stride, self.padding, self.dilation)
    }
}

/// 1D Convolution (for sequences/time series)
pub struct Conv1d { /* similar structure */ }

/// Transposed convolution (upsampling)
pub struct ConvTranspose2d { /* similar structure */ }
```

### 2.4 Normalization Layers

```rust
/// Batch Normalization (Ioffe & Szegedy, 2015)
///
/// Normalizes inputs across batch dimension during training.
/// Uses running statistics during inference.
pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    affine: bool,

    // Learnable parameters (if affine=true)
    gamma: Option<Tensor>,  // Scale
    beta: Option<Tensor>,   // Shift

    // Running statistics (not learnable)
    running_mean: Tensor,
    running_var: Tensor,

    // Mode
    training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            gamma: Some(Tensor::ones(&[num_features]).requires_grad_(true)),
            beta: Some(Tensor::zeros(&[num_features]).requires_grad_(true)),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            training: true,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training {
            // Compute batch statistics
            let mean = input.mean_dim(0);
            let var = input.var_dim(0);

            // Update running statistics
            self.running_mean = self.running_mean * (1.0 - self.momentum)
                              + mean * self.momentum;
            self.running_var = self.running_var * (1.0 - self.momentum)
                             + var * self.momentum;

            // Normalize
            let normalized = (input - mean) / (var + self.eps).sqrt();

            // Scale and shift
            self.gamma * normalized + self.beta
        } else {
            // Use running statistics
            let normalized = (input - self.running_mean)
                           / (self.running_var + self.eps).sqrt();
            self.gamma * normalized + self.beta
        }
    }
}

/// Layer Normalization (for transformers)
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f32,
    gamma: Tensor,
    beta: Tensor,
}

/// Instance Normalization (for style transfer)
pub struct InstanceNorm2d { /* similar */ }

/// Group Normalization
pub struct GroupNorm { /* similar */ }
```

### 2.5 Dropout

```rust
/// Dropout regularization (Srivastava et al., 2014)
///
/// Randomly zeroes elements during training to prevent co-adaptation.
pub struct Dropout {
    p: f32,  // Probability of element being zeroed
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");
        Self { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training && self.p > 0.0 {
            // Generate mask: Bernoulli(1-p)
            let mask = Tensor::bernoulli_like(input, 1.0 - self.p);
            // Scale by 1/(1-p) to maintain expected value
            input * mask / (1.0 - self.p)
        } else {
            input.clone()
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]  // No learnable parameters
    }
}

/// Dropout for 2D feature maps (drops entire channels)
pub struct Dropout2d { /* similar */ }
```

### 2.6 Activation Modules

```rust
pub struct ReLU;
pub struct LeakyReLU { negative_slope: f32 }
pub struct PReLU { weight: Tensor }  // Learnable slope (He et al., 2015)
pub struct Sigmoid;
pub struct Tanh;
pub struct GELU;
pub struct Softmax { dim: usize }
pub struct LogSoftmax { dim: usize }

// Functional API
pub mod functional {
    pub fn relu(x: &Tensor) -> Tensor;
    pub fn leaky_relu(x: &Tensor, negative_slope: f32) -> Tensor;
    pub fn sigmoid(x: &Tensor) -> Tensor;
    pub fn tanh(x: &Tensor) -> Tensor;
    pub fn gelu(x: &Tensor) -> Tensor;
    pub fn softmax(x: &Tensor, dim: usize) -> Tensor;
    pub fn log_softmax(x: &Tensor, dim: usize) -> Tensor;
}
```

### 2.7 Container Modules

```rust
/// Sequential container (linear chain of modules)
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { modules: vec![] }
    }

    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.modules.iter().fold(input.clone(), |x, m| m.forward(&x))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

/// Module list (indexed access)
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

/// Module dict (named access)
pub struct ModuleDict {
    modules: HashMap<String, Box<dyn Module>>,
}
```

### 2.8 Transformer Components (Vaswani et al., 2017)

```rust
/// Multi-Head Attention
///
/// Implements scaled dot-product attention with multiple heads.
pub struct MultiHeadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,

    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,

    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0);
        let head_dim = embed_dim / num_heads;

        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            dropout: Dropout::new(0.0),
        }
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        // Project Q, K, V
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        // Reshape for multi-head: [batch, seq, embed] -> [batch, heads, seq, head_dim]
        let q = q.reshape_multihead(self.num_heads);
        let k = k.reshape_multihead(self.num_heads);
        let v = v.reshape_multihead(self.num_heads);

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)) / scale;

        // Apply mask (for causal attention)
        let scores = match attn_mask {
            Some(mask) => scores + mask,
            None => scores,
        };

        let attn_weights = functional::softmax(&scores, -1);
        let attn_weights = self.dropout.forward(&attn_weights);

        // Weighted sum
        let attn_output = attn_weights.matmul(&v);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        let attn_output = attn_output.reshape_from_multihead();

        // Output projection
        let output = self.out_proj.forward(&attn_output);

        (output, attn_weights)
    }
}

/// Transformer Encoder Layer
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    activation: GELU,
}

/// Transformer Decoder Layer
pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
}
```

## Module 3: Loss Functions

```rust
/// Cross-entropy loss (for classification)
///
/// Combines log_softmax and nll_loss for numerical stability.
pub struct CrossEntropyLoss {
    weight: Option<Tensor>,  // Class weights
    reduction: Reduction,
    label_smoothing: f32,
}

impl CrossEntropyLoss {
    pub fn new() -> Self;
    pub fn with_weight(self, weight: Tensor) -> Self;
    pub fn with_label_smoothing(self, smoothing: f32) -> Self;

    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        let log_probs = functional::log_softmax(input, -1);
        functional::nll_loss(&log_probs, target, self.weight.as_ref(), self.reduction)
    }
}

/// Mean Squared Error (for regression)
pub struct MSELoss {
    reduction: Reduction,
}

/// Binary Cross-Entropy with Logits
pub struct BCEWithLogitsLoss {
    weight: Option<Tensor>,
    pos_weight: Option<Tensor>,
    reduction: Reduction,
}

/// Negative Log Likelihood
pub struct NLLLoss {
    weight: Option<Tensor>,
    reduction: Reduction,
}

/// L1 Loss (Mean Absolute Error)
pub struct L1Loss {
    reduction: Reduction,
}

/// Smooth L1 Loss (Huber loss)
pub struct SmoothL1Loss {
    beta: f32,
    reduction: Reduction,
}

#[derive(Clone, Copy)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}
```

## Module 4: Optimizers

### 4.1 Optimizer Trait Extension

```rust
/// Extended optimizer trait for autograd integration
pub trait AutogradOptimizer {
    /// Perform optimization step using accumulated gradients
    fn step(&mut self, parameters: &mut [&mut Tensor]);

    /// Zero out gradients for all parameters
    fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
        for param in parameters {
            param.zero_grad_();
        }
    }
}
```

### 4.2 SGD with Momentum

```rust
/// Stochastic Gradient Descent with momentum
///
/// v_t = momentum * v_{t-1} + grad
/// param = param - lr * v_t
pub struct SGD {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,

    // State: velocity buffers per parameter
    velocity: HashMap<TensorId, Tensor>,
}

impl AutogradOptimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Tensor]) {
        for param in parameters {
            if let Some(grad) = param.grad() {
                // Weight decay (L2 regularization)
                let grad = if self.weight_decay > 0.0 {
                    grad + param * self.weight_decay
                } else {
                    grad.clone()
                };

                // Momentum
                let v = self.velocity.entry(param.id)
                    .or_insert_with(|| Tensor::zeros_like(param));

                *v = v * self.momentum + grad * (1.0 - self.dampening);

                // Nesterov momentum
                let update = if self.nesterov {
                    grad + v * self.momentum
                } else {
                    v.clone()
                };

                // Parameter update
                *param = param - update * self.lr;
            }
        }
    }
}
```

### 4.3 Adam (Kingma & Ba, 2015)

```rust
/// Adam optimizer with bias correction
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,

    // State
    step_count: usize,
    m: HashMap<TensorId, Tensor>,  // First moment
    v: HashMap<TensorId, Tensor>,  // Second moment
    v_max: HashMap<TensorId, Tensor>,  // For AMSGrad
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            v_max: HashMap::new(),
        }
    }
}

impl AutogradOptimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut Tensor]) {
        self.step_count += 1;
        let t = self.step_count as f32;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        for param in parameters {
            if let Some(grad) = param.grad() {
                // Weight decay (AdamW style)
                if self.weight_decay > 0.0 {
                    *param = param - param * (self.lr * self.weight_decay);
                }

                // First moment estimate
                let m = self.m.entry(param.id)
                    .or_insert_with(|| Tensor::zeros_like(param));
                *m = m * self.beta1 + grad * (1.0 - self.beta1);

                // Second moment estimate
                let v = self.v.entry(param.id)
                    .or_insert_with(|| Tensor::zeros_like(param));
                *v = v * self.beta2 + (grad * grad) * (1.0 - self.beta2);

                // Bias-corrected estimates
                let m_hat = m / bias_correction1;
                let v_hat = if self.amsgrad {
                    let v_max = self.v_max.entry(param.id)
                        .or_insert_with(|| Tensor::zeros_like(param));
                    *v_max = v_max.maximum(&v);
                    v_max / bias_correction2
                } else {
                    v / bias_correction2
                };

                // Update
                *param = param - m_hat * self.lr / (v_hat.sqrt() + self.eps);
            }
        }
    }
}
```

### 4.4 Learning Rate Schedulers

```rust
pub trait LRScheduler {
    fn get_lr(&self) -> f32;
    fn step(&mut self);
}

/// Step decay: lr * gamma every step_size epochs
pub struct StepLR {
    optimizer_lr: f32,
    step_size: usize,
    gamma: f32,
    current_step: usize,
}

/// Cosine annealing: lr oscillates following cosine curve
pub struct CosineAnnealingLR {
    optimizer_lr: f32,
    t_max: usize,
    eta_min: f32,
    current_step: usize,
}

/// Linear warmup followed by decay
pub struct LinearWarmupScheduler {
    optimizer_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

/// Reduce on plateau: reduce lr when metric stops improving
pub struct ReduceLROnPlateau {
    optimizer_lr: f32,
    factor: f32,
    patience: usize,
    threshold: f32,
    mode: Mode,  // Min or Max
    best: f32,
    num_bad_epochs: usize,
}
```

## Module 5: Data Integration (alimentar)

```rust
use alimentar::{ArrowDataset, DataLoader};

/// Training loop with alimentar DataLoader
pub fn train_epoch<M: Module>(
    model: &mut M,
    loader: &DataLoader<ArrowDataset>,
    optimizer: &mut impl AutogradOptimizer,
    criterion: &CrossEntropyLoss,
) -> f32 {
    model.train();
    let mut total_loss = 0.0;
    let mut num_batches = 0;

    for batch in loader {
        // Zero gradients
        optimizer.zero_grad(&mut model.parameters_mut());

        // Forward pass
        let x = Tensor::from_arrow(batch.column("features"));
        let y = Tensor::from_arrow(batch.column("labels"));

        let output = model.forward(&x);
        let loss = criterion.forward(&output, &y);

        // Backward pass
        loss.backward();

        // Optimizer step
        optimizer.step(&mut model.parameters_mut());

        total_loss += loss.item();
        num_batches += 1;
    }

    total_loss / num_batches as f32
}
```

## Module 6: trueno Backend Integration

```rust
/// Bridge between aprender Tensor and trueno primitives
impl Tensor {
    /// Create from trueno tensor (zero-copy when possible)
    pub fn from_trueno(t: trueno::Tensor<f32>) -> Self;

    /// Export to trueno tensor
    pub fn to_trueno(&self) -> trueno::Tensor<f32>;

    /// Execute matmul on trueno backend (SIMD/GPU)
    fn matmul_backend(&self, other: &Tensor) -> Tensor {
        let a = self.to_trueno();
        let b = other.to_trueno();
        let result = trueno::ops::matmul(&a, &b);  // Uses SIMD/GPU
        Tensor::from_trueno(result)
    }
}

/// Feature flags for backend selection
#[cfg(feature = "simd")]
mod simd_backend {
    // AVX-512, AVX2, SSE2, NEON, WASM SIMD128
}

#[cfg(feature = "gpu")]
mod gpu_backend {
    // wgpu: Vulkan, Metal, DX12, WebGPU
}
```

## Quality Standards

### Testing Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| Line coverage | ≥90% | Certeza methodology |
| Mutation score | ≥85% | cargo-mutants |
| Gradient tests | 100% ops | Numerical gradient checking |
| Benchmark | vs PyTorch | Performance parity validation |

### Gradient Verification

```rust
/// Numerical gradient check for autograd correctness
///
/// Uses central difference: (f(x+h) - f(x-h)) / 2h
#[cfg(test)]
fn check_gradient(f: impl Fn(&Tensor) -> Tensor, x: &Tensor, eps: f32) -> bool {
    let analytical = {
        let x = x.clone().requires_grad_(true);
        let y = f(&x);
        y.backward();
        x.grad().unwrap().clone()
    };

    let numerical = {
        let mut grad = Tensor::zeros_like(x);
        for i in 0..x.numel() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus.data_mut()[i] += eps;
            x_minus.data_mut()[i] -= eps;
            grad.data_mut()[i] = (f(&x_plus).item() - f(&x_minus).item()) / (2.0 * eps);
        }
        grad
    };

    (analytical - numerical).abs().max() < eps * 10.0
}
```

## Roadmap

### Phase 1: Autograd Engine (v0.9.0)
- [ ] Tensor with gradient tracking
- [ ] Computational graph (tape-based)
- [ ] Backward pass implementation
- [ ] Core differentiable ops (20+ ops)
- [ ] Gradient verification tests
- [ ] trueno backend integration

### Phase 2: nn Module (v0.10.0)
- [ ] Module trait
- [ ] Linear, Conv2d, Conv1d
- [ ] BatchNorm, LayerNorm, GroupNorm
- [ ] Dropout, Dropout2d
- [ ] Activation modules
- [ ] Sequential, ModuleList, ModuleDict

### Phase 3: Loss & Optimizers (v0.11.0)
- [ ] Differentiable loss functions
- [ ] Optimizer autograd integration
- [ ] Learning rate schedulers
- [ ] Gradient clipping utilities

### Phase 4: Transformers (v0.12.0)
- [ ] MultiHeadAttention
- [ ] TransformerEncoder/Decoder
- [ ] Positional encodings
- [ ] Pre-built architectures (BERT-like, GPT-like)

### Phase 5: WASM & Production (v1.0.0)
- [ ] WASM compatibility verification
- [ ] Model serialization (safetensors)
- [ ] Performance benchmarks vs PyTorch
- [ ] Documentation & examples

## Dependencies

```toml
[dependencies]
trueno = "0.7"           # SIMD/GPU compute backend
alimentar = "0.1"        # Data loading (when available)

[dev-dependencies]
proptest = "1.5"         # Property-based testing
criterion = "0.5"        # Benchmarking
```

## References

See [Citations](#citations) section for complete bibliography.

## License

MIT License - Pragmatic AI Labs
