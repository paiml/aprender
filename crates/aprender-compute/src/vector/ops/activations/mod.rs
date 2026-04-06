//! Activation functions for Vector<f32>
//!
//! This module provides neural network activation functions optimized with
//! multi-backend SIMD support (Scalar, SSE2, AVX2, AVX-512, NEON, WASM SIMD).
//!
//! ## Activation Functions
//!
//! - [`softmax`](crate::Vector::softmax): Softmax normalization for classification
//! - [`log_softmax`](crate::Vector::log_softmax): Numerically stable log-softmax
//! - [`relu`](crate::Vector::relu): Rectified Linear Unit
//! - [`sigmoid`](crate::Vector::sigmoid): Logistic sigmoid
//! - [`leaky_relu`](crate::Vector::leaky_relu): Leaky ReLU with configurable slope
//! - [`elu`](crate::Vector::elu): Exponential Linear Unit
//! - [`gelu`](crate::Vector::gelu): Gaussian Error Linear Unit
//! - [`swish`](crate::Vector::swish): Self-gated activation (SiLU)
//! - [`hardswish`](crate::Vector::hardswish): Efficient hardware-friendly swish
//! - [`mish`](crate::Vector::mish): Self-regularizing activation
//! - [`selu`](crate::Vector::selu): Scaled Exponential Linear Unit

mod advanced;

#[cfg(test)]
mod tests;

use crate::backends::scalar::ScalarBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{Backend, Result, TruenoError};

/// Backend dispatch macro for unary operations - centralizes platform-specific SIMD dispatch
/// to eliminate code duplication across activation functions.
///
/// # Safety
/// The macro wraps unsafe backend calls internally, so callers don't need unsafe blocks.
macro_rules! dispatch_unary_op {
    ($backend:expr, $op:ident, $input:expr, $output:expr) => {{
        #[cfg(target_arch = "x86_64")]
        use crate::backends::{avx2::Avx2Backend, sse2::Sse2Backend};
        // SAFETY: CPU features verified at runtime before backend selection
        unsafe {
            match $backend {
                Backend::Scalar => ScalarBackend::$op($input, $output),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::$op($input, $output),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::$op($input, $output),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::$op($input, $output)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    use crate::backends::neon::NeonBackend;
                    NeonBackend::$op($input, $output)
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::$op($input, $output),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    use crate::backends::wasm::WasmBackend;
                    WasmBackend::$op($input, $output)
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::$op($input, $output),
                Backend::GPU | Backend::Auto => ScalarBackend::$op($input, $output),
            }
        }
    }};
}

// Re-export macro for use in advanced submodule
pub(crate) use dispatch_unary_op;

impl Vector<f32> {
    /// Softmax activation function
    ///
    /// Converts a vector of real values into a probability distribution.
    /// Formula: softmax(x)\[i\] = exp(x\[i\] - max(x)) / sum(exp(x\[j\] - max(x)))
    ///
    /// Uses the numerically stable version with max subtraction to prevent overflow.
    /// The output is a probability distribution: all values in [0, 1] and sum to 1.
    ///
    /// This is the standard activation function for multi-class classification in neural networks.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let probs = logits.softmax()?;
    ///
    /// // Verify sum ≈ 1
    /// let sum: f32 = probs.as_slice().iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    ///
    /// // Verify all values in [0, 1]
    /// for &p in probs.as_slice() {
    ///     assert!(p >= 0.0 && p <= 1.0);
    /// }
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors (cannot compute softmax).
    pub fn softmax(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Medium - GPU threshold: >10K elements (multi-pass overhead)
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 4-368x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.softmax(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Multi-pass softmax for numerical stability
        // Find max for numerical stability (prevents overflow in exp)
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize by sum (guard against sum=0 from underflow)
        let safe_sum = sum_exp.max(f32::EPSILON);
        let data: Vec<f32> = exp_vals.iter().map(|&e| e / safe_sum).collect();

        Ok(Vector::from_vec(data))
    }

    /// Log-softmax activation function
    ///
    /// Computes the logarithm of the softmax function in a numerically stable way.
    /// Formula: log_softmax(x)\[i\] = x\[i\] - max(x) - log(sum(exp(x\[j\] - max(x))))
    ///
    /// This is more numerically stable than computing log(softmax(x)) and is commonly
    /// used in neural networks for computing cross-entropy loss.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let log_probs = logits.log_softmax()?;
    ///
    /// // Verify exp(log_softmax) = softmax
    /// let probs_from_log: Vec<f32> = log_probs.as_slice().iter().map(|&x| x.exp()).collect();
    /// let sum: f32 = probs_from_log.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors.
    pub fn log_softmax(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Medium - GPU threshold: >10K elements (multi-pass overhead)
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 4-368x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.log_softmax(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Multi-pass log_softmax for numerical stability
        // Find max for numerical stability
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute log of sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();
        let log_sum_exp = sum_exp.max(f32::EPSILON).ln();

        // log_softmax(x)[i] = x[i] - max - log_sum_exp
        let data: Vec<f32> = self.data.iter().map(|&x| x - max_val - log_sum_exp).collect();

        Ok(Vector::from_vec(data))
    }

    /// ReLU (Rectified Linear Unit) activation function
    ///
    /// Computes the element-wise ReLU: max(0, x).
    /// ReLU is one of the most widely used activation functions in neural networks.
    ///
    /// # Formula
    ///
    /// ```text
    /// relu(x)[i] = max(0, x\[i\])
    ///            = x\[i\]  if x\[i\] > 0
    ///            = 0     otherwise
    /// ```
    ///
    /// # Properties
    ///
    /// - **Non-linearity**: Introduces non-linearity while preserving linearity for positive values
    /// - **Sparsity**: Produces exactly zero for negative inputs (sparse activations)
    /// - **Gradient**: Derivative is 1 for positive inputs, 0 for negative (solves vanishing gradient)
    /// - **Computational efficiency**: Simple max operation, no exponentials
    ///
    /// # Applications
    ///
    /// - **Deep neural networks**: Default activation for hidden layers
    /// - **Convolutional networks**: Standard activation in CNNs
    /// - **Feature learning**: Encourages sparse representations
    ///
    /// # Performance
    ///
    /// This operation is memory-bound. SIMD provides modest speedups since
    /// the computation (comparison and selection) is simpler than memory access.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.relu()?;
    /// assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn relu(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.relu(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Uninit: dispatch_unary_op writes every element before any read.
        let n = self.len();
        let mut result: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend activation writes all elements before any read.
        unsafe {
            result.set_len(n);
        }

        // Use parallel processing for very large arrays (reduces TLB pressure and improves cache utilization)
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 500_000; // Increased to avoid overhead at smaller sizes
            const CHUNK_SIZE: usize = 65536; // 64K elements = 256KB, cache-friendly

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data.par_chunks(CHUNK_SIZE).zip(result.par_chunks_mut(CHUNK_SIZE)).for_each(
                    |(chunk_in, chunk_out)| {
                        dispatch_unary_op!(self.backend, relu, chunk_in, chunk_out);
                    },
                );

                return Ok(Vector::from_vec(result)); // Use from_vec to avoid extra copy
            }
        }

        // Sequential processing for small arrays or when parallel feature disabled
        dispatch_unary_op!(self.backend, relu, &self.data, &mut result);

        Ok(Vector::from_vec(result)) // Use from_vec to avoid extra copy
    }

    /// Sigmoid (logistic) activation function
    ///
    /// Computes the element-wise sigmoid: σ(x) = 1 / (1 + e^(-x)).
    /// Sigmoid is a classic activation function that squashes inputs to the range (0, 1).
    ///
    /// # Formula
    ///
    /// ```text
    /// sigmoid(x)[i] = 1 / (1 + exp(-x\[i\]))
    ///               = exp(x\[i\]) / (1 + exp(x\[i\]))
    /// ```
    ///
    /// # Properties
    ///
    /// - **Bounded output**: Maps all inputs to (0, 1) range
    /// - **Smooth**: Infinitely differentiable (C^∞)
    /// - **Symmetric**: σ(-x) = 1 - σ(x)
    /// - **Derivative**: σ'(x) = σ(x) * (1 - σ(x))
    /// - **Interpretable**: Output can be interpreted as probability
    ///
    /// # Applications
    ///
    /// - **Binary classification**: Final layer for binary output (0 or 1)
    /// - **Logistic regression**: Traditional ML algorithm
    /// - **Gating mechanisms**: LSTM/GRU gates (input, forget, output)
    /// - **Attention mechanisms**: Soft attention weights
    ///
    /// # Numerical Considerations
    ///
    /// For very large negative inputs (x < -50), exp(-x) overflows to infinity.
    /// However, sigmoid(x) approaches 0, so we return 0 for numerical stability.
    /// For very large positive inputs (x > 50), exp(-x) underflows to 0,
    /// and sigmoid(x) approaches 1.
    ///
    /// # Performance
    ///
    /// This operation is compute-bound due to the exp() operation. SIMD provides
    /// modest speedups, but the exponential is the bottleneck.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, 0.0, 2.0]);
    /// let result = v.sigmoid()?;
    ///
    /// // sigmoid(-2) ≈ 0.119, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.881
    /// assert!((result.as_slice()[0] - 0.119).abs() < 0.001);
    /// assert!((result.as_slice()[1] - 0.5).abs() < 0.001);
    /// assert!((result.as_slice()[2] - 0.881).abs() < 0.001);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn sigmoid(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.sigmoid(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Uninit: dispatch_unary_op writes every element before any read.
        let n = self.len();
        let mut result: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend activation writes all elements before any read.
        unsafe {
            result.set_len(n);
        }

        // Dispatch to appropriate backend
        dispatch_unary_op!(self.backend, sigmoid, &self.data, &mut result);

        Ok(Vector::from_vec(result))
    }
}
