//! Smooth self-gated activation functions: gelu, swish, hardswish, mish
//!
//! These activations are smooth, non-monotonic, and use self-gating mechanisms.
//! They are the preferred activations in modern transformer and vision architectures.

use crate::backends::scalar::ScalarBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{Backend, Result, TruenoError};

use super::super::dispatch_unary_op;

impl Vector<f32> {
    /// GELU (Gaussian Error Linear Unit) activation function
    ///
    /// Computes the element-wise GELU activation using the tanh approximation.
    /// GELU is the activation function used in transformers (BERT, GPT, etc.).
    ///
    /// # Formula
    ///
    /// ```text
    /// gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// ```
    ///
    /// This is the tanh approximation which is faster than the exact form
    /// involving the error function (erf).
    ///
    /// # Properties
    ///
    /// - **Smooth**: Infinitely differentiable everywhere
    /// - **Non-monotonic**: Unlike ReLU variants, has slight non-monotonicity near zero
    /// - **Stochastic regularizer**: Can be viewed as adaptive dropout
    /// - **Zero-centered**: Mean activation close to zero
    /// - **Bounded below**: Approaches 0 as x → -∞
    /// - **Unbounded above**: Linear growth for large positive x
    ///
    /// # Applications
    ///
    /// - **Transformers**: BERT, GPT-2, GPT-3, GPT-4 (default activation)
    /// - **Vision transformers**: ViT, DINO, MAE
    /// - **Modern architectures**: State-of-the-art NLP and vision models
    /// - **Better than ReLU**: Empirically outperforms ReLU in many tasks
    ///
    /// # Performance
    ///
    /// This operation is compute-intensive (tanh, x³ calculations).
    /// More expensive than ReLU but comparable to ELU.
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
    /// let result = v.gelu()?;
    ///
    /// // GELU is smooth and non-monotonic near zero
    /// assert!(result.as_slice()[0] < 0.0); // Negative inputs → small negative outputs
    /// assert_eq!(result.as_slice()[2], 0.0); // gelu(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → ~linear
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn gelu(&self) -> Result<Self> {
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
                    if gpu.gelu(&self.data, &mut result).is_ok() {
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
        dispatch_unary_op!(self.backend, gelu, &self.data, &mut result);

        Ok(Vector::from_vec(result))
    }

    /// Swish activation function (also known as SiLU - Sigmoid Linear Unit)
    ///
    /// Applies the Swish activation element-wise: swish(x) = x * sigmoid(x) = x / (1 + e^(-x)).
    ///
    /// Swish is a smooth, non-monotonic activation function that consistently matches or
    /// outperforms ReLU in deep networks. It's used in EfficientNet, MobileNet v3, and
    /// many modern architectures. The function is self-gated: it adaptively gates the
    /// input based on its value.
    ///
    /// Properties:
    /// - Smooth and differentiable everywhere
    /// - Non-monotonic: has a slight "dip" for negative values
    /// - swish(0) = 0
    /// - swish(x) ≈ x for large positive x (linear)
    /// - swish(x) ≈ 0 for large negative x
    /// - Unbounded above, bounded below by ≈ -0.278 at x ≈ -1.278
    ///
    /// # Performance
    ///
    /// Compute-bound operation requiring exponential and division.
    /// Future SIMD optimizations planned for Phase 9 (GPU backend).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.swish()?;
    ///
    /// // Swish is smooth and self-gated
    /// assert!(result.as_slice()[0] < 0.0); // Negative inputs → small negative outputs
    /// assert_eq!(result.as_slice()[2], 0.0); // swish(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → ~linear
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Ramachandran et al. (2017): "Searching for Activation Functions"
    /// - Also known as SiLU (Sigmoid Linear Unit): Elfwing et al. (2018)
    pub fn swish(&self) -> Result<Self> {
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
                    if gpu.swish(&self.data, &mut result).is_ok() {
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

        // Dispatch to appropriate SIMD backend
        dispatch_unary_op!(self.backend, swish, &self.data, &mut result);

        Ok(Vector::from_vec(result))
    }

    /// Hard Swish activation function
    ///
    /// Applies the hardswish activation element-wise: hardswish(x) = x * relu6(x + 3) / 6
    ///
    /// Hardswish is a piece-wise linear approximation to swish, designed for efficient
    /// computation in mobile neural networks. It's used in MobileNetV3 and avoids the
    /// expensive sigmoid computation of standard swish.
    ///
    /// Properties:
    /// - Piece-wise linear: efficient to compute
    /// - hardswish(x) = 0 for x ≤ -3
    /// - hardswish(x) = x for x ≥ 3
    /// - hardswish(x) = x * (x + 3) / 6 for -3 < x < 3
    /// - hardswish(0) = 0
    /// - Smooth transitions at boundaries
    ///
    /// # Performance
    ///
    /// More efficient than swish as it uses only multiply/divide operations
    /// instead of expensive exponential functions. Ideal for inference on
    /// resource-constrained devices.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-4.0, -3.0, 0.0, 3.0, 4.0]);
    /// let result = v.hardswish()?;
    ///
    /// // Piece-wise linear behavior
    /// assert_eq!(result.as_slice()[0], 0.0); // x ≤ -3 → 0
    /// assert_eq!(result.as_slice()[1], 0.0); // x = -3 → 0
    /// assert_eq!(result.as_slice()[2], 0.0); // x = 0 → 0
    /// assert_eq!(result.as_slice()[3], 3.0); // x = 3 → x
    /// assert_eq!(result.as_slice()[4], 4.0); // x ≥ 3 → x
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Howard et al. (2019): "Searching for MobileNetV3"
    pub fn hardswish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Scalar implementation: hardswish(x) = x * relu6(x + 3) / 6
        // Simplified piece-wise:
        // - x <= -3: 0
        // - x >= 3: x
        // - else: x * (x + 3) / 6
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    x
                } else {
                    x * (x + 3.0) / 6.0
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Mish activation function
    ///
    /// Applies the mish activation element-wise: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    ///
    /// Mish is a self-regularizing non-monotonic activation function that often outperforms
    /// ReLU and swish in computer vision tasks. It's used in YOLOv4 and many modern architectures.
    ///
    /// Properties:
    /// - Smooth and non-monotonic (similar to swish)
    /// - Self-regularizing: prevents dying neurons
    /// - mish(0) ≈ 0 (small positive value)
    /// - mish(x) ≈ x for large positive x (nearly linear)
    /// - mish(x) ≈ 0 for large negative x
    /// - Bounded below by ≈ -0.31 at x ≈ -1.19
    ///
    /// # Performance
    ///
    /// Compute-bound operation requiring exponential, logarithm, and tanh.
    /// More expensive than ReLU/swish but often provides better accuracy.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.mish()?;
    ///
    /// // Mish is smooth and self-gated
    /// assert!(result.as_slice()[0] < 0.0); // Small negative output for negative inputs
    /// assert!(result.as_slice()[2].abs() < 1e-5); // mish(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → near linear
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Misra (2019): "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    pub fn mish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Scalar implementation: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                // Handle extreme values for numerical stability
                if x < -20.0 {
                    // For very negative x: softplus ≈ 0, tanh(0) ≈ 0, so mish ≈ 0
                    0.0
                } else if x > 20.0 {
                    // For very positive x: softplus ≈ x, tanh(x) ≈ 1, so mish ≈ x
                    x
                } else {
                    // Normal case: x * tanh(ln(1 + e^x))
                    let softplus = (1.0 + x.exp()).ln();
                    x * softplus.tanh()
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }
}
