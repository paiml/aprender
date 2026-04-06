//! Backend implementations for different SIMD instruction sets
//!
//! This module contains the actual SIMD implementations for each backend.
//! All backends implement the same trait-based interface to ensure API consistency.
//!
//! # Safety
//!
//! All `unsafe` code is isolated within backend implementations. The public API
//! remains 100% safe.
//!
//! # Backends
//!
//! - `scalar`: Portable baseline implementation (no SIMD)
//! - `sse2`: x86_64 baseline SIMD (128-bit)
//! - `avx2`: x86_64 advanced SIMD (256-bit with FMA)
//! - `avx512`: x86_64 maximum SIMD (512-bit)
//! - `neon`: ARM SIMD (128-bit)
//! - `wasm`: WebAssembly SIMD128

pub mod q4k;
pub mod q6k;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod sse2;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod avx2_tests;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub mod neon;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// GPU module - always available for TensorView/PartitionView abstractions
// Actual GPU compute requires "gpu" feature
pub mod gpu;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

/// Backend trait defining common operations
///
/// All backend implementations must implement this trait to ensure
/// consistent behavior across different SIMD instruction sets.
///
/// # Safety
///
/// Implementations may use unsafe SIMD intrinsics. Callers must ensure:
/// - Input slices are valid
/// - Result slice has sufficient capacity
/// - Slices `a` and `b` have the same length
pub trait VectorBackend {
    /// Element-wise addition: a\[i\] + b\[i\]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Element-wise subtraction: a\[i\] - b\[i\]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Element-wise multiplication: a\[i\] * b\[i\]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Element-wise division: a\[i\] / b\[i\]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Dot product: sum(a\[i\] * b\[i\])
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32;

    /// Sum reduction: sum(a\[i\])
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn sum(a: &[f32]) -> f32;

    /// Max reduction: max(a\[i\])
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn max(a: &[f32]) -> f32;

    /// Min reduction: min(a\[i\])
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn min(a: &[f32]) -> f32;

    /// Argmax: index of maximum value
    ///
    /// Returns the index of the first occurrence of the maximum value.
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn argmax(a: &[f32]) -> usize;

    /// Argmin: index of minimum value
    ///
    /// Returns the index of the first occurrence of the minimum value.
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn argmin(a: &[f32]) -> usize;

    /// Kahan summation: numerically stable sum(a\[i\])
    ///
    /// Uses the Kahan summation algorithm to reduce floating-point rounding errors
    /// when summing many numbers. Tracks a running compensation for lost low-order bits.
    ///
    /// # Safety
    ///
    /// - Can handle empty slice (returns 0.0)
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn sum_kahan(a: &[f32]) -> f32;

    /// L2 norm (Euclidean norm): sqrt(sum(a\[i\]^2))
    ///
    /// Computes the Euclidean length of the vector. This is equivalent to sqrt(dot(a, a)).
    ///
    /// # Safety
    ///
    /// - Can handle empty slice (returns 0.0)
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn norm_l2(a: &[f32]) -> f32;

    /// L1 norm (Manhattan norm): sum(|a\[i\]|)
    ///
    /// Computes the sum of absolute values of all elements.
    /// Used in machine learning (L1 regularization), distance metrics, and sparse modeling.
    ///
    /// # Safety
    ///
    /// - Can handle empty slice (returns 0.0)
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn norm_l1(a: &[f32]) -> f32;

    /// L-infinity norm (maximum absolute value): max(|a\[i\]|)
    ///
    /// Computes the maximum absolute value of all elements.
    /// Used in optimization (constraint checking), numerical analysis, and error bounds.
    ///
    /// # Safety
    ///
    /// - Can handle empty slice (returns 0.0)
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn norm_linf(a: &[f32]) -> f32;

    /// Scalar multiplication: result\[i\] = a\[i\] * scalar
    ///
    /// Multiplies all elements by a scalar value.
    /// Used in vector scaling, normalization, and linear transformations.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slice
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]);

    /// Absolute value: result\[i\] = |a\[i\]|
    ///
    /// Computes the absolute value of each element.
    /// Used in distance metrics (L1 norm), numerical stability, and signal processing.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slice
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn abs(a: &[f32], result: &mut [f32]);

    /// Clamp elements to range [min_val, max_val]: result\[i\] = max(min_val, min(a\[i\], max_val))
    ///
    /// Constrains each element to the specified range.
    /// Used in neural networks (gradient clipping), graphics (color clamping), and signal processing.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slice
    /// - Assumes min_val <= max_val (caller must validate)
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]);

    /// Linear interpolation: result\[i\] = a\[i\] + t * (b\[i\] - a\[i\])
    ///
    /// Computes element-wise linear interpolation between two vectors.
    /// When t=0, returns a; when t=1, returns b; values outside \[0,1\] extrapolate.
    /// Used in graphics, animation, neural networks, and signal processing.
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]);

    /// Fused multiply-add: result\[i\] = a\[i\] * b\[i\] + c\[i\]
    ///
    /// Computes element-wise fused multiply-add operation.
    /// On hardware with FMA support, this is a single instruction with better performance
    /// and numerical accuracy (no intermediate rounding).
    /// Used in neural networks, matrix multiplication, and scientific computing.
    ///
    /// # Safety
    ///
    /// - `a`, `b`, and `c` must all have the same length
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]);

    /// ReLU activation: result\[i\] = max(0, a\[i\])
    ///
    /// Rectified Linear Unit - the most common activation function in neural networks.
    /// Sets negative values to zero, passes positive values unchanged.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn relu(a: &[f32], result: &mut [f32]);

    /// Exponential function: result\[i\] = exp(a\[i\])
    ///
    /// Computes e^x for each element using range reduction for numerical accuracy.
    /// Foundation for sigmoid, softmax, GELU, and other activation functions.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn exp(a: &[f32], result: &mut [f32]);

    /// Sigmoid activation: result\[i\] = 1 / (1 + exp(-a\[i\]))
    ///
    /// Logistic sigmoid function - maps inputs to (0, 1) range.
    /// Used in binary classification and as gating mechanism.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]);

    /// GELU activation: result\[i\] = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ///
    /// Gaussian Error Linear Unit - smooth non-monotonic activation.
    /// Used in BERT, GPT, and modern transformers.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn gelu(a: &[f32], result: &mut [f32]);

    /// Swish activation: result\[i\] = x * sigmoid(x) = x / (1 + exp(-x))
    ///
    /// Self-gated activation function (also called SiLU).
    /// Used in EfficientNet, MobileNetV3.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn swish(a: &[f32], result: &mut [f32]);

    /// Hyperbolic tangent activation: result\[i\] = tanh(a\[i\]) = (exp(2x) - 1) / (exp(2x) + 1)
    ///
    /// Hyperbolic tangent - maps inputs to (-1, 1) range.
    /// Classic activation function from early neural networks.
    /// Used in RNNs, LSTMs, and as smooth alternative to ReLU.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn tanh(a: &[f32], result: &mut [f32]);

    /// Square root: result\[i\] = sqrt(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn sqrt(a: &[f32], result: &mut [f32]);

    /// Reciprocal: result\[i\] = 1 / a\[i\]
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn recip(a: &[f32], result: &mut [f32]);

    /// Natural logarithm: result\[i\] = ln(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn ln(a: &[f32], result: &mut [f32]);

    /// Base-2 logarithm: result\[i\] = log2(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn log2(a: &[f32], result: &mut [f32]);

    /// Base-10 logarithm: result\[i\] = log10(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn log10(a: &[f32], result: &mut [f32]);

    /// Sine: result\[i\] = sin(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn sin(a: &[f32], result: &mut [f32]);

    /// Cosine: result\[i\] = cos(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn cos(a: &[f32], result: &mut [f32]);

    /// Tangent: result\[i\] = tan(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn tan(a: &[f32], result: &mut [f32]);

    /// Floor: result\[i\] = floor(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn floor(a: &[f32], result: &mut [f32]);

    /// Ceiling: result\[i\] = ceil(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn ceil(a: &[f32], result: &mut [f32]);

    /// Round: result\[i\] = round(a\[i\])
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slices
    // SAFETY: Caller must satisfy the documented preconditions for slice validity
    unsafe fn round(a: &[f32], result: &mut [f32]);
}
