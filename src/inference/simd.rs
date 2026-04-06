//! SIMD-accelerated operations for inference
//!
//! Provides high-performance primitive operations using trueno's SIMD backend.
//! All operations are designed for cache efficiency with tiled implementations.
//!
//! ## Operations
//!
//! - [`simd_matmul`] - Matrix-vector multiplication with SIMD dot products
//! - [`simd_dot`] - SIMD-accelerated dot product
//! - [`simd_add`] - Vector addition
//! - [`simd_mul`] - Element-wise multiplication
//! - [`simd_silu`] - SiLU activation (x * sigmoid(x))
//! - [`simd_gelu`] - GELU activation (approximate)
//! - [`simd_softmax`] - Numerically stable softmax
//!
//! ## Performance
//!
//! Uses trueno's Vector::dot for all dot products, enabling:
//! - AVX2/SSE on x86
//! - NEON on ARM
//! - WASM SIMD in browsers
//! - Scalar fallback everywhere else

use trueno::Vector;

/// Tile size for cache-efficient tiled matmul
const TILE_SIZE: usize = 64;

/// SIMD-accelerated matrix-vector multiplication
///
/// Uses trueno's optimized SIMD backend for maximum performance.
/// Falls back to scalar for non-SIMD architectures.
///
/// # Arguments
///
/// * `input` - Input vector of length `in_dim`
/// * `weight` - Weight matrix stored row-major [out_dim × in_dim]
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
///
/// # Returns
///
/// Output vector of length `out_dim`
///
/// # Example
///
/// ```
/// use realizar::inference::simd_matmul;
///
/// // 2x3 matrix times 3-vector = 2-vector
/// let input = vec![1.0, 2.0, 3.0];
/// let weight = vec![
///     1.0, 0.0, 0.0,  // row 0: extracts x
///     0.0, 1.0, 0.0,  // row 1: extracts y
/// ];
/// let output = simd_matmul(&input, &weight, 3, 2);
/// assert_eq!(output.len(), 2);
/// ```
#[must_use]
pub fn simd_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    // Convert to trueno types for SIMD acceleration
    let input_vec = Vector::from_slice(input);

    // Compute each output element using SIMD dot product
    let mut output = vec![0.0; out_dim];

    // Use tiled approach for better cache utilization
    for tile_start in (0..out_dim).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(out_dim);

        for row in tile_start..tile_end {
            let row_start = row * in_dim;
            let row_end = row_start + in_dim;
            let row_vec = Vector::from_slice(&weight[row_start..row_end]);
            output[row] = input_vec.dot(&row_vec).expect("dot product failed");
        }
    }

    output
}

/// SIMD-accelerated dot product
///
/// Uses trueno's SIMD backend for vectorized computation.
///
/// # Example
///
/// ```
/// use realizar::inference::simd_dot;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = simd_dot(&a, &b);
/// assert!((result - 32.0).abs() < 1e-5);
/// ```
#[inline]
#[must_use]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    Vector::from_slice(a)
        .dot(&Vector::from_slice(b))
        .expect("dot product failed")
}

/// SIMD-accelerated vector addition (a += b)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_add;
///
/// let mut a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// simd_add(&mut a, &b);
/// assert_eq!(a, vec![5.0, 7.0, 9.0]);
/// ```
#[inline]
pub fn simd_add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

/// SIMD-accelerated element-wise multiplication (a *= b)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_mul;
///
/// let mut a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// simd_mul(&mut a, &b);
/// assert_eq!(a, vec![4.0, 10.0, 18.0]);
/// ```
#[inline]
pub fn simd_mul(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= y;
    }
}

/// SIMD-accelerated SiLU activation (x * sigmoid(x))
///
/// Also known as Swish activation: f(x) = x / (1 + exp(-x))
///
/// # Example
///
/// ```
/// use realizar::inference::simd_silu;
///
/// let mut data = vec![0.0, 1.0, -1.0];
/// simd_silu(&mut data);
/// assert!((data[0] - 0.0).abs() < 1e-5);  // silu(0) = 0
/// assert!((data[1] - 0.7311).abs() < 0.01);  // silu(1) ≈ 0.731
/// ```
#[inline]
pub fn simd_silu(data: &mut [f32]) {
    // ONE PATH: Per-element delegates to trueno::silu_scalar (UCBD §4).
    for x in data.iter_mut() {
        *x = trueno::silu_scalar(*x);
    }
}

/// SIMD-accelerated GELU activation (approximate)
///
/// Uses the tanh approximation:
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// # Example
///
/// ```
/// use realizar::inference::simd_gelu;
///
/// let mut data = vec![0.0, 1.0, -1.0];
/// simd_gelu(&mut data);
/// assert!((data[0] - 0.0).abs() < 1e-5);  // gelu(0) = 0
/// assert!((data[1] - 0.8413).abs() < 0.01);  // gelu(1) ≈ 0.841
/// ```
#[inline]
pub fn simd_gelu(data: &mut [f32]) {
    // ONE PATH: Per-element delegates to trueno::gelu_scalar (UCBD §4).
    for x in data.iter_mut() {
        *x = trueno::gelu_scalar(*x);
    }
}

/// SIMD-accelerated softmax with numerical stability
///
/// Uses the max-subtraction trick to prevent overflow:
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
///
/// # Example
///
/// ```
/// use realizar::inference::simd_softmax;
///
/// let mut data = vec![1.0, 2.0, 3.0];
/// simd_softmax(&mut data);
///
/// // Probabilities should sum to 1
/// let sum: f32 = data.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
///
/// // Largest input should have largest probability
/// assert!(data[2] > data[1]);
/// assert!(data[1] > data[0]);
/// ```
pub fn simd_softmax(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for x in data.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for x in data.iter_mut() {
            *x *= inv_sum;
        }
    }
}

// ============================================================================
// BF16/F16 SIMD Conversion (T-QA-021 Optimization)
// ============================================================================

/// Fast BF16→F32 conversion using bit manipulation
///
/// BF16 is a truncated F32 (same exponent, fewer mantissa bits).
/// Conversion is just a 16-bit left shift.
///
/// # Arguments
///
/// * `input` - Raw BF16 bytes (2 bytes per value)
///
/// # Returns
///
/// F32 vector with converted values
///
/// # Performance
///
/// This implementation uses SIMD on x86_64 with AVX2 support,
/// processing 8 BF16 values in parallel.
///
/// # Example
///
/// ```
/// use realizar::inference::simd_bf16_to_f32;
///
/// let bf16_bytes = half::bf16::from_f32(1.5).to_le_bytes();
/// let f32_vals = simd_bf16_to_f32(&bf16_bytes);
/// assert!((f32_vals[0] - 1.5).abs() < 0.01);
/// ```
#[must_use]
pub fn simd_bf16_to_f32(input: &[u8]) -> Vec<f32> {
    let count = input.len() / 2;
    if count == 0 {
        return Vec::new();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        simd_bf16_to_f32_avx2(input, count)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        bf16_to_f32_fast(input, count)
    }
}

/// AVX2-accelerated BF16→F32 conversion
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn simd_bf16_to_f32_avx2(input: &[u8], count: usize) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut output = vec![0.0f32; count];
    let chunks = count / 8;
    let remainder = count % 8;

    // SAFETY: AVX2 target_feature is required by cfg, input bounds checked by chunks calculation,
    // output vector pre-allocated to count elements
    unsafe {
        for i in 0..chunks {
            let in_offset = i * 16;
            let out_offset = i * 8;

            // Load 8 BF16 values (16 bytes)
            let bf16_bytes = _mm_loadu_si128(input.as_ptr().add(in_offset) as *const __m128i);

            // Unpack lower 4 BF16 to F32 (zero-extend and shift left by 16)
            let lo = _mm_unpacklo_epi16(bf16_bytes, _mm_setzero_si128());
            let lo_shifted = _mm_slli_epi32(lo, 16);

            // Unpack upper 4 BF16 to F32
            let hi = _mm_unpackhi_epi16(bf16_bytes, _mm_setzero_si128());
            let hi_shifted = _mm_slli_epi32(hi, 16);

            // Store results
            _mm_storeu_ps(
                output.as_mut_ptr().add(out_offset),
                _mm_castsi128_ps(lo_shifted),
            );
            _mm_storeu_ps(
                output.as_mut_ptr().add(out_offset + 4),
                _mm_castsi128_ps(hi_shifted),
            );
        }
    }

    // Handle remainder with scalar
    let remainder_start = chunks * 8;
    for i in 0..remainder {
        let offset = (remainder_start + i) * 2;
        let bits = u16::from_le_bytes([input[offset], input[offset + 1]]) as u32;
        output[remainder_start + i] = f32::from_bits(bits << 16);
    }

    output
}

/// Fast scalar BF16→F32 conversion using bit manipulation
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn bf16_to_f32_fast(input: &[u8], count: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(count);
    for chunk in input.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        output.push(f32::from_bits(bits << 16));
    }
    output
}

/// Scalar fallback for non-AVX2 platforms
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn bf16_to_f32_fast(input: &[u8], count: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(count);
    for chunk in input.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        output.push(f32::from_bits(bits << 16));
    }
    output
}

/// Fast F16→F32 conversion using the half crate
///
/// Unlike BF16, F16 has a different exponent bias and requires
/// proper conversion (not just bit shifting).
///
/// # Arguments
///
/// * `input` - Raw F16 bytes (2 bytes per value)
///
/// # Returns
///
/// F32 vector with converted values
#[must_use]
pub fn simd_f16_to_f32(input: &[u8]) -> Vec<f32> {
    input
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// SIMD-accelerated BF16 dot product
///
/// Computes dot product of two BF16 vectors without full conversion.
/// Converts small chunks at a time to keep F32 data in L1 cache.
///
/// # Arguments
///
/// * `a` - First BF16 vector (raw bytes)
/// * `b` - Second BF16 vector (raw bytes)
///
/// # Returns
///
/// Dot product as F32
#[must_use]
pub fn simd_bf16_dot(a: &[u8], b: &[u8]) -> f32 {
    const CHUNK_SIZE: usize = 64; // 64 BF16 values = 128 bytes, fits in L1

    let count = a.len().min(b.len()) / 2;
    let mut sum = 0.0f32;

    for chunk_start in (0..count).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(count);
        let byte_start = chunk_start * 2;
        let byte_end = chunk_end * 2;

        // Convert chunk to F32
        let a_f32 = simd_bf16_to_f32(&a[byte_start..byte_end]);
        let b_f32 = simd_bf16_to_f32(&b[byte_start..byte_end]);

        // Compute dot product of chunk using SIMD
        sum += simd_dot(&a_f32, &b_f32);
    }

    sum
}

include!("simd_bf16_ops.rs");
include!("simd_bf16.rs");
