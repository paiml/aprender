//! Precision conversion functions and memory estimation utilities.

use super::Precision;

/// Convert f32 to bf16 (truncated)
///
/// BF16 uses the same exponent as f32 but only 7 mantissa bits.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    // Take upper 16 bits (sign + exponent + 7 mantissa bits)
    (bits >> 16) as u16
}

/// Convert bf16 to f32
pub fn bf16_to_f32(value: u16) -> f32 {
    // Place in upper 16 bits, lower 16 are zeros
    let bits = u32::from(value) << 16;
    f32::from_bits(bits)
}

/// Convert f32 to fp16 (IEEE half precision)
///
/// ONE PATH: Delegates to `trueno::f32_to_f16` (UCBD §4).
pub fn f32_to_fp16(value: f32) -> u16 {
    trueno::f32_to_f16(value)
}

/// Convert fp16 to f32
///
/// ONE PATH: Delegates to `trueno::f16_to_f32` (UCBD §4).
pub fn fp16_to_f32(value: u16) -> f32 {
    trueno::f16_to_f32(value)
}

/// Truncate an f32 value to BF16 precision (zero lower 16 mantissa bits).
///
/// Equivalent to f32 → bf16 → f32 round-trip via bit truncation (not rounding).
/// The result is a valid f32 with only 7 mantissa bits of precision.
///
/// # Contract (C-BF16GEMM-001)
///
/// - `bf16_truncate(x).to_bits() & 0x0000FFFF == 0` for all x
/// - `bf16_truncate(NaN).is_nan()` and `bf16_truncate(Inf).is_infinite()`
/// - `bf16_truncate(x) == bf16_to_f32(f32_to_bf16(x))` for all x
#[inline]
pub fn bf16_truncate(val: f32) -> f32 {
    f32::from_bits(val.to_bits() & 0xFFFF_0000)
}

/// CPU reference implementation of BF16-precision GEMM.
///
/// Computes C = A @ B where A is MxK, B is KxN, but truncates each operand
/// to BF16 precision before multiply, with FP32 accumulation.
///
/// This matches the precision characteristics of hardware BF16 tensor cores:
/// - BF16 multiply (7-bit mantissa)
/// - FP32 accumulation (23-bit mantissa)
///
/// Used for verification against GPU BF16 GEMM kernels.
pub fn gemm_bf16_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                let a_val = bf16_truncate(a[row * k + i]);
                let b_val = bf16_truncate(b[i * n + col]);
                acc = a_val.mul_add(b_val, acc);
            }
            c[row * n + col] = acc;
        }
    }
    c
}

/// Estimate memory savings from mixed precision
///
/// # Arguments
///
/// * `num_params` - Number of model parameters
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension
/// * `precision` - Target precision
///
/// # Returns
///
/// Tuple of (fp32_bytes, mixed_bytes, savings_ratio)
pub fn estimate_memory_savings(
    num_params: usize,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    precision: Precision,
) -> (usize, usize, f32) {
    // FP32 memory: params + activations + gradients
    let param_bytes_fp32 = num_params * 4;
    let activation_bytes_fp32 = batch_size * seq_len * hidden_size * 4;
    let grad_bytes_fp32 = num_params * 4;
    let total_fp32 = param_bytes_fp32 + activation_bytes_fp32 + grad_bytes_fp32;

    // Mixed precision: master weights (fp32) + activations (reduced) + gradients (reduced)
    let param_bytes_mixed = num_params * 4; // Master weights in fp32
    let activation_bytes_mixed = batch_size * seq_len * hidden_size * precision.size_bytes();
    let grad_bytes_mixed = num_params * precision.size_bytes();
    let total_mixed = param_bytes_mixed + activation_bytes_mixed + grad_bytes_mixed;

    let savings = 1.0 - (total_mixed as f32 / total_fp32 as f32);
    (total_fp32, total_mixed, savings)
}
