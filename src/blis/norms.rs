//! SIMD-accelerated normalization kernels.
//!
//! AVX2 implementations of RMSNorm and LayerNorm with scalar fallback.
//! Uses FMA for sum-of-squares accumulation and fused normalize+scale.
//!
//! # Algorithm
//!
//! RMSNorm: output_i = x_i / sqrt(mean(x^2) + eps) * gamma_i
//! LayerNorm: output_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
//!
//! Both use two-accumulator SIMD reduction for sum/sum-of-squares to hide
//! FMA latency, then vectorized normalize+scale pass.
//!
//! Contract: provable-contracts/contracts/rmsnorm-kernel-v1.yaml
//! Contract: provable-contracts/contracts/layernorm-kernel-v1.yaml
//!
//! # References
//!
//! - Zhang & Sennrich (2019). Root Mean Square Layer Normalization.
//! - Ba, Kiros & Hinton (2016). Layer Normalization.

use crate::error::TruenoError;

// ============================================================================
// RMSNorm
// ============================================================================

/// RMSNorm: output_i = x_i / sqrt(mean(x^2) + eps) * gamma_i
///
/// Uses AVX2 SIMD with FMA when available, scalar fallback otherwise.
///
/// Contract: rmsnorm-kernel-v1, equation "rmsnorm"
///
/// # Errors
///
/// Returns `Err` if input/gamma/output lengths don't match or are empty.
pub fn rms_norm(
    input: &[f32],
    gamma: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), TruenoError> {
    let n = input.len();
    if n == 0 || n != gamma.len() || n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "rms_norm size mismatch: input[{}], gamma[{}], output[{}]",
            n,
            gamma.len(),
            output.len()
        )));
    }

    // Contract: rmsnorm-kernel-v1.yaml precondition (pv codegen)
    contract_pre_rmsnorm!(input);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified by feature detection above.
            unsafe {
                rms_norm_avx2(input, gamma, eps, output);
            }
            contract_post_rmsnorm!(output);
            return Ok(());
        }
    }

    rms_norm_scalar(input, gamma, eps, output);
    contract_post_rmsnorm!(output);
    Ok(())
}

/// Scalar RMSNorm implementation.
fn rms_norm_scalar(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();

    // Phase 1: sum of squares
    let mut sum_sq = 0.0_f32;
    for &x in input {
        sum_sq += x * x;
    }

    // Phase 2: inverse RMS
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // Phase 3: normalize and scale
    for i in 0..n {
        output[i] = input[i] * inv_rms * gamma[i];
    }
}

/// AVX2+FMA RMSNorm implementation.
///
/// Two-accumulator reduction hides FMA latency (5 cycles on Zen3/4, 4 on Intel).
/// Single vectorized pass for normalize+scale.
///
/// # Safety
///
/// Requires AVX2 and FMA support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_avx2(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 16; // Process 16 elements (2×8) per iteration
    let remainder_16 = chunks * 16;

    unsafe {
        // Phase 1: sum of squares with two accumulators
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for i in 0..chunks {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(i * 16));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(i * 16 + 8));
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }

        // Handle 8-element remainder chunk
        let mut sum_sq;
        if remainder_16 + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(remainder_16));
            acc0 = _mm256_fmadd_ps(v, v, acc0);
            let combined = _mm256_add_ps(acc0, acc1);

            // Horizontal sum: 128-bit halves, then pairs, then scalar
            let hi = _mm256_extractf128_ps(combined, 1);
            let lo = _mm256_castps256_ps128(combined);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            sum_sq = _mm_cvtss_f32(sums2);

            // Scalar tail after remainder_16 + 8
            for i in (remainder_16 + 8)..n {
                sum_sq += input[i] * input[i];
            }
        } else {
            let combined = _mm256_add_ps(acc0, acc1);
            let hi = _mm256_extractf128_ps(combined, 1);
            let lo = _mm256_castps256_ps128(combined);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            sum_sq = _mm_cvtss_f32(sums2);

            for i in remainder_16..n {
                sum_sq += input[i] * input[i];
            }
        }

        // Phase 2: inverse RMS (scalar — single instruction, not worth SIMD)
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

        // Phase 3: normalize and scale using AVX2
        let inv_rms_vec = _mm256_set1_ps(inv_rms);
        let chunks_out = n / 8;
        let remainder_out = chunks_out * 8;

        for i in 0..chunks_out {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(i * 8));
            // output = x * inv_rms * gamma = (x * inv_rms) * gamma
            let normed = _mm256_mul_ps(x, inv_rms_vec);
            let scaled = _mm256_mul_ps(normed, g);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), scaled);
        }

        // Scalar tail
        for i in remainder_out..n {
            output[i] = input[i] * inv_rms * gamma[i];
        }
    }
}

// ============================================================================
// LayerNorm
// ============================================================================

/// LayerNorm: output_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
///
/// Uses AVX2 SIMD when available, scalar fallback otherwise.
///
/// Contract: layernorm-kernel-v1, equation "layernorm"
///
/// # Errors
///
/// Returns `Err` if input/gamma/beta/output lengths don't match or are empty.
pub fn layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), TruenoError> {
    let n = input.len();
    if n == 0 || n != gamma.len() || n != beta.len() || n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "layer_norm size mismatch: input[{}], gamma[{}], beta[{}], output[{}]",
            n,
            gamma.len(),
            beta.len(),
            output.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified by feature detection above.
            unsafe {
                layer_norm_avx2(input, gamma, beta, eps, output);
            }
            return Ok(());
        }
    }

    layer_norm_scalar(input, gamma, beta, eps, output);
    Ok(())
}

/// Scalar LayerNorm implementation.
fn layer_norm_scalar(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();

    // Phase 1: mean
    let mut sum = 0.0_f32;
    for &x in input {
        sum += x;
    }
    let mean = sum / n as f32;

    // Phase 2: variance
    let mut var_sum = 0.0_f32;
    for &x in input {
        let d = x - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

    // Phase 3: normalize + affine
    for i in 0..n {
        output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
    }
}

/// AVX2+FMA LayerNorm implementation.
///
/// Two-pass: (1) mean via SIMD sum, (2) variance via SIMD FMA,
/// then vectorized normalize + affine transform.
///
/// # Safety
///
/// Requires AVX2 and FMA support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 8;
    let remainder = chunks * 8;

    unsafe {
        // Phase 1: compute mean with AVX2
        let mut sum_vec = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            sum_vec = _mm256_add_ps(sum_vec, v);
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum_vec, 1);
        let lo = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        let mut sum = _mm_cvtss_f32(sums2);

        for i in remainder..n {
            sum += input[i];
        }
        let mean = sum / n as f32;

        // Phase 2: compute variance with AVX2 FMA
        let mean_vec = _mm256_set1_ps(mean);
        let mut var_vec0 = _mm256_setzero_ps();
        let mut var_vec1 = _mm256_setzero_ps();
        let chunks2 = n / 16;
        let remainder2 = chunks2 * 16;

        for i in 0..chunks2 {
            let v0 = _mm256_loadu_ps(input.as_ptr().add(i * 16));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(i * 16 + 8));
            let d0 = _mm256_sub_ps(v0, mean_vec);
            let d1 = _mm256_sub_ps(v1, mean_vec);
            var_vec0 = _mm256_fmadd_ps(d0, d0, var_vec0);
            var_vec1 = _mm256_fmadd_ps(d1, d1, var_vec1);
        }

        // Handle 8-element remainder
        let mut var_sum;
        if remainder2 + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(remainder2));
            let d = _mm256_sub_ps(v, mean_vec);
            var_vec0 = _mm256_fmadd_ps(d, d, var_vec0);

            let combined = _mm256_add_ps(var_vec0, var_vec1);
            let hi2 = _mm256_extractf128_ps(combined, 1);
            let lo2 = _mm256_castps256_ps128(combined);
            let s128 = _mm_add_ps(lo2, hi2);
            let sh = _mm_movehdup_ps(s128);
            let ss = _mm_add_ps(s128, sh);
            let sh2 = _mm_movehl_ps(ss, ss);
            let ss2 = _mm_add_ss(ss, sh2);
            var_sum = _mm_cvtss_f32(ss2);

            for i in (remainder2 + 8)..n {
                let d = input[i] - mean;
                var_sum += d * d;
            }
        } else {
            let combined = _mm256_add_ps(var_vec0, var_vec1);
            let hi2 = _mm256_extractf128_ps(combined, 1);
            let lo2 = _mm256_castps256_ps128(combined);
            let s128 = _mm_add_ps(lo2, hi2);
            let sh = _mm_movehdup_ps(s128);
            let ss = _mm_add_ps(s128, sh);
            let sh2 = _mm_movehl_ps(ss, ss);
            let ss2 = _mm_add_ss(ss, sh2);
            var_sum = _mm_cvtss_f32(ss2);

            for i in remainder2..n {
                let d = input[i] - mean;
                var_sum += d * d;
            }
        }

        let inv_std = 1.0 / (var_sum / n as f32 + eps).sqrt();

        // Phase 3: normalize + affine with AVX2
        let inv_std_vec = _mm256_set1_ps(inv_std);
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(i * 8));
            let b = _mm256_loadu_ps(beta.as_ptr().add(i * 8));
            let centered = _mm256_sub_ps(x, mean_vec);
            let normed = _mm256_mul_ps(centered, inv_std_vec);
            // output = gamma * normed + beta
            let result = _mm256_fmadd_ps(g, normed, b);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), result);
        }

        // Scalar tail
        for i in remainder..n {
            output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
        }
    }
}

// ============================================================================
// Allocating variants
// ============================================================================

/// RMSNorm with output allocation. Avoids zero-fill overhead.
///
/// # Panics
///
/// Panics if input and gamma have different lengths.
#[must_use]
pub fn rms_norm_alloc(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    rms_norm(input, gamma, eps, &mut output).expect("rms_norm_alloc: length mismatch");
    output
}

/// LayerNorm with output allocation. Avoids zero-fill overhead.
///
/// # Panics
///
/// Panics if input, gamma, and beta have different lengths.
#[must_use]
pub fn layer_norm_alloc(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    layer_norm(input, gamma, beta, eps, &mut output).expect("layer_norm_alloc: length mismatch");
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── RMSNorm tests ─────────────────────────────────────────────────────

    /// FALSIFY-RN-001: Finiteness
    #[test]
    fn test_rmsnorm_finiteness() {
        for n in [4, 8, 16, 32, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
            let gamma = vec![1.0f32; n];
            let mut output = vec![0.0f32; n];
            rms_norm(&input, &gamma, 1e-5, &mut output).unwrap();
            for (i, &o) in output.iter().enumerate() {
                assert!(o.is_finite(), "RMSNorm output[{i}] not finite for n={n}");
            }
        }
    }

    /// FALSIFY-RN-002: Scale invariance
    #[test]
    fn test_rmsnorm_scale_invariance() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let gamma = vec![1.0f32; 64];
        let mut out1 = vec![0.0f32; 64];
        let mut out2 = vec![0.0f32; 64];

        rms_norm(&input, &gamma, 1e-8, &mut out1).unwrap();

        let scaled: Vec<f32> = input.iter().map(|&x| x * 3.7).collect();
        rms_norm(&scaled, &gamma, 1e-8, &mut out2).unwrap();

        for i in 0..64 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-4,
                "Scale invariance failed at {i}: {} vs {}",
                out1[i],
                out2[i]
            );
        }
    }

    /// FALSIFY-RN-003: AVX2 vs scalar parity
    #[test]
    fn test_rmsnorm_avx2_scalar_parity() {
        for n in [4, 7, 8, 16, 31, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
            let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
            let mut scalar_out = vec![0.0f32; n];
            let mut dispatch_out = vec![0.0f32; n];

            rms_norm_scalar(&input, &gamma, 1e-5, &mut scalar_out);
            rms_norm(&input, &gamma, 1e-5, &mut dispatch_out).unwrap();

            for i in 0..n {
                let diff = (scalar_out[i] - dispatch_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "RMSNorm parity failed at [{i}] n={n}: scalar={} dispatch={} diff={}",
                    scalar_out[i],
                    dispatch_out[i],
                    diff
                );
            }
        }
    }

    /// FALSIFY-RN-004: Zero vector
    #[test]
    fn test_rmsnorm_zero_input() {
        let input = vec![0.0f32; 16];
        let gamma = vec![1.0f32; 16];
        let mut output = vec![0.0f32; 16];
        rms_norm(&input, &gamma, 1e-5, &mut output).unwrap();
        for (i, &o) in output.iter().enumerate() {
            assert!(o.is_finite(), "Zero input produced non-finite at {i}");
            assert!(o.abs() < 1e-2, "Zero input should produce ~0 at {i}, got {o}");
        }
    }

    /// FALSIFY-RN-005: Unit gamma normalized RMS ≈ 1
    #[test]
    fn test_rmsnorm_unit_gamma_normalized_rms() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let gamma = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 128];
        rms_norm(&input, &gamma, 1e-8, &mut output).unwrap();

        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let rms_out = (sum_sq / output.len() as f32).sqrt();
        assert!((rms_out - 1.0).abs() < 1e-3, "RMS of output = {rms_out}, expected ~1.0");
    }

    #[test]
    fn test_rmsnorm_error_on_mismatch() {
        let input = vec![1.0f32; 4];
        let gamma = vec![1.0f32; 3];
        let mut output = vec![0.0f32; 4];
        assert!(rms_norm(&input, &gamma, 1e-5, &mut output).is_err());
    }

    #[test]
    fn test_rmsnorm_error_on_empty() {
        let input: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        assert!(rms_norm(&input, &gamma, 1e-5, &mut output).is_err());
    }

    // ── LayerNorm tests ───────────────────────────────────────────────────

    /// FALSIFY-LN-001: Finiteness
    #[test]
    fn test_layernorm_finiteness() {
        for n in [4, 8, 16, 32, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
            let gamma = vec![1.0f32; n];
            let beta = vec![0.0f32; n];
            let mut output = vec![0.0f32; n];
            layer_norm(&input, &gamma, &beta, 1e-5, &mut output).unwrap();
            for (i, &o) in output.iter().enumerate() {
                assert!(o.is_finite(), "LayerNorm output[{i}] not finite for n={n}");
            }
        }
    }

    /// FALSIFY-LN-002: Zero mean (with gamma=1, beta=0)
    #[test]
    fn test_layernorm_zero_mean() {
        for n in [16, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
            let gamma = vec![1.0f32; n];
            let beta = vec![0.0f32; n];
            let mut output = vec![0.0f32; n];
            layer_norm(&input, &gamma, &beta, 1e-5, &mut output).unwrap();

            let mean: f32 = output.iter().sum::<f32>() / n as f32;
            assert!(mean.abs() < 1e-4, "LayerNorm output mean = {mean}, expected ~0 for n={n}");
        }
    }

    /// FALSIFY-LN-003: Unit variance (with gamma=1, beta=0)
    #[test]
    fn test_layernorm_unit_variance() {
        for n in [16, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
            let gamma = vec![1.0f32; n];
            let beta = vec![0.0f32; n];
            let mut output = vec![0.0f32; n];
            layer_norm(&input, &gamma, &beta, 1e-5, &mut output).unwrap();

            let mean: f32 = output.iter().sum::<f32>() / n as f32;
            let var: f32 = output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            assert!(
                (var - 1.0).abs() < 1e-2,
                "LayerNorm output var = {var}, expected ~1.0 for n={n}"
            );
        }
    }

    /// FALSIFY-LN-004: Shift invariance
    #[test]
    fn test_layernorm_shift_invariance() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let gamma = vec![1.0f32; 64];
        let beta = vec![0.0f32; 64];
        let mut out1 = vec![0.0f32; 64];
        let mut out2 = vec![0.0f32; 64];

        layer_norm(&input, &gamma, &beta, 1e-5, &mut out1).unwrap();

        let shifted: Vec<f32> = input.iter().map(|&x| x + 42.0).collect();
        layer_norm(&shifted, &gamma, &beta, 1e-5, &mut out2).unwrap();

        for i in 0..64 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-3,
                "Shift invariance failed at {i}: {} vs {}",
                out1[i],
                out2[i]
            );
        }
    }

    /// FALSIFY-LN-005: AVX2 vs scalar parity
    #[test]
    fn test_layernorm_avx2_scalar_parity() {
        for n in [4, 7, 8, 16, 31, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
            let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
            let beta: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 0.1 - 0.1).collect();
            let mut scalar_out = vec![0.0f32; n];
            let mut dispatch_out = vec![0.0f32; n];

            layer_norm_scalar(&input, &gamma, &beta, 1e-5, &mut scalar_out);
            layer_norm(&input, &gamma, &beta, 1e-5, &mut dispatch_out).unwrap();

            for i in 0..n {
                let diff = (scalar_out[i] - dispatch_out[i]).abs();
                assert!(
                    diff < 1e-4,
                    "LayerNorm parity failed at [{i}] n={n}: scalar={} dispatch={} diff={}",
                    scalar_out[i],
                    dispatch_out[i],
                    diff
                );
            }
        }
    }

    /// FALSIFY-LN-006: Constant input → output = beta
    #[test]
    fn test_layernorm_constant_input() {
        let input = vec![5.0f32; 32];
        let gamma = vec![1.0f32; 32];
        let beta: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; 32];
        layer_norm(&input, &gamma, &beta, 1e-5, &mut output).unwrap();
        for (i, (&o, &b)) in output.iter().zip(beta.iter()).enumerate() {
            assert!((o - b).abs() < 1e-3, "Constant input: output[{i}]={o}, expected ~beta={b}");
        }
    }

    #[test]
    fn test_layernorm_error_on_mismatch() {
        let input = vec![1.0f32; 4];
        let gamma = vec![1.0f32; 3];
        let beta = vec![0.0f32; 4];
        let mut output = vec![0.0f32; 4];
        assert!(layer_norm(&input, &gamma, &beta, 1e-5, &mut output).is_err());
    }

    #[test]
    fn test_layernorm_error_on_empty() {
        let input: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        assert!(layer_norm(&input, &gamma, &beta, 1e-5, &mut output).is_err());
    }
}
