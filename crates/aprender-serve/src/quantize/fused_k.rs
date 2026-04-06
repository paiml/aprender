//! Fused quantized operations for K-quantization formats (PMAT-802)
//!
//! Implements fused dequant+dot operations that eliminate intermediate f32 buffer
//! allocation for 8x memory bandwidth reduction.
//!
//! Per llama-cpp-style-performance-spec.md:
//! - Memory wall is the bottleneck (Wulf & McKee [10])
//! - Fused operations keep data in registers, avoid memory round-trips
//! - ULP tolerance of ≤4 for numerical equivalence (Goldberg [9])
//!
//! Functions:
//! - `fused_q4k_dot`, `fused_q4k_dot_simd` - Q4_K dot products
//! - `fused_q4k_q8_dot` - Q4_K with Q8 activations
//! - `fused_q4k_q8k_dot`, `fused_q4k_q8k_dot_simd` - Q4_K with Q8_K activations
//! - `fused_q4k_q8k_parallel_matvec_into` - Parallel matrix-vector multiply
//! - `fused_q4k_q8k_ffn_up_gate_into` - Fused FFN up/gate computation
//! - `fused_q6k_dot`, `fused_q6k_dot_simd` - Q6_K dot products
//! - `fused_q5k_dot`, `fused_q5k_dot_simd` - Q5_K dot products

use super::dequant::read_f16;
use super::simd::extract_scale_min;
use super::types::QK_K;
use crate::error::{RealizarError, Result};

/// Fused Q4_K dequantize + dot product
///
/// Computes the dot product of Q4_K quantized weights with f32 activations
/// WITHOUT allocating an intermediate f32 buffer. Dequantization happens
/// inline, accumulating directly into a register.
///
/// # Arguments
///
/// * `q4k_data` - Raw Q4_K quantized data (super-blocks of 144 bytes)
/// * `activations` - f32 activation values (must match dequantized length)
///
/// # Returns
///
/// The dot product as f32
///
/// # Errors
///
/// Returns error if:
/// - `q4k_data` length is not a multiple of 144 bytes (super-block size)
/// - `activations` length doesn't match the number of quantized values
///
/// # Performance
///
/// This function reduces memory traffic by 8x compared to separate
/// dequantize-then-dot operations:
/// - Naive: Read Q4_K (4.5 bits) → Write f32 (32 bits) → Read f32 → Compute
/// - Fused: Read Q4_K (4.5 bits) → Compute in registers
///
/// # Examples
///
/// ```rust,ignore
/// let weights_q4k = load_q4k_weights();
/// let activations = get_layer_activations();
/// let result = fused_q4k_dot(&weights_q4k, &activations)?;
/// ```
pub fn fused_q4k_dot(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    contract_pre_quantized_dot!();
    const SUPER_BLOCK_BYTES: usize = 144;

    // Validate Q4_K data length
    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    // Validate activation length matches
    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q4_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Degenerate case: zero super-blocks = zero dot product
    if num_super_blocks == 0 {
        return Ok(0.0);
    }

    // Contract: matmul-kernel-v1.yaml precondition (pv codegen)
    // Placed after validation + degenerate case guards
    contract_pre_matmul!(activations);

    // Accumulator for dot product result
    let mut acc = 0.0f32;
    let mut activation_idx = 0;
    // Track whether all weight scales are finite (precondition: scale > 0)
    // Non-finite scales (e.g. from artificial test data) are precondition violations.
    let mut all_scales_finite = true;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Read d (f16 -> f32)
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);

        // Read dmin (f16 -> f32)
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        if !d.is_finite() || !dmin.is_finite() {
            all_scales_finite = false;
        }

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Read qs (128 bytes)
        let qs_start = sb_start + 16;
        let qs = &q4k_data[qs_start..qs_start + 128];

        // PAR-001: Match dequantize_q4_k layout (llama.cpp/candle compatible)
        // Process 4 chunks of 64 values each (0, 64, 128, 192)
        // Each chunk: 32 low nibbles, then 32 high nibbles from 32 consecutive bytes
        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let d1 = d * sc1;
            let dm1 = dmin * m1;

            let (sc2, m2) = extract_scale_min(&scales, is + 1);
            let d2 = d * sc2;
            let dm2 = dmin * m2;

            // First pass: 32 low nibbles (use sc1, m1)
            for &byte in q {
                let q_val = (byte & 0x0F) as f32;
                let value = d1 * q_val - dm1;
                acc += value * activations[activation_idx];
                activation_idx += 1;
            }

            // Second pass: 32 high nibbles (use sc2, m2)
            for &byte in q {
                let q_val = (byte >> 4) as f32;
                let value = d2 * q_val - dm2;
                acc += value * activations[activation_idx];
                activation_idx += 1;
            }
        }
    }

    // Only fire postcondition when preconditions hold: finite activations AND finite scales.
    // NaN/Inf propagation from precondition-violating inputs is GIGO, not a contract failure.
    if all_scales_finite && activations.iter().all(|a| a.is_finite()) {
        contract_post_quantized_dot!(&acc);
    }
    Ok(acc)
}

/// Fused Q4_K dequantize + dot product with SIMD acceleration
///
/// This is the public, safe API that automatically dispatches to the best
/// available implementation (AVX2 when available, scalar fallback otherwise).
///
/// # Arguments
///
/// * `q4k_data` - Raw Q4_K quantized data (super-blocks of 144 bytes)
/// * `activations` - f32 activation values (must match dequantized length)
///
/// # Returns
///
/// The dot product as f32, matching `fused_q4k_dot` within 4 ULPs
///
/// # Errors
///
/// Returns error if:
/// - `q4k_data` length is not a multiple of 144 bytes (super-block size)
/// - `activations` length doesn't match the number of quantized values
///
/// # Performance
///
/// - AVX2: ~8x speedup over scalar via 256-bit SIMD + FMA
/// - Fused operation: 8x memory bandwidth reduction vs dequant-then-dot
/// - Combined potential: Up to 64x improvement for memory-bound operations
pub fn fused_q4k_dot_simd(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    contract_pre_simd_only_threshold!();
    // Runtime feature detection with fallback (per RustBelt pattern)
    #[cfg(target_arch = "x86_64")]
    {
        // PAR-126: AVX-512 VNNI requires pre-quantized activations (Q4K×Q8K format)
        // For now, use AVX2 which works with f32 activations directly.
        // Future optimization: pre-quantize activations to Q8_0 format once per matmul.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We've verified AVX2 and FMA are available at runtime
            // The unsafe function performs the same logical operation as scalar
            // SAFETY: Memory safety ensured by bounds checking and alignment
            return unsafe { fused_q4k_dot_avx2(q4k_data, activations) };
        }
    }

    // pmat-ignore: hardware-path (scalar fallback tested directly via fused_q4k_dot)
    // Fallback to scalar implementation
    fused_q4k_dot(q4k_data, activations)
}

/// Quantize f32 activations to i8 and compute integer dot product with q_nibbles.
///
/// Returns (integer_sum, activation_scale) for caller to apply FP32 scaling.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VNNI
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn avx512_quantize_dot(
    act_slice: &[f32],
    q_nibbles_256: std::arch::x86_64::__m256i,
) -> (i32, f32) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let act_max = act_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let act_min = act_slice.iter().copied().fold(f32::INFINITY, f32::min);
    let act_scale = if act_max > act_min {
        127.0 / (act_max - act_min)
    } else {
        1.0
    };

    let mut act_i8 = [0i8; 32];
    for (i, &a) in act_slice.iter().enumerate() {
        act_i8[i] = ((a - act_min) * act_scale).round() as i8;
    }

    let act_vec = _mm256_loadu_si256(act_i8.as_ptr().cast::<__m256i>());
    let q_512 = _mm512_cvtepu8_epi16(q_nibbles_256);
    let act_512 = _mm512_cvtepi8_epi16(act_vec);
    let prod = _mm512_mullo_epi16(q_512, act_512);

    let sum_256 = _mm256_add_epi16(
        _mm512_castsi512_si256(prod),
        _mm512_extracti64x4_epi64(prod, 1),
    );
    let sum_128 = _mm_add_epi16(
        _mm256_castsi256_si128(sum_256),
        _mm256_extracti128_si256(sum_256, 1),
    );
    let sum_32 = _mm256_cvtepi16_epi32(sum_128);
    let sum_arr: [i32; 8] = std::mem::transmute(sum_32);
    (sum_arr.iter().sum(), act_scale)
}

/// AVX-512 VNNI accelerated Q4_K dot product (PAR-126)
///
/// Uses vpdpbusd for int8×int8 accumulation which is 4x faster than FP32 FMA.
///
/// # Safety
/// Requires AVX-512F, AVX-512BW, and AVX-512VNNI
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4k_dot_avx512_vnni(q4k_data: &[u8], activations: &[f32]) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of super-block size {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if activations.len() != expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Activation length {} doesn't match Q4_K values count {}",
                activations.len(),
                expected_values
            ),
        });
    }

    // Final accumulator (FP32)
    let mut total_sum = 0.0f32;
    let mut activation_idx = 0;

    // Nibble mask for 4-bit extraction
    let nibble_mask = _mm512_set1_epi8(0x0F_i8);

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            let next_sb = (sb_idx + 1) * SUPER_BLOCK_BYTES;
            _mm_prefetch(q4k_data.as_ptr().add(next_sb).cast::<i8>(), _MM_HINT_T0);
        }

        // Read d and dmin (f16 → f32)
        let d = read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        // Read scales (12 bytes)
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        // Pointer to quantized data (128 bytes)
        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);

        // Process 64 values at a time (matches AVX-512 width of 64 bytes)
        for j in (0..QK_K).step_by(64) {
            let q_start = j / 2;

            // Get scales for the two 32-value halves
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Load 32 bytes of quantized data (64 nibbles)
            let q_bytes_256 = _mm256_loadu_si256(qs_ptr.add(q_start).cast::<__m256i>());

            // Expand to 512-bit for AVX-512 operations
            let q_bytes = _mm512_castsi256_si512(q_bytes_256);
            let q_bytes = _mm512_inserti64x4(q_bytes, q_bytes_256, 1);

            // Extract low and high nibbles
            let q_lo = _mm512_and_si512(q_bytes, nibble_mask);
            let q_hi = _mm512_and_si512(_mm512_srli_epi16(q_bytes, 4), nibble_mask);

            // Process low nibbles: quantize activations → integer dot → scale
            let q_lo_256 = _mm512_castsi512_si256(q_lo);
            let (int_sum, act_scale) =
                avx512_quantize_dot(&activations[activation_idx..activation_idx + 32], q_lo_256);
            total_sum += int_sum as f32 * d * sc1 / act_scale - (32.0 * dmin * m1);
            activation_idx += 32;

            // Process high nibbles: same pattern
            let q_hi_256 = _mm512_castsi512_si256(q_hi);
            let (int_sum2, act_scale2) =
                avx512_quantize_dot(&activations[activation_idx..activation_idx + 32], q_hi_256);
            total_sum += int_sum2 as f32 * d * sc2 / act_scale2 - (32.0 * dmin * m2);
            activation_idx += 32;
        }
    }

    Ok(total_sum)
}

include!("q4k_dot_avx2.rs");
include!("fused_q4k_q8k_dot_avx512vnni.rs");
include!("horizontal.rs");
include!("requires.rs");
include!("q4_q8_dot_avx2.rs");
#[cfg(target_arch = "x86_64")]
include!("q4_q8_dot_avx512.rs");
#[cfg(target_arch = "x86_64")]
include!("q4k_dot_ggml_style.rs");
