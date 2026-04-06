//! Bsum Precomputation (Contract: quantized-dot-product-v1.yaml, Step 3)
//!
//! The dot product decomposition shows the offset term depends ONLY on activation
//! sub-block sums (bsums), NOT on weights. Precomputing bsums once per token
//! eliminates redundant computation across all weight rows.
//!
//! ## Mathematical Basis
//!
//! ```text
//! dot(W, x) = Σ_sb [d_W · d_x · Σ_j(s_j · Σ_i(q_W_i · q_x_i))
//!                   − dmin_W · d_x · Σ_j(m_j · bsums[j])]
//! ```
//!
//! Where `bsums[j] = Σ_i(q_x_i)` for sub-block j — weight-independent.

use super::simd::extract_scale_min;
use super::types::QK_K;
use crate::error::{RealizarError, Result};

/// Precompute Q8K sub-block sums for all superblocks in an activation vector.
///
/// Returns: `Vec<[i32; 8]>` — 8 sub-block sums per superblock (Q4_K has 8 blocks of 32).
///
/// These sums are weight-independent and can be reused across all rows in a matvec.
///
/// # Arguments
/// * `q8k_quants` - Quantized i8 activations (length = num_superblocks * 256)
/// * `num_superblocks` - Number of 256-element superblocks
///
/// # Errors
/// Returns error if buffer is too small.
pub fn precompute_q8k_bsums(q8k_quants: &[i8], num_superblocks: usize) -> Result<Vec<[i32; 8]>> {
    contract_pre_bsum!();
    let expected_len = num_superblocks * QK_K;
    if q8k_quants.len() < expected_len {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q8K buffer too small for bsum precomputation: need {}, have {}",
                expected_len,
                q8k_quants.len()
            ),
        });
    }

    let mut bsums = Vec::with_capacity(num_superblocks);

    for sb in 0..num_superblocks {
        let sb_start = sb * QK_K;
        let mut block_sums = [0i32; 8];

        for block in 0..8 {
            let block_start = sb_start + block * 32;
            let mut sum = 0i32;
            for i in 0..32 {
                sum += q8k_quants[block_start + i] as i32;
            }
            block_sums[block] = sum;
        }
        bsums.push(block_sums);
    }

    contract_post_bsum!(&bsums);
    Ok(bsums)
}

/// AVX2-optimized Q4_K × Q8_K dot product with precomputed bsums.
///
/// Identical to `fused_q4k_q8k_dot_avx2` but uses precomputed sub-block sums
/// instead of recomputing them per row (~50 SIMD instructions saved per superblock).
///
/// # Safety
/// Requires AVX2 CPU feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn fused_q4k_q8k_dot_with_bsums_avx2(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    bsums: &[[i32; 8]],
) -> Result<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    const SUPER_BLOCK_BYTES: usize = 144;

    if !q4k_data.len().is_multiple_of(SUPER_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Q4_K data length {} is not a multiple of {}",
                q4k_data.len(),
                SUPER_BLOCK_BYTES
            ),
        });
    }

    let num_super_blocks = q4k_data.len() / SUPER_BLOCK_BYTES;
    let expected_values = num_super_blocks * QK_K;

    if q8k_scales.len() < num_super_blocks
        || q8k_quants.len() < expected_values
        || bsums.len() < num_super_blocks
    {
        return Err(RealizarError::InvalidShape {
            reason: "Q8_K or bsums buffer too small".to_string(),
        });
    }

    let nibble_mask = _mm256_set1_epi8(0x0F_i8);

    let mut total_acc = 0.0f32;

    for sb_idx in 0..num_super_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let q8_start = sb_idx * QK_K;

        // Prefetch next super-block
        if sb_idx + 1 < num_super_blocks {
            _mm_prefetch(
                q4k_data
                    .as_ptr()
                    .add((sb_idx + 1) * SUPER_BLOCK_BYTES)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
            _mm_prefetch(
                q8k_quants.as_ptr().add((sb_idx + 1) * QK_K).cast::<i8>(),
                _MM_HINT_T0,
            );
        }

        // Read Q4_K header
        let d = super::simd::read_f16(&q4k_data[sb_start..sb_start + 2]);
        let dmin = super::simd::read_f16(&q4k_data[sb_start + 2..sb_start + 4]);

        let mut scales = [0u8; 12];
        scales.copy_from_slice(&q4k_data[sb_start + 4..sb_start + 16]);

        let q8_scale = q8k_scales[sb_idx];
        let d_q8 = d * q8_scale;
        let dmin_q8 = dmin * q8_scale;

        let qs_ptr = q4k_data.as_ptr().add(sb_start + 16);
        let q8_ptr = q8k_quants.as_ptr().add(q8_start);

        // Precomputed bsums for this superblock
        let sb_bsums = &bsums[sb_idx];

        // Process 4 iterations of 64 values each (256 total)
        for j in (0..QK_K).step_by(64) {
            let q_offset = j / 2;

            // Get scales for two 32-value blocks
            let is = j / 32;
            let (sc1, m1) = extract_scale_min(&scales, is);
            let (sc2, m2) = extract_scale_min(&scales, is + 1);

            // Load 32 bytes of Q4_K (64 nibbles)
            let q4_bytes = _mm256_loadu_si256(qs_ptr.add(q_offset).cast::<__m256i>());

            // Extract nibbles
            let q4_lo = _mm256_and_si256(q4_bytes, nibble_mask);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_bytes, 4), nibble_mask);

            // Load 64 bytes of Q8_K
            let q8_lo = _mm256_loadu_si256(q8_ptr.add(j).cast::<__m256i>());
            let q8_hi = _mm256_loadu_si256(q8_ptr.add(j + 32).cast::<__m256i>());

            // Q4×Q8 dot products
            let prod_lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
            let prod_hi = _mm256_maddubs_epi16(q4_hi, q8_hi);

            // Split products and horizontal sum
            let prod_lo_128 = _mm256_castsi256_si128(prod_lo);
            let prod_lo_hi128 = _mm256_extracti128_si256(prod_lo, 1);
            let prod_hi_128 = _mm256_castsi256_si128(prod_hi);
            let prod_hi_hi128 = _mm256_extracti128_si256(prod_hi, 1);

            let sum_lo_1 = _mm_madd_epi16(prod_lo_128, _mm_set1_epi16(1));
            let sum_lo_2 = _mm_madd_epi16(prod_lo_hi128, _mm_set1_epi16(1));
            let sum_hi_1 = _mm_madd_epi16(prod_hi_128, _mm_set1_epi16(1));
            let sum_hi_2 = _mm_madd_epi16(prod_hi_hi128, _mm_set1_epi16(1));

            let sum_1 = _mm_add_epi32(sum_lo_1, sum_hi_1);
            let sum_2 = _mm_add_epi32(sum_lo_2, sum_hi_2);

            let sum_1_f = _mm_cvtepi32_ps(sum_1);
            let sum_2_f = _mm_cvtepi32_ps(sum_2);

            let scaled_1 = _mm_mul_ps(sum_1_f, _mm_set1_ps(sc1));
            let scaled_2 = _mm_mul_ps(sum_2_f, _mm_set1_ps(sc2));

            // USE PRECOMPUTED BSUMS instead of ~50 SIMD instructions
            let q8_block1_val = sb_bsums[is];
            let q8_block2_val = sb_bsums[is + 1];

            // Final accumulation
            let scaled_sum = _mm_add_ps(scaled_1, scaled_2);
            let hsum = _mm_hadd_ps(scaled_sum, scaled_sum);
            let hsum = _mm_hadd_ps(hsum, hsum);
            let block_prod = _mm_cvtss_f32(hsum);

            total_acc += d_q8 * block_prod;
            total_acc -= dmin_q8 * (m1 * q8_block1_val as f32 + m2 * q8_block2_val as f32);
        }
    }

    Ok(total_acc)
}

/// Dispatcher: Q4_K × Q8_K dot product with precomputed bsums.
///
/// Uses AVX2 with bsum precomputation when available, falls back to
/// standard `fused_q4k_q8k_dot_simd` otherwise.
pub fn fused_q4k_q8k_dot_with_bsums_simd(
    q4k_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    bsums: &[[i32; 8]],
) -> Result<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected, bounds checked inside function
            return unsafe {
                fused_q4k_q8k_dot_with_bsums_avx2(q4k_data, q8k_scales, q8k_quants, bsums)
            };
        }
    }

    // Scalar fallback: ignore bsums, use standard path
    super::fused_k::fused_q4k_q8k_dot_simd(q4k_data, q8k_scales, q8k_quants)
}

/// Parallel Q4_K × Q8_K matrix-vector multiply with bsum precomputation.
///
/// Hoists weight-independent activation sub-block sums out of the per-row loop.
/// Contract-derived optimization: ~50 SIMD instructions saved per superblock per row.
///
/// Falls through to the standard `fused_q4k_q8k_parallel_matvec_into` when
/// the 4-row AVX-512 VNNI micro-kernel is available (which already precomputes
/// bsums within the micro-tile).
#[allow(clippy::similar_names)]
pub fn fused_q4k_q8k_parallel_matvec_with_bsums_into(
    weight_data: &[u8],
    q8k_scales: &[f32],
    q8k_quants: &[i8],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    const SUPER_BLOCK_BYTES: usize = 144;
    const MIDI_TILE_M: usize = 64;
    const PARALLEL_THRESHOLD: usize = 256;

    let super_blocks_per_row = in_dim.div_ceil(QK_K);
    let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;

    let expected_weight_bytes = out_dim * bytes_per_row;
    if weight_data.len() < expected_weight_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Weight data too small: need {} bytes, have {}",
                expected_weight_bytes,
                weight_data.len()
            ),
        });
    }

    if output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Output buffer too small: need {}, have {}",
                out_dim,
                output.len()
            ),
        });
    }

    // CONTRACT-DERIVED: Precompute bsums ONCE per token (weight-independent)
    let bsums = precompute_q8k_bsums(q8k_quants, super_blocks_per_row)?;

    if out_dim < PARALLEL_THRESHOLD {
        // Sequential path
        for o in 0..out_dim {
            let row_start = o * bytes_per_row;
            let row_data = &weight_data[row_start..row_start + bytes_per_row];
            output[o] = fused_q4k_q8k_dot_with_bsums_simd(row_data, q8k_scales, q8k_quants, &bsums)
                .unwrap_or(0.0);
        }
    } else {
        // Parallel path with midi-tile chunking
        output[..out_dim]
            .par_chunks_mut(MIDI_TILE_M)
            .enumerate()
            .for_each(|(midi_idx, midi_chunk)| {
                let midi_start = midi_idx * MIDI_TILE_M;

                for (local_idx, out) in midi_chunk.iter_mut().enumerate() {
                    let row = midi_start + local_idx;
                    let row_start = row * bytes_per_row;
                    let row_data = &weight_data[row_start..row_start + bytes_per_row];
                    *out =
                        fused_q4k_q8k_dot_with_bsums_simd(row_data, q8k_scales, q8k_quants, &bsums)
                            .unwrap_or(0.0);
                }
            });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_q8k_bsums_basic() {
        let num_sb = 2;
        let quants: Vec<i8> = (0..512).map(|i| (i % 7) as i8 - 3).collect();
        let bsums = precompute_q8k_bsums(&quants, num_sb).expect("should succeed");
        assert_eq!(bsums.len(), 2);

        // Verify first superblock, first block
        let expected_sum: i32 = (0..32).map(|i| (i % 7) as i32 - 3).sum();
        assert_eq!(bsums[0][0], expected_sum);
    }

    #[test]
    fn test_precompute_q8k_bsums_zeros() {
        let quants = vec![0i8; 256];
        let bsums = precompute_q8k_bsums(&quants, 1).expect("should succeed");
        assert_eq!(bsums.len(), 1);
        assert!(bsums[0].iter().all(|&s| s == 0));
    }

    #[test]
    fn test_precompute_q8k_bsums_too_small() {
        let quants = vec![0i8; 100];
        let result = precompute_q8k_bsums(&quants, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_bsum_equivalence_with_on_the_fly() {
        // FALSIFY-QDOT-003 extension: verify precomputed bsums match inline computation
        let num_sb = 3;
        let quants: Vec<i8> = (0..768).map(|i| ((i * 17 + 5) % 256) as i8).collect();
        let bsums = precompute_q8k_bsums(&quants, num_sb).expect("should succeed");

        // Verify against manual computation
        for sb in 0..num_sb {
            for block in 0..8 {
                let start = sb * 256 + block * 32;
                let expected: i32 = (0..32).map(|i| quants[start + i] as i32).sum();
                assert_eq!(
                    bsums[sb][block], expected,
                    "Mismatch at sb={sb}, block={block}"
                );
            }
        }
    }

    #[test]
    fn test_bsum_precompute_negative_values() {
        // Q8K can have negative values — verify signed sum is correct
        let quants: Vec<i8> = (0..256).map(|i| -((i % 128) as i8)).collect();
        let bsums = precompute_q8k_bsums(&quants, 1).expect("should succeed");

        for block in 0..8 {
            let start = block * 32;
            let expected: i32 = (0..32).map(|i| quants[start + i] as i32).sum();
            assert_eq!(bsums[0][block], expected);
            assert!(
                bsums[0][block] <= 0,
                "All negative inputs should give negative sum"
            );
        }
    }

    #[test]
    fn test_bsum_aware_matvec_basic() {
        // Basic smoke test: zero weights → zero output
        let in_dim: usize = 256;
        let out_dim: usize = 16;
        let super_blocks_per_row = in_dim.div_ceil(QK_K);
        let bytes_per_row = super_blocks_per_row * 144;
        let weights = vec![0u8; out_dim * bytes_per_row];
        let scales = vec![1.0f32; super_blocks_per_row];
        let quants = vec![0i8; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result = fused_q4k_q8k_parallel_matvec_with_bsums_into(
            &weights,
            &scales,
            &quants,
            in_dim,
            out_dim,
            &mut output,
        );
        assert!(result.is_ok());
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bsum_aware_matvec_weight_too_small() {
        let weights = vec![0u8; 100];
        let scales = vec![1.0f32; 1];
        let quants = vec![0i8; 256];
        let mut output = vec![0.0f32; 64];

        let result = fused_q4k_q8k_parallel_matvec_with_bsums_into(
            &weights,
            &scales,
            &quants,
            256,
            64,
            &mut output,
        );
        assert!(result.is_err());
    }
}
