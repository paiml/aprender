//! Generic Scalar Reference Dot Product (Contract: quantized-dot-product-v1.yaml)
//!
//! Implements the 6-phase kernel structure from the contract for ANY blocked
//! quantization format via the `QuantBlockFormat` trait. This is the
//! **mathematical reference** — every SIMD kernel must match it within
//! `F::ULP_TOLERANCE`.
//!
//! ## Dot Product Decomposition
//!
//! For quantized weight W and f32 activation x:
//! ```text
//! dot(W, x) = Σ_sb [
//!   SCALE TERM:  d * Σ_j( s_j * Σ_i( dequant(q_W_i) * x_i ) )
//! ]
//! ```
//!
//! When `has_dmin=true`, the dequantization includes the offset correction.
//! The generic implementation uses `dequant_value()` which handles all formats.

use super::format_trait::QuantBlockFormat;
use crate::error::{RealizarError, Result};

/// Generic fused dequant+dot product for any blocked quantization format.
///
/// This is the scalar reference implementation that follows the 6-phase
/// kernel structure defined in the contract:
///
/// 1. **Validate**: Check data length is a multiple of super-block size
/// 2. **Preprocess**: Verify activation length matches
/// 3. **Bsum precompute**: (implicit — scalar version folds into loop)
/// 4. **Super-block loop**: Dequantize and accumulate
/// 5. **Combine terms**: Scale and offset terms combined per-value
/// 6. **Hsum**: No SIMD reduction needed (scalar accumulator)
///
/// # Arguments
///
/// * `weight_data` - Raw quantized weight data (sequence of super-blocks)
/// * `activations` - f32 activation values (length must match quantized values)
///
/// # Returns
///
/// The dot product result as f32
///
/// # Errors
///
/// Returns error if:
/// - `weight_data` length is not a multiple of `F::SUPERBLOCK_BYTES`
/// - `activations` length doesn't match the number of quantized values
pub fn generic_fused_dot_scalar<F: QuantBlockFormat>(
    weight_data: &[u8],
    activations: &[f32],
) -> Result<f32> {
    // Phase 1: Validate
    let num_superblocks = F::validate_data_length(weight_data)?;
    let expected_values = num_superblocks * F::ELEMENTS_PER_SUPERBLOCK;

    // Phase 2: Preprocess (validate activation length)
    if activations.len() < expected_values {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "{}: activation length {} is less than expected {} values ({} super-blocks × {})",
                F::FORMAT_ID,
                activations.len(),
                expected_values,
                num_superblocks,
                F::ELEMENTS_PER_SUPERBLOCK,
            ),
        });
    }

    // Phases 3-5: Super-block loop with fused dequant+dot
    let mut acc = 0.0f32;

    for sb_idx in 0..num_superblocks {
        let sb_start = sb_idx * F::SUPERBLOCK_BYTES;
        let sb_end = sb_start + F::SUPERBLOCK_BYTES;
        let superblock = &weight_data[sb_start..sb_end];
        let act_start = sb_idx * F::ELEMENTS_PER_SUPERBLOCK;

        // Dequantize each value and accumulate dot product
        for i in 0..F::ELEMENTS_PER_SUPERBLOCK {
            let w = F::dequant_value(superblock, i);
            acc += w * activations[act_start + i];
        }
    }

    // Phase 6: Hsum (no-op for scalar)
    Ok(acc)
}

/// Compute sub-block activation sums (bsums) for formats with `has_dmin=true`.
///
/// The key mathematical insight from the contract: the offset term
/// `dmin * Σ_j(m_j * Σ_i(q_x_i))` depends ONLY on activation values,
/// not on weights. Therefore bsums can be precomputed once and reused
/// across all weight rows in a matvec.
///
/// # Arguments
///
/// * `activations` - f32 activation values
/// * `elements_per_subblock` - sub-block size (32 for Q4_K/Q5_K)
///
/// # Returns
///
/// Vector of sub-block sums, one per sub-block
#[must_use]
pub fn compute_bsums(activations: &[f32], elements_per_subblock: usize) -> Vec<f32> {
    activations
        .chunks(elements_per_subblock)
        .map(|chunk| chunk.iter().sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::format_trait::{Q4_0Fmt, Q8_0Fmt, Q4K, Q5K, Q6K};

    // =========================================================================
    // Zero-data tests (sanity check)
    // =========================================================================

    #[test]
    fn test_generic_dot_q4k_zero_data() {
        let data = vec![0u8; 144]; // 1 super-block of zeros
        let acts = vec![1.0f32; 256];
        let result = generic_fused_dot_scalar::<Q4K>(&data, &acts);
        assert!(result.is_ok());
        // Zero data means zero scales and zero values → dot = 0
        assert_eq!(result.expect("should succeed"), 0.0);
    }

    #[test]
    fn test_generic_dot_q6k_zero_data() {
        // Q6_K: d at offset 208, all zeros → dot = 0
        let data = vec![0u8; 210];
        let acts = vec![1.0f32; 256];
        let result = generic_fused_dot_scalar::<Q6K>(&data, &acts);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed"), 0.0);
    }

    #[test]
    fn test_generic_dot_q4_0_zero_data() {
        let data = vec![0u8; 18];
        let acts = vec![1.0f32; 32];
        let result = generic_fused_dot_scalar::<Q4_0Fmt>(&data, &acts);
        assert!(result.is_ok());
        // Zero scale → all dequantized values are 0
        assert_eq!(result.expect("should succeed"), 0.0);
    }

    #[test]
    fn test_generic_dot_q8_0_zero_data() {
        let data = vec![0u8; 34];
        let acts = vec![1.0f32; 32];
        let result = generic_fused_dot_scalar::<Q8_0Fmt>(&data, &acts);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed"), 0.0);
    }

    // =========================================================================
    // Validation tests
    // =========================================================================

    #[test]
    fn test_generic_dot_invalid_data_length() {
        let data = vec![0u8; 100]; // Not a multiple of 144
        let acts = vec![0.0f32; 256];
        assert!(generic_fused_dot_scalar::<Q4K>(&data, &acts).is_err());
    }

    #[test]
    fn test_generic_dot_empty_data() {
        let data = vec![];
        let acts = vec![0.0f32; 256];
        assert!(generic_fused_dot_scalar::<Q4K>(&data, &acts).is_err());
    }

    #[test]
    fn test_generic_dot_activations_too_short() {
        let data = vec![0u8; 144];
        let acts = vec![0.0f32; 100]; // Need 256
        assert!(generic_fused_dot_scalar::<Q4K>(&data, &acts).is_err());
    }

    // =========================================================================
    // Multiple super-blocks
    // =========================================================================

    #[test]
    fn test_generic_dot_multiple_superblocks() {
        let data = vec![0u8; 144 * 3]; // 3 super-blocks
        let acts = vec![0.0f32; 256 * 3];
        let result = generic_fused_dot_scalar::<Q4K>(&data, &acts);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Bsum computation
    // =========================================================================

    #[test]
    fn test_compute_bsums_q4k_subblock_size() {
        let acts: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let bsums = compute_bsums(&acts, 32);
        // 256 / 32 = 8 sub-blocks
        assert_eq!(bsums.len(), 8);
        // First sub-block: sum(0..32) = 31*32/2 = 496
        assert!((bsums[0] - 496.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_bsums_uniform() {
        let acts = vec![1.0f32; 256];
        let bsums = compute_bsums(&acts, 32);
        assert_eq!(bsums.len(), 8);
        for &b in &bsums {
            assert!((b - 32.0).abs() < 0.001);
        }
    }

    // =========================================================================
    // Q8_0 known-value test
    // =========================================================================

    #[test]
    fn test_generic_dot_q8_0_known_values() {
        // Construct a Q8_0 block with scale=1.0 and all quants=1
        let mut data = [0u8; 34];
        // f16 for 1.0 = 0x3C00 (little-endian: 0x00, 0x3C)
        data[0] = 0x00;
        data[1] = 0x3C;
        // Set all 32 quants to 1
        for i in 0..32 {
            data[2 + i] = 1;
        }

        // Activations all 1.0
        let acts = vec![1.0f32; 32];
        let result = generic_fused_dot_scalar::<Q8_0Fmt>(&data, &acts);
        assert!(result.is_ok());
        // dot = sum(1.0 * 1.0 * 1.0) for 32 values = 32.0
        let val = result.expect("should succeed");
        assert!((val - 32.0).abs() < 0.1, "Expected ~32.0, got {val}");
    }

    // =========================================================================
    // Q5_K smoke test
    // =========================================================================

    #[test]
    fn test_generic_dot_q5k_zero_data() {
        let data = vec![0u8; 176];
        let acts = vec![1.0f32; 256];
        let result = generic_fused_dot_scalar::<Q5K>(&data, &acts);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed"), 0.0);
    }
}
