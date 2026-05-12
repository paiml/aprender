// SHIP-TWO-001 — `q4k-q6k-superblock-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QS-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/q4k-q6k-superblock-v1.yaml`.

// ===========================================================================
// Canonical Q4K / Q6K superblock byte sizes (per llama.cpp ggml format)
// ===========================================================================

pub const AC_QS_001_Q4K_SUPERBLOCK_BYTES: u64 = 144;
pub const AC_QS_001_Q6K_SUPERBLOCK_BYTES: u64 = 210;
pub const AC_QS_001_QK_K: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType { Q4K, Q6K }

#[must_use]
pub const fn superblock_bytes(qt: QuantType) -> u64 {
    match qt {
        QuantType::Q4K => AC_QS_001_Q4K_SUPERBLOCK_BYTES,
        QuantType::Q6K => AC_QS_001_Q6K_SUPERBLOCK_BYTES,
    }
}

/// Total bytes for a tensor of `rows × cols` quantized as `qt`.
/// Formula: rows × ceil(cols / QK_K) × superblock_bytes.
#[must_use]
pub const fn tensor_quantized_bytes(rows: u64, cols: u64, qt: QuantType) -> u64 {
    if rows == 0 || cols == 0 { return 0; }
    let blocks_per_row = cols.div_ceil(AC_QS_001_QK_K);
    rows * blocks_per_row * superblock_bytes(qt)
}

// ===========================================================================
// QS-001 — Superblock sizes: Q4K = 144, Q6K = 210
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qs001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_superblock_size(qt: QuantType, observed: u64) -> Qs001Verdict {
    if observed == superblock_bytes(qt) { Qs001Verdict::Pass } else { Qs001Verdict::Fail }
}

// ===========================================================================
// QS-002 — Total bytes monotonic in cols
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qs002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_byte_count_monotone(rows: u64, c1: u64, c2: u64, qt: QuantType) -> Qs002Verdict {
    if rows == 0 || c1 >= c2 { return Qs002Verdict::Fail; }
    let b1 = tensor_quantized_bytes(rows, c1, qt);
    let b2 = tensor_quantized_bytes(rows, c2, qt);
    if b1 <= b2 { Qs002Verdict::Pass } else { Qs002Verdict::Fail }
}

// ===========================================================================
// QS-003 — Dequant finite for valid superblock fields
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qs003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_dequant_finite(dequantized: &[f32]) -> Qs003Verdict {
    if dequantized.is_empty() { return Qs003Verdict::Fail; }
    if dequantized.iter().all(|v| v.is_finite()) { Qs003Verdict::Pass } else { Qs003Verdict::Fail }
}

// ===========================================================================
// QS-004 — Offset vanishing: Q6K has no offset term (scale-only)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qs004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_offset_vanishing(qt: QuantType, observed_offset: f32) -> Qs004Verdict {
    match qt {
        QuantType::Q6K => {
            if observed_offset.abs() < f32::EPSILON { Qs004Verdict::Pass } else { Qs004Verdict::Fail }
        }
        QuantType::Q4K => {
            // Q4K legitimately has an offset; this gate is vacuously
            // true for non-Q6K formats.
            Qs004Verdict::Pass
        }
    }
}

// ===========================================================================
// QS-005 — bsum weight-independence: bsums depend only on activations
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qs005Verdict { Pass, Fail }

/// Pass iff `bsum_weights_a == bsum_weights_b` when computed with the
/// same activation but different weights — i.e. the bsum vector is
/// independent of the weight tensor.
#[must_use]
pub fn verdict_from_bsum_weight_independence(
    bsum_with_weights_a: &[f32],
    bsum_with_weights_b: &[f32],
) -> Qs005Verdict {
    if bsum_with_weights_a.len() != bsum_with_weights_b.len() || bsum_with_weights_a.is_empty() {
        return Qs005Verdict::Fail;
    }
    for (a, b) in bsum_with_weights_a.iter().zip(bsum_with_weights_b.iter()) {
        if a.to_bits() != b.to_bits() { return Qs005Verdict::Fail; }
    }
    Qs005Verdict::Pass
}

// ===========================================================================
// QS-006 — Byte layout consistency: superblock total bytes == documented
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qs006Verdict { Pass, Fail }

/// Pass iff the sum of `field_sizes` equals the canonical superblock
/// byte count for `qt`.
#[must_use]
pub fn verdict_from_byte_layout(qt: QuantType, field_sizes: &[u64]) -> Qs006Verdict {
    if field_sizes.is_empty() { return Qs006Verdict::Fail; }
    let sum: u64 = field_sizes.iter().sum();
    if sum == superblock_bytes(qt) { Qs006Verdict::Pass } else { Qs006Verdict::Fail }
}

// ===========================================================================
// QS-007 — SIMD vs scalar dequant equivalence: ULP bounded
// ===========================================================================

pub const AC_QS_007_MAX_ULP: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qs007Verdict { Pass, Fail }

fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if !a.is_finite() || !b.is_finite() { return None; }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    if (ai < 0) != (bi < 0) { return Some(ai.unsigned_abs() + bi.unsigned_abs()); }
    Some((ai - bi).unsigned_abs())
}

#[must_use]
pub fn verdict_from_simd_dequant_equivalence(simd: &[f32], scalar: &[f32]) -> Qs007Verdict {
    if simd.len() != scalar.len() || simd.is_empty() { return Qs007Verdict::Fail; }
    for (a, b) in simd.iter().zip(scalar.iter()) {
        match ulp_distance(*a, *b) {
            Some(d) if d < AC_QS_007_MAX_ULP => {}
            _ => return Qs007Verdict::Fail,
        }
    }
    Qs007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference impl spot checks
    #[test] fn ref_q4k_aligned_tensor_bytes() {
        // 256-col tensor, 1 row, Q4K → 144 bytes.
        assert_eq!(tensor_quantized_bytes(1, 256, QuantType::Q4K), 144);
    }
    #[test] fn ref_q4k_partial_block_rounds_up() {
        // 1 col, 1 row → 1 superblock = 144 bytes.
        assert_eq!(tensor_quantized_bytes(1, 1, QuantType::Q4K), 144);
        // 257 col → 2 superblocks = 288 bytes.
        assert_eq!(tensor_quantized_bytes(1, 257, QuantType::Q4K), 288);
    }
    #[test] fn ref_q6k_512_col_4_row() {
        // 4 rows × 2 superblocks × 210 = 1680 bytes.
        assert_eq!(tensor_quantized_bytes(4, 512, QuantType::Q6K), 1680);
    }

    // QS-001
    #[test] fn qs001_pass_q4k() {
        assert_eq!(verdict_from_superblock_size(QuantType::Q4K, 144), Qs001Verdict::Pass);
    }
    #[test] fn qs001_pass_q6k() {
        assert_eq!(verdict_from_superblock_size(QuantType::Q6K, 210), Qs001Verdict::Pass);
    }
    #[test] fn qs001_fail_q4k_drift() {
        assert_eq!(verdict_from_superblock_size(QuantType::Q4K, 143), Qs001Verdict::Fail);
    }
    #[test] fn qs001_fail_q6k_drift() {
        assert_eq!(verdict_from_superblock_size(QuantType::Q6K, 144), Qs001Verdict::Fail);
    }

    // QS-002
    #[test] fn qs002_pass_q4k() {
        assert_eq!(verdict_from_byte_count_monotone(1, 256, 512, QuantType::Q4K), Qs002Verdict::Pass);
    }
    #[test] fn qs002_pass_partial_blocks() {
        // c1=257, c2=300 — both round up to 2 superblocks.
        assert_eq!(verdict_from_byte_count_monotone(1, 257, 300, QuantType::Q4K), Qs002Verdict::Pass);
    }
    #[test] fn qs002_fail_swapped() {
        assert_eq!(verdict_from_byte_count_monotone(1, 512, 256, QuantType::Q4K), Qs002Verdict::Fail);
    }

    // QS-003
    #[test] fn qs003_pass_finite() {
        let v = vec![1.0_f32, 2.0, -3.0, 0.5];
        assert_eq!(verdict_from_dequant_finite(&v), Qs003Verdict::Pass);
    }
    #[test] fn qs003_fail_nan() {
        let v = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_dequant_finite(&v), Qs003Verdict::Fail);
    }
    #[test] fn qs003_fail_inf() {
        let v = vec![f32::INFINITY];
        assert_eq!(verdict_from_dequant_finite(&v), Qs003Verdict::Fail);
    }
    #[test] fn qs003_fail_empty() {
        assert_eq!(verdict_from_dequant_finite(&[]), Qs003Verdict::Fail);
    }

    // QS-004
    #[test] fn qs004_pass_q6k_zero() {
        assert_eq!(verdict_from_offset_vanishing(QuantType::Q6K, 0.0), Qs004Verdict::Pass);
    }
    #[test] fn qs004_fail_q6k_nonzero() {
        assert_eq!(verdict_from_offset_vanishing(QuantType::Q6K, 0.1), Qs004Verdict::Fail);
    }
    #[test] fn qs004_pass_q4k_with_offset() {
        // Q4K legitimately has offset; vacuously Pass for any value.
        assert_eq!(verdict_from_offset_vanishing(QuantType::Q4K, 0.5), Qs004Verdict::Pass);
    }

    // QS-005
    #[test] fn qs005_pass_independent() {
        let bsum_a = vec![1.0_f32, 2.0, 3.0];
        let bsum_b = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_bsum_weight_independence(&bsum_a, &bsum_b), Qs005Verdict::Pass);
    }
    #[test] fn qs005_fail_weight_dependent() {
        let bsum_a = vec![1.0_f32, 2.0];
        let bsum_b = vec![1.5_f32, 2.5];
        assert_eq!(verdict_from_bsum_weight_independence(&bsum_a, &bsum_b), Qs005Verdict::Fail);
    }
    #[test] fn qs005_fail_length_mismatch() {
        let bsum_a = vec![1.0_f32];
        let bsum_b = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_bsum_weight_independence(&bsum_a, &bsum_b), Qs005Verdict::Fail);
    }

    // QS-006
    #[test] fn qs006_pass_q4k() {
        // Sample Q4K layout: scale 12 + d/dmin 4 + qs 128 = 144.
        let fields = vec![12_u64, 4, 128];
        assert_eq!(verdict_from_byte_layout(QuantType::Q4K, &fields), Qs006Verdict::Pass);
    }
    #[test] fn qs006_pass_q6k() {
        // Sample Q6K layout: ql 128 + qh 64 + scales 16 + d 2 = 210.
        let fields = vec![128_u64, 64, 16, 2];
        assert_eq!(verdict_from_byte_layout(QuantType::Q6K, &fields), Qs006Verdict::Pass);
    }
    #[test] fn qs006_fail_short() {
        let fields = vec![100_u64];
        assert_eq!(verdict_from_byte_layout(QuantType::Q4K, &fields), Qs006Verdict::Fail);
    }
    #[test] fn qs006_fail_empty() {
        assert_eq!(verdict_from_byte_layout(QuantType::Q4K, &[]), Qs006Verdict::Fail);
    }

    // QS-007
    #[test] fn qs007_pass_exact() {
        let s = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_simd_dequant_equivalence(&s, &s), Qs007Verdict::Pass);
    }
    #[test] fn qs007_pass_within_4_ulp() {
        let scalar = [1.0_f32];
        let simd = [f32::from_bits(scalar[0].to_bits() + 3)];
        assert_eq!(verdict_from_simd_dequant_equivalence(&simd, &scalar), Qs007Verdict::Pass);
    }
    #[test] fn qs007_fail_too_far() {
        let scalar = [1.0_f32];
        let simd = [f32::from_bits(scalar[0].to_bits() + 100)];
        assert_eq!(verdict_from_simd_dequant_equivalence(&simd, &scalar), Qs007Verdict::Fail);
    }

    // Provenance pins
    #[test] fn provenance_constants() {
        assert_eq!(AC_QS_001_Q4K_SUPERBLOCK_BYTES, 144);
        assert_eq!(AC_QS_001_Q6K_SUPERBLOCK_BYTES, 210);
        assert_eq!(AC_QS_001_QK_K, 256);
        assert_eq!(AC_QS_007_MAX_ULP, 4);
    }
}
