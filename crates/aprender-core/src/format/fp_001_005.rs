// SHIP-TWO-001 — `format-parity-v1` algorithm-level PARTIAL discharge
// for FALSIFY-FP-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/format-parity-v1.yaml`.
// Spec: Cross-format tensor equivalence (GGUF, SafeTensors, APR).

// ===========================================================================
// FP-001 — Transpose involution: swap(swap([a, b])) == [a, b]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp001Verdict { Pass, Fail }

#[must_use]
pub fn swap_2d(shape: &[u64]) -> Vec<u64> {
    if shape.len() != 2 { return vec![]; }
    vec![shape[1], shape[0]]
}

#[must_use]
pub fn verdict_from_transpose_involution(shape: &[u64]) -> Fp001Verdict {
    if shape.len() != 2 { return Fp001Verdict::Fail; }
    if shape[0] == 0 || shape[1] == 0 { return Fp001Verdict::Fail; }
    let once = swap_2d(shape);
    let twice = swap_2d(&once);
    if twice == shape { Fp001Verdict::Pass } else { Fp001Verdict::Fail }
}

// ===========================================================================
// FP-002 — Element count: product(gguf_shape) == product(apr_shape)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp002Verdict { Pass, Fail }

#[must_use]
pub fn shape_product(shape: &[u64]) -> Option<u64> {
    if shape.is_empty() { return None; }
    let mut p: u64 = 1;
    for &d in shape {
        if d == 0 { return None; }
        p = p.checked_mul(d)?;
    }
    Some(p)
}

#[must_use]
pub fn verdict_from_element_count_preservation(
    gguf_shape: &[u64],
    apr_shape: &[u64],
) -> Fp002Verdict {
    let g = match shape_product(gguf_shape) {
        Some(p) => p,
        None => return Fp002Verdict::Fail,
    };
    let a = match shape_product(apr_shape) {
        Some(p) => p,
        None => return Fp002Verdict::Fail,
    };
    if g == a { Fp002Verdict::Pass } else { Fp002Verdict::Fail }
}

// ===========================================================================
// FP-003 — 1D identity: len(shape) == 1 ⟹ apr_shape == gguf_shape
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_identity_1d(gguf_shape: &[u64], apr_shape: &[u64]) -> Fp003Verdict {
    if gguf_shape.len() != 1 || apr_shape.len() != 1 { return Fp003Verdict::Fail; }
    if gguf_shape[0] == 0 || apr_shape[0] == 0 { return Fp003Verdict::Fail; }
    if gguf_shape == apr_shape { Fp003Verdict::Pass } else { Fp003Verdict::Fail }
}

// ===========================================================================
// FP-004 — Roundtrip: GGUF → APR → GGUF preserves shape byte-exactly
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_roundtrip_shape(
    original_gguf: &[u64],
    apr_intermediate: &[u64],
    final_gguf: &[u64],
) -> Fp004Verdict {
    if original_gguf.is_empty() || apr_intermediate.is_empty() || final_gguf.is_empty() {
        return Fp004Verdict::Fail;
    }
    // Element count must be preserved at every step.
    let orig_count = match shape_product(original_gguf) {
        Some(p) => p,
        None => return Fp004Verdict::Fail,
    };
    let mid_count = match shape_product(apr_intermediate) {
        Some(p) => p,
        None => return Fp004Verdict::Fail,
    };
    let final_count = match shape_product(final_gguf) {
        Some(p) => p,
        None => return Fp004Verdict::Fail,
    };
    if orig_count != mid_count || mid_count != final_count {
        return Fp004Verdict::Fail;
    }
    // Final shape must equal original shape byte-exactly.
    if original_gguf == final_gguf { Fp004Verdict::Pass } else { Fp004Verdict::Fail }
}

// ===========================================================================
// FP-005 — SIMD format parity: byte-exact (contract tolerance=0.0)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp005Verdict { Pass, Fail }

/// Pass iff the scalar and SIMD format-conversion outputs are byte-exact
/// (the contract specifies tolerance=0.0 for SIMD format equivalence).
#[must_use]
pub fn verdict_from_simd_format_parity(scalar: &[u64], simd: &[u64]) -> Fp005Verdict {
    if scalar.is_empty() || simd.is_empty() { return Fp005Verdict::Fail; }
    if scalar.len() != simd.len() { return Fp005Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if s != v { return Fp005Verdict::Fail; }
    }
    Fp005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // FP-001 (transpose involution)
    #[test] fn fp001_pass_canonical() {
        let shape = vec![4096_u64, 4096];
        assert_eq!(verdict_from_transpose_involution(&shape), Fp001Verdict::Pass);
    }
    #[test] fn fp001_pass_rectangular() {
        let shape = vec![152064_u64, 4096]; // lm_head: vocab × hidden
        assert_eq!(verdict_from_transpose_involution(&shape), Fp001Verdict::Pass);
    }
    #[test] fn fp001_fail_non_2d() {
        let shape = vec![1024_u64];
        assert_eq!(verdict_from_transpose_involution(&shape), Fp001Verdict::Fail);
    }
    #[test] fn fp001_fail_3d() {
        let shape = vec![16_u64, 16, 64];
        assert_eq!(verdict_from_transpose_involution(&shape), Fp001Verdict::Fail);
    }
    #[test] fn fp001_fail_zero_dim() {
        let shape = vec![0_u64, 4096];
        assert_eq!(verdict_from_transpose_involution(&shape), Fp001Verdict::Fail);
    }
    #[test] fn swap_2d_reverses() {
        assert_eq!(swap_2d(&[3, 5]), vec![5, 3]);
        assert_eq!(swap_2d(&[1024, 2048]), vec![2048, 1024]);
    }

    // FP-002 (element count)
    #[test] fn fp002_pass_same_product() {
        // Transposed shape: same product, different layout.
        let gguf = vec![4096_u64, 11008];
        let apr = vec![11008_u64, 4096];
        assert_eq!(verdict_from_element_count_preservation(&gguf, &apr), Fp002Verdict::Pass);
    }
    #[test] fn fp002_pass_reshaped() {
        // Different rank, same product.
        let gguf = vec![32_u64, 32];
        let apr = vec![1024_u64];
        assert_eq!(verdict_from_element_count_preservation(&gguf, &apr), Fp002Verdict::Pass);
    }
    #[test] fn fp002_fail_drift() {
        let gguf = vec![4096_u64, 4096];
        let apr = vec![4096_u64, 4097]; // off by one
        assert_eq!(verdict_from_element_count_preservation(&gguf, &apr), Fp002Verdict::Fail);
    }
    #[test] fn fp002_fail_overflow() {
        let gguf = vec![u64::MAX, 2];
        let apr = vec![1_u64];
        assert_eq!(verdict_from_element_count_preservation(&gguf, &apr), Fp002Verdict::Fail);
    }
    #[test] fn fp002_fail_empty() {
        assert_eq!(verdict_from_element_count_preservation(&[], &[]), Fp002Verdict::Fail);
    }

    // FP-003 (1D identity)
    #[test] fn fp003_pass_canonical() {
        // Bias vector: same in both formats.
        let s = vec![4096_u64];
        assert_eq!(verdict_from_identity_1d(&s, &s), Fp003Verdict::Pass);
    }
    #[test] fn fp003_fail_drift() {
        let gguf = vec![4096_u64];
        let apr = vec![4097_u64];
        assert_eq!(verdict_from_identity_1d(&gguf, &apr), Fp003Verdict::Fail);
    }
    #[test] fn fp003_fail_2d() {
        let gguf = vec![4096_u64, 1];
        let apr = vec![4096_u64];
        // Rejected: not both 1D.
        assert_eq!(verdict_from_identity_1d(&gguf, &apr), Fp003Verdict::Fail);
    }
    #[test] fn fp003_fail_zero() {
        assert_eq!(verdict_from_identity_1d(&[0_u64], &[0_u64]), Fp003Verdict::Fail);
    }

    // FP-004 (roundtrip)
    #[test] fn fp004_pass_canonical() {
        // GGUF [4096, 11008] → APR [11008, 4096] → GGUF [4096, 11008].
        let original = vec![4096_u64, 11008];
        let intermediate = vec![11008_u64, 4096];
        let final_ = vec![4096_u64, 11008];
        assert_eq!(
            verdict_from_roundtrip_shape(&original, &intermediate, &final_),
            Fp004Verdict::Pass
        );
    }
    #[test] fn fp004_fail_element_count_drift() {
        let original = vec![4096_u64, 11008];
        let intermediate = vec![11008_u64, 4096];
        let final_ = vec![4097_u64, 11008]; // count drifted
        assert_eq!(
            verdict_from_roundtrip_shape(&original, &intermediate, &final_),
            Fp004Verdict::Fail
        );
    }
    #[test] fn fp004_fail_shape_mismatch() {
        // Element count preserved but shape didn't roundtrip.
        let original = vec![4096_u64, 11008];
        let intermediate = vec![11008_u64, 4096];
        let final_ = vec![11008_u64, 4096]; // didn't transpose back
        assert_eq!(
            verdict_from_roundtrip_shape(&original, &intermediate, &final_),
            Fp004Verdict::Fail
        );
    }
    #[test] fn fp004_fail_intermediate_count_drift() {
        let original = vec![4096_u64, 11008];
        let intermediate = vec![4096_u64, 11009]; // wrong intermediate
        let final_ = vec![4096_u64, 11008];
        assert_eq!(
            verdict_from_roundtrip_shape(&original, &intermediate, &final_),
            Fp004Verdict::Fail
        );
    }
    #[test] fn fp004_fail_empty() {
        assert_eq!(verdict_from_roundtrip_shape(&[], &[], &[]), Fp004Verdict::Fail);
    }

    // FP-005 (SIMD parity)
    #[test] fn fp005_pass_identical() {
        let shape = vec![1024_u64, 4096];
        assert_eq!(verdict_from_simd_format_parity(&shape, &shape), Fp005Verdict::Pass);
    }
    #[test] fn fp005_fail_drift() {
        let scalar = vec![1024_u64, 4096];
        let simd = vec![1024_u64, 4097];
        assert_eq!(verdict_from_simd_format_parity(&scalar, &simd), Fp005Verdict::Fail);
    }
    #[test] fn fp005_fail_length() {
        let scalar = vec![1024_u64];
        let simd = vec![1024_u64, 4096];
        assert_eq!(verdict_from_simd_format_parity(&scalar, &simd), Fp005Verdict::Fail);
    }
    #[test] fn fp005_fail_empty() {
        assert_eq!(verdict_from_simd_format_parity(&[], &[]), Fp005Verdict::Fail);
    }

    // shape_product helper
    #[test] fn shape_product_canonical() {
        assert_eq!(shape_product(&[4_u64, 5, 6]), Some(120));
    }
    #[test] fn shape_product_zero_dim_returns_none() {
        assert_eq!(shape_product(&[4_u64, 0]), None);
    }
    #[test] fn shape_product_overflow_returns_none() {
        assert_eq!(shape_product(&[u64::MAX, 2]), None);
    }
}
