// SHIP-TWO-001 — `tensor-inventory-v1` algorithm-level PARTIAL
// discharge for FALSIFY-TI-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/tensor-inventory-v1.yaml`.
// Spec: Tensor inventory algebra and parameter count decomposition
// (Qwen3 Performance Parity Spec; Vaswani 2017 parameter analysis).

// ===========================================================================
// TI-001 — Tensor count formula: total = base + L * per_layer
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ti001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_tensor_count(
    base: u64,
    num_layers: u64,
    per_layer: u64,
    observed_total: u64,
) -> Ti001Verdict {
    if num_layers == 0 || per_layer == 0 { return Ti001Verdict::Fail; }
    if base == 0 { return Ti001Verdict::Fail; }
    let expected = base + num_layers * per_layer;
    if observed_total == expected { Ti001Verdict::Pass } else { Ti001Verdict::Fail }
}

// ===========================================================================
// TI-002 — Architecture delta: delta(A, B) = L * (per_layer_B - per_layer_A)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ti002Verdict { Pass, Fail }

/// Pass iff delta computed by formula matches observed delta. Both per-layer
/// values are u64; delta is signed (i64). Bases must be equal (no
/// architectural change at the top-level).
#[must_use]
pub const fn verdict_from_architecture_delta(
    base_a: u64,
    base_b: u64,
    num_layers: u64,
    per_layer_a: u64,
    per_layer_b: u64,
    observed_delta: i64,
) -> Ti002Verdict {
    if num_layers == 0 { return Ti002Verdict::Fail; }
    if base_a != base_b { return Ti002Verdict::Fail; } // delta predicate scoped to per-layer changes
    let expected = (per_layer_b as i64 - per_layer_a as i64) * (num_layers as i64);
    if observed_delta == expected { Ti002Verdict::Pass } else { Ti002Verdict::Fail }
}

// ===========================================================================
// TI-003 — Quantization byte ordering: Q4K < Q6K < Q8 < F16 < F32 (same params)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ti003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_quant_byte_ordering(
    q4k: u64,
    q6k: u64,
    q8: u64,
    f16: u64,
    f32_: u64,
) -> Ti003Verdict {
    if q4k == 0 || q6k == 0 || q8 == 0 || f16 == 0 || f32_ == 0 {
        return Ti003Verdict::Fail;
    }
    if q4k < q6k && q6k < q8 && q8 < f16 && f16 < f32_ {
        Ti003Verdict::Pass
    } else {
        Ti003Verdict::Fail
    }
}

// ===========================================================================
// TI-004 — Parameter decomposition: sum(per-layer) + embed + head == total
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ti004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_param_decomposition(
    embed_params: u64,
    layer_params: &[u64],
    head_params: u64,
    observed_total: u64,
) -> Ti004Verdict {
    if layer_params.is_empty() { return Ti004Verdict::Fail; }
    if embed_params == 0 || head_params == 0 { return Ti004Verdict::Fail; }
    let mut sum: u64 = embed_params.checked_add(head_params).unwrap_or(0);
    if sum == 0 { return Ti004Verdict::Fail; }
    for &lp in layer_params {
        if lp == 0 { return Ti004Verdict::Fail; }
        sum = match sum.checked_add(lp) {
            Some(s) => s,
            None => return Ti004Verdict::Fail,
        };
    }
    if sum == observed_total { Ti004Verdict::Pass } else { Ti004Verdict::Fail }
}

// ===========================================================================
// TI-005 — Tied embedding count: tied=true ⇒ count(untied) - count(tied) == 1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ti005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_tied_embedding_count(
    untied_tensor_count: u64,
    tied_tensor_count: u64,
) -> Ti005Verdict {
    if untied_tensor_count == 0 || tied_tensor_count == 0 { return Ti005Verdict::Fail; }
    if untied_tensor_count <= tied_tensor_count { return Ti005Verdict::Fail; }
    if untied_tensor_count - tied_tensor_count == 1 { Ti005Verdict::Pass } else { Ti005Verdict::Fail }
}

// ===========================================================================
// TI-006 — SIMD inventory parity: byte-exact (contract tolerance=0.0)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ti006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_inventory_parity(scalar: &[u64], simd: &[u64]) -> Ti006Verdict {
    if scalar.is_empty() || simd.is_empty() { return Ti006Verdict::Fail; }
    if scalar.len() != simd.len() { return Ti006Verdict::Fail; }
    for (&s, &v) in scalar.iter().zip(simd.iter()) {
        if s != v { return Ti006Verdict::Fail; }
    }
    Ti006Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // TI-001 (tensor count formula)
    #[test] fn ti001_pass_canonical() {
        // base=3 (embed + lm_head + final_norm), L=28, per_layer=12.
        // Total = 3 + 28 * 12 = 339.
        assert_eq!(verdict_from_tensor_count(3, 28, 12, 339), Ti001Verdict::Pass);
    }
    #[test] fn ti001_fail_omitted_bias() {
        // The contract's stated falsifier: "Omit a bias tensor to break count"
        // → per-layer count off by 1 → total off by L.
        assert_eq!(verdict_from_tensor_count(3, 28, 12, 311), Ti001Verdict::Fail);
    }
    #[test] fn ti001_fail_zero_layers() {
        assert_eq!(verdict_from_tensor_count(3, 0, 12, 3), Ti001Verdict::Fail);
    }
    #[test] fn ti001_fail_zero_per_layer() {
        assert_eq!(verdict_from_tensor_count(3, 28, 0, 3), Ti001Verdict::Fail);
    }

    // TI-002 (architecture delta)
    #[test] fn ti002_pass_zero_delta() {
        // Same architectures → delta = 0.
        assert_eq!(
            verdict_from_architecture_delta(3, 3, 28, 12, 12, 0),
            Ti002Verdict::Pass
        );
    }
    #[test] fn ti002_pass_canonical_delta() {
        // L=28, per_layer differs by 2 → delta = 56.
        assert_eq!(
            verdict_from_architecture_delta(3, 3, 28, 12, 14, 56),
            Ti002Verdict::Pass
        );
    }
    #[test] fn ti002_pass_negative_delta() {
        // Architecture B has fewer per-layer tensors than A.
        assert_eq!(
            verdict_from_architecture_delta(3, 3, 28, 14, 12, -56),
            Ti002Verdict::Pass
        );
    }
    #[test] fn ti002_fail_wrong_delta() {
        assert_eq!(
            verdict_from_architecture_delta(3, 3, 28, 12, 14, 100),
            Ti002Verdict::Fail
        );
    }
    #[test] fn ti002_fail_base_mismatch() {
        // Different base → predicate doesn't apply.
        assert_eq!(
            verdict_from_architecture_delta(3, 4, 28, 12, 12, 1),
            Ti002Verdict::Fail
        );
    }

    // TI-003 (quantization byte ordering)
    #[test] fn ti003_pass_canonical() {
        // For 1B params: Q4K=500MB, Q6K=750MB, Q8=1GB, F16=2GB, F32=4GB.
        assert_eq!(
            verdict_from_quant_byte_ordering(500, 750, 1000, 2000, 4000),
            Ti003Verdict::Pass
        );
    }
    #[test] fn ti003_fail_q4k_above_q6k() {
        assert_eq!(
            verdict_from_quant_byte_ordering(800, 750, 1000, 2000, 4000),
            Ti003Verdict::Fail
        );
    }
    #[test] fn ti003_fail_q8_above_f16() {
        // Q8 must be < F16. If Q8 reports more bytes than F16, formula wrong.
        assert_eq!(
            verdict_from_quant_byte_ordering(500, 750, 3000, 2000, 4000),
            Ti003Verdict::Fail
        );
    }
    #[test] fn ti003_fail_f16_above_f32() {
        assert_eq!(
            verdict_from_quant_byte_ordering(500, 750, 1000, 5000, 4000),
            Ti003Verdict::Fail
        );
    }
    #[test] fn ti003_fail_zero() {
        assert_eq!(
            verdict_from_quant_byte_ordering(0, 750, 1000, 2000, 4000),
            Ti003Verdict::Fail
        );
    }

    // TI-004 (parameter decomposition)
    #[test] fn ti004_pass_canonical() {
        // embed=100M, layer params summed=600M, head=100M → total=800M.
        let layers = vec![300_000_000_u64, 300_000_000];
        assert_eq!(
            verdict_from_param_decomposition(100_000_000, &layers, 100_000_000, 800_000_000),
            Ti004Verdict::Pass
        );
    }
    #[test] fn ti004_pass_uniform_layers() {
        // 28 identical layers of 100M each = 2.8B + embed + head.
        let layers = vec![100_000_000_u64; 28];
        let total = 100_000_000 + 100_000_000 + 28 * 100_000_000;
        assert_eq!(
            verdict_from_param_decomposition(100_000_000, &layers, 100_000_000, total),
            Ti004Verdict::Pass
        );
    }
    #[test] fn ti004_fail_drift() {
        // Sum doesn't match observed total — missing tensor in decomposition.
        let layers = vec![300_000_000_u64];
        assert_eq!(
            verdict_from_param_decomposition(100_000_000, &layers, 100_000_000, 999_999_999),
            Ti004Verdict::Fail
        );
    }
    #[test] fn ti004_fail_empty_layers() {
        assert_eq!(
            verdict_from_param_decomposition(100_000_000, &[], 100_000_000, 200_000_000),
            Ti004Verdict::Fail
        );
    }
    #[test] fn ti004_fail_zero_embed() {
        let layers = vec![100_000_000_u64];
        assert_eq!(
            verdict_from_param_decomposition(0, &layers, 100_000_000, 200_000_000),
            Ti004Verdict::Fail
        );
    }

    // TI-005 (tied embedding count)
    #[test] fn ti005_pass_canonical() {
        // Untied: 339 tensors; tied: 338 (lm_head shares with embed).
        assert_eq!(verdict_from_tied_embedding_count(339, 338), Ti005Verdict::Pass);
    }
    #[test] fn ti005_fail_no_reduction() {
        // Tied claim but count didn't decrease — tied weights double-counted.
        assert_eq!(verdict_from_tied_embedding_count(339, 339), Ti005Verdict::Fail);
    }
    #[test] fn ti005_fail_too_much_reduction() {
        // Tied claim reduced count by 2 — possibly tied AND missing another tensor.
        assert_eq!(verdict_from_tied_embedding_count(339, 337), Ti005Verdict::Fail);
    }
    #[test] fn ti005_fail_tied_above_untied() {
        // Tied count cannot exceed untied count.
        assert_eq!(verdict_from_tied_embedding_count(338, 339), Ti005Verdict::Fail);
    }
    #[test] fn ti005_fail_zero() {
        assert_eq!(verdict_from_tied_embedding_count(0, 0), Ti005Verdict::Fail);
    }

    // TI-006 (SIMD inventory parity)
    #[test] fn ti006_pass_identical() {
        let inv = vec![339_u64, 28, 1, 152064];
        assert_eq!(verdict_from_simd_inventory_parity(&inv, &inv), Ti006Verdict::Pass);
    }
    #[test] fn ti006_fail_drift() {
        let scalar = vec![339_u64, 28];
        let simd = vec![338_u64, 28];
        assert_eq!(verdict_from_simd_inventory_parity(&scalar, &simd), Ti006Verdict::Fail);
    }
    #[test] fn ti006_fail_length() {
        let scalar = vec![339_u64];
        let simd = vec![339_u64, 28];
        assert_eq!(verdict_from_simd_inventory_parity(&scalar, &simd), Ti006Verdict::Fail);
    }
    #[test] fn ti006_fail_empty() {
        assert_eq!(verdict_from_simd_inventory_parity(&[], &[]), Ti006Verdict::Fail);
    }
}
