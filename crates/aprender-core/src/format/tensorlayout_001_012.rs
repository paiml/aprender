// `tensor-layout-v1` algorithm-level PARTIAL discharge for
// FALSIFY-001..012.
//
// Contract: `contracts/tensor-layout-v1.yaml` (the canonical
// source-of-truth contract for tensor layout invariants).
//
// 12 falsification gates:
//   001: density — embedding rejected when > 50% zeros
//   002: type-enforcement — AprTransformer cannot bypass ValidatedEmbedding
//   003: NaN rejection — ValidatedWeight rejects any NaN
//   004: spot check — 94.5% leading zeros rejected (PMAT-234 bug)
//   005: lm_head shape — must be [vocab, hidden]
//   006: cross-crate parity — aprender + realizar agree on validation
//   007: quant dispatch exhaustive — no `_ =>` catchall
//   008: wrong-kernel detection — Q6K via Q4K kernel = garbage
//   009: SafeTensors→APR Q4K roundtrip == F32 inference
//   010: APR self-contained tokenizer — embedded BPE present
//   011: APR GPU within 20% of GGUF GPU
//   012: SafeTensors GPU within 20% of GGUF GPU

/// FALSIFY-001: maximum allowed zero-fraction for ValidatedEmbedding.
pub const AC_TL_MAX_ZERO_FRACTION: f32 = 0.50;
/// FALSIFY-004: tighter spot-check fraction (PMAT-234 used 94.5%).
pub const AC_TL_SPOT_CHECK_MAX_ZERO_FRACTION: f32 = 0.50;
/// FALSIFY-011/012: GPU throughput parity slack (within 20%).
pub const AC_TL_GPU_PARITY_FLOOR: f32 = 0.80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlVerdict {
    Pass,
    Fail,
}

/// FALSIFY-001: embedding density gate.
///
/// `zero_count` = count of f32 elements bit-equal to 0.0.
/// `total` = total element count.
/// Pass iff `zero_count / total <= AC_TL_MAX_ZERO_FRACTION`.
#[must_use]
pub fn verdict_from_embedding_density(zero_count: u64, total: u64) -> TlVerdict {
    if total == 0 {
        return TlVerdict::Fail;
    }
    let frac = zero_count as f32 / total as f32;
    if frac <= AC_TL_MAX_ZERO_FRACTION {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-002: type enforcement — caller must use ValidatedEmbedding.
///
/// `path_uses_validated_embedding` is true when AprTransformer is
/// constructed via the ValidatedEmbedding type. Algorithm-level
/// captures the boolean predicate; the actual mistake-proofing is
/// the type system itself, which we cannot exercise at runtime.
#[must_use]
pub fn verdict_from_type_enforcement(path_uses_validated_embedding: bool) -> TlVerdict {
    if path_uses_validated_embedding {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-003: NaN rejection — every f32 element must be finite.
#[must_use]
pub fn verdict_from_nan_rejection(weights: &[f32]) -> TlVerdict {
    if weights.is_empty() {
        return TlVerdict::Fail;
    }
    if weights.iter().all(|w| w.is_finite()) {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-004: PMAT-234 spot check — leading-zeros density.
///
/// `leading_zero_fraction` is the fraction of leading rows that are
/// all-zero. Pass iff `<= AC_TL_SPOT_CHECK_MAX_ZERO_FRACTION`.
#[must_use]
pub fn verdict_from_spot_check_density(leading_zero_fraction: f32) -> TlVerdict {
    if !leading_zero_fraction.is_finite() {
        return TlVerdict::Fail;
    }
    if !(0.0..=1.0).contains(&leading_zero_fraction) {
        return TlVerdict::Fail;
    }
    if leading_zero_fraction <= AC_TL_SPOT_CHECK_MAX_ZERO_FRACTION {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-005: lm_head shape MUST be `[vocab, hidden]`.
#[must_use]
pub fn verdict_from_lm_head_shape(
    rows: usize,
    cols: usize,
    expected_vocab: usize,
    expected_hidden: usize,
) -> TlVerdict {
    if rows == 0 || cols == 0 || expected_vocab == 0 || expected_hidden == 0 {
        return TlVerdict::Fail;
    }
    if rows == expected_vocab && cols == expected_hidden {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-006: aprender vs realizar validation must agree.
#[must_use]
pub fn verdict_from_cross_crate_parity(
    aprender_result_is_err: bool,
    realizar_result_is_err: bool,
    aprender_msg: &str,
    realizar_msg: &str,
) -> TlVerdict {
    if aprender_result_is_err != realizar_result_is_err {
        return TlVerdict::Fail;
    }
    if aprender_msg != realizar_msg {
        return TlVerdict::Fail;
    }
    TlVerdict::Pass
}

/// FALSIFY-007: every dispatch site must be exhaustive on WeightQuantType.
///
/// `dispatch_sites_with_catchall` = number of sites containing `_ =>`.
/// Pass iff equal to 0.
#[must_use]
pub fn verdict_from_dispatch_exhaustive(dispatch_sites_with_catchall: u32) -> TlVerdict {
    if dispatch_sites_with_catchall == 0 {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-008: Q6K through Q4K kernel must produce garbage.
///
/// `output_max_abs` is the max absolute value of the kernel output.
/// `output_has_nan` true if any element is NaN.
/// Pass iff `output_has_nan` OR `output_max_abs > 1e6` (clearly garbage).
#[must_use]
pub fn verdict_from_wrong_kernel_garbage(
    output_max_abs: f32,
    output_has_nan: bool,
) -> TlVerdict {
    if output_has_nan {
        return TlVerdict::Pass;
    }
    if !output_max_abs.is_finite() {
        return TlVerdict::Pass;
    }
    if output_max_abs > 1e6 {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-009: SafeTensors→APR Q4K roundtrip — output coherent.
///
/// `f32_output_token_ids` and `q4k_output_token_ids` are the first-N
/// generated tokens. Pass iff slices have equal length, are non-empty,
/// and have ≥ 70% prefix match (allowing slight quantization drift).
#[must_use]
pub fn verdict_from_q4k_roundtrip_coherent(
    f32_output: &[u32],
    q4k_output: &[u32],
) -> TlVerdict {
    if f32_output.is_empty() || q4k_output.is_empty() || f32_output.len() != q4k_output.len() {
        return TlVerdict::Fail;
    }
    let matches: u32 = f32_output
        .iter()
        .zip(q4k_output.iter())
        .map(|(a, b)| if a == b { 1 } else { 0 })
        .sum();
    let frac = matches as f32 / f32_output.len() as f32;
    if frac >= 0.70 {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

/// FALSIFY-010: APR self-contained tokenizer present.
#[must_use]
pub fn verdict_from_embedded_tokenizer(
    has_embedded_tokenizer: bool,
    embedded_vocab_size: u32,
) -> TlVerdict {
    if !has_embedded_tokenizer {
        return TlVerdict::Fail;
    }
    if embedded_vocab_size == 0 {
        return TlVerdict::Fail;
    }
    TlVerdict::Pass
}

/// FALSIFY-011/012: GPU within 20% of GGUF GPU throughput.
#[must_use]
pub fn verdict_from_gpu_parity(gguf_tps: f32, candidate_tps: f32) -> TlVerdict {
    if !gguf_tps.is_finite() || !candidate_tps.is_finite() {
        return TlVerdict::Fail;
    }
    if gguf_tps <= 0.0 || candidate_tps <= 0.0 {
        return TlVerdict::Fail;
    }
    if candidate_tps >= AC_TL_GPU_PARITY_FLOOR * gguf_tps {
        TlVerdict::Pass
    } else {
        TlVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_TL_MAX_ZERO_FRACTION, 0.50);
        assert_eq!(AC_TL_SPOT_CHECK_MAX_ZERO_FRACTION, 0.50);
        assert_eq!(AC_TL_GPU_PARITY_FLOOR, 0.80);
    }

    // -----------------------------------------------------------------
    // Section 2: FALSIFY-001..004 density / type / NaN / spot-check.
    // -----------------------------------------------------------------
    #[test]
    fn ftl001_pass_under_50pct_zeros() {
        let v = verdict_from_embedding_density(40, 100);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl001_fail_94_5_pct_zeros() {
        // PMAT-234 regression: 94.5% zeros
        let v = verdict_from_embedding_density(945, 1000);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl001_fail_100_pct_zeros() {
        let v = verdict_from_embedding_density(1000, 1000);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl001_fail_zero_total() {
        let v = verdict_from_embedding_density(0, 0);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl002_pass_path_uses_validated() {
        let v = verdict_from_type_enforcement(true);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl002_fail_path_bypasses_validated() {
        let v = verdict_from_type_enforcement(false);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl003_pass_all_finite() {
        let v = verdict_from_nan_rejection(&[1.0, 2.0, -3.0, 0.0]);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl003_fail_one_nan() {
        let v = verdict_from_nan_rejection(&[1.0, f32::NAN, 3.0]);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl003_fail_infinity() {
        let v = verdict_from_nan_rejection(&[1.0, f32::INFINITY]);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl003_fail_empty() {
        let v = verdict_from_nan_rejection(&[]);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl004_pass_30_pct_leading() {
        let v = verdict_from_spot_check_density(0.30);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl004_fail_pmat_234_value() {
        let v = verdict_from_spot_check_density(0.945);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl004_fail_negative() {
        let v = verdict_from_spot_check_density(-0.1);
        assert_eq!(v, TlVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: FALSIFY-005..007 lm_head / cross-crate / dispatch.
    // -----------------------------------------------------------------
    #[test]
    fn ftl005_pass_qwen2_lm_head() {
        let v = verdict_from_lm_head_shape(151_936, 896, 151_936, 896);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl005_fail_transposed() {
        // [hidden, vocab] is wrong
        let v = verdict_from_lm_head_shape(896, 151_936, 151_936, 896);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl005_fail_zero_dim() {
        let v = verdict_from_lm_head_shape(0, 896, 151_936, 896);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl006_pass_both_err_same_msg() {
        let v = verdict_from_cross_crate_parity(true, true, "DENSITY", "DENSITY");
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl006_pass_both_ok() {
        let v = verdict_from_cross_crate_parity(false, false, "", "");
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl006_fail_one_ok_other_err() {
        let v = verdict_from_cross_crate_parity(true, false, "DENSITY", "");
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl006_fail_msg_drift() {
        let v = verdict_from_cross_crate_parity(true, true, "DENSITY", "TOO MANY ZEROS");
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl007_pass_no_catchall() {
        let v = verdict_from_dispatch_exhaustive(0);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl007_fail_one_catchall() {
        let v = verdict_from_dispatch_exhaustive(1);
        assert_eq!(v, TlVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: FALSIFY-008..010 wrong-kernel / roundtrip / tokenizer.
    // -----------------------------------------------------------------
    #[test]
    fn ftl008_pass_garbage_huge_magnitude() {
        let v = verdict_from_wrong_kernel_garbage(1e10, false);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl008_pass_garbage_nan() {
        let v = verdict_from_wrong_kernel_garbage(0.5, true);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl008_fail_clean_output() {
        // Kernels that DON'T produce garbage on wrong-format input
        // are themselves a contract violation per FALSIFY-008.
        let v = verdict_from_wrong_kernel_garbage(0.5, false);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl009_pass_8_of_8_match() {
        let v = verdict_from_q4k_roundtrip_coherent(&[1, 2, 3, 4, 5, 6, 7, 8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl009_pass_70pct_match() {
        // 7 of 10 match = 70%
        let v = verdict_from_q4k_roundtrip_coherent(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[1, 2, 3, 4, 5, 6, 7, 99, 99, 99],
        );
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl009_fail_below_70pct() {
        // 5 of 10 = 50% — below threshold
        let v = verdict_from_q4k_roundtrip_coherent(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[1, 2, 3, 4, 5, 99, 99, 99, 99, 99],
        );
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl010_pass_embedded_present() {
        let v = verdict_from_embedded_tokenizer(true, 151_936);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl010_fail_no_embedded() {
        let v = verdict_from_embedded_tokenizer(false, 0);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl010_fail_zero_vocab() {
        let v = verdict_from_embedded_tokenizer(true, 0);
        assert_eq!(v, TlVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: FALSIFY-011..012 GPU parity.
    // -----------------------------------------------------------------
    #[test]
    fn ftl011_pass_within_20pct() {
        let v = verdict_from_gpu_parity(100.0, 85.0); // 85% of GGUF
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl011_pass_at_threshold() {
        let v = verdict_from_gpu_parity(100.0, 80.0);
        assert_eq!(v, TlVerdict::Pass);
    }

    #[test]
    fn ftl011_fail_below_threshold() {
        let v = verdict_from_gpu_parity(100.0, 79.99);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl011_fail_380x_slower() {
        // GH-87 regression class
        let v = verdict_from_gpu_parity(380.0, 1.0);
        assert_eq!(v, TlVerdict::Fail);
    }

    #[test]
    fn ftl011_fail_nan() {
        let v = verdict_from_gpu_parity(f32::NAN, 100.0);
        assert_eq!(v, TlVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_density_band() {
        let total: u64 = 1000;
        for pct in [40_u64, 45, 50, 51, 60, 90, 95] {
            let zeros = total * pct / 100;
            let v = verdict_from_embedding_density(zeros, total);
            let want = if pct <= 50 {
                TlVerdict::Pass
            } else {
                TlVerdict::Fail
            };
            assert_eq!(v, want, "pct={pct}");
        }
    }

    #[test]
    fn mutation_survey_011_parity_band() {
        let gguf = 100.0_f32;
        for ratio_x100 in [70_u32, 79, 80, 81, 90, 100, 200] {
            let cand = gguf * (ratio_x100 as f32 / 100.0);
            let v = verdict_from_gpu_parity(gguf, cand);
            let want = if cand >= AC_TL_GPU_PARITY_FLOOR * gguf {
                TlVerdict::Pass
            } else {
                TlVerdict::Fail
            };
            assert_eq!(v, want, "ratio={ratio_x100}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_12() {
        let v1 = verdict_from_embedding_density(40, 100);
        let v2 = verdict_from_type_enforcement(true);
        let v3 = verdict_from_nan_rejection(&[0.1, 0.2]);
        let v4 = verdict_from_spot_check_density(0.20);
        let v5 = verdict_from_lm_head_shape(151_936, 896, 151_936, 896);
        let v6 = verdict_from_cross_crate_parity(true, true, "DENSITY", "DENSITY");
        let v7 = verdict_from_dispatch_exhaustive(0);
        let v8 = verdict_from_wrong_kernel_garbage(1e10, false);
        let v9 = verdict_from_q4k_roundtrip_coherent(&[1, 2, 3, 4], &[1, 2, 3, 4]);
        let v10 = verdict_from_embedded_tokenizer(true, 151_936);
        let v11 = verdict_from_gpu_parity(440.0, 432.0); // ~98%
        let v12 = verdict_from_gpu_parity(440.0, 360.0); // ~82%
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12] {
            assert_eq!(v, TlVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_12_failures() {
        // Regression class — every gate trips simultaneously.
        let v1 = verdict_from_embedding_density(945, 1000); // PMAT-234
        let v2 = verdict_from_type_enforcement(false);
        let v3 = verdict_from_nan_rejection(&[1.0, f32::NAN]);
        let v4 = verdict_from_spot_check_density(0.945);
        let v5 = verdict_from_lm_head_shape(896, 151_936, 151_936, 896); // transposed
        let v6 = verdict_from_cross_crate_parity(true, false, "DENSITY", "");
        let v7 = verdict_from_dispatch_exhaustive(3);
        let v8 = verdict_from_wrong_kernel_garbage(0.5, false); // clean despite wrong kernel
        let v9 = verdict_from_q4k_roundtrip_coherent(&[1, 2, 3, 4], &[99, 99, 99, 99]);
        let v10 = verdict_from_embedded_tokenizer(false, 0);
        let v11 = verdict_from_gpu_parity(380.0, 1.0); // GH-87 380x slower
        let v12 = verdict_from_gpu_parity(380.0, 30.0); // GH-88 13x slower
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12] {
            assert_eq!(v, TlVerdict::Fail);
        }
    }
}
