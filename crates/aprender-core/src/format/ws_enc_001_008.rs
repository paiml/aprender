// Bundles two sister contracts in one verdict module:
//
//   `cpu-work-stealing-v1` (FALSIFY-WS-001..004)
//   `encoder-forward-v1` (FALSIFY-ENC-001..004)
//
// WS-001: dispatch overhead < 1ms per forward pass
// WS-002: L1 cache miss rate < 5%
// WS-003: matvec parity vs Rayon within 1e-6
// WS-004: 4-thread throughput ≥ 3.5× single-thread
// ENC-001: 12 encoder layers preserve (n, 768) shape
// ENC-002: no NaN/Inf for inputs in [-10, 10]
// ENC-003: entrenar output matches HF reference within 1e-4
// ENC-004: CLS pooling extracts encoder_output[0]

/// WS-001 dispatch overhead ceiling (ms).
pub const AC_WS_DISPATCH_OVERHEAD_MAX_MS: f32 = 1.0;
/// WS-002 L1 cache miss rate ceiling (%).
pub const AC_WS_L1_MISS_RATE_MAX_PCT: f32 = 5.0;
/// WS-003 matvec parity tolerance.
pub const AC_WS_MATVEC_TOLERANCE: f32 = 1e-6;
/// WS-004 scaling efficiency floor (4 threads ≥ 3.5× single).
pub const AC_WS_SCALING_FLOOR: f32 = 3.5;
/// ENC-001 hidden dim.
pub const AC_ENC_HIDDEN_DIM: usize = 768;
/// ENC-001 layer count.
pub const AC_ENC_LAYER_COUNT: u32 = 12;
/// ENC-003 reference parity tolerance.
pub const AC_ENC_REFERENCE_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsEncVerdict {
    Pass,
    Fail,
}

// ----------------------------------------------------------------
// WS-001..004
// ----------------------------------------------------------------

#[must_use]
pub fn verdict_from_dispatch_overhead(overhead_ms: f32) -> WsEncVerdict {
    if !overhead_ms.is_finite() || overhead_ms < 0.0 {
        return WsEncVerdict::Fail;
    }
    if overhead_ms < AC_WS_DISPATCH_OVERHEAD_MAX_MS {
        WsEncVerdict::Pass
    } else {
        WsEncVerdict::Fail
    }
}

#[must_use]
pub fn verdict_from_l1_miss_rate(miss_rate_pct: f32) -> WsEncVerdict {
    if !miss_rate_pct.is_finite() || miss_rate_pct < 0.0 || miss_rate_pct > 100.0 {
        return WsEncVerdict::Fail;
    }
    if miss_rate_pct < AC_WS_L1_MISS_RATE_MAX_PCT {
        WsEncVerdict::Pass
    } else {
        WsEncVerdict::Fail
    }
}

#[must_use]
pub fn verdict_from_matvec_parity(workst: &[f32], rayon: &[f32]) -> WsEncVerdict {
    if workst.is_empty() || workst.len() != rayon.len() {
        return WsEncVerdict::Fail;
    }
    for (a, b) in workst.iter().zip(rayon.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return WsEncVerdict::Fail;
        }
        if (a - b).abs() > AC_WS_MATVEC_TOLERANCE {
            return WsEncVerdict::Fail;
        }
    }
    WsEncVerdict::Pass
}

#[must_use]
pub fn verdict_from_scaling_efficiency(t1_tps: f32, t4_tps: f32) -> WsEncVerdict {
    if !t1_tps.is_finite() || !t4_tps.is_finite() {
        return WsEncVerdict::Fail;
    }
    if t1_tps <= 0.0 || t4_tps <= 0.0 {
        return WsEncVerdict::Fail;
    }
    if t4_tps >= AC_WS_SCALING_FLOOR * t1_tps {
        WsEncVerdict::Pass
    } else {
        WsEncVerdict::Fail
    }
}

// ----------------------------------------------------------------
// ENC-001..004
// ----------------------------------------------------------------

/// ENC-001: 12 encoder layers preserve (n, 768) shape.
#[must_use]
pub fn verdict_from_shape_preservation(
    layer_count: u32,
    in_seq_len: usize,
    in_hidden: usize,
    out_seq_len: usize,
    out_hidden: usize,
) -> WsEncVerdict {
    if layer_count != AC_ENC_LAYER_COUNT {
        return WsEncVerdict::Fail;
    }
    if in_hidden != AC_ENC_HIDDEN_DIM || out_hidden != AC_ENC_HIDDEN_DIM {
        return WsEncVerdict::Fail;
    }
    if in_seq_len == out_seq_len && in_seq_len > 0 {
        WsEncVerdict::Pass
    } else {
        WsEncVerdict::Fail
    }
}

/// ENC-002: every output element is finite.
#[must_use]
pub fn verdict_from_finite_output(output: &[f32]) -> WsEncVerdict {
    if output.is_empty() {
        return WsEncVerdict::Fail;
    }
    if output.iter().all(|x| x.is_finite()) {
        WsEncVerdict::Pass
    } else {
        WsEncVerdict::Fail
    }
}

/// ENC-003: max|aprender_out - hf_reference| ≤ 1e-4.
#[must_use]
pub fn verdict_from_hf_reference_parity(
    aprender_out: &[f32],
    hf_reference: &[f32],
) -> WsEncVerdict {
    if aprender_out.is_empty() || aprender_out.len() != hf_reference.len() {
        return WsEncVerdict::Fail;
    }
    for (a, b) in aprender_out.iter().zip(hf_reference.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return WsEncVerdict::Fail;
        }
        if (a - b).abs() > AC_ENC_REFERENCE_TOLERANCE {
            return WsEncVerdict::Fail;
        }
    }
    WsEncVerdict::Pass
}

/// ENC-004: CLS pooling extracts row 0 of the encoder output.
#[must_use]
pub fn verdict_from_cls_pooling(
    encoder_output_first_row: &[f32],
    cls_embedding: &[f32],
) -> WsEncVerdict {
    if encoder_output_first_row.is_empty()
        || encoder_output_first_row.len() != cls_embedding.len()
    {
        return WsEncVerdict::Fail;
    }
    for (a, b) in encoder_output_first_row.iter().zip(cls_embedding.iter()) {
        if a.to_bits() != b.to_bits() {
            return WsEncVerdict::Fail;
        }
    }
    WsEncVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_WS_DISPATCH_OVERHEAD_MAX_MS, 1.0);
        assert_eq!(AC_WS_L1_MISS_RATE_MAX_PCT, 5.0);
        assert_eq!(AC_WS_MATVEC_TOLERANCE, 1e-6);
        assert_eq!(AC_WS_SCALING_FLOOR, 3.5);
        assert_eq!(AC_ENC_HIDDEN_DIM, 768);
        assert_eq!(AC_ENC_LAYER_COUNT, 12);
        assert_eq!(AC_ENC_REFERENCE_TOLERANCE, 1e-4);
    }

    // -----------------------------------------------------------------
    // Section 2: WS-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn fws001_pass_under_1ms() {
        let v = verdict_from_dispatch_overhead(0.5);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fws001_fail_at_1ms() {
        let v = verdict_from_dispatch_overhead(1.0); // strict <
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fws001_fail_negative() {
        let v = verdict_from_dispatch_overhead(-0.1);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fws002_pass_low_miss_rate() {
        let v = verdict_from_l1_miss_rate(2.0);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fws002_fail_high_miss_rate() {
        let v = verdict_from_l1_miss_rate(10.0);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fws003_pass_within_tolerance() {
        let workst = vec![1.0_f32, 2.0, 3.0];
        let rayon = vec![1.0000001_f32, 1.9999999, 3.0000001];
        let v = verdict_from_matvec_parity(&workst, &rayon);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fws003_fail_drift() {
        let workst = vec![1.0_f32, 2.0];
        let rayon = vec![1.0_f32, 2.5];
        let v = verdict_from_matvec_parity(&workst, &rayon);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fws004_pass_3_5x() {
        let v = verdict_from_scaling_efficiency(100.0, 350.0);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fws004_fail_only_2x() {
        let v = verdict_from_scaling_efficiency(100.0, 200.0);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fws004_fail_zero_baseline() {
        let v = verdict_from_scaling_efficiency(0.0, 100.0);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: ENC-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn fenc001_pass_canonical_shape() {
        let v = verdict_from_shape_preservation(12, 50, 768, 50, 768);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fenc001_fail_dropped_layer() {
        let v = verdict_from_shape_preservation(11, 50, 768, 50, 768);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fenc001_fail_wrong_hidden() {
        let v = verdict_from_shape_preservation(12, 50, 768, 50, 1024);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fenc001_fail_seq_len_changed() {
        let v = verdict_from_shape_preservation(12, 50, 768, 49, 768);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fenc002_pass_finite_output() {
        let v = verdict_from_finite_output(&[1.0, 2.0, -3.0, 0.0]);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fenc002_fail_nan() {
        let v = verdict_from_finite_output(&[1.0, f32::NAN]);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fenc002_fail_infinity() {
        let v = verdict_from_finite_output(&[1.0, f32::INFINITY]);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fenc003_pass_within_1e_4() {
        let apr = vec![1.0_f32, 2.0, 3.0];
        let hf = vec![1.00005_f32, 1.99995, 3.00001];
        let v = verdict_from_hf_reference_parity(&apr, &hf);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fenc003_fail_drift() {
        let apr = vec![1.0_f32, 2.0];
        let hf = vec![1.0_f32, 2.5];
        let v = verdict_from_hf_reference_parity(&apr, &hf);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fenc004_pass_first_row_match() {
        let row0 = vec![1.0_f32, 2.0, 3.0];
        let cls = row0.clone();
        let v = verdict_from_cls_pooling(&row0, &cls);
        assert_eq!(v, WsEncVerdict::Pass);
    }

    #[test]
    fn fenc004_fail_one_ulp_drift() {
        let row0 = vec![1.0_f32, 2.0];
        let bumped = f32::from_bits(2.0_f32.to_bits() + 1);
        let cls = vec![1.0_f32, bumped];
        let v = verdict_from_cls_pooling(&row0, &cls);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    #[test]
    fn fenc004_fail_length_mismatch() {
        let row0 = vec![1.0_f32, 2.0];
        let cls = vec![1.0_f32];
        let v = verdict_from_cls_pooling(&row0, &cls);
        assert_eq!(v, WsEncVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_ws_scaling_band() {
        let baseline = 100.0_f32;
        for ratio_x10 in [10_u32, 30, 35, 36, 40, 80] {
            let t4 = baseline * (ratio_x10 as f32 / 10.0);
            let v = verdict_from_scaling_efficiency(baseline, t4);
            let want = if t4 >= AC_WS_SCALING_FLOOR * baseline {
                WsEncVerdict::Pass
            } else {
                WsEncVerdict::Fail
            };
            assert_eq!(v, want, "ratio={ratio_x10}");
        }
    }

    #[test]
    fn mutation_survey_enc_layer_count_band() {
        for n in [10_u32, 11, 12, 13, 24] {
            let v = verdict_from_shape_preservation(n, 50, 768, 50, 768);
            let want = if n == 12 {
                WsEncVerdict::Pass
            } else {
                WsEncVerdict::Fail
            };
            assert_eq!(v, want, "n={n}");
        }
    }

    // -----------------------------------------------------------------
    // Section 5: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_8() {
        let v1 = verdict_from_dispatch_overhead(0.3);
        let v2 = verdict_from_l1_miss_rate(2.5);
        let v3 = verdict_from_matvec_parity(&[1.0_f32, 2.0], &[1.0_f32, 2.0]);
        let v4 = verdict_from_scaling_efficiency(100.0, 380.0);
        let v5 = verdict_from_shape_preservation(12, 50, 768, 50, 768);
        let v6 = verdict_from_finite_output(&[1.0, 2.0, 3.0]);
        let v7 = verdict_from_hf_reference_parity(&[1.0_f32, 2.0], &[1.00005, 1.99995]);
        let v8 = verdict_from_cls_pooling(&[1.0_f32, 2.0], &[1.0, 2.0]);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8] {
            assert_eq!(v, WsEncVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Pre-fix regressions.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_all_8_failures() {
        let v1 = verdict_from_dispatch_overhead(2.0); // 2ms — atomic contention
        let v2 = verdict_from_l1_miss_rate(15.0); // tile too big
        let v3 = verdict_from_matvec_parity(&[1.0_f32], &[1.5_f32]); // race
        let v4 = verdict_from_scaling_efficiency(100.0, 200.0); // only 2×
        let v5 = verdict_from_shape_preservation(11, 50, 768, 50, 768); // dropped layer
        let v6 = verdict_from_finite_output(&[1.0, f32::NAN]);
        let v7 = verdict_from_hf_reference_parity(&[1.0_f32], &[2.0]);
        let v8 = verdict_from_cls_pooling(&[1.0_f32], &[2.0]);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8] {
            assert_eq!(v, WsEncVerdict::Fail);
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Edge cases.
    // -----------------------------------------------------------------
    #[test]
    fn edge_empty_inputs_fail_closed() {
        let v = verdict_from_matvec_parity(&[], &[]);
        assert_eq!(v, WsEncVerdict::Fail);
        let v = verdict_from_finite_output(&[]);
        assert_eq!(v, WsEncVerdict::Fail);
    }
}
