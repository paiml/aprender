// Bundles two sister contracts in one verdict module:
//
//   `encoder-roundtrip-v1` (FALSIFY-RT-001..003)
//   `gateway-contract-v1` (FALSIFY-GW-001..004)
//
// RT-001: APR write→read preserves F32 bit-exactly
// RT-002: GGUF export→import preserves tensor shape names+ranks+dims
// RT-003: SafeTensors save→load preserves metadata key-value pairs
// GW-001: any gateway failure zeros MQS
// GW-002: two-phase execution (G0 sub-gates complete before scenarios)
// GW-003: G4 passes when garbage_count <= evidence_count / 4
// GW-004: check_gateways always returns exactly 5 items

use std::collections::HashMap;

/// GW-004 expected gateway count.
pub const AC_GW_GATEWAY_COUNT: usize = 5;
/// GW-003 garbage ratio: 25% (≤ evidence/4).
pub const AC_GW_GARBAGE_RATIO_NUM: u32 = 1;
pub const AC_GW_GARBAGE_RATIO_DEN: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtGwVerdict {
    Pass,
    Fail,
}

// ----------------------------------------------------------------
// RT-001..003
// ----------------------------------------------------------------

/// RT-001: APR write→read F32 bit-exact preservation.
#[must_use]
pub fn verdict_from_apr_f32_roundtrip(
    written: &[f32],
    read_back: &[f32],
) -> RtGwVerdict {
    if written.is_empty() || written.len() != read_back.len() {
        return RtGwVerdict::Fail;
    }
    for (a, b) in written.iter().zip(read_back.iter()) {
        if a.to_bits() != b.to_bits() {
            return RtGwVerdict::Fail;
        }
    }
    RtGwVerdict::Pass
}

/// RT-002: GGUF export→import preserves names + ranks + dims.
///
/// `original` and `roundtripped` are slices of (name, shape) pairs.
#[must_use]
pub fn verdict_from_gguf_shape_roundtrip(
    original: &[(&str, &[usize])],
    roundtripped: &[(&str, &[usize])],
) -> RtGwVerdict {
    if original.is_empty() || original.len() != roundtripped.len() {
        return RtGwVerdict::Fail;
    }
    for ((on, os), (rn, rs)) in original.iter().zip(roundtripped.iter()) {
        if on != rn {
            return RtGwVerdict::Fail;
        }
        if os.len() != rs.len() {
            return RtGwVerdict::Fail;
        }
        if os != rs {
            return RtGwVerdict::Fail;
        }
    }
    RtGwVerdict::Pass
}

/// RT-003: SafeTensors metadata roundtrip — full HashMap equality.
#[must_use]
pub fn verdict_from_safetensors_metadata_roundtrip<S: std::hash::BuildHasher>(
    original: &HashMap<String, String, S>,
    roundtripped: &HashMap<String, String, S>,
) -> RtGwVerdict {
    if original.is_empty() {
        return RtGwVerdict::Fail;
    }
    if original == roundtripped {
        RtGwVerdict::Pass
    } else {
        RtGwVerdict::Fail
    }
}

// ----------------------------------------------------------------
// GW-001..004
// ----------------------------------------------------------------

/// GW-001: any gateway failure zeros MQS.
///
/// `gateway_pass` is a slice of 5 booleans (one per gateway).
/// `mqs` is the resulting score.
/// Pass iff: `all(gateway_pass) ↔ mqs > 0`.
#[must_use]
pub fn verdict_from_mqs_zero_on_failure(
    gateway_pass: &[bool],
    mqs: f32,
) -> RtGwVerdict {
    if gateway_pass.len() != AC_GW_GATEWAY_COUNT {
        return RtGwVerdict::Fail;
    }
    if !mqs.is_finite() || mqs < 0.0 {
        return RtGwVerdict::Fail;
    }
    let all_pass = gateway_pass.iter().all(|&p| p);
    if (all_pass && mqs > 0.0) || (!all_pass && mqs == 0.0) {
        RtGwVerdict::Pass
    } else {
        RtGwVerdict::Fail
    }
}

/// GW-002: G0 sub-gates complete before scenarios run.
#[must_use]
pub fn verdict_from_two_phase_execution(
    g0_complete_before_scenarios: bool,
    scenarios_started_only_after_g0: bool,
) -> RtGwVerdict {
    if g0_complete_before_scenarios && scenarios_started_only_after_g0 {
        RtGwVerdict::Pass
    } else {
        RtGwVerdict::Fail
    }
}

/// GW-003: G4 passes iff `garbage_count <= evidence_count / 4`.
#[must_use]
pub fn verdict_from_g4_garbage_threshold(
    garbage_count: u32,
    evidence_count: u32,
    g4_actual_pass: bool,
) -> RtGwVerdict {
    if evidence_count == 0 {
        return RtGwVerdict::Fail;
    }
    let limit = evidence_count * AC_GW_GARBAGE_RATIO_NUM / AC_GW_GARBAGE_RATIO_DEN;
    let expected_pass = garbage_count <= limit;
    if expected_pass == g4_actual_pass {
        RtGwVerdict::Pass
    } else {
        RtGwVerdict::Fail
    }
}

/// GW-004: `check_gateways` returns exactly 5 items.
#[must_use]
pub fn verdict_from_gateway_count(actual_count: usize) -> RtGwVerdict {
    if actual_count == AC_GW_GATEWAY_COUNT {
        RtGwVerdict::Pass
    } else {
        RtGwVerdict::Fail
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
        assert_eq!(AC_GW_GATEWAY_COUNT, 5);
        assert_eq!(AC_GW_GARBAGE_RATIO_NUM, 1);
        assert_eq!(AC_GW_GARBAGE_RATIO_DEN, 4);
    }

    // -----------------------------------------------------------------
    // Section 2: RT-001..003.
    // -----------------------------------------------------------------
    #[test]
    fn frt001_pass_bit_exact() {
        let w = vec![1.0_f32, 2.5, -3.0, 0.0];
        let r = w.clone();
        let v = verdict_from_apr_f32_roundtrip(&w, &r);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn frt001_fail_one_ulp_drift() {
        let w = vec![1.0_f32];
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let r = vec![bumped];
        let v = verdict_from_apr_f32_roundtrip(&w, &r);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn frt001_fail_length_mismatch() {
        let v = verdict_from_apr_f32_roundtrip(&[1.0], &[1.0, 2.0]);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn frt002_pass_canonical_roundtrip() {
        let orig: &[(&str, &[usize])] = &[
            ("token_embd.weight", &[151_936, 1024]),
            ("output_norm.weight", &[1024]),
        ];
        let rt: &[(&str, &[usize])] = &[
            ("token_embd.weight", &[151_936, 1024]),
            ("output_norm.weight", &[1024]),
        ];
        let v = verdict_from_gguf_shape_roundtrip(orig, rt);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn frt002_fail_dim_drift() {
        let orig: &[(&str, &[usize])] = &[("a", &[100, 200])];
        let rt: &[(&str, &[usize])] = &[("a", &[100, 201])];
        let v = verdict_from_gguf_shape_roundtrip(orig, rt);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn frt002_fail_name_drift() {
        let orig: &[(&str, &[usize])] = &[("a", &[100])];
        let rt: &[(&str, &[usize])] = &[("b", &[100])];
        let v = verdict_from_gguf_shape_roundtrip(orig, rt);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn frt002_fail_rank_drift() {
        let orig: &[(&str, &[usize])] = &[("a", &[100, 200])];
        let rt: &[(&str, &[usize])] = &[("a", &[100])];
        let v = verdict_from_gguf_shape_roundtrip(orig, rt);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn frt003_pass_metadata_match() {
        let mut orig = HashMap::new();
        orig.insert("license".to_string(), "MIT".to_string());
        orig.insert("data_source".to_string(), "csn-python".to_string());
        let rt = orig.clone();
        let v = verdict_from_safetensors_metadata_roundtrip(&orig, &rt);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn frt003_fail_value_drift() {
        let mut orig = HashMap::new();
        orig.insert("license".to_string(), "MIT".to_string());
        let mut rt = HashMap::new();
        rt.insert("license".to_string(), "Apache-2.0".to_string());
        let v = verdict_from_safetensors_metadata_roundtrip(&orig, &rt);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn frt003_fail_key_dropped() {
        let mut orig = HashMap::new();
        orig.insert("a".to_string(), "1".to_string());
        orig.insert("b".to_string(), "2".to_string());
        let mut rt = HashMap::new();
        rt.insert("a".to_string(), "1".to_string());
        let v = verdict_from_safetensors_metadata_roundtrip(&orig, &rt);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: GW-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn fgw001_pass_all_gates_pass_nonzero_mqs() {
        let v = verdict_from_mqs_zero_on_failure(&[true; 5], 75.0);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn fgw001_pass_one_fails_zero_mqs() {
        let v = verdict_from_mqs_zero_on_failure(&[true, false, true, true, true], 0.0);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn fgw001_fail_partial_credit_bug() {
        // Gateway failed but MQS > 0 — the regression class.
        let v = verdict_from_mqs_zero_on_failure(&[true, false, true, true, true], 50.0);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn fgw001_fail_wrong_gateway_count() {
        let v = verdict_from_mqs_zero_on_failure(&[true; 3], 50.0);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn fgw002_pass_two_phase() {
        let v = verdict_from_two_phase_execution(true, true);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn fgw002_fail_scenarios_first() {
        let v = verdict_from_two_phase_execution(false, false);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn fgw003_pass_at_threshold() {
        // 100 evidence × 25% = 25 garbage tolerated.
        let v = verdict_from_g4_garbage_threshold(25, 100, true);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn fgw003_pass_one_more_should_fail() {
        // 26/100 — actual_pass should be false; if false → Pass (correct).
        let v = verdict_from_g4_garbage_threshold(26, 100, false);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn fgw003_fail_pass_when_should_fail() {
        // 50/100 garbage — should fail but actual says pass.
        let v = verdict_from_g4_garbage_threshold(50, 100, true);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn fgw003_fail_zero_evidence() {
        let v = verdict_from_g4_garbage_threshold(0, 0, true);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn fgw004_pass_exactly_5() {
        let v = verdict_from_gateway_count(5);
        assert_eq!(v, RtGwVerdict::Pass);
    }

    #[test]
    fn fgw004_fail_4() {
        let v = verdict_from_gateway_count(4);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    #[test]
    fn fgw004_fail_6() {
        let v = verdict_from_gateway_count(6);
        assert_eq!(v, RtGwVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_gw001_5_gates_truth_table() {
        // Every 32-element subset; only [true; 5] yields nonzero MQS.
        for mask in 0_u32..32 {
            let gates: Vec<bool> = (0..5).map(|i| (mask >> i) & 1 != 0).collect();
            let all_pass = gates.iter().all(|&p| p);
            let mqs = if all_pass { 80.0 } else { 0.0 };
            let v = verdict_from_mqs_zero_on_failure(&gates, mqs);
            assert_eq!(v, RtGwVerdict::Pass, "mask={mask:05b}");
        }
    }

    #[test]
    fn mutation_survey_gw003_threshold_band() {
        for garb in [0_u32, 24, 25, 26, 50, 100] {
            let expected_pass = garb <= 25;
            let v = verdict_from_g4_garbage_threshold(garb, 100, expected_pass);
            assert_eq!(v, RtGwVerdict::Pass, "garb={garb}");
        }
    }

    // -----------------------------------------------------------------
    // Section 5: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_7() {
        let v1 = verdict_from_apr_f32_roundtrip(&[1.0, 2.0], &[1.0, 2.0]);
        let v2 = verdict_from_gguf_shape_roundtrip(
            &[("a", &[100, 200])],
            &[("a", &[100, 200])],
        );
        let mut m = HashMap::new();
        m.insert("k".to_string(), "v".to_string());
        let v3 = verdict_from_safetensors_metadata_roundtrip(&m, &m);
        let v4 = verdict_from_mqs_zero_on_failure(&[true; 5], 80.0);
        let v5 = verdict_from_two_phase_execution(true, true);
        let v6 = verdict_from_g4_garbage_threshold(20, 100, true);
        let v7 = verdict_from_gateway_count(5);
        for v in [v1, v2, v3, v4, v5, v6, v7] {
            assert_eq!(v, RtGwVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Pre-fix regressions.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_all_7_failures() {
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let v1 = verdict_from_apr_f32_roundtrip(&[1.0], &[bumped]);
        let v2 = verdict_from_gguf_shape_roundtrip(
            &[("a", &[100, 200])],
            &[("a", &[100, 201])],
        );
        let mut a = HashMap::new();
        a.insert("k".to_string(), "v".to_string());
        let mut b = HashMap::new();
        b.insert("k".to_string(), "DIFFERENT".to_string());
        let v3 = verdict_from_safetensors_metadata_roundtrip(&a, &b);
        let v4 = verdict_from_mqs_zero_on_failure(&[true, false, true, true, true], 50.0);
        let v5 = verdict_from_two_phase_execution(false, false);
        let v6 = verdict_from_g4_garbage_threshold(50, 100, true);
        let v7 = verdict_from_gateway_count(4);
        for v in [v1, v2, v3, v4, v5, v6, v7] {
            assert_eq!(v, RtGwVerdict::Fail);
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Edge cases.
    // -----------------------------------------------------------------
    #[test]
    fn edge_empty_inputs_fail() {
        let v1 = verdict_from_apr_f32_roundtrip(&[], &[]);
        let v2 = verdict_from_gguf_shape_roundtrip(&[], &[]);
        let v3 = verdict_from_safetensors_metadata_roundtrip(&HashMap::new(), &HashMap::new());
        for v in [v1, v2, v3] {
            assert_eq!(v, RtGwVerdict::Fail);
        }
    }
}
