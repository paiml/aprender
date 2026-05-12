// `decode-gpu-resident-sampling-v1` algorithm-level PARTIAL discharge
// for FALSIFY-GRS-001..004.
//
// Contract: `contracts/decode-gpu-resident-sampling-v1.yaml` (FALSIFIED).
//
// Pure Rust verdicts for the 4 falsification gates:
//   GRS-001: per-token ID parity vs golden snapshot
//   GRS-002: median Ollama-parity ratio crosses 1.50×
//   GRS-003: stop-token latency bounded by stop_pos + N
//   GRS-004: Non-Kernel Host Overhead ≤ 20.0%
//
// Note: contract is FALSIFIED in the field. The verdict modules below
// codify the *decision rule* the contract used to falsify itself; they
// remain valid as algorithm-level discharge regardless of whether the
// field measurement passed or failed. The dogfooding evidence at the
// time was a Fail on GRS-002 / GRS-004 — that evidence is preserved
// inside `realistic_falsified_field_evidence` so future regressions
// can never silently re-claim a Pass.

/// Required minimum median Ollama-parity ratio (1.50×).
pub const AC_GRS_PARITY_THRESHOLD: f32 = 1.50;

/// Maximum allowed Non-Kernel Host Overhead percentage (20.0%).
pub const AC_GRS_HOST_OVERHEAD_MAX_PCT: f32 = 20.0;

/// Default stop-check stride (every N tokens).
pub const AC_GRS_DEFAULT_STOP_CHECK_N: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrsVerdict {
    Pass,
    Fail,
}

/// GRS-001: per-token ID parity vs golden snapshot.
///
/// Pass iff every observed token ID matches its golden snapshot
/// at the same index. Length mismatch → Fail. Empty → Fail.
#[must_use]
pub fn verdict_from_token_parity(observed: &[u32], golden: &[u32]) -> GrsVerdict {
    if observed.is_empty() || golden.is_empty() {
        return GrsVerdict::Fail;
    }
    if observed.len() != golden.len() {
        return GrsVerdict::Fail;
    }
    if observed == golden {
        GrsVerdict::Pass
    } else {
        GrsVerdict::Fail
    }
}

/// GRS-002: median throughput parity ratio crosses 1.50×.
///
/// Pass iff the median of three runs is `≥ AC_GRS_PARITY_THRESHOLD`.
/// Non-finite values → Fail. Empty / wrong-length → Fail.
#[must_use]
pub fn verdict_from_parity_ratio(samples: [f32; 3]) -> GrsVerdict {
    if samples.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return GrsVerdict::Fail;
    }
    let mut sorted = samples;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[1];
    if median >= AC_GRS_PARITY_THRESHOLD {
        GrsVerdict::Pass
    } else {
        GrsVerdict::Fail
    }
}

/// GRS-003: stop-token latency bounded by `stop_pos + stop_check_n`.
///
/// Pass iff `generated_tokens <= stop_pos + stop_check_n`.
/// Note: `stop_pos` and `stop_check_n` are token counts (u32).
/// Overflow → Fail (defensive).
#[must_use]
pub fn verdict_from_stop_latency(
    generated_tokens: u32,
    stop_pos: u32,
    stop_check_n: u32,
) -> GrsVerdict {
    let Some(bound) = stop_pos.checked_add(stop_check_n) else {
        return GrsVerdict::Fail;
    };
    if generated_tokens <= bound {
        GrsVerdict::Pass
    } else {
        GrsVerdict::Fail
    }
}

/// GRS-004: Non-Kernel Host Overhead percentage ≤ 20.0%.
///
/// Pass iff `pct <= AC_GRS_HOST_OVERHEAD_MAX_PCT`.
/// Non-finite → Fail. Negative → Fail. > 100% → Fail (sanity).
#[must_use]
pub fn verdict_from_host_overhead_pct(pct: f32) -> GrsVerdict {
    if !pct.is_finite() || pct < 0.0 || pct > 100.0 {
        return GrsVerdict::Fail;
    }
    if pct <= AC_GRS_HOST_OVERHEAD_MAX_PCT {
        GrsVerdict::Pass
    } else {
        GrsVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_parity_threshold_is_1_50() {
        assert_eq!(AC_GRS_PARITY_THRESHOLD, 1.50);
    }

    #[test]
    fn provenance_host_overhead_max_is_20_pct() {
        assert_eq!(AC_GRS_HOST_OVERHEAD_MAX_PCT, 20.0);
    }

    #[test]
    fn provenance_default_stop_check_is_8() {
        assert_eq!(AC_GRS_DEFAULT_STOP_CHECK_N, 8);
    }

    // -----------------------------------------------------------------
    // Section 2: GRS-001 token parity.
    // -----------------------------------------------------------------
    #[test]
    fn fgrs001_pass_exact_match() {
        let v = verdict_from_token_parity(&[1, 2, 3, 4], &[1, 2, 3, 4]);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs001_fail_off_by_one() {
        let v = verdict_from_token_parity(&[1, 2, 3, 5], &[1, 2, 3, 4]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs001_fail_length_mismatch() {
        let v = verdict_from_token_parity(&[1, 2, 3], &[1, 2, 3, 4]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs001_fail_observed_empty() {
        let v = verdict_from_token_parity(&[], &[1, 2, 3]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs001_fail_golden_empty() {
        let v = verdict_from_token_parity(&[1, 2, 3], &[]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs001_fail_first_token_mismatch() {
        let v = verdict_from_token_parity(&[99, 2, 3], &[1, 2, 3]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: GRS-002 parity ratio.
    // -----------------------------------------------------------------
    #[test]
    fn fgrs002_pass_at_threshold() {
        let v = verdict_from_parity_ratio([1.50, 1.50, 1.50]);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs002_pass_clearly_above() {
        let v = verdict_from_parity_ratio([1.55, 1.60, 1.58]);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs002_fail_field_falsified_value() {
        // 1.434× was the actual median when the contract was falsified.
        let v = verdict_from_parity_ratio([1.434, 1.42, 1.45]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs002_fail_just_below_threshold() {
        let v = verdict_from_parity_ratio([1.49, 1.49, 1.49]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs002_fail_negative_or_zero() {
        let v = verdict_from_parity_ratio([1.5, 0.0, 1.5]);
        assert_eq!(v, GrsVerdict::Fail);
        let v = verdict_from_parity_ratio([-1.0, 1.5, 1.5]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs002_fail_nan() {
        let v = verdict_from_parity_ratio([f32::NAN, 1.5, 1.5]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs002_fail_infinite() {
        let v = verdict_from_parity_ratio([f32::INFINITY, 1.5, 1.5]);
        assert_eq!(v, GrsVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: GRS-003 stop-token latency.
    // -----------------------------------------------------------------
    #[test]
    fn fgrs003_pass_at_bound() {
        // generated == stop_pos + N
        let v = verdict_from_stop_latency(13, 5, 8);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs003_pass_under_bound() {
        let v = verdict_from_stop_latency(7, 5, 8);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs003_pass_exactly_at_stop() {
        let v = verdict_from_stop_latency(5, 5, 8);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs003_fail_overshoot_by_one() {
        let v = verdict_from_stop_latency(14, 5, 8);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs003_fail_overshoot_far() {
        let v = verdict_from_stop_latency(100, 5, 8);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs003_fail_overflow() {
        let v = verdict_from_stop_latency(0, u32::MAX, 1);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs003_pass_zero_stop_pos_zero_n() {
        // stop at index 0, no slack — only generated=0 passes.
        let v = verdict_from_stop_latency(0, 0, 0);
        assert_eq!(v, GrsVerdict::Pass);
        let v = verdict_from_stop_latency(1, 0, 0);
        assert_eq!(v, GrsVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: GRS-004 host overhead.
    // -----------------------------------------------------------------
    #[test]
    fn fgrs004_pass_at_threshold() {
        let v = verdict_from_host_overhead_pct(20.0);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs004_pass_well_under() {
        let v = verdict_from_host_overhead_pct(5.0);
        assert_eq!(v, GrsVerdict::Pass);
    }

    #[test]
    fn fgrs004_fail_field_falsified_value() {
        // 37.9% was the actual measurement when the contract was falsified.
        let v = verdict_from_host_overhead_pct(37.9);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs004_fail_just_over() {
        let v = verdict_from_host_overhead_pct(20.01);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs004_fail_negative() {
        let v = verdict_from_host_overhead_pct(-1.0);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs004_fail_above_100() {
        let v = verdict_from_host_overhead_pct(101.0);
        assert_eq!(v, GrsVerdict::Fail);
    }

    #[test]
    fn fgrs004_fail_nan() {
        let v = verdict_from_host_overhead_pct(f32::NAN);
        assert_eq!(v, GrsVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation survey.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_002_threshold_inclusive() {
        // Walk values around the 1.50 threshold.
        for ratio_x100 in [148_u32, 149, 150, 151, 152] {
            let r = ratio_x100 as f32 / 100.0;
            let v = verdict_from_parity_ratio([r, r, r]);
            let expected = if r >= AC_GRS_PARITY_THRESHOLD {
                GrsVerdict::Pass
            } else {
                GrsVerdict::Fail
            };
            assert_eq!(v, expected, "r={r}");
        }
    }

    #[test]
    fn mutation_survey_004_threshold_inclusive() {
        for pct_x10 in [180_u32, 195, 200, 201, 220, 379] {
            let p = pct_x10 as f32 / 10.0;
            let v = verdict_from_host_overhead_pct(p);
            let expected = if p <= AC_GRS_HOST_OVERHEAD_MAX_PCT {
                GrsVerdict::Pass
            } else {
                GrsVerdict::Fail
            };
            assert_eq!(v, expected, "pct={p}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic vectors.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_grs_passes_all_4() {
        let v1 = verdict_from_token_parity(&[100, 200, 300], &[100, 200, 300]);
        let v2 = verdict_from_parity_ratio([1.55, 1.58, 1.60]);
        let v3 = verdict_from_stop_latency(7, 5, 8);
        let v4 = verdict_from_host_overhead_pct(15.0);
        assert_eq!(v1, GrsVerdict::Pass);
        assert_eq!(v2, GrsVerdict::Pass);
        assert_eq!(v3, GrsVerdict::Pass);
        assert_eq!(v4, GrsVerdict::Pass);
    }

    #[test]
    fn realistic_falsified_field_evidence() {
        // Locks in the falsification evidence preserved in the contract:
        //   - GRS-002: median ratio 1.434× < 1.50×
        //   - GRS-004: host overhead 37.9% > 20.0%
        let v2 = verdict_from_parity_ratio([1.434, 1.42, 1.45]);
        let v4 = verdict_from_host_overhead_pct(37.9);
        assert_eq!(v2, GrsVerdict::Fail);
        assert_eq!(v4, GrsVerdict::Fail);
    }
}
