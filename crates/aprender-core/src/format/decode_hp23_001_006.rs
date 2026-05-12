// `decode-hot-path-first-tokens-diagnostic-v1` (HP2) and
// `decode-hot-path-prefix-cache-diagnostic-v1` (HP3) algorithm-level
// PARTIAL discharge — closes 6 falsification gates in one module.
//
// Contracts:
//   contracts/decode-hot-path-first-tokens-diagnostic-v1.yaml
//   contracts/decode-hot-path-prefix-cache-diagnostic-v1.yaml
//
// Both contracts share the same shape: source-level lint gate +
// throughput+parity gate + behaviour-preservation gate.
//
//   HP2-001: zero unconditional eprintln in par-062 hot path
//   HP2-002: TPS ≥ 380 AND Ollama parity ≥ 1.25×
//   HP2-003: Golden Output gate is 2/2 PASS
//   HP3-001: every PMAT-450 eprintln is `if config.trace`-gated
//   HP3-002: TPS ≥ 390 AND Ollama parity ≥ 1.25×
//   HP3-003: HIT / INSERT / ERROR paths share symmetric `config.trace` wrap

/// Pinned per-contract minimum TPS thresholds.
pub const AC_HP2_MIN_TPS: f32 = 380.0;
pub const AC_HP3_MIN_TPS: f32 = 390.0;
/// Shared Ollama-parity floor.
pub const AC_DECODE_HP_MIN_PARITY: f32 = 1.25;
/// HP2-003: golden output PASS encodes 2/2.
pub const AC_HP2_GOLDEN_PASS_NUM: u32 = 2;
pub const AC_HP2_GOLDEN_PASS_DEN: u32 = 2;
/// HP3-003: number of cache paths that must share the gating
/// (HIT, INSERT, ERROR).
pub const AC_HP3_CACHE_PATHS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeHpVerdict {
    Pass,
    Fail,
}

/// HP2-001: zero unconditional eprintln in the hot path.
///
/// Pass iff:
/// - 0 matches for `realizr#198` in the included scope
/// - 0 matches for `REPLAY: pos=` in the included scope
/// - 0 unconditional `eprintln!` lines in `par-062.rs`
#[must_use]
pub fn verdict_from_zero_eprintln(
    realizr_198_matches: u32,
    replay_pos_matches: u32,
    par062_unconditional_eprintln: u32,
) -> DecodeHpVerdict {
    if realizr_198_matches == 0
        && replay_pos_matches == 0
        && par062_unconditional_eprintln == 0
    {
        DecodeHpVerdict::Pass
    } else {
        DecodeHpVerdict::Fail
    }
}

/// HP2-002 / HP3-002: throughput + Ollama parity composite gate.
///
/// Pass iff `tps >= min_tps` AND `parity >= AC_DECODE_HP_MIN_PARITY`.
/// Non-finite or non-positive → Fail.
#[must_use]
pub fn verdict_from_tps_parity(tps: f32, parity: f32, min_tps: f32) -> DecodeHpVerdict {
    if !tps.is_finite() || !parity.is_finite() {
        return DecodeHpVerdict::Fail;
    }
    if tps <= 0.0 || parity <= 0.0 {
        return DecodeHpVerdict::Fail;
    }
    if tps >= min_tps && parity >= AC_DECODE_HP_MIN_PARITY {
        DecodeHpVerdict::Pass
    } else {
        DecodeHpVerdict::Fail
    }
}

/// HP2-003: golden output gate is 2/2 PASS.
#[must_use]
pub fn verdict_from_golden_output(pass: u32, total: u32) -> DecodeHpVerdict {
    if total != AC_HP2_GOLDEN_PASS_DEN {
        return DecodeHpVerdict::Fail;
    }
    if pass == AC_HP2_GOLDEN_PASS_NUM {
        DecodeHpVerdict::Pass
    } else {
        DecodeHpVerdict::Fail
    }
}

/// HP3-001: every PMAT-450 eprintln site is preceded by `if config.trace`.
///
/// `pmat450_sites_total` = total number of `eprintln!("[PMAT-450]` sites.
/// `pmat450_sites_gated` = subset with `if config.trace` guard.
/// Pass iff total > 0 AND gated == total.
#[must_use]
pub fn verdict_from_pmat450_gating(
    pmat450_sites_total: u32,
    pmat450_sites_gated: u32,
) -> DecodeHpVerdict {
    if pmat450_sites_total == 0 {
        // If the diagnostic was removed entirely, the contract is not
        // applicable — but the spec lists 3 known sites; missing them
        // is a regression class, so Fail.
        return DecodeHpVerdict::Fail;
    }
    if pmat450_sites_gated == pmat450_sites_total {
        DecodeHpVerdict::Pass
    } else {
        DecodeHpVerdict::Fail
    }
}

/// HP3-003: HIT, INSERT, ERROR paths all share the gating wrap.
///
/// `paths_with_gating` should contain at least `AC_HP3_CACHE_PATHS`
/// entries (3) and all must be `true`.
#[must_use]
pub fn verdict_from_symmetric_gating(paths_with_gating: &[bool]) -> DecodeHpVerdict {
    if paths_with_gating.len() < AC_HP3_CACHE_PATHS {
        return DecodeHpVerdict::Fail;
    }
    if paths_with_gating.iter().all(|&b| b) {
        DecodeHpVerdict::Pass
    } else {
        DecodeHpVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_hp2_min_tps_is_380() {
        assert_eq!(AC_HP2_MIN_TPS, 380.0);
    }

    #[test]
    fn provenance_hp3_min_tps_is_390() {
        assert_eq!(AC_HP3_MIN_TPS, 390.0);
    }

    #[test]
    fn provenance_min_parity_is_1_25() {
        assert_eq!(AC_DECODE_HP_MIN_PARITY, 1.25);
    }

    #[test]
    fn provenance_golden_2_of_2() {
        assert_eq!(AC_HP2_GOLDEN_PASS_NUM, 2);
        assert_eq!(AC_HP2_GOLDEN_PASS_DEN, 2);
    }

    #[test]
    fn provenance_hp3_cache_paths_is_3() {
        assert_eq!(AC_HP3_CACHE_PATHS, 3);
    }

    // -----------------------------------------------------------------
    // Section 2: HP2-001 zero-eprintln.
    // -----------------------------------------------------------------
    #[test]
    fn fhp2_001_pass_clean_hot_path() {
        let v = verdict_from_zero_eprintln(0, 0, 0);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    #[test]
    fn fhp2_001_fail_realizr_198_present() {
        let v = verdict_from_zero_eprintln(1, 0, 0);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp2_001_fail_replay_pos_present() {
        let v = verdict_from_zero_eprintln(0, 1, 0);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp2_001_fail_par062_unconditional_eprintln() {
        let v = verdict_from_zero_eprintln(0, 0, 1);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp2_001_fail_all_three_diagnostics_present() {
        let v = verdict_from_zero_eprintln(5, 5, 5);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: HP2-002 / HP3-002 TPS+parity composite.
    // -----------------------------------------------------------------
    #[test]
    fn fhp2_002_pass_at_thresholds() {
        let v = verdict_from_tps_parity(380.0, 1.25, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    #[test]
    fn fhp2_002_pass_well_above() {
        let v = verdict_from_tps_parity(450.0, 1.45, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    #[test]
    fn fhp2_002_fail_low_tps() {
        let v = verdict_from_tps_parity(379.9, 1.30, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp2_002_fail_low_parity() {
        let v = verdict_from_tps_parity(400.0, 1.20, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp3_002_pass_at_threshold() {
        let v = verdict_from_tps_parity(390.0, 1.25, AC_HP3_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    #[test]
    fn fhp3_002_fail_below_390() {
        let v = verdict_from_tps_parity(385.0, 1.30, AC_HP3_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp_002_fail_nan() {
        let v = verdict_from_tps_parity(f32::NAN, 1.30, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Fail);
        let v = verdict_from_tps_parity(400.0, f32::NAN, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp_002_fail_zero_or_negative() {
        let v = verdict_from_tps_parity(0.0, 1.30, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Fail);
        let v = verdict_from_tps_parity(400.0, -1.0, AC_HP2_MIN_TPS);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: HP2-003 golden output.
    // -----------------------------------------------------------------
    #[test]
    fn fhp2_003_pass_2_of_2() {
        let v = verdict_from_golden_output(2, 2);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    #[test]
    fn fhp2_003_fail_1_of_2() {
        let v = verdict_from_golden_output(1, 2);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp2_003_fail_0_of_2() {
        let v = verdict_from_golden_output(0, 2);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp2_003_fail_total_not_2() {
        let v = verdict_from_golden_output(3, 3);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: HP3-001 PMAT-450 gating.
    // -----------------------------------------------------------------
    #[test]
    fn fhp3_001_pass_all_3_gated() {
        let v = verdict_from_pmat450_gating(3, 3);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    #[test]
    fn fhp3_001_fail_one_ungated() {
        let v = verdict_from_pmat450_gating(3, 2);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp3_001_fail_zero_total() {
        // diagnostic completely missing — guard regression
        let v = verdict_from_pmat450_gating(0, 0);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp3_001_fail_gated_exceeds_total() {
        // Defensive: gated > total is impossible but produces Fail.
        let v = verdict_from_pmat450_gating(2, 3);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: HP3-003 symmetric gating.
    // -----------------------------------------------------------------
    #[test]
    fn fhp3_003_pass_all_three_paths_gated() {
        let v = verdict_from_symmetric_gating(&[true, true, true]);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    #[test]
    fn fhp3_003_fail_only_hit_gated() {
        let v = verdict_from_symmetric_gating(&[true, false, false]);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp3_003_fail_too_few_paths() {
        let v = verdict_from_symmetric_gating(&[true, true]);
        assert_eq!(v, DecodeHpVerdict::Fail);
    }

    #[test]
    fn fhp3_003_pass_more_than_three() {
        let v = verdict_from_symmetric_gating(&[true; 5]);
        assert_eq!(v, DecodeHpVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation survey + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_only_zero_zero_zero_passes() {
        for r in 0_u32..3 {
            for p in 0_u32..3 {
                for e in 0_u32..3 {
                    let v = verdict_from_zero_eprintln(r, p, e);
                    let expected = if r == 0 && p == 0 && e == 0 {
                        DecodeHpVerdict::Pass
                    } else {
                        DecodeHpVerdict::Fail
                    };
                    assert_eq!(v, expected, "(r,p,e)=({r},{p},{e})");
                }
            }
        }
    }

    #[test]
    fn realistic_healthy_decode_hot_path_passes_all_6() {
        // HP2-001
        let v1 = verdict_from_zero_eprintln(0, 0, 0);
        // HP2-002 (1.5B Q4_K_M post-fix throughput)
        let v2 = verdict_from_tps_parity(440.4, 1.434, AC_HP2_MIN_TPS);
        // HP2-003
        let v3 = verdict_from_golden_output(2, 2);
        // HP3-001 (3 known PMAT-450 sites, all gated)
        let v4 = verdict_from_pmat450_gating(3, 3);
        // HP3-002 (post-prefix-cache fix throughput)
        let v5 = verdict_from_tps_parity(391.8, 1.36, AC_HP3_MIN_TPS);
        // HP3-003
        let v6 = verdict_from_symmetric_gating(&[true, true, true]);
        assert_eq!(v1, DecodeHpVerdict::Pass);
        assert_eq!(v2, DecodeHpVerdict::Pass);
        assert_eq!(v3, DecodeHpVerdict::Pass);
        assert_eq!(v4, DecodeHpVerdict::Pass);
        assert_eq!(v5, DecodeHpVerdict::Pass);
        assert_eq!(v6, DecodeHpVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // Pre-fix regression class:
        //   - 5 unconditional eprintlns in hot path
        //   - 184 tok/s baseline (way under threshold)
        //   - 0/2 golden output
        //   - 0 PMAT-450 sites gated
        //   - HIT path gated but INSERT/ERROR not
        let v1 = verdict_from_zero_eprintln(2, 2, 5);
        let v2 = verdict_from_tps_parity(184.0, 0.6, AC_HP2_MIN_TPS);
        let v3 = verdict_from_golden_output(0, 2);
        let v4 = verdict_from_pmat450_gating(3, 0);
        let v5 = verdict_from_tps_parity(184.0, 0.6, AC_HP3_MIN_TPS);
        let v6 = verdict_from_symmetric_gating(&[true, false, false]);
        assert_eq!(v1, DecodeHpVerdict::Fail);
        assert_eq!(v2, DecodeHpVerdict::Fail);
        assert_eq!(v3, DecodeHpVerdict::Fail);
        assert_eq!(v4, DecodeHpVerdict::Fail);
        assert_eq!(v5, DecodeHpVerdict::Fail);
        assert_eq!(v6, DecodeHpVerdict::Fail);
    }
}
