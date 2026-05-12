// SHIP-TWO-001 — `gpu-decode-profiling-v1` algorithm-level PARTIAL
// discharge for FALSIFY-GDP-001..015 (closes 15/15 sweep).
//
// Contract: `contracts/gpu-decode-profiling-v1.yaml`.
//
// Bundles 15 verdict fns over the GPU decode profiler / cbtop report
// invariants. The runtime-level falsifiers exist as cbtop integration
// tests; this file pins the algorithm-level rule against drift.

// ===========================================================================
// GDP-001 — Brick coverage threshold
// ===========================================================================

pub const AC_GDP_001_MIN_COVERAGE: f64 = 0.85;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp001Verdict { Pass, Fail }

/// Pass iff `brick_total_ns / wall_ns >= 0.85`.
#[must_use]
pub fn verdict_from_brick_coverage(brick_total_ns: u64, wall_ns: u64) -> Gdp001Verdict {
    if wall_ns == 0 { return Gdp001Verdict::Fail; }
    let cov = brick_total_ns as f64 / wall_ns as f64;
    if cov >= AC_GDP_001_MIN_COVERAGE { Gdp001Verdict::Pass } else { Gdp001Verdict::Fail }
}

// ===========================================================================
// GDP-002 — Graph-enabled coverage upper bound
// ===========================================================================

pub const AC_GDP_002_MAX_COVERAGE_GRAPH_ENABLED: f64 = 0.50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp002Verdict { Pass, Fail }

/// Pass iff `brick_total_ns / wall_ns < 0.50` when CUDA graph is
/// enabled (graph replay hides brick instrumentation).
#[must_use]
pub fn verdict_from_graph_enabled_coverage(brick_total_ns: u64, wall_ns: u64) -> Gdp002Verdict {
    if wall_ns == 0 { return Gdp002Verdict::Fail; }
    let cov = brick_total_ns as f64 / wall_ns as f64;
    if cov < AC_GDP_002_MAX_COVERAGE_GRAPH_ENABLED { Gdp002Verdict::Pass } else { Gdp002Verdict::Fail }
}

// ===========================================================================
// GDP-003 — Sync mode dichotomy: Immediate > 100us, Deferred < 100us
// ===========================================================================

pub const AC_GDP_003_SYNC_BOUNDARY_US: f64 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode { Immediate, Deferred }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp003Verdict { Pass, Fail }

/// Pass iff `Immediate => avg_us > 100` AND `Deferred => avg_us < 100`.
#[must_use]
pub fn verdict_from_sync_mode_split(mode: SyncMode, avg_us: f64) -> Gdp003Verdict {
    if !avg_us.is_finite() || avg_us < 0.0 { return Gdp003Verdict::Fail; }
    match mode {
        SyncMode::Immediate => {
            if avg_us > AC_GDP_003_SYNC_BOUNDARY_US { Gdp003Verdict::Pass } else { Gdp003Verdict::Fail }
        }
        SyncMode::Deferred => {
            if avg_us < AC_GDP_003_SYNC_BOUNDARY_US { Gdp003Verdict::Pass } else { Gdp003Verdict::Fail }
        }
    }
}

// ===========================================================================
// GDP-004 — LmHead call count == decoded tokens (exactly 1 per token)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp004Verdict { Pass, Fail }

/// Pass iff `lm_head_count == decoded_tokens` AND `decoded_tokens > 0`.
#[must_use]
pub const fn verdict_from_lm_head_count(lm_head_count: u64, decoded_tokens: u64) -> Gdp004Verdict {
    if decoded_tokens == 0 { return Gdp004Verdict::Fail; }
    if lm_head_count == decoded_tokens { Gdp004Verdict::Pass } else { Gdp004Verdict::Fail }
}

// ===========================================================================
// GDP-005 — Brick per-call avg ordering: LmHead > Gate > RmsNorm
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp005Verdict { Pass, Fail }

/// Pass iff `lm_head_avg_us > gate_avg_us > rms_norm_avg_us` strictly.
#[must_use]
pub fn verdict_from_brick_ordering(
    lm_head_avg_us: f64,
    gate_avg_us: f64,
    rms_norm_avg_us: f64,
) -> Gdp005Verdict {
    if !lm_head_avg_us.is_finite() || !gate_avg_us.is_finite() || !rms_norm_avg_us.is_finite() {
        return Gdp005Verdict::Fail;
    }
    if lm_head_avg_us > gate_avg_us && gate_avg_us > rms_norm_avg_us {
        Gdp005Verdict::Pass
    } else {
        Gdp005Verdict::Fail
    }
}

// ===========================================================================
// GDP-006 — Profiler total_tokens > decoded_tokens (different units)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp006Verdict { Pass, Fail }

/// Pass iff `profiler_total_tokens > decoded_tokens` AND
/// `decoded_tokens > 0`. The profiler measures *brick elements*, not
/// decoded tokens, so for a 28-layer model emitting 32 tokens the
/// profiler should report ~9920 (28 * 32 * something).
#[must_use]
pub const fn verdict_from_token_unit_separation(
    profiler_total_tokens: u64,
    decoded_tokens: u64,
) -> Gdp006Verdict {
    if decoded_tokens == 0 { return Gdp006Verdict::Fail; }
    if profiler_total_tokens > decoded_tokens { Gdp006Verdict::Pass } else { Gdp006Verdict::Fail }
}

// ===========================================================================
// GDP-007 — Coverage upper bound: bricks are subset of wall time
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp007Verdict { Pass, Fail }

/// Pass iff `brick_total_ns <= wall_ns`. The brick profiler must
/// never report more time than the elapsed wall time.
#[must_use]
pub const fn verdict_from_coverage_upper_bound(brick_total_ns: u64, wall_ns: u64) -> Gdp007Verdict {
    if wall_ns == 0 { return Gdp007Verdict::Fail; }
    if brick_total_ns <= wall_ns { Gdp007Verdict::Pass } else { Gdp007Verdict::Fail }
}

// ===========================================================================
// GDP-008 — Reproducibility: CV of per-token time < 0.15
// ===========================================================================

pub const AC_GDP_008_MAX_CV: f64 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp008Verdict { Pass, Fail }

/// Compute the coefficient of variation (std-dev / mean) of the
/// per-token time series. Pass iff `cv < 0.15` AND series.len() >= 3.
#[must_use]
pub fn verdict_from_per_token_cv(per_token_us: &[f64]) -> Gdp008Verdict {
    if per_token_us.len() < 3 { return Gdp008Verdict::Fail; }
    if per_token_us.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Gdp008Verdict::Fail;
    }
    let n = per_token_us.len() as f64;
    let mean = per_token_us.iter().sum::<f64>() / n;
    if mean <= 0.0 { return Gdp008Verdict::Fail; }
    let var = per_token_us.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let cv = var.sqrt() / mean;
    if cv < AC_GDP_008_MAX_CV { Gdp008Verdict::Pass } else { Gdp008Verdict::Fail }
}

// ===========================================================================
// GDP-009 — Report fidelity: actual_us within 1% of profiler avg
// ===========================================================================

pub const AC_GDP_009_TOLERANCE_FRAC: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp009Verdict { Pass, Fail }

/// Pass iff `|actual_us - profiler_avg_us| / profiler_avg_us <= 0.01`.
#[must_use]
pub fn verdict_from_actual_us_fidelity(actual_us: f64, profiler_avg_us: f64) -> Gdp009Verdict {
    if !actual_us.is_finite() || !profiler_avg_us.is_finite() { return Gdp009Verdict::Fail; }
    if profiler_avg_us <= 0.0 { return Gdp009Verdict::Fail; }
    let rel = (actual_us - profiler_avg_us).abs() / profiler_avg_us;
    if rel <= AC_GDP_009_TOLERANCE_FRAC { Gdp009Verdict::Pass } else { Gdp009Verdict::Fail }
}

// ===========================================================================
// GDP-010 — Not all brick scores hardcoded to 100
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp010Verdict { Pass, Fail }

/// Pass iff at least one entry in `scores` differs from 100.0.
/// The contract says all-100 indicates `compute_brick_score` was
/// never called (B3 regression).
#[must_use]
pub fn verdict_from_score_diversity(scores: &[f64]) -> Gdp010Verdict {
    if scores.is_empty() { return Gdp010Verdict::Fail; }
    if scores.iter().any(|x| !x.is_finite()) { return Gdp010Verdict::Fail; }
    let any_non_hundred = scores.iter().any(|s| (s - 100.0).abs() > 1e-9);
    if any_non_hundred { Gdp010Verdict::Pass } else { Gdp010Verdict::Fail }
}

// ===========================================================================
// GDP-011 — Brick count == 11 for Qwen 1.5B
// ===========================================================================

pub const AC_GDP_011_QWEN_1_5B_BRICK_COUNT: u64 = 11;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp011Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_brick_count(observed: u64) -> Gdp011Verdict {
    if observed == AC_GDP_011_QWEN_1_5B_BRICK_COUNT { Gdp011Verdict::Pass } else { Gdp011Verdict::Fail }
}

// ===========================================================================
// GDP-012 — Falsification accounting: total_points == len(brick_scores)
//                                     AND passed + failed == total_points
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp012Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_falsification_accounting(
    total_points: u64,
    brick_scores_len: u64,
    passed: u64,
    failed: u64,
) -> Gdp012Verdict {
    if total_points == 0 { return Gdp012Verdict::Fail; }
    if total_points != brick_scores_len { return Gdp012Verdict::Fail; }
    if passed.saturating_add(failed) != total_points { return Gdp012Verdict::Fail; }
    Gdp012Verdict::Pass
}

// ===========================================================================
// GDP-013 — LmHead per-decoded-token cost > 100us
// ===========================================================================

pub const AC_GDP_013_MIN_LM_HEAD_PER_TOK_US: f64 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp013Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_lm_head_per_decoded_tok(lm_head_per_decoded_tok_us: f64) -> Gdp013Verdict {
    if !lm_head_per_decoded_tok_us.is_finite() || lm_head_per_decoded_tok_us < 0.0 {
        return Gdp013Verdict::Fail;
    }
    if lm_head_per_decoded_tok_us > AC_GDP_013_MIN_LM_HEAD_PER_TOK_US {
        Gdp013Verdict::Pass
    } else {
        Gdp013Verdict::Fail
    }
}

// ===========================================================================
// GDP-014 — Report metadata: pmat scores all == 0.0 (no magic constants)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp014Verdict { Pass, Fail }

/// Pass iff all three pmat scores are exactly 0.0 (i.e., not the
/// hardcoded 173.9 / 98.1 / 95.2 magic constants).
#[must_use]
pub fn verdict_from_pmat_score_zero(
    rust_project_score: f64,
    tdg_score: f64,
    cuda_tdg_score: f64,
) -> Gdp014Verdict {
    if !rust_project_score.is_finite() || !tdg_score.is_finite() || !cuda_tdg_score.is_finite() {
        return Gdp014Verdict::Fail;
    }
    if rust_project_score == 0.0 && tdg_score == 0.0 && cuda_tdg_score == 0.0 {
        Gdp014Verdict::Pass
    } else {
        Gdp014Verdict::Fail
    }
}

// ===========================================================================
// GDP-015 — Falsification.total_points must NOT equal magic 137
// ===========================================================================

pub const AC_GDP_015_FORBIDDEN_TOTAL_POINTS: u64 = 137;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gdp015Verdict { Pass, Fail }

/// Pass iff `total_points != 137` (the hardcoded magic constant
/// flagged by B10 regression) AND `total_points > 0`.
#[must_use]
pub const fn verdict_from_no_137_constant(total_points: u64) -> Gdp015Verdict {
    if total_points == 0 { return Gdp015Verdict::Fail; }
    if total_points == AC_GDP_015_FORBIDDEN_TOTAL_POINTS { Gdp015Verdict::Fail } else { Gdp015Verdict::Pass }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- GDP-001 -----------------------------------------------------------

    #[test]
    fn gdp001_pass_above_threshold() {
        assert_eq!(verdict_from_brick_coverage(900, 1000), Gdp001Verdict::Pass);
    }

    #[test]
    fn gdp001_pass_at_exact_threshold() {
        assert_eq!(verdict_from_brick_coverage(850, 1000), Gdp001Verdict::Pass);
    }

    #[test]
    fn gdp001_fail_below_threshold() {
        assert_eq!(verdict_from_brick_coverage(800, 1000), Gdp001Verdict::Fail);
    }

    #[test]
    fn gdp001_fail_zero_wall() {
        assert_eq!(verdict_from_brick_coverage(0, 0), Gdp001Verdict::Fail);
    }

    // ----- GDP-002 -----------------------------------------------------------

    #[test]
    fn gdp002_pass_low_coverage_with_graph() {
        assert_eq!(verdict_from_graph_enabled_coverage(400, 1000), Gdp002Verdict::Pass);
    }

    #[test]
    fn gdp002_fail_above_threshold() {
        assert_eq!(verdict_from_graph_enabled_coverage(600, 1000), Gdp002Verdict::Fail);
    }

    #[test]
    fn gdp002_fail_at_exact_threshold() {
        assert_eq!(verdict_from_graph_enabled_coverage(500, 1000), Gdp002Verdict::Fail);
    }

    // ----- GDP-003 -----------------------------------------------------------

    #[test]
    fn gdp003_immediate_pass_above_100() {
        assert_eq!(
            verdict_from_sync_mode_split(SyncMode::Immediate, 595.0),
            Gdp003Verdict::Pass
        );
    }

    #[test]
    fn gdp003_immediate_fail_below_100() {
        assert_eq!(
            verdict_from_sync_mode_split(SyncMode::Immediate, 50.0),
            Gdp003Verdict::Fail
        );
    }

    #[test]
    fn gdp003_deferred_pass_below_100() {
        assert_eq!(
            verdict_from_sync_mode_split(SyncMode::Deferred, 1.9),
            Gdp003Verdict::Pass
        );
    }

    #[test]
    fn gdp003_deferred_fail_above_100() {
        assert_eq!(
            verdict_from_sync_mode_split(SyncMode::Deferred, 200.0),
            Gdp003Verdict::Fail
        );
    }

    #[test]
    fn gdp003_fail_negative() {
        assert_eq!(
            verdict_from_sync_mode_split(SyncMode::Immediate, -1.0),
            Gdp003Verdict::Fail
        );
    }

    // ----- GDP-004 -----------------------------------------------------------

    #[test]
    fn gdp004_pass_one_per_token() {
        assert_eq!(verdict_from_lm_head_count(32, 32), Gdp004Verdict::Pass);
    }

    #[test]
    fn gdp004_fail_zero_decoded() {
        assert_eq!(verdict_from_lm_head_count(0, 0), Gdp004Verdict::Fail);
    }

    #[test]
    fn gdp004_fail_extra_calls() {
        assert_eq!(verdict_from_lm_head_count(33, 32), Gdp004Verdict::Fail);
    }

    #[test]
    fn gdp004_fail_missing_call() {
        assert_eq!(verdict_from_lm_head_count(31, 32), Gdp004Verdict::Fail);
    }

    // ----- GDP-005 -----------------------------------------------------------

    #[test]
    fn gdp005_pass_canonical() {
        assert_eq!(verdict_from_brick_ordering(595.0, 100.0, 5.0), Gdp005Verdict::Pass);
    }

    #[test]
    fn gdp005_fail_lm_head_below_gate() {
        assert_eq!(verdict_from_brick_ordering(50.0, 100.0, 5.0), Gdp005Verdict::Fail);
    }

    #[test]
    fn gdp005_fail_equal_pair() {
        // Strict inequality required.
        assert_eq!(verdict_from_brick_ordering(100.0, 100.0, 5.0), Gdp005Verdict::Fail);
    }

    // ----- GDP-006 -----------------------------------------------------------

    #[test]
    fn gdp006_pass_28_layer() {
        // 28 layers × 32 tokens × ~11 bricks ≈ 9920.
        assert_eq!(verdict_from_token_unit_separation(9920, 32), Gdp006Verdict::Pass);
    }

    #[test]
    fn gdp006_fail_equal() {
        // API-misuse signal: profiler reporting decoded_tokens.
        assert_eq!(verdict_from_token_unit_separation(32, 32), Gdp006Verdict::Fail);
    }

    #[test]
    fn gdp006_fail_zero_decoded() {
        assert_eq!(verdict_from_token_unit_separation(100, 0), Gdp006Verdict::Fail);
    }

    // ----- GDP-007 -----------------------------------------------------------

    #[test]
    fn gdp007_pass_subset() {
        assert_eq!(verdict_from_coverage_upper_bound(900, 1000), Gdp007Verdict::Pass);
    }

    #[test]
    fn gdp007_pass_equal_max() {
        assert_eq!(verdict_from_coverage_upper_bound(1000, 1000), Gdp007Verdict::Pass);
    }

    #[test]
    fn gdp007_fail_overcounting() {
        // Bricks claim more time than wall — must Fail.
        assert_eq!(verdict_from_coverage_upper_bound(1500, 1000), Gdp007Verdict::Fail);
    }

    // ----- GDP-008 -----------------------------------------------------------

    #[test]
    fn gdp008_pass_low_cv() {
        let series = vec![100.0, 101.0, 99.0, 100.5, 100.2];
        assert_eq!(verdict_from_per_token_cv(&series), Gdp008Verdict::Pass);
    }

    #[test]
    fn gdp008_fail_high_cv() {
        let series = vec![100.0, 200.0, 50.0];
        assert_eq!(verdict_from_per_token_cv(&series), Gdp008Verdict::Fail);
    }

    #[test]
    fn gdp008_fail_too_few_samples() {
        let series = vec![100.0, 101.0];
        assert_eq!(verdict_from_per_token_cv(&series), Gdp008Verdict::Fail);
    }

    #[test]
    fn gdp008_fail_nan() {
        let series = vec![100.0, f64::NAN, 99.0];
        assert_eq!(verdict_from_per_token_cv(&series), Gdp008Verdict::Fail);
    }

    // ----- GDP-009 -----------------------------------------------------------

    #[test]
    fn gdp009_pass_within_1pct() {
        assert_eq!(verdict_from_actual_us_fidelity(595.0, 595.0), Gdp009Verdict::Pass);
        assert_eq!(verdict_from_actual_us_fidelity(595.5, 595.0), Gdp009Verdict::Pass);
    }

    #[test]
    fn gdp009_fail_above_tolerance() {
        // 1.9 vs 595 — the B1 regression (per-element vs per-call).
        assert_eq!(verdict_from_actual_us_fidelity(1.9, 595.0), Gdp009Verdict::Fail);
    }

    #[test]
    fn gdp009_fail_zero_avg() {
        assert_eq!(verdict_from_actual_us_fidelity(0.0, 0.0), Gdp009Verdict::Fail);
    }

    // ----- GDP-010 -----------------------------------------------------------

    #[test]
    fn gdp010_pass_diverse_scores() {
        let scores = vec![100.0, 95.0, 88.0, 100.0];
        assert_eq!(verdict_from_score_diversity(&scores), Gdp010Verdict::Pass);
    }

    #[test]
    fn gdp010_fail_all_hundred() {
        let scores = vec![100.0, 100.0, 100.0];
        assert_eq!(verdict_from_score_diversity(&scores), Gdp010Verdict::Fail);
    }

    #[test]
    fn gdp010_fail_empty() {
        assert_eq!(verdict_from_score_diversity(&[]), Gdp010Verdict::Fail);
    }

    // ----- GDP-011 -----------------------------------------------------------

    #[test]
    fn gdp011_pass_eleven_bricks() {
        assert_eq!(verdict_from_brick_count(11), Gdp011Verdict::Pass);
    }

    #[test]
    fn gdp011_fail_off_by_one() {
        assert_eq!(verdict_from_brick_count(10), Gdp011Verdict::Fail);
        assert_eq!(verdict_from_brick_count(12), Gdp011Verdict::Fail);
    }

    #[test]
    fn gdp011_provenance() {
        assert_eq!(AC_GDP_011_QWEN_1_5B_BRICK_COUNT, 11);
    }

    // ----- GDP-012 -----------------------------------------------------------

    #[test]
    fn gdp012_pass_consistent() {
        assert_eq!(
            verdict_from_falsification_accounting(11, 11, 8, 3),
            Gdp012Verdict::Pass
        );
    }

    #[test]
    fn gdp012_fail_total_mismatch() {
        assert_eq!(
            verdict_from_falsification_accounting(11, 12, 8, 3),
            Gdp012Verdict::Fail
        );
    }

    #[test]
    fn gdp012_fail_passed_failed_dont_sum() {
        assert_eq!(
            verdict_from_falsification_accounting(11, 11, 8, 4),
            Gdp012Verdict::Fail
        );
    }

    #[test]
    fn gdp012_fail_zero_total() {
        assert_eq!(
            verdict_from_falsification_accounting(0, 0, 0, 0),
            Gdp012Verdict::Fail
        );
    }

    // ----- GDP-013 -----------------------------------------------------------

    #[test]
    fn gdp013_pass_above_100() {
        assert_eq!(verdict_from_lm_head_per_decoded_tok(595.0), Gdp013Verdict::Pass);
    }

    #[test]
    fn gdp013_fail_at_or_below_100() {
        assert_eq!(verdict_from_lm_head_per_decoded_tok(100.0), Gdp013Verdict::Fail);
        assert_eq!(verdict_from_lm_head_per_decoded_tok(1.9), Gdp013Verdict::Fail);
    }

    #[test]
    fn gdp013_fail_negative() {
        assert_eq!(verdict_from_lm_head_per_decoded_tok(-1.0), Gdp013Verdict::Fail);
    }

    // ----- GDP-014 -----------------------------------------------------------

    #[test]
    fn gdp014_pass_all_zero() {
        assert_eq!(verdict_from_pmat_score_zero(0.0, 0.0, 0.0), Gdp014Verdict::Pass);
    }

    #[test]
    fn gdp014_fail_hardcoded_constants() {
        assert_eq!(
            verdict_from_pmat_score_zero(173.9, 98.1, 95.2),
            Gdp014Verdict::Fail
        );
    }

    #[test]
    fn gdp014_fail_one_nonzero() {
        assert_eq!(verdict_from_pmat_score_zero(0.0, 1.0, 0.0), Gdp014Verdict::Fail);
    }

    #[test]
    fn gdp014_fail_nan() {
        assert_eq!(
            verdict_from_pmat_score_zero(f64::NAN, 0.0, 0.0),
            Gdp014Verdict::Fail
        );
    }

    // ----- GDP-015 -----------------------------------------------------------

    #[test]
    fn gdp015_pass_canonical_count() {
        assert_eq!(verdict_from_no_137_constant(11), Gdp015Verdict::Pass);
    }

    #[test]
    fn gdp015_fail_magic_137() {
        assert_eq!(verdict_from_no_137_constant(137), Gdp015Verdict::Fail);
    }

    #[test]
    fn gdp015_fail_zero() {
        assert_eq!(verdict_from_no_137_constant(0), Gdp015Verdict::Fail);
    }

    #[test]
    fn gdp015_provenance() {
        assert_eq!(AC_GDP_015_FORBIDDEN_TOTAL_POINTS, 137);
    }
}
