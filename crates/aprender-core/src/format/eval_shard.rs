// SHIP-TWO-001 — `eval-sharding-v1` algorithm-level PARTIAL
// discharge for FALSIFY-SHARD-001..004 (closes 4/4).
//
// Contract: `contracts/eval-sharding-v1.yaml`.

// ===========================================================================
// SHARD-001 — Completeness: every benchmark task appears in some shard
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard001Verdict { Pass, Fail }

/// Pass iff `union_size == bench_size AND missing_count == 0`.
#[must_use]
pub fn verdict_from_completeness(
    bench_size: u64,
    union_size: u64,
    missing_count: u64,
) -> Shard001Verdict {
    if bench_size == 0 { return Shard001Verdict::Fail; }
    if union_size == bench_size && missing_count == 0 {
        Shard001Verdict::Pass
    } else {
        Shard001Verdict::Fail
    }
}

// ===========================================================================
// SHARD-002 — Disjointness: no task_id appears in two shards
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard002Verdict { Pass, Fail }

/// Pass iff `total_pairwise_intersection_size == 0`.
#[must_use]
pub fn verdict_from_disjointness(total_pairwise_intersection_size: u64) -> Shard002Verdict {
    if total_pairwise_intersection_size == 0 {
        Shard002Verdict::Pass
    } else {
        Shard002Verdict::Fail
    }
}

// ===========================================================================
// SHARD-003 — Determinism: T=0 byte-identical on same host twice
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard003Verdict { Pass, Fail }

/// Pass iff `run_a == run_b` (both non-empty).
#[must_use]
pub fn verdict_from_determinism(run_a: &[u8], run_b: &[u8]) -> Shard003Verdict {
    if run_a.is_empty() || run_b.is_empty() { return Shard003Verdict::Fail; }
    if run_a == run_b { Shard003Verdict::Pass } else { Shard003Verdict::Fail }
}

// ===========================================================================
// SHARD-004 — Merged-score identity within 0.01 pp
// ===========================================================================

pub const AC_SHARD_004_TOLERANCE_PP: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard004Verdict { Pass, Fail }

/// Pass iff `|reference_pp - merged_pp| <= 0.01`.
#[must_use]
pub fn verdict_from_merged_score_identity(
    reference_pp: f64,
    merged_pp: f64,
) -> Shard004Verdict {
    if !reference_pp.is_finite() || !merged_pp.is_finite() {
        return Shard004Verdict::Fail;
    }
    if (reference_pp - merged_pp).abs() <= AC_SHARD_004_TOLERANCE_PP {
        Shard004Verdict::Pass
    } else {
        Shard004Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // SHARD-001
    #[test] fn s001_pass_complete() { assert_eq!(verdict_from_completeness(164, 164, 0), Shard001Verdict::Pass); }
    #[test] fn s001_fail_missing() { assert_eq!(verdict_from_completeness(164, 163, 1), Shard001Verdict::Fail); }
    #[test] fn s001_fail_extra() { assert_eq!(verdict_from_completeness(164, 165, 0), Shard001Verdict::Fail); }
    #[test] fn s001_fail_empty_bench() { assert_eq!(verdict_from_completeness(0, 0, 0), Shard001Verdict::Fail); }

    // SHARD-002
    #[test] fn s002_pass_disjoint() { assert_eq!(verdict_from_disjointness(0), Shard002Verdict::Pass); }
    #[test] fn s002_fail_overlap() { assert_eq!(verdict_from_disjointness(1), Shard002Verdict::Fail); }
    #[test] fn s002_fail_many_overlaps() { assert_eq!(verdict_from_disjointness(50), Shard002Verdict::Fail); }

    // SHARD-003
    #[test] fn s003_pass_identical() { assert_eq!(verdict_from_determinism(b"abc", b"abc"), Shard003Verdict::Pass); }
    #[test] fn s003_fail_drift() { assert_eq!(verdict_from_determinism(b"abc", b"abd"), Shard003Verdict::Fail); }
    #[test] fn s003_fail_empty() { assert_eq!(verdict_from_determinism(&[], &[]), Shard003Verdict::Fail); }

    // SHARD-004
    #[test] fn s004_pass_exact() { assert_eq!(verdict_from_merged_score_identity(86.0, 86.0), Shard004Verdict::Pass); }
    #[test] fn s004_pass_within_tolerance() { assert_eq!(verdict_from_merged_score_identity(86.0, 86.01), Shard004Verdict::Pass); }
    #[test] fn s004_fail_above_tolerance() { assert_eq!(verdict_from_merged_score_identity(86.0, 86.02), Shard004Verdict::Fail); }
    #[test] fn s004_fail_nan() { assert_eq!(verdict_from_merged_score_identity(f64::NAN, 86.0), Shard004Verdict::Fail); }

    #[test] fn provenance_tolerance() { assert_eq!(AC_SHARD_004_TOLERANCE_PP, 0.01); }
}
