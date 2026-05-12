// SHIP-TWO-001 — `crux-competitive-research-ux-v1` algorithm-level
// PARTIAL discharge for FALSIFY-CRUX-001..010 (closes 10/10 sweep).
//
// Contract: `contracts/crux-competitive-research-ux-v1.yaml`.

// ===========================================================================
// CRUX-001 — stories[] contains exactly 250 entries
// ===========================================================================

pub const AC_CRUX_001_REQUIRED_STORY_COUNT: usize = 250;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_story_count(observed: usize) -> Crux001Verdict {
    if observed == AC_CRUX_001_REQUIRED_STORY_COUNT { Crux001Verdict::Pass } else { Crux001Verdict::Fail }
}

// ===========================================================================
// CRUX-002 — Every story.contract path exists in contracts/
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux002Verdict { Pass, Fail }

/// Pass iff every name in `referenced_contracts` is present in
/// `existing_contracts` AND `referenced_contracts` is non-empty.
#[must_use]
pub fn verdict_from_contract_paths_exist(
    referenced: &[String],
    existing: &[String],
) -> Crux002Verdict {
    if referenced.is_empty() { return Crux002Verdict::Fail; }
    for c in referenced {
        if !existing.iter().any(|e| e == c) { return Crux002Verdict::Fail; }
    }
    Crux002Verdict::Pass
}

// ===========================================================================
// CRUX-003 — `status=supported` story has a golden test
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux003Verdict { Pass, Fail }

/// Pass iff every story marked `supported` has `has_golden == true`.
#[must_use]
pub fn verdict_from_supported_have_golden(stories: &[(bool, bool)]) -> Crux003Verdict {
    if stories.is_empty() { return Crux003Verdict::Fail; }
    for &(supported, has_golden) in stories {
        if supported && !has_golden { return Crux003Verdict::Fail; }
    }
    Crux003Verdict::Pass
}

// ===========================================================================
// CRUX-004 — Category J entries all have `interpretation: pending`
//            (until OpenCLAW resolved)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux004Verdict { Pass, Fail }

/// `category_j_interpretations[i]` is the interpretation field of the
/// i-th category-J story. Pass iff every entry equals `"pending"`.
#[must_use]
pub fn verdict_from_category_j_pending(category_j_interpretations: &[&str]) -> Crux004Verdict {
    if category_j_interpretations.is_empty() { return Crux004Verdict::Fail; }
    for s in category_j_interpretations {
        if *s != "pending" { return Crux004Verdict::Fail; }
    }
    Crux004Verdict::Pass
}

// ===========================================================================
// CRUX-005 — Coverage monotonicity: ✅→🔨 or ✅→❌ requires defect ticket
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CruxStatus { Supported, InProgress, Missing, Pending }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux005Verdict { Pass, Fail }

/// Pass iff a regression (`Supported → !Supported`) is accompanied by
/// a defect ticket. Forward progress (`!Supported → Supported`) is
/// always Pass.
#[must_use]
pub const fn verdict_from_coverage_monotone(
    prev: CruxStatus,
    next: CruxStatus,
    has_defect_ticket: bool,
) -> Crux005Verdict {
    let is_regression = matches!(prev, CruxStatus::Supported) && !matches!(next, CruxStatus::Supported);
    if is_regression && !has_defect_ticket { Crux005Verdict::Fail } else { Crux005Verdict::Pass }
}

// ===========================================================================
// CRUX-006 — SDK compatibility: K-01/K-02 stories supported ⇒ tests pass
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux006Verdict { Pass, Fail }

/// Pass iff `(supported_claim) → (sdk_tests_pass)`. If supported is
/// not claimed, the gate is vacuously true.
#[must_use]
pub const fn verdict_from_sdk_compatibility(
    supported_claim: bool,
    sdk_tests_pass: bool,
) -> Crux006Verdict {
    if supported_claim && !sdk_tests_pass { Crux006Verdict::Fail } else { Crux006Verdict::Pass }
}

// ===========================================================================
// CRUX-007 — `status=missing` ⇒ pmat work ticket exists
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux007Verdict { Pass, Fail }

/// Pass iff every `(status, has_ticket)` pair satisfies:
///   `status == Missing` ⇒ `has_ticket == true`.
#[must_use]
pub fn verdict_from_missing_has_ticket(stories: &[(CruxStatus, bool)]) -> Crux007Verdict {
    if stories.is_empty() { return Crux007Verdict::Fail; }
    for &(status, has_ticket) in stories {
        if matches!(status, CruxStatus::Missing) && !has_ticket {
            return Crux007Verdict::Fail;
        }
    }
    Crux007Verdict::Pass
}

// ===========================================================================
// CRUX-008 — pmat work priority matches demand_score
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux008Verdict { Pass, Fail }

/// Canonical mapping: 5 → critical, 4 → high, 3 → medium,
/// 2 → low, 1 → minor.
#[must_use]
pub fn expected_priority(demand_score: u8) -> Option<&'static str> {
    match demand_score {
        5 => Some("critical"),
        4 => Some("high"),
        3 => Some("medium"),
        2 => Some("low"),
        1 => Some("minor"),
        _ => None,
    }
}

#[must_use]
pub fn verdict_from_priority_alignment(demand_score: u8, priority: &str) -> Crux008Verdict {
    match expected_priority(demand_score) {
        Some(expected) if expected == priority => Crux008Verdict::Pass,
        _ => Crux008Verdict::Fail,
    }
}

// ===========================================================================
// CRUX-009 — demand_score in [1, 5]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux009Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_demand_score_range(demand_score: u8) -> Crux009Verdict {
    if matches!(demand_score, 1..=5) { Crux009Verdict::Pass } else { Crux009Verdict::Fail }
}

// ===========================================================================
// CRUX-010 — Subspec coverage table matches stories[] status counts
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crux010Verdict { Pass, Fail }

/// `(supported, in_progress, missing, pending)` counts from each source.
/// Pass iff yaml and md tuples match exactly.
#[must_use]
pub fn verdict_from_coverage_table_match(
    yaml_counts: (u64, u64, u64, u64),
    md_counts: (u64, u64, u64, u64),
) -> Crux010Verdict {
    if yaml_counts == md_counts { Crux010Verdict::Pass } else { Crux010Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CRUX-001
    #[test] fn crux001_pass_250() { assert_eq!(verdict_from_story_count(250), Crux001Verdict::Pass); }
    #[test] fn crux001_fail_249() { assert_eq!(verdict_from_story_count(249), Crux001Verdict::Fail); }
    #[test] fn crux001_fail_251() { assert_eq!(verdict_from_story_count(251), Crux001Verdict::Fail); }
    #[test] fn crux001_fail_zero() { assert_eq!(verdict_from_story_count(0), Crux001Verdict::Fail); }

    // CRUX-002
    fn s(items: &[&str]) -> Vec<String> { items.iter().map(|i| i.to_string()).collect() }

    #[test] fn crux002_pass_match() {
        let refd = s(&["a-v1.yaml", "b-v1.yaml"]);
        let exists = s(&["a-v1.yaml", "b-v1.yaml", "c-v1.yaml"]);
        assert_eq!(verdict_from_contract_paths_exist(&refd, &exists), Crux002Verdict::Pass);
    }
    #[test] fn crux002_fail_missing() {
        let refd = s(&["a-v1.yaml", "missing-v1.yaml"]);
        let exists = s(&["a-v1.yaml"]);
        assert_eq!(verdict_from_contract_paths_exist(&refd, &exists), Crux002Verdict::Fail);
    }
    #[test] fn crux002_fail_empty_refs() {
        let exists = s(&["a-v1.yaml"]);
        assert_eq!(verdict_from_contract_paths_exist(&[], &exists), Crux002Verdict::Fail);
    }

    // CRUX-003
    #[test] fn crux003_pass_supported_with_golden() {
        let stories = vec![(true, true), (false, false), (true, true)];
        assert_eq!(verdict_from_supported_have_golden(&stories), Crux003Verdict::Pass);
    }
    #[test] fn crux003_fail_supported_no_golden() {
        let stories = vec![(true, true), (true, false)];
        assert_eq!(verdict_from_supported_have_golden(&stories), Crux003Verdict::Fail);
    }
    #[test] fn crux003_pass_unsupported_no_golden() {
        // Vacuously true: not supported → no golden requirement.
        let stories = vec![(false, false), (false, false)];
        assert_eq!(verdict_from_supported_have_golden(&stories), Crux003Verdict::Pass);
    }
    #[test] fn crux003_fail_empty() {
        assert_eq!(verdict_from_supported_have_golden(&[]), Crux003Verdict::Fail);
    }

    // CRUX-004
    #[test] fn crux004_pass_all_pending() {
        let interps = ["pending", "pending", "pending"];
        assert_eq!(verdict_from_category_j_pending(&interps), Crux004Verdict::Pass);
    }
    #[test] fn crux004_fail_resolved_too_early() {
        let interps = ["pending", "openclaw_resolved", "pending"];
        assert_eq!(verdict_from_category_j_pending(&interps), Crux004Verdict::Fail);
    }
    #[test] fn crux004_fail_empty() {
        assert_eq!(verdict_from_category_j_pending(&[]), Crux004Verdict::Fail);
    }

    // CRUX-005
    #[test] fn crux005_pass_no_regression() {
        assert_eq!(
            verdict_from_coverage_monotone(CruxStatus::Missing, CruxStatus::Supported, false),
            Crux005Verdict::Pass
        );
    }
    #[test] fn crux005_pass_regression_with_ticket() {
        assert_eq!(
            verdict_from_coverage_monotone(CruxStatus::Supported, CruxStatus::Missing, true),
            Crux005Verdict::Pass
        );
    }
    #[test] fn crux005_fail_regression_no_ticket() {
        // ✅ → ❌ without defect ticket — Fail.
        assert_eq!(
            verdict_from_coverage_monotone(CruxStatus::Supported, CruxStatus::Missing, false),
            Crux005Verdict::Fail
        );
    }
    #[test] fn crux005_fail_regression_in_progress_no_ticket() {
        assert_eq!(
            verdict_from_coverage_monotone(CruxStatus::Supported, CruxStatus::InProgress, false),
            Crux005Verdict::Fail
        );
    }

    // CRUX-006
    #[test] fn crux006_pass_supported_and_passing() {
        assert_eq!(verdict_from_sdk_compatibility(true, true), Crux006Verdict::Pass);
    }
    #[test] fn crux006_pass_not_claimed() {
        assert_eq!(verdict_from_sdk_compatibility(false, false), Crux006Verdict::Pass);
    }
    #[test] fn crux006_fail_supported_but_failing() {
        assert_eq!(verdict_from_sdk_compatibility(true, false), Crux006Verdict::Fail);
    }

    // CRUX-007
    #[test] fn crux007_pass_missing_has_ticket() {
        let stories = vec![
            (CruxStatus::Missing, true),
            (CruxStatus::Supported, false),
            (CruxStatus::Missing, true),
        ];
        assert_eq!(verdict_from_missing_has_ticket(&stories), Crux007Verdict::Pass);
    }
    #[test] fn crux007_fail_missing_no_ticket() {
        let stories = vec![(CruxStatus::Missing, false)];
        assert_eq!(verdict_from_missing_has_ticket(&stories), Crux007Verdict::Fail);
    }
    #[test] fn crux007_fail_empty() {
        assert_eq!(verdict_from_missing_has_ticket(&[]), Crux007Verdict::Fail);
    }

    // CRUX-008
    #[test] fn crux008_pass_critical() { assert_eq!(verdict_from_priority_alignment(5, "critical"), Crux008Verdict::Pass); }
    #[test] fn crux008_pass_high() { assert_eq!(verdict_from_priority_alignment(4, "high"), Crux008Verdict::Pass); }
    #[test] fn crux008_pass_medium() { assert_eq!(verdict_from_priority_alignment(3, "medium"), Crux008Verdict::Pass); }
    #[test] fn crux008_pass_low() { assert_eq!(verdict_from_priority_alignment(2, "low"), Crux008Verdict::Pass); }
    #[test] fn crux008_pass_minor() { assert_eq!(verdict_from_priority_alignment(1, "minor"), Crux008Verdict::Pass); }
    #[test] fn crux008_fail_score_5_high() { assert_eq!(verdict_from_priority_alignment(5, "high"), Crux008Verdict::Fail); }
    #[test] fn crux008_fail_invalid_score() { assert_eq!(verdict_from_priority_alignment(6, "critical"), Crux008Verdict::Fail); }

    // CRUX-009
    #[test] fn crux009_pass_in_range() {
        for score in 1..=5_u8 {
            assert_eq!(verdict_from_demand_score_range(score), Crux009Verdict::Pass);
        }
    }
    #[test] fn crux009_fail_zero() { assert_eq!(verdict_from_demand_score_range(0), Crux009Verdict::Fail); }
    #[test] fn crux009_fail_six() { assert_eq!(verdict_from_demand_score_range(6), Crux009Verdict::Fail); }
    #[test] fn crux009_fail_max() { assert_eq!(verdict_from_demand_score_range(255), Crux009Verdict::Fail); }

    // CRUX-010
    #[test] fn crux010_pass_match() {
        let yaml = (50, 30, 20, 150);
        let md = (50, 30, 20, 150);
        assert_eq!(verdict_from_coverage_table_match(yaml, md), Crux010Verdict::Pass);
    }
    #[test] fn crux010_fail_drift_supported() {
        let yaml = (50, 30, 20, 150);
        let md = (51, 30, 20, 150);
        assert_eq!(verdict_from_coverage_table_match(yaml, md), Crux010Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_story_count() {
        assert_eq!(AC_CRUX_001_REQUIRED_STORY_COUNT, 250);
    }
}
