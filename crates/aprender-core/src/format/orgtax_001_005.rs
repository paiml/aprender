// `apr-org-taxonomy-v1` algorithm-level PARTIAL discharge for the 5
// falsification conditions (orphan classification, archive-deadline,
// stale-archive-deadline, legacy-redirect, category-count sum).
//
// Contract: `contracts/apr-org-taxonomy-v1.yaml`.
//
// The contract has 5 unkeyed `falsification:` entries. We pin local IDs
// FALSIFY-ORGTAX-001..005 to give each a stable verdict function:
//
//   001 — every repo classified (no orphan in `gh repo list`)
//   002 — SHOULD-MERGE repo archived within 7 days
//   003 — STALE repo archived within 7 days
//   004 — repo description contains legacy name without MOVED redirect
//   005 — sum(category_counts) == total_repos
//
// Live discharge: `gh repo list paiml --limit 300 --json` + a
// classification map. Algorithm-level pinning prevents drift on the
// invariant predicates regardless of how the live shell harness fetches
// the org state.

use std::collections::HashSet;

/// Total repos enumerated in the contract metadata.
pub const AC_ORGTAX_TOTAL_REPOS: u32 = 205;

/// Number of categories in the taxonomy.
pub const AC_ORGTAX_CATEGORIES: u32 = 15;

/// Archive deadline for SHOULD-MERGE / STALE repos.
pub const AC_ORGTAX_ARCHIVE_DEADLINE_DAYS: u32 = 7;

/// Canonical category names. The contract `categories:` map MUST contain
/// exactly these 15 keys.
pub const AC_ORGTAX_CATEGORY_NAMES: [&str; 15] = [
    "monorepo",
    "merged",
    "should_merge",
    "active_tool",
    "model_training",
    "poc_benchmark",
    "course_demo",
    "legacy_book",
    "legacy_library",
    "ground_truth",
    "transpiler",
    "ruchy",
    "infra_platform",
    "pmat",
    "stale_archive",
];

/// Canonical category counts from contract v1.0.0.
pub const AC_ORGTAX_CATEGORY_COUNTS: [(&str, u32); 15] = [
    ("monorepo", 1),
    ("merged", 14),
    ("should_merge", 6),
    ("active_tool", 21),
    ("model_training", 3),
    ("poc_benchmark", 8),
    ("course_demo", 34),
    ("legacy_book", 15),
    ("legacy_library", 18),
    ("ground_truth", 13),
    ("transpiler", 9),
    ("ruchy", 8),
    ("infra_platform", 25),
    ("pmat", 6),
    ("stale_archive", 24),
];

/// Legacy names whose appearance in repo descriptions REQUIRES a MOVED
/// redirect (consolidated into aprender per APR-MONO).
pub const AC_ORGTAX_LEGACY_NAMES: [&str; 4] = ["trueno", "realizar", "entrenar", "batuta"];

// =============================================================================
// FALSIFY-ORGTAX-001 — every repo classified
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrphanClassificationVerdict {
    /// No repo from the live `gh repo list` is missing a category.
    Pass,
    /// At least one repo has no category assignment.
    Fail,
}

#[must_use]
pub fn verdict_from_orphan_classification(
    live_repos: &[&str],
    classified_repos: &[&str],
) -> OrphanClassificationVerdict {
    let classified: HashSet<&&str> = classified_repos.iter().collect();
    for repo in live_repos {
        if !classified.contains(repo) {
            return OrphanClassificationVerdict::Fail;
        }
    }
    OrphanClassificationVerdict::Pass
}

// =============================================================================
// FALSIFY-ORGTAX-002 — SHOULD-MERGE repo archived within 7 days
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShouldMergeArchiveVerdict {
    /// Repo archived OR within deadline.
    Pass,
    /// Repo not archived AND age > 7 days.
    Fail,
}

#[must_use]
pub fn verdict_from_should_merge_archive(
    archived: bool,
    days_since_classification: u32,
) -> ShouldMergeArchiveVerdict {
    if archived {
        return ShouldMergeArchiveVerdict::Pass;
    }
    if days_since_classification > AC_ORGTAX_ARCHIVE_DEADLINE_DAYS {
        ShouldMergeArchiveVerdict::Fail
    } else {
        ShouldMergeArchiveVerdict::Pass
    }
}

// =============================================================================
// FALSIFY-ORGTAX-003 — STALE repo archived within 7 days
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleArchiveVerdict {
    /// Repo archived OR within deadline.
    Pass,
    /// Repo not archived AND age > 7 days.
    Fail,
}

#[must_use]
pub fn verdict_from_stale_archive(
    archived: bool,
    days_since_classification: u32,
) -> StaleArchiveVerdict {
    if archived {
        return StaleArchiveVerdict::Pass;
    }
    if days_since_classification > AC_ORGTAX_ARCHIVE_DEADLINE_DAYS {
        StaleArchiveVerdict::Fail
    } else {
        StaleArchiveVerdict::Pass
    }
}

// =============================================================================
// FALSIFY-ORGTAX-004 — legacy name in description requires MOVED redirect
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyRedirectVerdict {
    /// Description either has no legacy name OR includes "MOVED" tag.
    Pass,
    /// Legacy name present without "MOVED" redirect.
    Fail,
}

#[must_use]
pub fn verdict_from_legacy_redirect(description: &str) -> LegacyRedirectVerdict {
    let lower = description.to_lowercase();
    let mentions_legacy = AC_ORGTAX_LEGACY_NAMES.iter().any(|n| lower.contains(n));
    if !mentions_legacy {
        return LegacyRedirectVerdict::Pass;
    }
    if lower.contains("moved") {
        LegacyRedirectVerdict::Pass
    } else {
        LegacyRedirectVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-ORGTAX-005 — category counts sum to total_repos
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategorySumVerdict {
    /// sum(category_counts) == total_repos.
    Pass,
    /// Sum mismatch (recount required).
    Fail,
}

#[must_use]
pub fn verdict_from_category_sum(category_counts: &[(&str, u32)], total_repos: u32) -> CategorySumVerdict {
    let sum: u32 = category_counts.iter().map(|(_, n)| n).sum();
    if sum == total_repos {
        CategorySumVerdict::Pass
    } else {
        CategorySumVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_total_repos_205() {
        assert_eq!(AC_ORGTAX_TOTAL_REPOS, 205);
    }

    #[test]
    fn provenance_categories_15() {
        assert_eq!(AC_ORGTAX_CATEGORIES, 15);
        assert_eq!(AC_ORGTAX_CATEGORY_NAMES.len(), 15);
        assert_eq!(AC_ORGTAX_CATEGORY_COUNTS.len(), 15);
    }

    #[test]
    fn provenance_canonical_counts_sum_to_total() {
        let sum: u32 = AC_ORGTAX_CATEGORY_COUNTS.iter().map(|(_, n)| n).sum();
        assert_eq!(sum, AC_ORGTAX_TOTAL_REPOS, "contract category counts must sum to {}", AC_ORGTAX_TOTAL_REPOS);
    }

    #[test]
    fn provenance_archive_deadline_7() {
        assert_eq!(AC_ORGTAX_ARCHIVE_DEADLINE_DAYS, 7);
    }

    #[test]
    fn provenance_legacy_names() {
        assert_eq!(AC_ORGTAX_LEGACY_NAMES.len(), 4);
        for n in AC_ORGTAX_LEGACY_NAMES {
            assert!(["trueno", "realizar", "entrenar", "batuta"].contains(&n));
        }
    }

    // -------------------------------------------------------------------------
    // Section 2: ORGTAX-001 orphan classification.
    // -------------------------------------------------------------------------
    #[test]
    fn ot001_pass_all_classified() {
        let live = ["aprender", "ruchy", "pmat"];
        let classified = ["aprender", "ruchy", "pmat", "older-archive"];
        assert_eq!(
            verdict_from_orphan_classification(&live, &classified),
            OrphanClassificationVerdict::Pass
        );
    }

    #[test]
    fn ot001_fail_orphan_repo() {
        let live = ["aprender", "new-repo", "pmat"];
        let classified = ["aprender", "pmat"];
        assert_eq!(
            verdict_from_orphan_classification(&live, &classified),
            OrphanClassificationVerdict::Fail
        );
    }

    #[test]
    fn ot001_pass_empty_live() {
        // No live repos ⇒ vacuously classified.
        let classified = ["aprender"];
        assert_eq!(
            verdict_from_orphan_classification(&[], &classified),
            OrphanClassificationVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: ORGTAX-002 SHOULD-MERGE archive deadline.
    // -------------------------------------------------------------------------
    #[test]
    fn ot002_pass_archived() {
        assert_eq!(verdict_from_should_merge_archive(true, 0), ShouldMergeArchiveVerdict::Pass);
    }

    #[test]
    fn ot002_pass_archived_late() {
        assert_eq!(verdict_from_should_merge_archive(true, 100), ShouldMergeArchiveVerdict::Pass);
    }

    #[test]
    fn ot002_pass_within_deadline() {
        assert_eq!(verdict_from_should_merge_archive(false, 7), ShouldMergeArchiveVerdict::Pass);
    }

    #[test]
    fn ot002_fail_past_deadline() {
        assert_eq!(verdict_from_should_merge_archive(false, 8), ShouldMergeArchiveVerdict::Fail);
    }

    #[test]
    fn ot002_fail_long_overdue() {
        assert_eq!(verdict_from_should_merge_archive(false, 30), ShouldMergeArchiveVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: ORGTAX-003 STALE archive deadline.
    // -------------------------------------------------------------------------
    #[test]
    fn ot003_pass_archived() {
        assert_eq!(verdict_from_stale_archive(true, 0), StaleArchiveVerdict::Pass);
    }

    #[test]
    fn ot003_pass_within_deadline() {
        assert_eq!(verdict_from_stale_archive(false, 7), StaleArchiveVerdict::Pass);
    }

    #[test]
    fn ot003_fail_past_deadline() {
        assert_eq!(verdict_from_stale_archive(false, 14), StaleArchiveVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: ORGTAX-004 legacy redirect.
    // -------------------------------------------------------------------------
    #[test]
    fn ot004_pass_no_legacy_name() {
        assert_eq!(verdict_from_legacy_redirect("ML framework"), LegacyRedirectVerdict::Pass);
    }

    #[test]
    fn ot004_pass_legacy_with_moved() {
        let d = "trueno (MOVED to aprender — see paiml/aprender)";
        assert_eq!(verdict_from_legacy_redirect(d), LegacyRedirectVerdict::Pass);
    }

    #[test]
    fn ot004_pass_moved_lowercase() {
        let d = "realizar — moved to aprender";
        assert_eq!(verdict_from_legacy_redirect(d), LegacyRedirectVerdict::Pass);
    }

    #[test]
    fn ot004_fail_legacy_no_redirect() {
        let d = "Trueno SIMD primitives library";
        assert_eq!(verdict_from_legacy_redirect(d), LegacyRedirectVerdict::Fail);
    }

    #[test]
    fn ot004_fail_each_legacy_name() {
        for name in AC_ORGTAX_LEGACY_NAMES {
            let d = format!("Active {name} development.");
            assert_eq!(
                verdict_from_legacy_redirect(&d),
                LegacyRedirectVerdict::Fail,
                "legacy name {name} without MOVED must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: ORGTAX-005 category sum.
    // -------------------------------------------------------------------------
    #[test]
    fn ot005_pass_canonical_counts() {
        assert_eq!(
            verdict_from_category_sum(&AC_ORGTAX_CATEGORY_COUNTS, AC_ORGTAX_TOTAL_REPOS),
            CategorySumVerdict::Pass
        );
    }

    #[test]
    fn ot005_pass_minimal() {
        let counts = [("a", 5), ("b", 3)];
        assert_eq!(verdict_from_category_sum(&counts, 8), CategorySumVerdict::Pass);
    }

    #[test]
    fn ot005_fail_undercount() {
        let counts = [("a", 1), ("b", 1)];
        assert_eq!(verdict_from_category_sum(&counts, 5), CategorySumVerdict::Fail);
    }

    #[test]
    fn ot005_fail_overcount() {
        let counts = [("a", 100)];
        assert_eq!(verdict_from_category_sum(&counts, 50), CategorySumVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full org snapshot passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_org_passes_all_5() {
        // 001: every live repo is classified.
        let live = ["aprender", "ruchy", "pmat"];
        let classified = ["aprender", "ruchy", "pmat", "old-archive"];
        assert_eq!(
            verdict_from_orphan_classification(&live, &classified),
            OrphanClassificationVerdict::Pass
        );
        // 002: SHOULD-MERGE archived 1 day after classification → Pass.
        assert_eq!(verdict_from_should_merge_archive(true, 1), ShouldMergeArchiveVerdict::Pass);
        // 003: STALE archived → Pass.
        assert_eq!(verdict_from_stale_archive(true, 0), StaleArchiveVerdict::Pass);
        // 004: legacy mention with MOVED → Pass.
        assert_eq!(
            verdict_from_legacy_redirect("trueno (MOVED to aprender)"),
            LegacyRedirectVerdict::Pass
        );
        // 005: contract category counts sum to 205.
        assert_eq!(
            verdict_from_category_sum(&AC_ORGTAX_CATEGORY_COUNTS, AC_ORGTAX_TOTAL_REPOS),
            CategorySumVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: orphan repo not classified.
        let live = ["new-orphan"];
        let classified: [&str; 0] = [];
        assert_eq!(
            verdict_from_orphan_classification(&live, &classified),
            OrphanClassificationVerdict::Fail
        );
        // 002: 30 days past deadline, still not archived.
        assert_eq!(verdict_from_should_merge_archive(false, 30), ShouldMergeArchiveVerdict::Fail);
        // 003: stale not archived past deadline.
        assert_eq!(verdict_from_stale_archive(false, 30), StaleArchiveVerdict::Fail);
        // 004: legacy name without MOVED.
        assert_eq!(
            verdict_from_legacy_redirect("entrenar — training framework"),
            LegacyRedirectVerdict::Fail
        );
        // 005: counts don't add up.
        let bad_counts = [("a", 100)];
        assert_eq!(verdict_from_category_sum(&bad_counts, 205), CategorySumVerdict::Fail);
    }
}
