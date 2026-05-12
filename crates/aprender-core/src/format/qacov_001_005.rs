// `apr-qa-coverage-v1` algorithm-level PARTIAL discharge for the 5
// coverage-completeness falsifiers (category coverage, untested-surface
// tracking, complexity gate, SATD zero, dogfood exercise map).
//
// Contract: `contracts/apr-qa-coverage-v1.yaml`.
//
// ## Disambiguation
//
// `apr-cli-coverage-v1.yaml` (task #235, COV-001) is a different
// contract — it pins ONE coverage gate (overall apr-cli line coverage
// ≥ 95%). This contract — apr-qa-coverage-v1 — adds 5 finer-grained
// gates on per-category, per-function, and per-module coverage.
// Module suffix `qacov_` disambiguates from the existing `cov_` family.

/// Minimum per-category command coverage (80% per F-COV-001).
pub const AC_QACOV_MIN_CATEGORY_COVERAGE: f64 = 0.80;

/// Inference category coverage requirement (100% — critical path).
pub const AC_QACOV_INFERENCE_COVERAGE: f64 = 1.00;

/// Transform category coverage requirement (90% — data integrity).
pub const AC_QACOV_TRANSFORM_COVERAGE: f64 = 0.90;

/// Impact-score threshold above which an untested function MUST have a
/// tracking issue (F-COV-002).
pub const AC_QACOV_HIGH_IMPACT_THRESHOLD: f64 = 0.80;

/// Cyclomatic complexity threshold per F-COV-003.
pub const AC_QACOV_MAX_CC: u32 = 15;

/// CC threshold above which a dedicated test is required even if < 15.
pub const AC_QACOV_CC_REQUIRES_TEST: u32 = 10;

/// Maximum allowed High-severity SATD items.
pub const AC_QACOV_MAX_HIGH_SATD: u32 = 0;

/// Critical modules per F-COV-005.
pub const AC_QACOV_CRITICAL_MODULES: [&str; 6] = ["hex", "profile", "cbtop", "train", "chat", "serve"];

// =============================================================================
// F-COV-001 — command category coverage
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoryCoverageVerdict {
    /// Every category meets its threshold (general 80%, inference 100%,
    /// transform 90%).
    Pass,
    /// At least one category below threshold.
    Fail,
}

#[must_use]
pub fn verdict_from_category_coverage(category_coverage: &[(&str, f64)]) -> CategoryCoverageVerdict {
    for (category, ratio) in category_coverage {
        let threshold = match *category {
            "inference" => AC_QACOV_INFERENCE_COVERAGE,
            "transform" => AC_QACOV_TRANSFORM_COVERAGE,
            _ => AC_QACOV_MIN_CATEGORY_COVERAGE,
        };
        if *ratio + 1e-9 < threshold {
            return CategoryCoverageVerdict::Fail;
        }
    }
    CategoryCoverageVerdict::Pass
}

// =============================================================================
// F-COV-002 — untested surface tracking
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UntestedSurfaceVerdict {
    /// No untested function has impact_score > 0.8 without a tracking issue.
    Pass,
    /// At least one high-impact untested function has no issue.
    Fail,
}

/// `(impact_score, has_tracking_issue)` per untested function.
#[must_use]
pub fn verdict_from_untested_surface(uncovered: &[(f64, bool)]) -> UntestedSurfaceVerdict {
    for (score, has_issue) in uncovered {
        if *score > AC_QACOV_HIGH_IMPACT_THRESHOLD && !*has_issue {
            return UntestedSurfaceVerdict::Fail;
        }
    }
    UntestedSurfaceVerdict::Pass
}

// =============================================================================
// F-COV-003 — complexity gate
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityGateVerdict {
    /// All functions: CC ≤ 15 AND (CC ≤ 10 OR has dedicated test).
    Pass,
    /// CC > 15 without tracking, OR CC > 10 without dedicated test.
    Fail,
}

#[must_use]
pub fn verdict_from_complexity_gate(
    function_metrics: &[(u32, bool, bool)],
) -> ComplexityGateVerdict {
    // (cc, has_dedicated_test, has_refactoring_issue)
    for (cc, has_test, has_issue) in function_metrics {
        if *cc > AC_QACOV_MAX_CC && !*has_issue {
            return ComplexityGateVerdict::Fail;
        }
        if *cc > AC_QACOV_CC_REQUIRES_TEST && !*has_test {
            return ComplexityGateVerdict::Fail;
        }
    }
    ComplexityGateVerdict::Pass
}

// =============================================================================
// F-COV-004 — SATD zero
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatdZeroVerdict {
    /// Zero High-severity SATD items.
    Pass,
    /// At least one High-severity SATD item.
    Fail,
}

#[must_use]
pub fn verdict_from_satd_zero(high_severity_count: u32) -> SatdZeroVerdict {
    if high_severity_count == AC_QACOV_MAX_HIGH_SATD {
        SatdZeroVerdict::Pass
    } else {
        SatdZeroVerdict::Fail
    }
}

// =============================================================================
// F-COV-005 — dogfood exercise map
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DogfoodMapVerdict {
    /// All 6 critical modules exercised AND none panicked.
    Pass,
    /// At least one critical module skipped or panicked.
    Fail,
}

/// `(module_name, was_exercised, panicked)` per module under test.
#[must_use]
pub fn verdict_from_dogfood_map(module_results: &[(&str, bool, bool)]) -> DogfoodMapVerdict {
    use std::collections::HashSet;
    let mut exercised: HashSet<&&str> = HashSet::new();
    for (name, was_exercised, panicked) in module_results {
        if *was_exercised {
            if *panicked {
                return DogfoodMapVerdict::Fail;
            }
            exercised.insert(name);
        }
    }
    for required in AC_QACOV_CRITICAL_MODULES {
        if !exercised.contains(&required) {
            return DogfoodMapVerdict::Fail;
        }
    }
    DogfoodMapVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_category_coverage_080() {
        assert!((AC_QACOV_MIN_CATEGORY_COVERAGE - 0.80).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_inference_coverage_100() {
        assert!((AC_QACOV_INFERENCE_COVERAGE - 1.00).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_transform_coverage_090() {
        assert!((AC_QACOV_TRANSFORM_COVERAGE - 0.90).abs() < f64::EPSILON);
    }

    #[test]
    fn provenance_max_cc_15() {
        assert_eq!(AC_QACOV_MAX_CC, 15);
    }

    #[test]
    fn provenance_max_high_satd_0() {
        assert_eq!(AC_QACOV_MAX_HIGH_SATD, 0);
    }

    #[test]
    fn provenance_critical_modules_count_6() {
        assert_eq!(AC_QACOV_CRITICAL_MODULES.len(), 6);
    }

    // -------------------------------------------------------------------------
    // Section 2: F-COV-001 category coverage.
    // -------------------------------------------------------------------------
    #[test]
    fn fc001_pass_all_above_threshold() {
        let cov = [
            ("inspection", 0.91),
            ("inference", 1.00),
            ("transform", 1.00),
            ("training", 0.85),
        ];
        assert_eq!(verdict_from_category_coverage(&cov), CategoryCoverageVerdict::Pass);
    }

    #[test]
    fn fc001_pass_inference_at_100() {
        let cov = [("inference", 1.00)];
        assert_eq!(verdict_from_category_coverage(&cov), CategoryCoverageVerdict::Pass);
    }

    #[test]
    fn fc001_fail_inference_below_100() {
        // Inference is critical-path: 99% fails.
        let cov = [("inference", 0.99)];
        assert_eq!(verdict_from_category_coverage(&cov), CategoryCoverageVerdict::Fail);
    }

    #[test]
    fn fc001_fail_transform_below_90() {
        let cov = [("transform", 0.89)];
        assert_eq!(verdict_from_category_coverage(&cov), CategoryCoverageVerdict::Fail);
    }

    #[test]
    fn fc001_fail_general_below_80() {
        let cov = [("misc", 0.79)];
        assert_eq!(verdict_from_category_coverage(&cov), CategoryCoverageVerdict::Fail);
    }

    #[test]
    fn fc001_pass_general_at_exactly_80() {
        let cov = [("misc", 0.80)];
        assert_eq!(verdict_from_category_coverage(&cov), CategoryCoverageVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: F-COV-002 untested surface.
    // -------------------------------------------------------------------------
    #[test]
    fn fc002_pass_no_high_impact_uncovered() {
        let uncovered = [(0.5, false), (0.7, false), (0.79, false)];
        assert_eq!(verdict_from_untested_surface(&uncovered), UntestedSurfaceVerdict::Pass);
    }

    #[test]
    fn fc002_pass_high_impact_with_issue() {
        let uncovered = [(0.95, true)];
        assert_eq!(verdict_from_untested_surface(&uncovered), UntestedSurfaceVerdict::Pass);
    }

    #[test]
    fn fc002_fail_high_impact_no_issue() {
        let uncovered = [(0.85, false)];
        assert_eq!(verdict_from_untested_surface(&uncovered), UntestedSurfaceVerdict::Fail);
    }

    #[test]
    fn fc002_pass_at_threshold() {
        // > 0.8 — exactly 0.8 passes (strict greater-than).
        let uncovered = [(0.80, false)];
        assert_eq!(verdict_from_untested_surface(&uncovered), UntestedSurfaceVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 4: F-COV-003 complexity gate.
    // -------------------------------------------------------------------------
    #[test]
    fn fc003_pass_low_complexity() {
        let metrics = [(8, false, false), (5, false, false)];
        assert_eq!(verdict_from_complexity_gate(&metrics), ComplexityGateVerdict::Pass);
    }

    #[test]
    fn fc003_pass_cc_11_with_dedicated_test() {
        let metrics = [(11, true, false)];
        assert_eq!(verdict_from_complexity_gate(&metrics), ComplexityGateVerdict::Pass);
    }

    #[test]
    fn fc003_fail_cc_11_no_test() {
        let metrics = [(11, false, false)];
        assert_eq!(verdict_from_complexity_gate(&metrics), ComplexityGateVerdict::Fail);
    }

    #[test]
    fn fc003_fail_cc_16_no_issue() {
        let metrics = [(16, true, false)];
        assert_eq!(verdict_from_complexity_gate(&metrics), ComplexityGateVerdict::Fail);
    }

    #[test]
    fn fc003_pass_cc_16_with_issue_and_test() {
        // CC > 15 needs refactoring issue; CC > 10 needs test.
        let metrics = [(16, true, true)];
        assert_eq!(verdict_from_complexity_gate(&metrics), ComplexityGateVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: F-COV-004 SATD zero.
    // -------------------------------------------------------------------------
    #[test]
    fn fc004_pass_zero_high() {
        assert_eq!(verdict_from_satd_zero(0), SatdZeroVerdict::Pass);
    }

    #[test]
    fn fc004_fail_one_high() {
        assert_eq!(verdict_from_satd_zero(1), SatdZeroVerdict::Fail);
    }

    #[test]
    fn fc004_fail_many_high() {
        assert_eq!(verdict_from_satd_zero(50), SatdZeroVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: F-COV-005 dogfood exercise map.
    // -------------------------------------------------------------------------
    #[test]
    fn fc005_pass_all_6_exercised() {
        let results = [
            ("hex", true, false),
            ("profile", true, false),
            ("cbtop", true, false),
            ("train", true, false),
            ("chat", true, false),
            ("serve", true, false),
        ];
        assert_eq!(verdict_from_dogfood_map(&results), DogfoodMapVerdict::Pass);
    }

    #[test]
    fn fc005_fail_one_module_panicked() {
        let results = [
            ("hex", true, true), // panic!
            ("profile", true, false),
            ("cbtop", true, false),
            ("train", true, false),
            ("chat", true, false),
            ("serve", true, false),
        ];
        assert_eq!(verdict_from_dogfood_map(&results), DogfoodMapVerdict::Fail);
    }

    #[test]
    fn fc005_fail_one_module_skipped() {
        let results = [
            ("hex", true, false),
            ("profile", true, false),
            ("cbtop", true, false),
            ("train", true, false),
            ("chat", true, false),
            // serve missing
        ];
        assert_eq!(verdict_from_dogfood_map(&results), DogfoodMapVerdict::Fail);
    }

    #[test]
    fn fc005_fail_empty() {
        let results: [(&str, bool, bool); 0] = [];
        assert_eq!(verdict_from_dogfood_map(&results), DogfoodMapVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic + family.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_dogfood_passes_all_5() {
        let cov = [
            ("inspection", 0.91),
            ("inference", 1.00),
            ("transform", 0.95),
            ("training", 0.82),
            ("registry", 0.85),
            ("hardware", 0.80),
            ("qa", 0.83),
            ("ui", 0.84),
            ("pipeline", 0.81),
            ("misc", 0.86),
        ];
        assert_eq!(verdict_from_category_coverage(&cov), CategoryCoverageVerdict::Pass);

        let uncovered = [(0.5, false), (0.85, true)];
        assert_eq!(verdict_from_untested_surface(&uncovered), UntestedSurfaceVerdict::Pass);

        let metrics = [(8, false, false), (12, true, false), (16, true, true)];
        assert_eq!(verdict_from_complexity_gate(&metrics), ComplexityGateVerdict::Pass);

        assert_eq!(verdict_from_satd_zero(0), SatdZeroVerdict::Pass);

        let results = [
            ("hex", true, false),
            ("profile", true, false),
            ("cbtop", true, false),
            ("train", true, false),
            ("chat", true, false),
            ("serve", true, false),
        ];
        assert_eq!(verdict_from_dogfood_map(&results), DogfoodMapVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: inference < 100%.
        assert_eq!(
            verdict_from_category_coverage(&[("inference", 0.95)]),
            CategoryCoverageVerdict::Fail
        );
        // 002: high-impact uncovered without issue.
        assert_eq!(
            verdict_from_untested_surface(&[(0.92, false)]),
            UntestedSurfaceVerdict::Fail
        );
        // 003: CC=20 untested.
        assert_eq!(
            verdict_from_complexity_gate(&[(20, false, false)]),
            ComplexityGateVerdict::Fail
        );
        // 004: SATD non-zero.
        assert_eq!(verdict_from_satd_zero(3), SatdZeroVerdict::Fail);
        // 005: hex module panicked.
        let bad = [
            ("hex", true, true),
            ("profile", true, false),
            ("cbtop", true, false),
            ("train", true, false),
            ("chat", true, false),
            ("serve", true, false),
        ];
        assert_eq!(verdict_from_dogfood_map(&bad), DogfoodMapVerdict::Fail);
    }
}
