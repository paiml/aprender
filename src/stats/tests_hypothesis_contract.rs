// =========================================================================
// FALSIFY-HT: hypothesis testing contract (aprender stats)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-HT-* tests for hypothesis tests
//   Why 2: hypothesis tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for hypothesis testing yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: t-test was "obviously correct" (textbook statistics)
//
// References:
//   - Student (1908) "The Probable Error of a Mean"
//   - Pearson (1900) "On the criterion that a given system of deviations..."
// =========================================================================

use super::*;

/// FALSIFY-HT-001: One-sample t-test p-value is in [0, 1]
#[test]
fn falsify_ht_001_ttest_pvalue_bounded() {
    let sample = vec![2.0, 2.5, 3.0, 3.5, 4.0];
    let result = ttest_1samp(&sample, 3.0).expect("valid input");

    assert!(
        (0.0..=1.0).contains(&result.pvalue),
        "FALSIFIED HT-001: p-value={} outside [0,1]",
        result.pvalue
    );
}

/// FALSIFY-HT-002: Two-sample t-test detects significant difference
#[test]
fn falsify_ht_002_ttest_ind_detects_difference() {
    // Two clearly different groups
    let group1 = vec![1.0, 1.1, 1.2, 0.9, 1.0, 1.1, 0.95, 1.05];
    let group2 = vec![5.0, 5.1, 5.2, 4.9, 5.0, 5.1, 4.95, 5.05];
    let result = ttest_ind(&group1, &group2, true).expect("valid input");

    assert!(
        result.pvalue < 0.05,
        "FALSIFIED HT-002: p-value={} >= 0.05 for clearly different groups",
        result.pvalue
    );
}

/// FALSIFY-HT-003: t-test statistic is finite
#[test]
fn falsify_ht_003_ttest_finite_statistic() {
    let sample = vec![10.0, 12.0, 11.5, 13.0, 9.5];
    let result = ttest_1samp(&sample, 11.0).expect("valid input");

    assert!(
        result.statistic.is_finite(),
        "FALSIFIED HT-003: t-statistic is not finite"
    );
    assert!(
        result.df.is_finite(),
        "FALSIFIED HT-003: degrees of freedom is not finite"
    );
}

/// FALSIFY-HT-004: Chi-square p-value is in [0, 1]
#[test]
fn falsify_ht_004_chisq_pvalue_bounded() {
    let observed = vec![50.0, 30.0, 20.0];
    let expected = vec![40.0, 35.0, 25.0];
    let result = chisquare(&observed, &expected).expect("valid input");

    assert!(
        (0.0..=1.0).contains(&result.pvalue),
        "FALSIFIED HT-004: chi-square p-value={} outside [0,1]",
        result.pvalue
    );
}
