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

// =========================================================================
// FALSIFY-HT-FINITE: finite p-values for large degrees of freedom (PMAT-904)
//
// Root cause: raw-space `gamma(z)` overflows f32 at z >= ~36 (Gamma(36) ~ 1e40).
// For df >= ~72 the chi-square `incomplete_gamma` and the t/F `beta_function`
// prefactors divide by an overflowed gamma -> Inf/Inf = NaN p-values.
//
// Oracle: scipy.stats (chi2.sf / t.sf / f.sf), assert |apr_p - scipy_p| < TOL.
// TOL = 1e-5: f32 machine epsilon is ~1.2e-7; the series/continued-fraction
// accumulation on order-1 p-values (~0.12..0.48) bottoms out near 1e-6 absolute
// error, so 1e-5 is the numerically-honest f64-vs-f32 agreement bound. The PRE-FIX
// failure mode was NaN (non-finite), not a near-miss tolerance, so the
// `is_finite` assertion below is the load-bearing falsifier; TOL only guards
// against a wrong-but-finite regression.
// =========================================================================

/// scipy-vs-f32 agreement tolerance (see module note above).
const PVALUE_SCIPY_TOL: f32 = 1e-5;

/// OBLIG-CHISQUARE-PVALUE-FINITE / FALSIFY-HT-CHI2-FINITE:
/// chi-square survival p-value is finite and matches scipy for df in {72,100,200}.
#[test]
fn falsify_ht_chi2_pvalue_finite_large_df() {
    // scipy.stats.chi2.sf(df, df) for chi2 == df:
    //   df=72  -> 4.7783333265e-01
    //   df=100 -> 4.8119168453e-01
    //   df=200 -> 4.8670120172e-01
    let cases: [(usize, f32); 3] = [
        (72, 4.778_333_3e-01),
        (100, 4.811_917e-01),
        (200, 4.867_012e-01),
    ];
    for (df, scipy_p) in cases {
        let chi2 = df as f32;
        let p = chi_square_pvalue(chi2, df);
        assert!(
            p.is_finite(),
            "FALSIFIED HT-CHI2-FINITE: chi-square p-value is non-finite ({p}) at df={df} (raw-space gamma overflow)"
        );
        assert!(
            (p - scipy_p).abs() < PVALUE_SCIPY_TOL,
            "FALSIFIED HT-CHI2-FINITE: chi-square p={p} != scipy {scipy_p} at df={df} (|diff|={})",
            (p - scipy_p).abs()
        );
    }
}

/// OBLIG-HYPOTHESIS-PVALUE-FINITE / FALSIFY-HT-T-FINITE:
/// two-tailed t-distribution p-value is finite and matches scipy for df in {72,100,200}.
#[test]
fn falsify_ht_t_pvalue_finite_large_df() {
    // 2*scipy.stats.t.sf(2.0, df):
    //   df=72  -> 4.9273158278e-02
    //   df=100 -> 4.8212178731e-02
    //   df=200 -> 4.6853186187e-02
    let cases: [(f32, f32); 3] = [
        (72.0, 4.927_315_8e-02),
        (100.0, 4.821_217_9e-02),
        (200.0, 4.685_318_6e-02),
    ];
    for (df, scipy_p) in cases {
        let p = t_distribution_pvalue(2.0, df);
        assert!(
            p.is_finite(),
            "FALSIFIED HT-T-FINITE: t p-value is non-finite ({p}) at df={df} (raw-space gamma overflow)"
        );
        assert!(
            (p - scipy_p).abs() < PVALUE_SCIPY_TOL,
            "FALSIFIED HT-T-FINITE: t p={p} != scipy {scipy_p} at df={df} (|diff|={})",
            (p - scipy_p).abs()
        );
    }
}

/// OBLIG-HYPOTHESIS-PVALUE-FINITE / FALSIFY-HT-F-FINITE:
/// one-way ANOVA F-distribution p-value is finite and matches scipy for df2 in {72,100,200}.
#[test]
fn falsify_ht_f_pvalue_finite_large_df() {
    // scipy.stats.f.sf(2.0, 3, df2):
    //   df2=72  -> 1.2161927540e-01
    //   df2=100 -> 1.1884246789e-01
    //   df2=200 -> 1.1524280145e-01
    let cases: [(usize, f32); 3] = [
        (72, 1.216_192_8e-01),
        (100, 1.188_424_7e-01),
        (200, 1.152_428e-01),
    ];
    let df1 = 3usize;
    for (df2, scipy_p) in cases {
        let p = f_distribution_pvalue(2.0, df1, df2);
        assert!(
            p.is_finite(),
            "FALSIFIED HT-F-FINITE: F p-value is non-finite ({p}) at df2={df2} (raw-space gamma overflow)"
        );
        assert!(
            (p - scipy_p).abs() < PVALUE_SCIPY_TOL,
            "FALSIFIED HT-F-FINITE: F p={p} != scipy {scipy_p} at df2={df2} (|diff|={})",
            (p - scipy_p).abs()
        );
    }
}

mod ht_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // FALSIFY-HT-001-prop: t-test p-value in [0, 1] for random samples
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_ht_001_prop_pvalue_bounded(
            n in 5..=20usize,
            seed in 0..500u32,
        ) {
            let sample: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let result = ttest_1samp(&sample, 0.0).expect("valid");
            prop_assert!(
                (0.0..=1.0).contains(&result.pvalue),
                "FALSIFIED HT-001-prop: p-value={} outside [0,1]",
                result.pvalue
            );
        }
    }

    // FALSIFY-HT-003-prop: t-statistic is finite for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_ht_003_prop_finite_statistic(
            n in 5..=20usize,
            seed in 0..500u32,
        ) {
            let sample: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0 + 5.0)
                .collect();
            let result = ttest_1samp(&sample, 5.0).expect("valid");
            prop_assert!(
                result.statistic.is_finite(),
                "FALSIFIED HT-003-prop: t-statistic not finite"
            );
        }
    }
}
