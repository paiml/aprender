//! Statistical Hypothesis Testing
//!
//! Implements classical hypothesis tests for comparing distributions and testing relationships.
//!
//! # Tests
//!
//! - **t-tests**: Compare means (one-sample, two-sample, paired)
//! - **chi-square**: Test categorical distributions (goodness-of-fit, independence)
//! - **ANOVA**: Compare multiple group means (F-test)
//!
//! # Example
//!
//! ```ignore
//! use aprender::stats::hypothesis::{ttest_ind, TTestResult};
//!
//! let group1 = vec![2.3, 2.5, 2.7, 2.9, 3.1];
//! let group2 = vec![3.2, 3.4, 3.6, 3.8, 4.0];
//!
//! let result = ttest_ind(&group1, &group2, true).expect("valid t-test inputs");
//! println!("t-statistic: {:.4}, p-value: {:.4}", result.statistic, result.pvalue);
//! ```

use crate::error::{AprenderError, Result};
use std::f32::consts::PI;

/// Result of a t-test.
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// t-statistic
    pub statistic: f32,

    /// p-value (two-tailed)
    pub pvalue: f32,

    /// Degrees of freedom
    pub df: f32,
}

/// Result of a chi-square test.
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    /// Chi-square statistic
    pub statistic: f32,

    /// p-value
    pub pvalue: f32,

    /// Degrees of freedom
    pub df: usize,
}

/// Result of an ANOVA F-test.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// F-statistic
    pub statistic: f32,

    /// p-value
    pub pvalue: f32,

    /// Between-groups degrees of freedom
    pub df_between: usize,

    /// Within-groups degrees of freedom
    pub df_within: usize,
}

/// One-sample t-test: Tests if sample mean differs from population mean.
///
/// H₀: μ = `population_mean`
/// H₁: μ ≠ `population_mean`
///
/// # Arguments
///
/// * `sample` - Sample data
/// * `population_mean` - Hypothesized population mean
///
/// # Returns
///
/// `TTestResult` with statistic, p-value, and degrees of freedom
pub fn ttest_1samp(sample: &[f32], population_mean: f32) -> Result<TTestResult> {
    let n = sample.len();
    if n < 2 {
        return Err(AprenderError::Other(
            "t-test requires at least 2 samples".into(),
        ));
    }

    // Compute sample mean
    let sample_mean = sample.iter().sum::<f32>() / n as f32;

    // Compute sample standard deviation
    let variance = sample
        .iter()
        .map(|&x| (x - sample_mean).powi(2))
        .sum::<f32>()
        / (n - 1) as f32;
    let std = variance.sqrt();

    // Compute t-statistic: t = (x̄ - μ₀) / (s / √n)
    let se = std / (n as f32).sqrt();
    let t_stat = (sample_mean - population_mean) / se;

    // Degrees of freedom
    let df = (n - 1) as f32;

    // Compute p-value (two-tailed)
    let pvalue = t_distribution_pvalue(t_stat.abs(), df);

    Ok(TTestResult {
        statistic: t_stat,
        pvalue,
        df,
    })
}

/// Independent two-sample t-test: Tests if two independent samples have different means.
///
/// H₀: μ₁ = μ₂
/// H₁: μ₁ ≠ μ₂
///
/// # Arguments
///
/// * `sample1` - First sample
/// * `sample2` - Second sample
/// * `equal_var` - Assume equal variances (pooled t-test) or not (Welch's t-test)
///
/// # Returns
///
/// `TTestResult` with statistic, p-value, and degrees of freedom
pub fn ttest_ind(sample1: &[f32], sample2: &[f32], equal_var: bool) -> Result<TTestResult> {
    let n1 = sample1.len();
    let n2 = sample2.len();

    if n1 < 2 || n2 < 2 {
        return Err(AprenderError::Other(
            "Each sample must have at least 2 observations".into(),
        ));
    }

    // Compute means
    let mean1 = sample1.iter().sum::<f32>() / n1 as f32;
    let mean2 = sample2.iter().sum::<f32>() / n2 as f32;

    // Compute variances
    let var1 = sample1.iter().map(|&x| (x - mean1).powi(2)).sum::<f32>() / (n1 - 1) as f32;
    let var2 = sample2.iter().map(|&x| (x - mean2).powi(2)).sum::<f32>() / (n2 - 1) as f32;

    let (t_stat, df) = if equal_var {
        // Pooled t-test (Student's t-test)
        let pooled_var = ((n1 - 1) as f32 * var1 + (n2 - 1) as f32 * var2) / (n1 + n2 - 2) as f32;
        let se = (pooled_var * (1.0 / n1 as f32 + 1.0 / n2 as f32)).sqrt();
        let t = (mean1 - mean2) / se;
        let df = (n1 + n2 - 2) as f32;
        (t, df)
    } else {
        // Welch's t-test (unequal variances)
        let se = (var1 / n1 as f32 + var2 / n2 as f32).sqrt();
        let t = (mean1 - mean2) / se;

        // Welch-Satterthwaite degrees of freedom
        let numerator = (var1 / n1 as f32 + var2 / n2 as f32).powi(2);
        let denominator = (var1 / n1 as f32).powi(2) / (n1 - 1) as f32
            + (var2 / n2 as f32).powi(2) / (n2 - 1) as f32;
        let df = numerator / denominator;
        (t, df)
    };

    let pvalue = t_distribution_pvalue(t_stat.abs(), df);

    Ok(TTestResult {
        statistic: t_stat,
        pvalue,
        df,
    })
}

/// Paired t-test: Tests if paired samples have different means.
///
/// H₀: `μ_diff` = 0
/// H₁: `μ_diff` ≠ 0
///
/// # Arguments
///
/// * `sample1` - First sample (before)
/// * `sample2` - Second sample (after)
///
/// # Returns
///
/// `TTestResult` with statistic, p-value, and degrees of freedom
pub fn ttest_rel(sample1: &[f32], sample2: &[f32]) -> Result<TTestResult> {
    if sample1.len() != sample2.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{} samples in sample1", sample1.len()),
            actual: format!("{} samples in sample2", sample2.len()),
        });
    }

    // Compute differences
    let diffs: Vec<f32> = sample1
        .iter()
        .zip(sample2.iter())
        .map(|(&x1, &x2)| x1 - x2)
        .collect();

    // Perform one-sample t-test on differences
    ttest_1samp(&diffs, 0.0)
}

// ====================================================================================
// CLAIMS-LAYER f64 PAIRED STATISTICS (plan 05-04, D-05/D-06)
//
// The benchmark's published numbers must be EXACTLY recomputable (EVAL-04), which is a
// bit-level requirement. That drove three choices, all of them visible below:
//
//   1. f64 mirrors of the closed forms already in this module. Same arithmetic, wider
//      type, because f32 rounding is visible in a CI half-width.
//   2. A FROZEN t critical value rather than an inverse CDF. df is fixed at 9 by the
//      ten-seed design, so exactly one constant is needed and no root-finder enters the
//      claims path (Ph1 D-14 pattern).
//   3. No RNG anywhere. A bootstrap would put a resampling stream inside a number that
//      is supposed to be recomputable by hand (D-06, T-05-04-02).
//
// Reference values live in scripts/setfit_fixtures/claims_stats/ and are asserted in
// tests_claims_stats.rs.
// ====================================================================================

/// The paired design's sample size: ten contracted seeds, hence df = 9.
pub const PAIRED_DESIGN_N: usize = 10;

/// Student-t two-tailed 95% critical value at df = 9, frozen.
///
/// Source of truth, computed in the pinned uv environment (scipy 1.18.0) by
/// `scripts/setfit_fixtures/gen_claims_fixtures.py` and recorded at full binary64 repr
/// in `scripts/setfit_fixtures/claims_stats/t_critical.json`:
///
/// ```text
/// >>> scipy.stats.t.ppf(0.975, 9)
/// 2.262157162798205
/// ```
///
/// df is fixed at 9 by the benchmark design (ten seeds per shot level), so this is the
/// only critical value the claims path needs and no inverse CDF is implemented. A test
/// asserts BIT equality against the fixture, so drift turns red rather than quietly
/// widening every published interval.
pub const T_CRIT_975_DF9: f64 = 2.262_157_162_798_205;

/// Result of an f64 t-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TTestResultF64 {
    /// t-statistic
    pub statistic: f64,
    /// p-value (two-tailed)
    pub pvalue: f64,
    /// Degrees of freedom
    pub df: f64,
}

/// A paired confidence interval and the moments it was built from.
///
/// Every field is reported so a reader can recompute the interval by hand from the
/// stored rows — that is what EVAL-04's "exactly recompute" asks for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedCi {
    /// Number of pairs
    pub n: usize,
    /// Mean of the paired differences, `d̄`
    pub mean_diff: f64,
    /// (n-1) sample standard deviation of the differences, `s_d`
    pub std_diff: f64,
    /// Standard error, `s_d / √n`
    pub std_err: f64,
    /// `t_crit · s_d / √n`
    pub half_width: f64,
    /// `d̄ − half_width`
    pub low: f64,
    /// `d̄ + half_width`
    pub high: f64,
}

/// Arithmetic mean, or `None` for an empty slice.
#[must_use]
pub fn mean_f64(sample: &[f64]) -> Option<f64> {
    if sample.is_empty() {
        return None;
    }
    Some(sample.iter().sum::<f64>() / sample.len() as f64)
}

/// (n-1) sample standard deviation, or `None` for fewer than two observations.
///
/// The `(n-1)` divisor is the one D-05 contracts ("mean ± sample (n−1) std"), matching
/// the SetFit paper's reporting convention and `numpy.std(..., ddof=1)`.
#[must_use]
pub fn sample_std_f64(sample: &[f64]) -> Option<f64> {
    if sample.len() < 2 {
        return None;
    }
    let mean = sample.iter().sum::<f64>() / sample.len() as f64;
    let variance =
        sample.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (sample.len() - 1) as f64;
    Some(variance.sqrt())
}

/// Minimum and maximum, or `None` for an empty slice.
#[must_use]
pub fn min_max_f64(sample: &[f64]) -> Option<(f64, f64)> {
    let first = *sample.first()?;
    Some(
        sample
            .iter()
            .fold((first, first), |(lo, hi), &x| (lo.min(x), hi.max(x))),
    )
}

/// Mean and (n-1) sample std, or the typed refusal when the slice has no variance.
///
/// ONE PATH: both `ttest_1samp_f64` and `paired_ci` compute their moments here, so the
/// statistic and the interval can never disagree about `d̄` or `s_d` (OPS-03).
///
/// Two distinct degenerate shapes are caught:
///
/// - **Exactly constant** — every value is bit-identical. Checked directly rather than
///   inferred from `std == 0.0`, because for a value with no exact binary representation
///   (`0.1`, say) the computed variance of ten identical copies is a ~1e-34 rounding
///   artefact, not zero, and an inferred check would let it through and return a
///   meaningless 1e17-scale statistic.
/// - **Numerically constant** — the values differ, but their squared deviations underflow
///   to zero (reachable at ~1e-200 scale). There is still no variance to divide by.
fn moments_or_zero_variance(values: &[f64]) -> Result<(f64, f64)> {
    let n = values.len();
    debug_assert!(n >= 2, "caller must reject n < 2 before computing moments");

    let first = values[0];
    if values.iter().all(|&v| v == first) {
        return Err(AprenderError::ZeroVarianceDifferences {
            n,
            constant_value: first,
        });
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
    let std = variance.sqrt();
    if std == 0.0 {
        return Err(AprenderError::ZeroVarianceDifferences {
            n,
            constant_value: mean,
        });
    }
    Ok((mean, std))
}

/// Element-wise `sample1 - sample2`, or `DimensionMismatch`.
fn paired_differences(sample1: &[f64], sample2: &[f64]) -> Result<Vec<f64>> {
    if sample1.len() != sample2.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{} samples in sample1", sample1.len()),
            actual: format!("{} samples in sample2", sample2.len()),
        });
    }
    Ok(sample1
        .iter()
        .zip(sample2.iter())
        .map(|(&x1, &x2)| x1 - x2)
        .collect())
}

/// One-sample t-test in f64 — the same closed form as [`ttest_1samp`], wider type.
///
/// ```text
/// t = (x̄ − μ₀) / (s / √n),   df = n − 1,   p = 2 · P(T_df > |t|)
/// ```
///
/// # Errors
///
/// - `Other` if fewer than two observations are supplied.
/// - [`AprenderError::ZeroVarianceDifferences`] if the sample has no variance. The
///   statistic would be `x/0` or `0/0`; a typed refusal is returned instead of a
///   non-finite `f64`, which `serde_json` would render as `null` in a published row.
pub fn ttest_1samp_f64(sample: &[f64], population_mean: f64) -> Result<TTestResultF64> {
    let n = sample.len();
    if n < 2 {
        return Err(AprenderError::Other(
            "t-test requires at least 2 samples".into(),
        ));
    }

    let (mean, std) = moments_or_zero_variance(sample)?;
    let se = std / (n as f64).sqrt();
    let statistic = (mean - population_mean) / se;
    let df = (n - 1) as f64;
    let pvalue = t_distribution_pvalue_f64(statistic.abs(), df);

    Ok(TTestResultF64 {
        statistic,
        pvalue,
        df,
    })
}

/// Paired t-test in f64 — a thin wrapper over [`ttest_1samp_f64`] on the differences.
///
/// Deliberately NOT a second implementation of the closed form: the paired statistic and
/// the one-sample statistic on the same differences are bit-identical, and a test asserts
/// exactly that (OPS-03).
///
/// # Errors
///
/// - `DimensionMismatch` if the two samples differ in length.
/// - Whatever [`ttest_1samp_f64`] returns, including
///   [`AprenderError::ZeroVarianceDifferences`] when every difference is the same.
pub fn ttest_rel_f64(sample1: &[f64], sample2: &[f64]) -> Result<TTestResultF64> {
    let diffs = paired_differences(sample1, sample2)?;
    ttest_1samp_f64(&diffs, 0.0)
}

/// Paired 95%-style confidence interval `d̄ ± t_crit · s_d / √n` (D-06).
///
/// `t_crit` is a PARAMETER rather than a lookup: this module implements no inverse CDF,
/// so the caller supplies a critical value that was frozen from a reference environment.
/// The claims path passes [`T_CRIT_975_DF9`] via [`paired_ci95_df9`].
///
/// # Errors
///
/// - `DimensionMismatch` if the two samples differ in length.
/// - `Other` if fewer than two pairs are supplied.
/// - [`AprenderError::ZeroVarianceDifferences`] if every difference is the same. The
///   interval would have no finite width; the claims layer refuses to publish an
///   undefined interval rather than serialising a non-finite bound.
pub fn paired_ci(sample1: &[f64], sample2: &[f64], t_crit: f64) -> Result<PairedCi> {
    let diffs = paired_differences(sample1, sample2)?;
    let n = diffs.len();
    if n < 2 {
        return Err(AprenderError::Other(
            "paired confidence interval requires at least 2 pairs".into(),
        ));
    }

    let (mean_diff, std_diff) = moments_or_zero_variance(&diffs)?;
    let std_err = std_diff / (n as f64).sqrt();
    let half_width = t_crit * std_err;

    Ok(PairedCi {
        n,
        mean_diff,
        std_diff,
        std_err,
        half_width,
        low: mean_diff - half_width,
        high: mean_diff + half_width,
    })
}

/// [`paired_ci`] at the benchmark's fixed design: ten pairs, df = 9, [`T_CRIT_975_DF9`].
///
/// Refuses any other sample size. The frozen critical value is only correct at df = 9, so
/// silently applying it to a different `n` would produce a plausible, wrong interval —
/// the exact failure the constant was frozen to prevent.
///
/// # Errors
///
/// - `DimensionMismatch` if `sample1` does not hold exactly [`PAIRED_DESIGN_N`]
///   observations, or if the two samples differ in length.
/// - Whatever [`paired_ci`] returns, including
///   [`AprenderError::ZeroVarianceDifferences`].
pub fn paired_ci95_df9(sample1: &[f64], sample2: &[f64]) -> Result<PairedCi> {
    if sample1.len() != PAIRED_DESIGN_N {
        return Err(AprenderError::DimensionMismatch {
            expected: format!(
                "{PAIRED_DESIGN_N} paired observations (df = 9, the design T_CRIT_975_DF9 was frozen for)"
            ),
            actual: format!("{} observations", sample1.len()),
        });
    }
    paired_ci(sample1, sample2, T_CRIT_975_DF9)
}

/// One-sample 95%-style interval `x̄ ± t_crit · s / √n` at the benchmark's fixed design.
///
/// THE SEED-DISPERSION INTERVAL of `setfit-benchmark-claims-v1` 2.0.0
/// (`equations.claims_statistics.seed_dispersion_ci95`): the active, single-method scope has
/// no second arm to difference against, so uncertainty is reported as how far a score moves
/// when only the sampling seed moves, at fixed data and protocol. It is NOT a population
/// interval and NOT a comparison.
///
/// NOT A SECOND IMPLEMENTATION (OPS-03). It delegates to [`paired_ci`] against an all-zero
/// comparator, which is the exact one-sample specialisation — `values - 0 == values` — so the
/// mean and the (n − 1) standard deviation come from the SAME
/// `moments_or_zero_variance` the paired path uses. There is no second mean and no second std
/// anywhere in the claims layer, and therefore nothing that can drift. The degrees of freedom
/// are 9 for both, because both are over the same ten contracted seeds, so the one frozen
/// [`T_CRIT_975_DF9`] serves both and no inverse CDF is implemented.
///
/// # Errors
///
/// - `DimensionMismatch` if `values` does not hold exactly [`PAIRED_DESIGN_N`] observations.
///   The frozen critical value is only correct at df = 9; applying it to another `n` would
///   produce a plausible, wrong interval.
/// - [`AprenderError::ZeroVarianceDifferences`] if all ten seeds scored identically. The
///   claims layer renders that as a typed no-interval shape with a stated reason — never a
///   NaN, and never a serde `null` a reader would take for a missing measurement (CR-03).
pub fn ci95_one_sample_df9(values: &[f64]) -> Result<PairedCi> {
    if values.len() != PAIRED_DESIGN_N {
        return Err(AprenderError::DimensionMismatch {
            expected: format!(
                "{PAIRED_DESIGN_N} observations (df = 9, the design T_CRIT_975_DF9 was frozen for)"
            ),
            actual: format!("{} observations", values.len()),
        });
    }
    let zeros = vec![0.0_f64; values.len()];
    paired_ci(values, &zeros, T_CRIT_975_DF9)
}

/// Chi-square goodness-of-fit test: Tests if observed frequencies match expected.
///
/// H₀: Observed frequencies follow expected distribution
/// H₁: Observed frequencies do not follow expected distribution
///
/// # Arguments
///
/// * `observed` - Observed frequencies
/// * `expected` - Expected frequencies
///
/// # Returns
///
/// `ChiSquareResult` with statistic, p-value, and degrees of freedom
pub fn chisquare(observed: &[f32], expected: &[f32]) -> Result<ChiSquareResult> {
    if observed.len() != expected.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{} categories in expected", expected.len()),
            actual: format!("{} categories in observed", observed.len()),
        });
    }

    let k = observed.len();
    if k < 2 {
        return Err(AprenderError::Other(
            "Chi-square test requires at least 2 categories".into(),
        ));
    }

    // Check for negative or zero expected frequencies
    for &exp in expected {
        if exp <= 0.0 {
            return Err(AprenderError::Other(
                "Expected frequencies must be positive".into(),
            ));
        }
    }

    // Compute chi-square statistic: χ² = Σ (O - E)² / E
    let chi2_stat = observed
        .iter()
        .zip(expected.iter())
        .map(|(&obs, &exp)| (obs - exp).powi(2) / exp)
        .sum::<f32>();

    let df = k - 1;
    let pvalue = chi_square_pvalue(chi2_stat, df);

    Ok(ChiSquareResult {
        statistic: chi2_stat,
        pvalue,
        df,
    })
}

/// One-way ANOVA: Tests if multiple groups have the same mean.
///
/// H₀: μ₁ = μ₂ = ... = μₖ
/// H₁: At least one mean is different
///
/// # Arguments
///
/// * `groups` - Vector of samples (each group is a `Vec<f32>`)
///
/// # Returns
///
/// `AnovaResult` with F-statistic, p-value, and degrees of freedom
pub fn f_oneway(groups: &[Vec<f32>]) -> Result<AnovaResult> {
    let k = groups.len();
    if k < 2 {
        return Err(AprenderError::Other(
            "ANOVA requires at least 2 groups".into(),
        ));
    }

    // Check each group has at least 1 observation
    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(AprenderError::Other(format!(
                "Group {i} is empty. All groups must have at least 1 observation"
            )));
        }
    }

    // Compute group means and overall mean
    let group_means: Vec<f32> = groups
        .iter()
        .map(|g| g.iter().sum::<f32>() / g.len() as f32)
        .collect();

    let n_total: usize = groups.iter().map(Vec::len).sum();
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f32>() / n_total as f32;

    // Between-group sum of squares: SSB = Σ n_i * (ȳ_i - ȳ)²
    let ss_between = groups
        .iter()
        .zip(group_means.iter())
        .map(|(group, &mean)| group.len() as f32 * (mean - grand_mean).powi(2))
        .sum::<f32>();

    // Within-group sum of squares: SSW = Σ Σ (y_ij - ȳ_i)²
    let ss_within = groups
        .iter()
        .zip(group_means.iter())
        .map(|(group, &mean)| group.iter().map(|&val| (val - mean).powi(2)).sum::<f32>())
        .sum::<f32>();

    // Degrees of freedom
    let df_between = k - 1;
    let df_within = n_total - k;

    if df_within == 0 {
        return Err(AprenderError::Other(
            "Not enough observations for within-group variance".into(),
        ));
    }

    // Mean squares
    let ms_between = ss_between / df_between as f32;
    let ms_within = ss_within / df_within as f32;

    // F-statistic: F = MS_between / MS_within
    let f_stat = ms_between / ms_within;

    // Compute p-value
    let pvalue = f_distribution_pvalue(f_stat, df_between, df_within);

    Ok(AnovaResult {
        statistic: f_stat,
        pvalue,
        df_between,
        df_within,
    })
}

// ============================================================================
// Distribution p-value approximations
// ============================================================================

/// Computes the exact two-tailed p-value for a Student's t-distribution.
///
/// Uses the regularized incomplete beta identity for the t-distribution tail,
/// matching `scipy.stats.t.sf` for ALL degrees of freedom:
///
///   two-tailed p = I_x(df/2, 1/2),  where  x = df / (df + t²)
///
/// PMAT-853: a previous `df > 30.0` shortcut returned the standard-normal
/// approximation `2 * normal_cdf(-|t|)`, which understates the moderate-df
/// t-tail and flips significance decisions near alpha. The exact path below
/// (now accurate after the PMAT-827 `incomplete_beta` correction) is used for
/// every df, so the shortcut has been removed.
fn t_distribution_pvalue(t: f32, df: f32) -> f32 {
    // P(|T| > |t|) = I_x(df/2, 1/2) with x = df/(df + t²).
    let x = df / (df + t * t);
    let p_one_tail = 0.5 * incomplete_beta(df / 2.0, 0.5, x);
    (2.0 * p_one_tail).clamp(0.0, 1.0)
}

// ---- f64 t-tail, for the claims layer (plan 05-04) -----------------------------------
//
// The f32 path above is scipy-oracle tested but only to f32 resolution; a published CI
// and a published p-value need more than that. These are the SAME closed forms in f64,
// not a different method: the two-tailed t tail via the regularized incomplete beta,
// `p = I_x(df/2, 1/2)` with `x = df/(df + t²)`.

/// Lanczos coefficients, g = 7, n = 9 (relative accuracy ~1e-15).
///
/// The f32 `ln_gamma` above uses the classic Numerical-Recipes six-term series, whose
/// ~1e-10 accuracy is ample for f32 and is not for a p-value asserted to 1e-9.
const LANCZOS_G7_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// `ln Γ(z)` in f64 (Lanczos, g = 7), valid for `z >= 0.5`.
///
/// No reflection branch: every call site here passes `df/2`, `1/2` or their sum, all of
/// which are `>= 0.5`. An unreachable branch is an untested branch, so the precondition
/// is asserted instead of being silently handled.
fn ln_gamma_f64(z: f64) -> f64 {
    debug_assert!(
        z >= 0.5,
        "ln_gamma_f64 is the z >= 0.5 Lanczos branch; got {z}"
    );
    let z = z - 1.0;
    let mut series = LANCZOS_G7_COEFFS[0];
    for (i, &c) in LANCZOS_G7_COEFFS.iter().enumerate().skip(1) {
        series += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// Continued fraction for the incomplete beta in f64 (Lentz's algorithm).
fn beta_continued_fraction_f64(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 300;
    const EPS: f64 = 3e-16;
    const TINY: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    h
}

/// Regularized incomplete beta `I_x(a, b)` in f64.
///
/// The prefactor is combined in LOG space for the same reason the f32 path does it
/// (PMAT-904): `Γ(a)` overflows long before `ln Γ(a)` does.
fn incomplete_beta_f64(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_bt =
        a * x.ln() + b * (1.0 - x).ln() + ln_gamma_f64(a + b) - ln_gamma_f64(a) - ln_gamma_f64(b);
    let bt = ln_bt.exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction_f64(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction_f64(b, a, 1.0 - x) / b
    }
}

/// Exact two-tailed p-value for Student's t in f64, matching `scipy.stats.t.sf` × 2.
///
/// ```text
/// P(|T| > |t|) = I_x(df/2, 1/2),   x = df / (df + t²)
/// ```
fn t_distribution_pvalue_f64(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let p_one_tail = 0.5 * incomplete_beta_f64(df / 2.0, 0.5, x);
    (2.0 * p_one_tail).clamp(0.0, 1.0)
}

/// Approximates the p-value for a chi-square distribution.
fn chi_square_pvalue(chi2: f32, df: usize) -> f32 {
    // P(χ² > x) ≈ 1 - I_x(df/2, 1) using incomplete gamma
    let k = df as f32 / 2.0;
    1.0 - incomplete_gamma(k, chi2 / 2.0)
}

/// Approximates the p-value for an F-distribution.
fn f_distribution_pvalue(f: f32, df1: usize, df2: usize) -> f32 {
    // P(F > x) using beta distribution relationship
    // x_beta = df2 / (df2 + df1 * F)
    let x = df2 as f32 / (df2 as f32 + df1 as f32 * f);
    incomplete_beta(df2 as f32 / 2.0, df1 as f32 / 2.0, x).clamp(0.0, 1.0)
}

/// Standard normal CDF approximation (using error function).
///
/// Test-only since PMAT-853 removed the `df > 30` normal-approximation
/// shortcut from `t_distribution_pvalue`; retained as a reference oracle for
/// the falsification tests that contrast the exact t-tail against the normal
/// approximation.
#[cfg(test)]
fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x / 2.0_f32.sqrt()))
}

/// Error function approximation (delegates to batuta-common).
#[cfg(test)]
fn erf(x: f32) -> f32 {
    batuta_common::math::erf_f32(x)
}

/// Incomplete gamma function approximation (series expansion).
fn incomplete_gamma(a: f32, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if a <= 0.0 {
        return 1.0;
    }

    // Series expansion: γ(a,x)/Γ(a) = e^(-x) * x^a * Σ x^n / Γ(a+n+1).
    // The bounded `sum` factors Γ(a) back out; the e^(-x) x^a / Γ(a) prefactor
    // is combined in LOG space (PMAT-904) so that for large `a` (df ≳ 72) the
    // x^a and Γ(a) terms — each individually Inf in raw f32 — never materialize:
    //   prefactor = exp(-x + a·ln(x) − ln Γ(a)).
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..100 {
        term *= x / (a + n as f32);
        sum += term;
        if term.abs() < 1e-7 {
            break;
        }
    }

    let ln_prefactor = -x + a * x.ln() - ln_gamma(a);
    (ln_prefactor.exp() * sum).clamp(0.0, 1.0)
}

/// Incomplete beta function approximation.
fn incomplete_beta(a: f32, b: f32, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Numerical-Recipes `betai` prefactor: bt = x^a (1-x)^b / B(a,b).
    // The 1/a and 1/b normalizers belong ONLY in the final returns below — folding a
    // `/a` in here double-divided the if-branch (and `b/a`-scaled the else-branch),
    // making I_x(a,b) wrong for a != 1 (it was correct only at the a==1 identity).
    //
    // PMAT-904: combine the whole prefactor in LOG space. With B(a,b) =
    // Γ(a)Γ(b)/Γ(a+b), ln(bt) = a·ln(x) + b·ln(1−x) + ln Γ(a+b) − ln Γ(a) − ln Γ(b).
    // For large df (a or b ≳ 36) each raw Γ overflows f32 to Inf, so the old
    // `x^a/B(a,b)` form yielded Inf/Inf = NaN p-values; the log-space form does
    // a single bounded `exp` at the end and never overflows.
    let ln_bt = a * x.ln() + b * (1.0 - x).ln() + ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let bt = ln_bt.exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
///
/// PMAT-904: production `incomplete_beta` now builds its prefactor in log space
/// via `ln_gamma`, so the raw-space `beta_function`/`gamma` pair (which overflows
/// f32 for large arguments) is retained only as a small-argument reference for
/// the unit tests.
#[cfg(test)]
fn beta_function(a: f32, b: f32) -> f32 {
    gamma(a) * gamma(b) / gamma(a + b)
}

include!("beta_continued_fraction.rs");
include!("hypothesis_tests.rs");

#[cfg(test)]
#[path = "tests_hypothesis_contract.rs"]
mod tests_hypothesis_contract;

#[cfg(test)]
#[path = "tests_claims_stats.rs"]
mod tests_claims_stats;
