//! Data drift detection for ML pipelines
//!
//! Detects distribution changes between dataset versions or time periods.
//! Implements Jidoka—building quality in at the data layer before training.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::drift::{DriftDetector, DriftTest};
//!
//! let detector = DriftDetector::new(reference_dataset)
//!     .with_test(DriftTest::KolmogorovSmirnov)
//!     .with_test(DriftTest::PSI)
//!     .with_alpha(0.05);
//!
//! let report = detector.detect(&current_dataset)?;
//! if report.drift_detected {
//!     println!("Drift detected in columns: {:?}", report.drifted_columns());
//! }
//! ```

// Statistical computation requires casts, similar variable names, and float literals
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::suboptimal_flops)]

use std::collections::HashMap;

use crate::{
    dataset::{ArrowDataset, Dataset},
    error::{Error, Result},
};

/// Statistical tests for drift detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriftTest {
    /// Kolmogorov-Smirnov test for continuous features
    KolmogorovSmirnov,
    /// Chi-squared test for categorical features
    ChiSquared,
    /// Population Stability Index (PSI)
    PSI,
    /// Jensen-Shannon divergence
    JensenShannon,
}

impl DriftTest {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::KolmogorovSmirnov => "Kolmogorov-Smirnov",
            Self::ChiSquared => "Chi-Squared",
            Self::PSI => "Population Stability Index",
            Self::JensenShannon => "Jensen-Shannon Divergence",
        }
    }

    /// Check if test is suitable for continuous data
    pub fn is_continuous(&self) -> bool {
        matches!(self, Self::KolmogorovSmirnov | Self::JensenShannon)
    }

    /// Check if test is suitable for categorical data
    pub fn is_categorical(&self) -> bool {
        matches!(self, Self::ChiSquared | Self::PSI)
    }
}

/// Severity of detected drift
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DriftSeverity {
    /// No drift detected
    None,
    /// Low drift (p > 0.01)
    Low,
    /// Medium drift (0.001 < p <= 0.01)
    Medium,
    /// High drift (p <= 0.001)
    High,
    /// Critical - distribution fundamentally changed
    Critical,
}

impl DriftSeverity {
    /// Create severity from p-value
    pub fn from_p_value(p_value: f64) -> Self {
        if p_value > 0.05 {
            Self::None
        } else if p_value > 0.01 {
            Self::Low
        } else if p_value > 0.001 {
            Self::Medium
        } else if p_value > 0.0001 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Create severity from PSI value
    pub fn from_psi(psi: f64) -> Self {
        if psi < 0.1 {
            Self::None
        } else if psi < 0.2 {
            Self::Low
        } else if psi < 0.25 {
            Self::Medium
        } else if psi < 0.5 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Check if this severity indicates drift
    pub fn is_drift(&self) -> bool {
        *self != Self::None
    }
}

/// Per-column drift result
#[derive(Debug, Clone)]
pub struct ColumnDrift {
    /// Column name
    pub column: String,
    /// Test used
    pub test: DriftTest,
    /// Test statistic value
    pub statistic: f64,
    /// P-value (if applicable)
    pub p_value: Option<f64>,
    /// Whether drift was detected for this column
    pub drift_detected: bool,
    /// Severity of drift
    pub severity: DriftSeverity,
}

impl ColumnDrift {
    /// Create a new column drift result
    pub fn new(
        column: impl Into<String>,
        test: DriftTest,
        statistic: f64,
        p_value: Option<f64>,
        severity: DriftSeverity,
    ) -> Self {
        Self {
            column: column.into(),
            test,
            statistic,
            p_value,
            drift_detected: severity.is_drift(),
            severity,
        }
    }
}

/// Overall drift detection report
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Per-column drift scores
    pub column_scores: HashMap<String, ColumnDrift>,
    /// Overall drift detected
    pub drift_detected: bool,
    /// Timestamp of analysis (Unix epoch seconds)
    pub timestamp: u64,
}

impl DriftReport {
    /// Create a new drift report from column results
    pub fn from_columns(columns: Vec<ColumnDrift>) -> Self {
        let drift_detected = columns.iter().any(|c| c.drift_detected);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Key by column_name:test_name to allow multiple tests per column
        let column_scores = columns
            .into_iter()
            .map(|c| (format!("{}:{:?}", c.column, c.test), c))
            .collect();

        Self {
            column_scores,
            drift_detected,
            timestamp,
        }
    }

    /// Get columns with detected drift
    pub fn drifted_columns(&self) -> Vec<&str> {
        self.column_scores
            .values()
            .filter(|c| c.drift_detected)
            .map(|c| c.column.as_str())
            .collect()
    }

    /// Get the maximum severity across all columns
    pub fn max_severity(&self) -> DriftSeverity {
        self.column_scores
            .values()
            .map(|c| c.severity)
            .max()
            .unwrap_or(DriftSeverity::None)
    }

    /// Get number of columns analyzed
    pub fn num_columns(&self) -> usize {
        self.column_scores.len()
    }

    /// Get number of columns with drift
    pub fn num_drifted(&self) -> usize {
        self.column_scores
            .values()
            .filter(|c| c.drift_detected)
            .count()
    }
}

/// Statistical drift detector
pub struct DriftDetector {
    /// Reference dataset (baseline distribution)
    reference: ArrowDataset,
    /// Statistical tests to apply
    tests: Vec<DriftTest>,
    /// Significance threshold (default: 0.05)
    alpha: f64,
}

impl DriftDetector {
    /// Create a new drift detector with a reference dataset
    pub fn new(reference: ArrowDataset) -> Self {
        Self {
            reference,
            tests: vec![DriftTest::KolmogorovSmirnov],
            alpha: 0.05,
        }
    }

    /// Add a statistical test
    #[must_use]
    pub fn with_test(mut self, test: DriftTest) -> Self {
        if !self.tests.contains(&test) {
            self.tests.push(test);
        }
        self
    }

    /// Set significance threshold
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set all tests at once
    #[must_use]
    pub fn with_tests(mut self, tests: Vec<DriftTest>) -> Self {
        self.tests = tests;
        self
    }

    /// Get the reference dataset
    pub fn reference(&self) -> &ArrowDataset {
        &self.reference
    }

    /// Get configured tests
    pub fn tests(&self) -> &[DriftTest] {
        &self.tests
    }

    /// Get significance threshold
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Compare current dataset against reference
    pub fn detect(&self, current: &ArrowDataset) -> Result<DriftReport> {
        // Verify schemas match
        if self.reference.schema() != current.schema() {
            return Err(Error::invalid_config(
                "Schema mismatch between reference and current dataset",
            ));
        }

        let schema = self.reference.schema();
        let mut results = Vec::new();

        // Extract data from datasets
        let ref_data = collect_dataset_data(&self.reference);
        let cur_data = collect_dataset_data(current);

        // Test each column
        for field in schema.fields() {
            let column_name = field.name();

            let ref_col = ref_data.get(column_name);
            let cur_col = cur_data.get(column_name);

            if let (Some(ref_values), Some(cur_values)) = (ref_col, cur_col) {
                // Run each configured test
                for test in &self.tests {
                    let result = run_test(*test, ref_values, cur_values, self.alpha)?;
                    results.push(ColumnDrift::new(
                        column_name,
                        *test,
                        result.statistic,
                        result.p_value,
                        result.severity,
                    ));
                }
            }
        }

        Ok(DriftReport::from_columns(results))
    }
}

/// Internal result from a statistical test
struct TestResult {
    statistic: f64,
    p_value: Option<f64>,
    severity: DriftSeverity,
}

/// Run a specific statistical test
fn run_test(test: DriftTest, reference: &[f64], current: &[f64], alpha: f64) -> Result<TestResult> {
    match test {
        DriftTest::KolmogorovSmirnov => ks_test(reference, current, alpha),
        DriftTest::ChiSquared => chi_squared_test(reference, current, alpha),
        DriftTest::PSI => psi_test(reference, current),
        DriftTest::JensenShannon => jensen_shannon_test(reference, current),
    }
}

/// Kolmogorov-Smirnov two-sample test
///
/// Tests whether two samples come from the same distribution.
/// The statistic D is the maximum absolute difference between CDFs.
fn ks_test(reference: &[f64], current: &[f64], alpha: f64) -> Result<TestResult> {
    if reference.is_empty() || current.is_empty() {
        return Err(Error::invalid_config(
            "Cannot perform KS test on empty data",
        ));
    }

    // Sort both samples
    let mut ref_sorted: Vec<f64> = reference
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let mut cur_sorted: Vec<f64> = current.iter().copied().filter(|x| x.is_finite()).collect();

    if ref_sorted.is_empty() || cur_sorted.is_empty() {
        return Err(Error::invalid_config("No finite values in data"));
    }

    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cur_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = ref_sorted.len() as f64;
    let n2 = cur_sorted.len() as f64;

    // Compute empirical CDFs and find maximum difference
    let d_statistic = compute_ks_statistic(&ref_sorted, &cur_sorted);

    // Compute approximate p-value using asymptotic distribution
    let en = (n1 * n2 / (n1 + n2)).sqrt();
    let p_value = ks_p_value(d_statistic * en);

    let severity = if p_value <= alpha {
        DriftSeverity::from_p_value(p_value)
    } else {
        DriftSeverity::None
    };

    Ok(TestResult {
        statistic: d_statistic,
        p_value: Some(p_value),
        severity,
    })
}

/// Compute KS statistic (maximum CDF difference)
fn compute_ks_statistic(ref_sorted: &[f64], cur_sorted: &[f64]) -> f64 {
    let n1 = ref_sorted.len();
    let n2 = cur_sorted.len();

    // Merge and compute CDF differences at each point
    let mut all_values: Vec<f64> = ref_sorted
        .iter()
        .chain(cur_sorted.iter())
        .copied()
        .collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_values.dedup();

    let mut max_diff = 0.0_f64;

    for &x in &all_values {
        // CDF of reference at x
        let cdf1 = ref_sorted.iter().filter(|&&v| v <= x).count() as f64 / n1 as f64;
        // CDF of current at x
        let cdf2 = cur_sorted.iter().filter(|&&v| v <= x).count() as f64 / n2 as f64;

        let diff = (cdf1 - cdf2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    max_diff
}

/// Approximate p-value for KS statistic using Kolmogorov distribution
fn ks_p_value(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    if z > 3.0 {
        return 0.0;
    }

    // Asymptotic formula: P(D > z) ≈ 2 * sum_{k=1}^inf (-1)^(k-1) * exp(-2*k^2*z^2)
    let mut p = 0.0;
    let z_sq = z * z;

    for k in 1..=100 {
        let k_f = f64::from(k);
        let term = (-1.0_f64).powi(k - 1) * (-2.0 * k_f * k_f * z_sq).exp();
        p += term;
        if term.abs() < 1e-12 {
            break;
        }
    }

    (2.0 * p).clamp(0.0, 1.0)
}

/// Chi-squared test for categorical data
///
/// Bins continuous data and tests for independence.
fn chi_squared_test(reference: &[f64], current: &[f64], alpha: f64) -> Result<TestResult> {
    if reference.is_empty() || current.is_empty() {
        return Err(Error::invalid_config(
            "Cannot perform chi-squared test on empty data",
        ));
    }

    // Bin the data
    let num_bins = ((reference.len() as f64).sqrt().ceil() as usize).clamp(5, 20);
    let (ref_bins, cur_bins) = bin_data(reference, current, num_bins)?;

    // Compute chi-squared statistic
    let n_ref = reference.len() as f64;
    let n_cur = current.len() as f64;
    let total = n_ref + n_cur;

    let mut chi_sq = 0.0;
    let mut df: usize = 0;

    for (r, c) in ref_bins.iter().zip(cur_bins.iter()) {
        let r = *r as f64;
        let c = *c as f64;
        let row_total = r + c;

        if row_total > 0.0 {
            let expected_r = row_total * n_ref / total;
            let expected_c = row_total * n_cur / total;

            if expected_r > 0.0 {
                chi_sq += (r - expected_r).powi(2) / expected_r;
            }
            if expected_c > 0.0 {
                chi_sq += (c - expected_c).powi(2) / expected_c;
            }
            df += 1;
        }
    }

    df = df.saturating_sub(1); // degrees of freedom = bins - 1

    // Approximate p-value using chi-squared distribution
    let p_value = chi_squared_p_value(chi_sq, df);

    let severity = if p_value <= alpha {
        DriftSeverity::from_p_value(p_value)
    } else {
        DriftSeverity::None
    };

    Ok(TestResult {
        statistic: chi_sq,
        p_value: Some(p_value),
        severity,
    })
}

/// Bin continuous data into histogram
fn bin_data(
    reference: &[f64],
    current: &[f64],
    num_bins: usize,
) -> Result<(Vec<usize>, Vec<usize>)> {
    // Find global min/max
    let all_data: Vec<f64> = reference
        .iter()
        .chain(current.iter())
        .copied()
        .filter(|x| x.is_finite())
        .collect();

    if all_data.is_empty() {
        return Err(Error::invalid_config("No finite values in data"));
    }

    let min_val = all_data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = all_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < f64::EPSILON {
        // All values are the same
        return Ok((vec![reference.len()], vec![current.len()]));
    }

    let bin_width = (max_val - min_val) / num_bins as f64;

    let bin_value = |v: f64| -> usize {
        if !v.is_finite() {
            return 0;
        }
        let bin = ((v - min_val) / bin_width).floor() as usize;
        bin.min(num_bins - 1)
    };

    let mut ref_bins = vec![0usize; num_bins];
    let mut cur_bins = vec![0usize; num_bins];

    for &v in reference {
        ref_bins[bin_value(v)] += 1;
    }
    for &v in current {
        cur_bins[bin_value(v)] += 1;
    }

    Ok((ref_bins, cur_bins))
}

/// Approximate chi-squared p-value using Wilson-Hilferty transformation
fn chi_squared_p_value(chi_sq: f64, df: usize) -> f64 {
    if df == 0 {
        return 1.0;
    }

    let k = df as f64;

    // Wilson-Hilferty approximation: transform to standard normal
    let z = ((chi_sq / k).cbrt() - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();

    // Standard normal CDF (approximation)
    1.0 - standard_normal_cdf(z)
}

/// Standard normal CDF approximation
fn standard_normal_cdf(z: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Population Stability Index (PSI)
///
/// Measures how much a distribution has shifted.
/// PSI < 0.1: No significant change
/// PSI 0.1-0.2: Moderate change
/// PSI > 0.2: Significant change
fn psi_test(reference: &[f64], current: &[f64]) -> Result<TestResult> {
    if reference.is_empty() || current.is_empty() {
        return Err(Error::invalid_config("Cannot compute PSI on empty data"));
    }

    // Bin the data (10 bins is standard for PSI)
    let num_bins = 10;
    let (ref_bins, cur_bins) = bin_data(reference, current, num_bins)?;

    let n_ref = reference.len() as f64;
    let n_cur = current.len() as f64;

    let mut psi = 0.0;

    for (r, c) in ref_bins.iter().zip(cur_bins.iter()) {
        // Convert counts to proportions with smoothing to avoid log(0)
        let p_ref = (*r as f64 + 0.5) / (n_ref + num_bins as f64 * 0.5);
        let p_cur = (*c as f64 + 0.5) / (n_cur + num_bins as f64 * 0.5);

        psi += (p_cur - p_ref) * (p_cur / p_ref).ln();
    }

    let severity = DriftSeverity::from_psi(psi);

    Ok(TestResult {
        statistic: psi,
        p_value: None, // PSI doesn't have a p-value
        severity,
    })
}

/// Jensen-Shannon divergence
///
/// Symmetric measure of distribution difference.
/// JSD = 0: Identical distributions
/// JSD = 1: Completely different distributions
fn jensen_shannon_test(reference: &[f64], current: &[f64]) -> Result<TestResult> {
    if reference.is_empty() || current.is_empty() {
        return Err(Error::invalid_config("Cannot compute JSD on empty data"));
    }

    // Bin the data
    let num_bins = 20;
    let (ref_bins, cur_bins) = bin_data(reference, current, num_bins)?;

    let n_ref = reference.len() as f64;
    let n_cur = current.len() as f64;

    // Convert to probability distributions with smoothing
    let p: Vec<f64> = ref_bins
        .iter()
        .map(|&c| (c as f64 + 0.5) / (n_ref + num_bins as f64 * 0.5))
        .collect();
    let q: Vec<f64> = cur_bins
        .iter()
        .map(|&c| (c as f64 + 0.5) / (n_cur + num_bins as f64 * 0.5))
        .collect();

    // M = (P + Q) / 2
    let m: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(pi, qi)| (pi + qi) / 2.0)
        .collect();

    // JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    let kl_pm: f64 = p
        .iter()
        .zip(m.iter())
        .map(|(pi, mi)| if *pi > 0.0 { pi * (pi / mi).ln() } else { 0.0 })
        .sum();

    let kl_qm: f64 = q
        .iter()
        .zip(m.iter())
        .map(|(qi, mi)| if *qi > 0.0 { qi * (qi / mi).ln() } else { 0.0 })
        .sum();

    let jsd = 0.5 * kl_pm + 0.5 * kl_qm;

    // JSD is in [0, ln(2)] for base e, normalize to [0, 1]
    let jsd_normalized = jsd / std::f64::consts::LN_2;

    // Map JSD to severity (similar thresholds to PSI)
    let severity = if jsd_normalized < 0.05 {
        DriftSeverity::None
    } else if jsd_normalized < 0.1 {
        DriftSeverity::Low
    } else if jsd_normalized < 0.2 {
        DriftSeverity::Medium
    } else if jsd_normalized < 0.4 {
        DriftSeverity::High
    } else {
        DriftSeverity::Critical
    };

    Ok(TestResult {
        statistic: jsd_normalized,
        p_value: None, // JSD doesn't have a traditional p-value
        severity,
    })
}

/// Extract non-null f64 values from a numeric Arrow array into the output
/// vector.
fn extract_numeric_values(
    array: &dyn arrow::array::Array,
    data_type: &arrow::datatypes::DataType,
    out: &mut Vec<f64>,
) {
    use arrow::{
        array::{Array, Float64Array, Int32Array, Int64Array},
        datatypes::DataType,
    };

    match data_type {
        DataType::Float64 => {
            if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
                out.extend(
                    (0..arr.len())
                        .filter(|&i| !arr.is_null(i))
                        .map(|i| arr.value(i)),
                );
            }
        }
        DataType::Float32 => {
            if let Some(arr) = array.as_any().downcast_ref::<arrow::array::Float32Array>() {
                out.extend(
                    (0..arr.len())
                        .filter(|&i| !arr.is_null(i))
                        .map(|i| f64::from(arr.value(i))),
                );
            }
        }
        DataType::Int32 => {
            if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                out.extend(
                    (0..arr.len())
                        .filter(|&i| !arr.is_null(i))
                        .map(|i| f64::from(arr.value(i))),
                );
            }
        }
        DataType::Int64 => {
            if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                out.extend(
                    (0..arr.len())
                        .filter(|&i| !arr.is_null(i))
                        .map(|i| arr.value(i) as f64),
                );
            }
        }
        _ => {}
    }
}

/// Collect all numeric column data from a dataset
fn collect_dataset_data(dataset: &ArrowDataset) -> HashMap<String, Vec<f64>> {
    use arrow::datatypes::DataType;

    let mut data: HashMap<String, Vec<f64>> = HashMap::new();
    let schema = dataset.schema();

    // Initialize vectors for each numeric column
    for field in schema.fields() {
        if matches!(
            field.data_type(),
            DataType::Int32 | DataType::Int64 | DataType::Float64 | DataType::Float32
        ) {
            data.insert(field.name().clone(), Vec::new());
        }
    }

    // Collect data from all batches
    for batch in dataset.iter() {
        for (col_idx, field) in schema.fields().iter().enumerate() {
            if let Some(col_data) = data.get_mut(field.name()) {
                extract_numeric_values(batch.column(col_idx), field.data_type(), col_data);
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float64Array, Int32Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;

    // ========== DriftTest enum tests ==========

    #[test]
    fn test_drift_test_name() {
        assert_eq!(DriftTest::KolmogorovSmirnov.name(), "Kolmogorov-Smirnov");
        assert_eq!(DriftTest::ChiSquared.name(), "Chi-Squared");
        assert_eq!(DriftTest::PSI.name(), "Population Stability Index");
        assert_eq!(DriftTest::JensenShannon.name(), "Jensen-Shannon Divergence");
    }

    #[test]
    fn test_drift_test_is_continuous() {
        assert!(DriftTest::KolmogorovSmirnov.is_continuous());
        assert!(DriftTest::JensenShannon.is_continuous());
        assert!(!DriftTest::ChiSquared.is_continuous());
        assert!(!DriftTest::PSI.is_continuous());
    }

    #[test]
    fn test_drift_test_is_categorical() {
        assert!(DriftTest::ChiSquared.is_categorical());
        assert!(DriftTest::PSI.is_categorical());
        assert!(!DriftTest::KolmogorovSmirnov.is_categorical());
        assert!(!DriftTest::JensenShannon.is_categorical());
    }

    // ========== DriftSeverity tests ==========

    #[test]
    fn test_drift_severity_from_p_value() {
        assert_eq!(DriftSeverity::from_p_value(0.1), DriftSeverity::None);
        assert_eq!(DriftSeverity::from_p_value(0.06), DriftSeverity::None);
        assert_eq!(DriftSeverity::from_p_value(0.04), DriftSeverity::Low);
        assert_eq!(DriftSeverity::from_p_value(0.005), DriftSeverity::Medium);
        assert_eq!(DriftSeverity::from_p_value(0.0005), DriftSeverity::High);
        assert_eq!(
            DriftSeverity::from_p_value(0.00001),
            DriftSeverity::Critical
        );
    }

    #[test]
    fn test_drift_severity_from_psi() {
        assert_eq!(DriftSeverity::from_psi(0.05), DriftSeverity::None);
        assert_eq!(DriftSeverity::from_psi(0.15), DriftSeverity::Low);
        assert_eq!(DriftSeverity::from_psi(0.22), DriftSeverity::Medium);
        assert_eq!(DriftSeverity::from_psi(0.35), DriftSeverity::High);
        assert_eq!(DriftSeverity::from_psi(0.6), DriftSeverity::Critical);
    }

    #[test]
    fn test_drift_severity_is_drift() {
        assert!(!DriftSeverity::None.is_drift());
        assert!(DriftSeverity::Low.is_drift());
        assert!(DriftSeverity::Medium.is_drift());
        assert!(DriftSeverity::High.is_drift());
        assert!(DriftSeverity::Critical.is_drift());
    }

    #[test]
    fn test_drift_severity_ordering() {
        assert!(DriftSeverity::None < DriftSeverity::Low);
        assert!(DriftSeverity::Low < DriftSeverity::Medium);
        assert!(DriftSeverity::Medium < DriftSeverity::High);
        assert!(DriftSeverity::High < DriftSeverity::Critical);
    }

    // ========== ColumnDrift tests ==========

    #[test]
    fn test_column_drift_new() {
        let drift = ColumnDrift::new(
            "age",
            DriftTest::KolmogorovSmirnov,
            0.15,
            Some(0.03),
            DriftSeverity::Low,
        );

        assert_eq!(drift.column, "age");
        assert_eq!(drift.test, DriftTest::KolmogorovSmirnov);
        assert!((drift.statistic - 0.15).abs() < f64::EPSILON);
        assert_eq!(drift.p_value, Some(0.03));
        assert!(drift.drift_detected);
        assert_eq!(drift.severity, DriftSeverity::Low);
    }

    #[test]
    fn test_column_drift_no_drift() {
        let drift = ColumnDrift::new("income", DriftTest::PSI, 0.05, None, DriftSeverity::None);

        assert!(!drift.drift_detected);
        assert_eq!(drift.severity, DriftSeverity::None);
    }

    // ========== DriftReport tests ==========

    #[test]
    fn test_drift_report_from_columns() {
        let columns = vec![
            ColumnDrift::new(
                "a",
                DriftTest::KolmogorovSmirnov,
                0.1,
                Some(0.5),
                DriftSeverity::None,
            ),
            ColumnDrift::new("b", DriftTest::PSI, 0.25, None, DriftSeverity::Medium),
        ];

        let report = DriftReport::from_columns(columns);

        assert!(report.drift_detected);
        assert_eq!(report.num_columns(), 2);
        assert_eq!(report.num_drifted(), 1);
        assert_eq!(report.max_severity(), DriftSeverity::Medium);
    }

    #[test]
    fn test_drift_report_no_drift() {
        let columns = vec![
            ColumnDrift::new(
                "a",
                DriftTest::KolmogorovSmirnov,
                0.05,
                Some(0.5),
                DriftSeverity::None,
            ),
            ColumnDrift::new("b", DriftTest::PSI, 0.05, None, DriftSeverity::None),
        ];

        let report = DriftReport::from_columns(columns);

        assert!(!report.drift_detected);
        assert_eq!(report.num_drifted(), 0);
        assert_eq!(report.max_severity(), DriftSeverity::None);
    }

    #[test]
    fn test_drift_report_drifted_columns() {
        let columns = vec![
            ColumnDrift::new(
                "a",
                DriftTest::KolmogorovSmirnov,
                0.1,
                Some(0.5),
                DriftSeverity::None,
            ),
            ColumnDrift::new("b", DriftTest::PSI, 0.3, None, DriftSeverity::High),
            ColumnDrift::new(
                "c",
                DriftTest::ChiSquared,
                50.0,
                Some(0.001),
                DriftSeverity::Medium,
            ),
        ];

        let report = DriftReport::from_columns(columns);
        let drifted = report.drifted_columns();

        assert_eq!(drifted.len(), 2);
        assert!(drifted.contains(&"b"));
        assert!(drifted.contains(&"c"));
    }

    // ========== KS test implementation tests ==========

    #[test]
    fn test_ks_identical_distributions() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let result = ks_test(&data, &data, 0.05).expect("ks test");

        assert!(
            result.statistic < 0.01,
            "KS statistic should be ~0 for identical data"
        );
        assert!(
            result.p_value.unwrap_or(0.0) > 0.05,
            "p-value should be high"
        );
        assert_eq!(result.severity, DriftSeverity::None);
    }

    #[test]
    fn test_ks_different_distributions() {
        // Uniform [0, 100] vs Uniform [50, 150]
        let ref_data: Vec<f64> = (0..1000).map(|i| i as f64 / 10.0).collect();
        let cur_data: Vec<f64> = (0..1000).map(|i| 50.0 + i as f64 / 10.0).collect();

        let result = ks_test(&ref_data, &cur_data, 0.05).expect("ks test");

        assert!(
            result.statistic > 0.3,
            "KS statistic should be large for shifted data"
        );
        assert!(
            result.p_value.unwrap_or(1.0) < 0.05,
            "p-value should be small"
        );
        assert!(result.severity.is_drift());
    }

    #[test]
    fn test_ks_empty_data_error() {
        let empty: Vec<f64> = vec![];
        let data = vec![1.0, 2.0, 3.0];

        assert!(ks_test(&empty, &data, 0.05).is_err());
        assert!(ks_test(&data, &empty, 0.05).is_err());
    }

    // ========== Chi-squared test implementation tests ==========

    #[test]
    fn test_chi_squared_identical_distributions() {
        let data: Vec<f64> = (0..1000).map(|i| (i % 10) as f64).collect();
        let result = chi_squared_test(&data, &data, 0.05).expect("chi-squared test");

        // Identical data should have chi-sq ≈ 0
        assert!(
            result.statistic < 1.0,
            "Chi-squared should be small for identical data"
        );
        assert!(result.p_value.unwrap_or(0.0) > 0.05);
        assert_eq!(result.severity, DriftSeverity::None);
    }

    #[test]
    fn test_chi_squared_different_distributions() {
        // Very different distributions
        let ref_data: Vec<f64> = (0..1000).map(|_| 0.0).collect();
        let cur_data: Vec<f64> = (0..1000).map(|_| 100.0).collect();

        let result = chi_squared_test(&ref_data, &cur_data, 0.05).expect("chi-squared test");

        assert!(result.statistic > 100.0, "Chi-squared should be large");
        assert!(result.p_value.unwrap_or(1.0) < 0.001);
        assert!(result.severity.is_drift());
    }

    // ========== PSI test implementation tests ==========

    #[test]
    fn test_psi_identical_distributions() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let result = psi_test(&data, &data).expect("psi test");

        assert!(
            result.statistic < 0.05,
            "PSI should be ~0 for identical data"
        );
        assert_eq!(result.severity, DriftSeverity::None);
    }

    #[test]
    fn test_psi_shifted_distribution() {
        let ref_data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let cur_data: Vec<f64> = (0..1000).map(|i| 500.0 + i as f64).collect();

        let result = psi_test(&ref_data, &cur_data).expect("psi test");

        assert!(
            result.statistic > 0.2,
            "PSI should indicate drift: {}",
            result.statistic
        );
        assert!(result.severity.is_drift());
    }

    #[test]
    fn test_psi_moderate_shift() {
        // Moderate shift - should have PSI between 0.1 and 0.25
        let ref_data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let cur_data: Vec<f64> = (0..1000).map(|i| i as f64 * 1.1 + 50.0).collect();

        let result = psi_test(&ref_data, &cur_data).expect("psi test");

        // PSI should be in moderate range
        assert!(result.statistic > 0.0, "PSI should be positive");
    }

    // ========== Jensen-Shannon test implementation tests ==========

    #[test]
    fn test_jsd_identical_distributions() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let result = jensen_shannon_test(&data, &data).expect("jsd test");

        assert!(
            result.statistic < 0.01,
            "JSD should be ~0 for identical data"
        );
        assert_eq!(result.severity, DriftSeverity::None);
    }

    #[test]
    fn test_jsd_different_distributions() {
        // Completely non-overlapping distributions
        let ref_data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let cur_data: Vec<f64> = (0..1000).map(|i| 10000.0 + i as f64).collect();

        let result = jensen_shannon_test(&ref_data, &cur_data).expect("jsd test");

        assert!(
            result.statistic > 0.5,
            "JSD should be high for non-overlapping: {}",
            result.statistic
        );
        assert!(result.severity.is_drift());
    }

    // ========== DriftDetector tests ==========

    fn make_test_dataset(values: Vec<f64>) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(values))],
        )
        .expect("batch");

        ArrowDataset::from_batch(batch).expect("dataset")
    }

    fn make_int_dataset(values: Vec<i32>) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )
        .expect("batch");

        ArrowDataset::from_batch(batch).expect("dataset")
    }

    #[test]
    fn test_drift_detector_new() {
        let dataset = make_test_dataset(vec![1.0, 2.0, 3.0]);
        let detector = DriftDetector::new(dataset);

        assert_eq!(detector.alpha(), 0.05);
        assert_eq!(detector.tests().len(), 1);
        assert_eq!(detector.tests()[0], DriftTest::KolmogorovSmirnov);
    }

    #[test]
    fn test_drift_detector_builder() {
        let dataset = make_test_dataset(vec![1.0, 2.0, 3.0]);
        let detector = DriftDetector::new(dataset)
            .with_test(DriftTest::PSI)
            .with_test(DriftTest::ChiSquared)
            .with_alpha(0.01);

        assert_eq!(detector.alpha(), 0.01);
        assert_eq!(detector.tests().len(), 3);
    }

    #[test]
    fn test_drift_detector_no_duplicate_tests() {
        let dataset = make_test_dataset(vec![1.0, 2.0, 3.0]);
        let detector = DriftDetector::new(dataset)
            .with_test(DriftTest::KolmogorovSmirnov) // duplicate
            .with_test(DriftTest::KolmogorovSmirnov); // duplicate

        assert_eq!(detector.tests().len(), 1);
    }

    #[test]
    fn test_drift_detector_detect_no_drift() {
        let ref_data: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let cur_data: Vec<f64> = (0..500).map(|i| i as f64).collect();

        let reference = make_test_dataset(ref_data);
        let current = make_test_dataset(cur_data);

        let detector = DriftDetector::new(reference);
        let report = detector.detect(&current).expect("detect");

        assert!(!report.drift_detected);
        assert_eq!(report.num_columns(), 1);
    }

    #[test]
    fn test_drift_detector_detect_drift() {
        let ref_data: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let cur_data: Vec<f64> = (0..500).map(|i| 1000.0 + i as f64).collect();

        let reference = make_test_dataset(ref_data);
        let current = make_test_dataset(cur_data);

        let detector = DriftDetector::new(reference);
        let report = detector.detect(&current).expect("detect");

        assert!(report.drift_detected);
        assert!(report.max_severity().is_drift());
    }

    #[test]
    fn test_drift_detector_schema_mismatch() {
        let ref_dataset = make_test_dataset(vec![1.0, 2.0, 3.0]);
        let cur_dataset = make_int_dataset(vec![1, 2, 3]);

        let detector = DriftDetector::new(ref_dataset);
        let result = detector.detect(&cur_dataset);

        assert!(result.is_err());
    }

    #[test]
    fn test_drift_detector_multiple_tests() {
        let ref_data: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let cur_data: Vec<f64> = (0..500).map(|i| 500.0 + i as f64).collect();

        let reference = make_test_dataset(ref_data);
        let current = make_test_dataset(cur_data);

        let detector = DriftDetector::new(reference)
            .with_test(DriftTest::PSI)
            .with_test(DriftTest::JensenShannon);

        let report = detector.detect(&current).expect("detect");

        // Should have results for each test
        assert_eq!(report.num_columns(), 3); // 1 column × 3 tests
    }

    // ========== Edge cases ==========

    #[test]
    fn test_ks_with_nan_values() {
        let ref_data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        let cur_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = ks_test(&ref_data, &cur_data, 0.05).expect("ks test");
        // Should handle NaN gracefully
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_psi_with_small_sample() {
        let ref_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cur_data = vec![1.0, 2.0, 3.0, 4.0, 6.0];

        let result = psi_test(&ref_data, &cur_data).expect("psi test");
        // Should work with small samples
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_bin_data_constant_values() {
        let ref_data = vec![5.0; 100];
        let cur_data = vec![5.0; 100];

        let result = bin_data(&ref_data, &cur_data, 10).expect("bin data");
        // Should handle constant data (all in one bin)
        assert_eq!(result.0.iter().sum::<usize>(), 100);
        assert_eq!(result.1.iter().sum::<usize>(), 100);
    }
}
