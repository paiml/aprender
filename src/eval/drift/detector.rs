//! Drift detector implementation.

use std::collections::HashMap;

use super::statistical::{bin_counts, chi_square_p_value, ks_p_value};
use super::types::{
    CategoricalBaseline, DriftCallback, DriftResult, DriftSummary, DriftTest, Severity,
};

/// Drift detector with statistical tests and callbacks
pub struct DriftDetector {
    tests: Vec<DriftTest>,
    baseline: Option<Vec<Vec<f64>>>,
    baseline_categorical: Option<CategoricalBaseline>,
    warning_multiplier: f64,
    callbacks: Vec<DriftCallback>,
}

impl DriftDetector {
    /// Create a new drift detector with specified tests
    pub fn new(tests: Vec<DriftTest>) -> Self {
        Self {
            tests,
            baseline: None,
            baseline_categorical: None,
            warning_multiplier: 0.8, // Warning at 80% of threshold
            callbacks: Vec::new(),
        }
    }

    /// Register callback for drift events (Andon Cord)
    ///
    /// Callbacks are invoked when drift is detected via `check_and_trigger`.
    pub fn on_drift<F>(&mut self, callback: F)
    where
        F: Fn(&[DriftResult]) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// Check for drift and trigger callbacks if drift detected
    ///
    /// Returns the drift results and invokes all registered callbacks
    /// if any feature shows drift.
    pub fn check_and_trigger(&self, current: &[Vec<f64>]) -> Vec<DriftResult> {
        let results = self.check(current);

        // Check if any drift detected
        let has_drift = results.iter().any(|r| r.drifted);

        if has_drift {
            for callback in &self.callbacks {
                callback(&results);
            }
        }

        results
    }

    /// Check categorical features for drift and trigger callbacks
    pub fn check_categorical_and_trigger(&self, current: &[Vec<usize>]) -> Vec<DriftResult> {
        let results = self.check_categorical(current);

        let has_drift = results.iter().any(|r| r.drifted);

        if has_drift {
            for callback in &self.callbacks {
                callback(&results);
            }
        }

        results
    }

    /// Set baseline distribution for continuous features
    /// Each row is a sample, each column is a feature
    pub fn set_baseline(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }
        // Transpose: store column-wise for easier feature comparison
        let n_features = data[0].len();
        let mut columns = vec![Vec::new(); n_features];
        for row in data {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    columns[i].push(val);
                }
            }
        }
        self.baseline = Some(columns);
    }

    /// Set baseline distribution for categorical features
    pub fn set_baseline_categorical(&mut self, data: &[Vec<usize>]) {
        if data.is_empty() {
            return;
        }
        let n_features = data[0].len();
        let mut histograms = vec![HashMap::new(); n_features];
        for row in data {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    *histograms[i].entry(val).or_insert(0) += 1;
                }
            }
        }
        self.baseline_categorical = Some(histograms);
    }

    /// Check new data for drift against baseline
    pub fn check(&self, current: &[Vec<f64>]) -> Vec<DriftResult> {
        let mut results = Vec::new();

        let baseline = match &self.baseline {
            Some(b) => b,
            None => return results,
        };

        if current.is_empty() {
            return results;
        }

        // Transpose current data to column-wise
        let n_features = current[0].len().min(baseline.len());
        let mut current_columns = vec![Vec::new(); n_features];
        for row in current {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    current_columns[i].push(val);
                }
            }
        }

        // Run tests on each feature
        for (feature_idx, (baseline_col, current_col)) in
            baseline.iter().zip(current_columns.iter()).enumerate()
        {
            for test in &self.tests {
                let result = match test {
                    DriftTest::KS { threshold } => {
                        self.ks_test(feature_idx, baseline_col, current_col, *threshold)
                    }
                    DriftTest::PSI { threshold } => {
                        self.psi_test(feature_idx, baseline_col, current_col, *threshold)
                    }
                    DriftTest::ChiSquare { .. } => continue, // Skip for continuous
                };
                results.push(result);
            }
        }

        results
    }

    /// Check categorical features for drift
    pub fn check_categorical(&self, current: &[Vec<usize>]) -> Vec<DriftResult> {
        let mut results = Vec::new();

        let baseline = match &self.baseline_categorical {
            Some(b) => b,
            None => return results,
        };

        if current.is_empty() {
            return results;
        }

        // Build current histograms
        let n_features = current[0].len().min(baseline.len());
        let mut current_histograms = vec![HashMap::new(); n_features];
        for row in current {
            for (i, &val) in row.iter().enumerate() {
                if i < n_features {
                    *current_histograms[i].entry(val).or_insert(0) += 1;
                }
            }
        }

        // Run chi-square test on each feature
        for (feature_idx, (baseline_hist, current_hist)) in
            baseline.iter().zip(current_histograms.iter()).enumerate()
        {
            for test in &self.tests {
                if let DriftTest::ChiSquare { threshold } = test {
                    let result =
                        self.chi_square_test(feature_idx, baseline_hist, current_hist, *threshold);
                    results.push(result);
                }
            }
        }

        results
    }

    /// Kolmogorov-Smirnov test for continuous features
    fn ks_test(
        &self,
        feature_idx: usize,
        baseline: &[f64],
        current: &[f64],
        threshold: f64,
    ) -> DriftResult {
        // Sort both distributions
        let mut sorted_baseline: Vec<f64> = baseline.to_vec();
        let mut sorted_current: Vec<f64> = current.to_vec();
        sorted_baseline.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_current.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute empirical CDFs and find maximum difference
        let n1 = sorted_baseline.len() as f64;
        let n2 = sorted_current.len() as f64;

        let mut d_max = 0.0f64;
        let mut i = 0usize;
        let mut j = 0usize;

        while i < sorted_baseline.len() && j < sorted_current.len() {
            let cdf1 = (i + 1) as f64 / n1;
            let cdf2 = (j + 1) as f64 / n2;

            let diff = (cdf1 - cdf2).abs();
            d_max = d_max.max(diff);

            if sorted_baseline[i] <= sorted_current[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        // Approximate p-value using asymptotic formula
        let n_eff = (n1 * n2) / (n1 + n2);
        let lambda = d_max * n_eff.sqrt();
        let p_value = ks_p_value(lambda);

        let (drifted, severity) = self.classify_result(p_value, threshold);

        DriftResult {
            feature: format!("feature_{feature_idx}"),
            test: DriftTest::KS { threshold },
            statistic: d_max,
            p_value,
            drifted,
            severity,
        }
    }

    /// Population Stability Index (PSI) test
    fn psi_test(
        &self,
        feature_idx: usize,
        baseline: &[f64],
        current: &[f64],
        threshold: f64,
    ) -> DriftResult {
        if baseline.is_empty() || current.is_empty() {
            return DriftResult {
                feature: format!("feature_{feature_idx}"),
                test: DriftTest::PSI { threshold },
                statistic: 0.0,
                p_value: 0.0,
                drifted: false,
                severity: Severity::None,
            };
        }

        // Create 10 bins based on baseline deciles
        let n_bins = 10;
        let mut sorted_baseline: Vec<f64> = baseline.to_vec();
        sorted_baseline.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate bin edges (deciles)
        let mut edges = Vec::with_capacity(n_bins + 1);
        edges.push(f64::NEG_INFINITY);
        for i in 1..n_bins {
            let idx = (sorted_baseline.len() * i / n_bins.max(1)).min(sorted_baseline.len() - 1);
            edges.push(sorted_baseline[idx]);
        }
        edges.push(f64::INFINITY);

        // Count samples in each bin
        let baseline_counts = bin_counts(baseline, &edges);
        let current_counts = bin_counts(current, &edges);

        // Calculate PSI
        let total_baseline = baseline.len() as f64;
        let total_current = current.len() as f64;

        let mut psi = 0.0;
        for (b_count, c_count) in baseline_counts.iter().zip(current_counts.iter()) {
            let b_pct = (*b_count as f64 + 0.0001) / (total_baseline + 0.001);
            let c_pct = (*c_count as f64 + 0.0001) / (total_current + 0.001);
            psi += (c_pct - b_pct) * (c_pct / b_pct).max(f64::MIN_POSITIVE).ln();
        }

        let (drifted, severity) = if psi >= threshold {
            (true, Severity::Critical)
        } else if psi >= threshold * self.warning_multiplier {
            (true, Severity::Warning)
        } else {
            (false, Severity::None)
        };

        DriftResult {
            feature: format!("feature_{feature_idx}"),
            test: DriftTest::PSI { threshold },
            statistic: psi,
            p_value: psi, // PSI doesn't use p-value, store the PSI value
            drifted,
            severity,
        }
    }

    /// Chi-square test for categorical features
    fn chi_square_test(
        &self,
        feature_idx: usize,
        baseline: &HashMap<usize, usize>,
        current: &HashMap<usize, usize>,
        threshold: f64,
    ) -> DriftResult {
        // Get all categories
        let mut categories: Vec<usize> = baseline.keys().chain(current.keys()).copied().collect();
        categories.sort_unstable();
        categories.dedup();

        let total_baseline: f64 = baseline.values().sum::<usize>() as f64;
        let total_current: f64 = current.values().sum::<usize>() as f64;

        if total_baseline == 0.0 || total_current == 0.0 {
            return DriftResult {
                feature: format!("feature_{feature_idx}"),
                test: DriftTest::ChiSquare { threshold },
                statistic: 0.0,
                p_value: 1.0,
                drifted: false,
                severity: Severity::None,
            };
        }

        // Calculate chi-square statistic
        let mut chi_sq = 0.0;
        let mut df: usize = 0;

        for &cat in &categories {
            let observed = *current.get(&cat).unwrap_or(&0) as f64;
            let baseline_pct = *baseline.get(&cat).unwrap_or(&0) as f64 / total_baseline;
            let expected = baseline_pct * total_current;

            if expected > 0.0 {
                chi_sq += (observed - expected).powi(2) / expected;
                df += 1;
            }
        }

        df = df.saturating_sub(1); // degrees of freedom = categories - 1
        let p_value = chi_square_p_value(chi_sq, df);

        let (drifted, severity) = self.classify_result(p_value, threshold);

        DriftResult {
            feature: format!("feature_{feature_idx}"),
            test: DriftTest::ChiSquare { threshold },
            statistic: chi_sq,
            p_value,
            drifted,
            severity,
        }
    }

    /// Classify result based on p-value and threshold
    fn classify_result(&self, p_value: f64, threshold: f64) -> (bool, Severity) {
        if p_value < threshold {
            (true, Severity::Critical)
        } else if p_value < threshold / self.warning_multiplier {
            (true, Severity::Warning)
        } else {
            (false, Severity::None)
        }
    }

    /// Get summary of drift results
    pub fn summary(results: &[DriftResult]) -> DriftSummary {
        let total = results.len();
        let drifted = results.iter().filter(|r| r.drifted).count();
        let warnings = results.iter().filter(|r| r.severity == Severity::Warning).count();
        let critical = results.iter().filter(|r| r.severity == Severity::Critical).count();

        DriftSummary { total_features: total, drifted_features: drifted, warnings, critical }
    }
}
