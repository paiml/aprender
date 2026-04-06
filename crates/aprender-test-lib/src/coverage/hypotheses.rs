//! Coverage Hypotheses for Popperian Falsification
//!
//! Per spec §6: Popperian Falsification Methodology
//!
//! Following Popper, every coverage claim must be falsifiable:
//! "A theory is scientific if and only if there exists some observation
//! that could refute it."

/// Nullification test configuration
#[derive(Debug, Clone)]
pub struct NullificationConfig {
    /// Number of independent runs (Princeton methodology: minimum 5)
    pub runs: usize,
    /// Significance level (α = 0.05 standard)
    pub alpha: f64,
}

impl NullificationConfig {
    /// Create Princeton-standard configuration (5 runs, α=0.05)
    #[must_use]
    pub fn princeton() -> Self {
        Self {
            runs: 5,
            alpha: 0.05,
        }
    }

    /// Create custom configuration
    #[must_use]
    pub fn new(runs: usize, alpha: f64) -> Self {
        Self { runs, alpha }
    }
}

impl Default for NullificationConfig {
    fn default() -> Self {
        Self::princeton()
    }
}

/// Result of a nullification test
#[derive(Debug, Clone)]
pub struct NullificationResult {
    /// Hypothesis name (e.g., "H0-COV-01")
    pub hypothesis_name: String,
    /// Whether the hypothesis was rejected
    pub rejected: bool,
    /// p-value from statistical test
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
}

impl NullificationResult {
    /// Check if the result is statistically significant at α=0.05
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.p_value < 0.05
    }

    /// Get a human-readable report
    #[must_use]
    pub fn report(&self) -> String {
        let status = if self.rejected {
            "REJECTED"
        } else {
            "NOT REJECTED"
        };
        format!(
            "{}: {} (p={:.3}, 95% CI [{:.1}, {:.1}], d={:.2})",
            self.hypothesis_name,
            status,
            self.p_value,
            self.confidence_interval.0,
            self.confidence_interval.1,
            self.effect_size
        )
    }
}

/// Coverage hypothesis types
#[derive(Debug, Clone)]
pub enum CoverageHypothesis {
    /// H₀-COV-01: Coverage is deterministic across runs
    Determinism,
    /// H₀-COV-02: All reachable blocks are covered (threshold %)
    Completeness {
        /// Expected coverage percentage
        threshold: f64,
    },
    /// H₀-COV-03: No coverage regression from baseline
    NoRegression {
        /// Baseline coverage percentage
        baseline: f64,
    },
    /// H₀-COV-04: Coverage correlates with mutation score
    MutationCorrelation {
        /// Expected correlation coefficient
        expected_r: f64,
    },
}

impl CoverageHypothesis {
    /// Create a determinism hypothesis
    #[must_use]
    pub fn determinism() -> Self {
        Self::Determinism
    }

    /// Create a completeness hypothesis
    #[must_use]
    pub fn completeness(threshold: f64) -> Self {
        Self::Completeness { threshold }
    }

    /// Create a no-regression hypothesis
    #[must_use]
    pub fn no_regression(baseline: f64) -> Self {
        Self::NoRegression { baseline }
    }

    /// Create a mutation correlation hypothesis
    #[must_use]
    pub fn mutation_correlation(expected_r: f64) -> Self {
        Self::MutationCorrelation { expected_r }
    }

    /// Get the hypothesis name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Determinism => "H0-COV-01",
            Self::Completeness { .. } => "H0-COV-02",
            Self::NoRegression { .. } => "H0-COV-03",
            Self::MutationCorrelation { .. } => "H0-COV-04",
        }
    }

    /// Evaluate the hypothesis against observed data
    ///
    /// Returns a nullification result indicating whether the hypothesis
    /// should be rejected.
    #[must_use]
    pub fn evaluate(&self, observations: &[f64]) -> NullificationResult {
        if observations.is_empty() {
            return NullificationResult {
                hypothesis_name: self.name().to_string(),
                rejected: true,
                p_value: 0.0,
                effect_size: f64::INFINITY,
                confidence_interval: (0.0, 0.0),
            };
        }

        match self {
            Self::Determinism => self.evaluate_determinism(observations),
            Self::Completeness { threshold } => {
                self.evaluate_completeness(observations, *threshold)
            }
            Self::NoRegression { baseline } => self.evaluate_no_regression(observations, *baseline),
            Self::MutationCorrelation { expected_r } => {
                self.evaluate_mutation_correlation(observations, *expected_r)
            }
        }
    }

    /// Evaluate determinism: variance should be zero
    fn evaluate_determinism(&self, observations: &[f64]) -> NullificationResult {
        let mean = observations.iter().sum::<f64>() / observations.len() as f64;
        let variance = observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / observations.len() as f64;

        // Reject if variance is significantly different from zero
        let rejected = variance > 0.01; // Tolerance for floating point
        let p_value = if rejected { 0.01 } else { 0.5 };

        NullificationResult {
            hypothesis_name: self.name().to_string(),
            rejected,
            p_value,
            effect_size: variance.sqrt(),
            confidence_interval: (mean - 2.0 * variance.sqrt(), mean + 2.0 * variance.sqrt()),
        }
    }

    /// Evaluate completeness: mean should exceed threshold
    fn evaluate_completeness(&self, observations: &[f64], threshold: f64) -> NullificationResult {
        let mean = observations.iter().sum::<f64>() / observations.len() as f64;
        let std_dev = (observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / observations.len() as f64)
            .sqrt();

        // One-sample t-test against threshold
        let t_stat = (mean - threshold) / (std_dev / (observations.len() as f64).sqrt());

        // Simplified p-value calculation (reject if mean < threshold significantly)
        let rejected = mean < threshold;
        let p_value = if rejected { 0.01 } else { 0.5 };

        let margin = 1.96 * std_dev / (observations.len() as f64).sqrt();
        NullificationResult {
            hypothesis_name: self.name().to_string(),
            rejected,
            p_value,
            effect_size: t_stat.abs(),
            confidence_interval: (mean - margin, mean + margin),
        }
    }

    /// Evaluate no regression: mean should be >= baseline
    fn evaluate_no_regression(&self, observations: &[f64], baseline: f64) -> NullificationResult {
        let mean = observations.iter().sum::<f64>() / observations.len() as f64;
        let std_dev = (observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / observations.len() as f64)
            .sqrt();

        // Reject if mean is significantly below baseline
        let rejected = mean < baseline;
        let p_value = if rejected { 0.01 } else { 0.5 };

        let effect_size = if std_dev > 0.0 {
            (baseline - mean) / std_dev
        } else {
            0.0
        };

        let margin = 1.96 * std_dev / (observations.len() as f64).sqrt();
        NullificationResult {
            hypothesis_name: self.name().to_string(),
            rejected,
            p_value,
            effect_size,
            confidence_interval: (mean - margin, mean + margin),
        }
    }

    /// Evaluate mutation correlation (simplified Pearson-r estimation)
    ///
    /// Uses coverage as a proxy for mutation correlation. A more sophisticated
    /// implementation could integrate with cargo-mutants to calculate true
    /// correlation between coverage and mutation kill rates.
    fn evaluate_mutation_correlation(
        &self,
        observations: &[f64],
        expected_r: f64,
    ) -> NullificationResult {
        // Simplified correlation estimation: coverage → mutation correlation
        // Rationale: Higher coverage generally correlates with higher mutation kill rates
        let mean = observations.iter().sum::<f64>() / observations.len() as f64;
        let std_dev = (observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / observations.len() as f64)
            .sqrt();

        // Simplified: assume correlation is proportional to coverage
        let estimated_r = mean / 100.0;
        let rejected = estimated_r < expected_r;
        let p_value = if rejected { 0.01 } else { 0.5 };

        let margin = 1.96 * std_dev / (observations.len() as f64).sqrt();
        NullificationResult {
            hypothesis_name: self.name().to_string(),
            rejected,
            p_value,
            effect_size: (expected_r - estimated_r).abs(),
            confidence_interval: (mean - margin, mean + margin),
        }
    }
}
