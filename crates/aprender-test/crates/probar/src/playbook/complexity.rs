//! Empirical computational complexity verification.
//!
//! Implements O(n) complexity detection via curve fitting.
//! Reference: Goldsmith et al., "Measuring Empirical Computational Complexity" (ESEC/FSE 2007)

use super::schema::ComplexityClass;

/// Result of complexity analysis.
#[derive(Debug, Clone)]
pub struct ComplexityResult {
    /// Detected complexity class
    pub detected_class: ComplexityClass,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// R² value for the best fit
    pub r_squared: f64,
    /// Whether the complexity violates the expected class
    pub is_violation: bool,
    /// Expected complexity (if specified)
    pub expected_class: Option<ComplexityClass>,
    /// Detailed fit results for each complexity class
    pub fit_results: Vec<FitResult>,
}

/// Result of fitting data to a complexity class.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Complexity class
    pub class: ComplexityClass,
    /// R² coefficient of determination
    pub r_squared: f64,
    /// Fitted coefficients
    pub coefficients: Vec<f64>,
}

/// Complexity analyzer using empirical curve fitting.
pub struct ComplexityAnalyzer {
    /// Input sizes (n values)
    input_sizes: Vec<f64>,
    /// Measured execution times
    execution_times: Vec<f64>,
}

impl ComplexityAnalyzer {
    /// Create a new analyzer with measurement data.
    ///
    /// # Arguments
    /// * `measurements` - Pairs of (input_size, execution_time)
    pub fn new(measurements: Vec<(usize, f64)>) -> Self {
        let (input_sizes, execution_times): (Vec<f64>, Vec<f64>) =
            measurements.into_iter().map(|(n, t)| (n as f64, t)).unzip();

        Self {
            input_sizes,
            execution_times,
        }
    }

    /// Analyze the measurements and determine complexity class.
    ///
    /// # Arguments
    /// * `expected` - Optional expected complexity class for violation detection
    pub fn analyze(&self, expected: Option<ComplexityClass>) -> ComplexityResult {
        if self.input_sizes.len() < 3 {
            return ComplexityResult {
                detected_class: ComplexityClass::Linear,
                confidence: 0.0,
                r_squared: 0.0,
                is_violation: false,
                expected_class: expected,
                fit_results: Vec::new(),
            };
        }

        // Fit data to each complexity class
        let fit_results = vec![
            self.fit_constant(),
            self.fit_logarithmic(),
            self.fit_linear(),
            self.fit_linearithmic(),
            self.fit_quadratic(),
        ];

        // Find best fit (highest R², prefer simpler models as tiebreaker)
        let best_fit = fit_results
            .iter()
            .max_by(|a, b| {
                match a.r_squared.partial_cmp(&b.r_squared) {
                    Some(std::cmp::Ordering::Equal) => {
                        // Prefer simpler models (lower complexity order) when R² is equal
                        complexity_order(b.class).cmp(&complexity_order(a.class))
                    }
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Equal,
                }
            })
            .expect("fit_results is non-empty");

        let detected_class = best_fit.class;
        let confidence = best_fit.r_squared;

        // Check for violation
        let is_violation = if let Some(exp) = expected {
            complexity_order(detected_class) > complexity_order(exp)
        } else {
            false
        };

        ComplexityResult {
            detected_class,
            confidence,
            r_squared: best_fit.r_squared,
            is_violation,
            expected_class: expected,
            fit_results,
        }
    }

    /// Fit to O(1) - constant time.
    fn fit_constant(&self) -> FitResult {
        // y = c
        let mean = self.execution_times.iter().sum::<f64>() / self.execution_times.len() as f64;
        let r_squared = self.compute_r_squared(|_| mean);

        FitResult {
            class: ComplexityClass::Constant,
            r_squared,
            coefficients: vec![mean],
        }
    }

    /// Fit to O(log n) - logarithmic.
    fn fit_logarithmic(&self) -> FitResult {
        // y = a * log(n) + b
        let log_n: Vec<f64> = self.input_sizes.iter().map(|n| n.ln()).collect();
        let (a, b) = self.linear_regression(&log_n, &self.execution_times);
        let r_squared = self.compute_r_squared(|n| a * n.ln() + b);

        FitResult {
            class: ComplexityClass::Logarithmic,
            r_squared,
            coefficients: vec![a, b],
        }
    }

    /// Fit to O(n) - linear.
    fn fit_linear(&self) -> FitResult {
        // y = a * n + b
        let (a, b) = self.linear_regression(&self.input_sizes, &self.execution_times);
        let r_squared = self.compute_r_squared(|n| a * n + b);

        FitResult {
            class: ComplexityClass::Linear,
            r_squared,
            coefficients: vec![a, b],
        }
    }

    /// Fit to O(n log n) - linearithmic.
    fn fit_linearithmic(&self) -> FitResult {
        // y = a * n * log(n) + b
        let n_log_n: Vec<f64> = self.input_sizes.iter().map(|n| n * n.ln()).collect();
        let (a, b) = self.linear_regression(&n_log_n, &self.execution_times);
        let r_squared = self.compute_r_squared(|n| a * n * n.ln() + b);

        FitResult {
            class: ComplexityClass::Linearithmic,
            r_squared,
            coefficients: vec![a, b],
        }
    }

    /// Fit to O(n²) - quadratic.
    fn fit_quadratic(&self) -> FitResult {
        // y = a * n² + b
        let n_squared: Vec<f64> = self.input_sizes.iter().map(|n| n * n).collect();
        let (a, b) = self.linear_regression(&n_squared, &self.execution_times);
        let r_squared = self.compute_r_squared(|n| a * n * n + b);

        FitResult {
            class: ComplexityClass::Quadratic,
            r_squared,
            coefficients: vec![a, b],
        }
    }

    /// Simple linear regression: y = ax + b
    fn linear_regression(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return (0.0, sum_y / n);
        }

        let a = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_y - a * sum_x) / n;

        (a, b)
    }

    /// Compute R² (coefficient of determination) for a model.
    fn compute_r_squared<F>(&self, model: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let y_mean = self.execution_times.iter().sum::<f64>() / self.execution_times.len() as f64;

        let ss_tot: f64 = self
            .execution_times
            .iter()
            .map(|y| (y - y_mean).powi(2))
            .sum();

        let ss_res: f64 = self
            .input_sizes
            .iter()
            .zip(self.execution_times.iter())
            .map(|(n, y)| (y - model(*n)).powi(2))
            .sum();

        if ss_tot.abs() < f64::EPSILON {
            return 1.0; // Perfect fit if no variance
        }

        1.0 - (ss_res / ss_tot)
    }
}

/// Get numerical order for complexity comparison.
fn complexity_order(class: ComplexityClass) -> u8 {
    match class {
        ComplexityClass::Constant => 0,
        ComplexityClass::Logarithmic => 1,
        ComplexityClass::Linear => 2,
        ComplexityClass::Linearithmic => 3,
        ComplexityClass::Quadratic => 4,
    }
}

/// Check if measured complexity violates expected complexity.
pub fn check_complexity_violation(
    measurements: Vec<(usize, f64)>,
    expected: ComplexityClass,
) -> ComplexityResult {
    let analyzer = ComplexityAnalyzer::new(measurements);
    analyzer.analyze(Some(expected))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_linear_complexity() {
        // Generate O(n) data: t = 2n + 5
        let measurements: Vec<(usize, f64)> = (1..=10)
            .map(|n| (n * 100, (2 * n * 100) as f64 + 5.0))
            .collect();

        let analyzer = ComplexityAnalyzer::new(measurements);
        let result = analyzer.analyze(None);

        assert_eq!(result.detected_class, ComplexityClass::Linear);
        assert!(result.r_squared > 0.99);
    }

    #[test]
    fn test_detect_quadratic_complexity() {
        // Generate O(n²) data: t = n² + 10
        let measurements: Vec<(usize, f64)> = (1..=10)
            .map(|n| (n * 10, ((n * 10) * (n * 10)) as f64 + 10.0))
            .collect();

        let analyzer = ComplexityAnalyzer::new(measurements);
        let result = analyzer.analyze(None);

        assert_eq!(result.detected_class, ComplexityClass::Quadratic);
        assert!(result.r_squared > 0.99);
    }

    #[test]
    fn test_detect_constant_complexity() {
        // Generate O(1) data: t = 100 (exactly constant)
        let measurements: Vec<(usize, f64)> = (1..=10).map(|n| (n * 100, 100.0)).collect();

        let analyzer = ComplexityAnalyzer::new(measurements);
        let result = analyzer.analyze(None);

        // Constant data should be detected as constant
        assert_eq!(result.detected_class, ComplexityClass::Constant);
    }

    #[test]
    fn test_violation_detection() {
        // Generate O(n²) data but expect O(n)
        let measurements: Vec<(usize, f64)> = (1..=10)
            .map(|n| (n * 10, ((n * 10) * (n * 10)) as f64))
            .collect();

        let result = check_complexity_violation(measurements, ComplexityClass::Linear);

        assert!(result.is_violation);
        assert_eq!(result.expected_class, Some(ComplexityClass::Linear));
        assert_eq!(result.detected_class, ComplexityClass::Quadratic);
    }

    #[test]
    fn test_no_violation_when_better() {
        // Generate O(n) data but expect O(n²)
        let measurements: Vec<(usize, f64)> =
            (1..=10).map(|n| (n * 100, (n * 100) as f64)).collect();

        let result = check_complexity_violation(measurements, ComplexityClass::Quadratic);

        assert!(!result.is_violation);
    }

    #[test]
    fn test_insufficient_data() {
        // Only 2 points - not enough for reliable analysis
        let measurements = vec![(100, 100.0), (200, 200.0)];

        let analyzer = ComplexityAnalyzer::new(measurements);
        let result = analyzer.analyze(None);

        assert_eq!(result.confidence, 0.0);
    }
}
