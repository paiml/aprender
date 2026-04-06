//! Jidoka Guard (Built-in Quality: Stop on Defect)
//!
//! Implements Toyota Production System's Jidoka principle:
//! stop production when a defect is detected.

/// Jidoka condition that triggers stop
#[derive(Debug, Clone, PartialEq)]
pub enum JidokaCondition {
    /// NaN detected in output
    NanDetected,
    /// Infinity detected in output
    InfDetected,
    /// Cross-backend divergence exceeds tolerance
    BackendDivergence {
        /// Tolerance threshold
        tolerance: f32,
    },
    /// Performance regression exceeds threshold
    PerformanceRegression {
        /// Threshold percentage
        threshold_pct: f32,
    },
    /// Determinism failure (same seed, different output)
    DeterminismFailure,
}

/// Jidoka action on condition trigger
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JidokaAction {
    /// Stop immediately and report
    Stop,
    /// Log and continue (soft Jidoka)
    LogAndContinue,
    /// Trigger visual diff report
    VisualReport,
}

/// Jidoka error types
#[derive(Debug, Clone)]
pub enum JidokaError {
    /// NaN values detected
    NanDetected {
        /// Context description
        context: String,
        /// Indices of NaN values
        indices: Vec<usize>,
    },
    /// Infinity values detected
    InfDetected {
        /// Context description
        context: String,
        /// Indices of infinite values
        indices: Vec<usize>,
    },
    /// Backend divergence detected
    BackendDivergence {
        /// Context description
        context: String,
        /// Maximum difference found
        max_diff: f32,
        /// Tolerance threshold
        tolerance: f32,
    },
    /// Performance regression detected
    PerformanceRegression {
        /// Context description
        context: String,
        /// Actual regression percentage
        regression_pct: f32,
        /// Threshold percentage
        threshold_pct: f32,
    },
    /// Determinism failure detected
    DeterminismFailure {
        /// Context description
        context: String,
        /// First differing index
        first_diff_index: usize,
    },
}

impl std::fmt::Display for JidokaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NanDetected { context, indices } => {
                write!(f, "Jidoka: NaN detected at {context} (indices: {indices:?})")
            }
            Self::InfDetected { context, indices } => {
                write!(f, "Jidoka: Infinity detected at {context} (indices: {indices:?})")
            }
            Self::BackendDivergence { context, max_diff, tolerance } => {
                write!(
                    f,
                    "Jidoka: Backend divergence at {context} (max_diff: {max_diff}, tolerance: {tolerance})"
                )
            }
            Self::PerformanceRegression { context, regression_pct, threshold_pct } => {
                write!(
                    f,
                    "Jidoka: Performance regression at {context} ({regression_pct:.2}% > {threshold_pct:.2}%)"
                )
            }
            Self::DeterminismFailure { context, first_diff_index } => {
                write!(
                    f,
                    "Jidoka: Determinism failure at {context} (first diff at index {first_diff_index})"
                )
            }
        }
    }
}

impl std::error::Error for JidokaError {}

/// Jidoka guard for simulation tests
///
/// Implements Toyota Production System's Jidoka principle:
/// stop production when a defect is detected.
#[derive(Debug, Clone)]
pub struct JidokaGuard {
    /// Condition that triggers stop
    pub condition: JidokaCondition,
    /// Action to take on trigger
    pub action: JidokaAction,
    /// Context for debugging
    pub context: String,
}

impl JidokaGuard {
    /// Create a new Jidoka guard
    #[must_use]
    pub fn new(
        condition: JidokaCondition,
        action: JidokaAction,
        context: impl Into<String>,
    ) -> Self {
        Self { condition, action, context: context.into() }
    }

    /// Create a NaN detection guard
    #[must_use]
    pub fn nan_guard(context: impl Into<String>) -> Self {
        Self::new(JidokaCondition::NanDetected, JidokaAction::Stop, context)
    }

    /// Create an infinity detection guard
    #[must_use]
    pub fn inf_guard(context: impl Into<String>) -> Self {
        Self::new(JidokaCondition::InfDetected, JidokaAction::Stop, context)
    }

    /// Create a backend divergence guard
    #[must_use]
    pub fn divergence_guard(tolerance: f32, context: impl Into<String>) -> Self {
        Self::new(JidokaCondition::BackendDivergence { tolerance }, JidokaAction::Stop, context)
    }

    /// Check output for NaN/Inf and return error if found
    ///
    /// # Errors
    ///
    /// Returns `JidokaError` if the condition is triggered
    pub fn check_output(&self, output: &[f32]) -> Result<(), JidokaError> {
        match &self.condition {
            JidokaCondition::NanDetected => {
                let nan_indices: Vec<usize> =
                    output.iter().enumerate().filter(|(_, x)| x.is_nan()).map(|(i, _)| i).collect();

                if !nan_indices.is_empty() {
                    return Err(JidokaError::NanDetected {
                        context: self.context.clone(),
                        indices: nan_indices,
                    });
                }
            }
            JidokaCondition::InfDetected => {
                let inf_indices: Vec<usize> = output
                    .iter()
                    .enumerate()
                    .filter(|(_, x)| x.is_infinite())
                    .map(|(i, _)| i)
                    .collect();

                if !inf_indices.is_empty() {
                    return Err(JidokaError::InfDetected {
                        context: self.context.clone(),
                        indices: inf_indices,
                    });
                }
            }
            JidokaCondition::BackendDivergence { .. }
            | JidokaCondition::PerformanceRegression { .. }
            | JidokaCondition::DeterminismFailure => {} // Handled by compare methods
        }
        Ok(())
    }

    /// Compare two outputs for backend divergence
    ///
    /// # Errors
    ///
    /// Returns `JidokaError` if divergence exceeds tolerance
    pub fn check_divergence(&self, a: &[f32], b: &[f32]) -> Result<(), JidokaError> {
        if let JidokaCondition::BackendDivergence { tolerance } = &self.condition {
            let max_diff =
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);

            if max_diff > *tolerance {
                return Err(JidokaError::BackendDivergence {
                    context: self.context.clone(),
                    max_diff,
                    tolerance: *tolerance,
                });
            }
        }
        Ok(())
    }

    /// Check for determinism (same inputs should produce same outputs)
    ///
    /// # Errors
    ///
    /// Returns `JidokaError` if outputs differ
    pub fn check_determinism(&self, a: &[f32], b: &[f32]) -> Result<(), JidokaError> {
        if let JidokaCondition::DeterminismFailure = &self.condition {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                // Use bitwise comparison for exact equality
                if x.to_bits() != y.to_bits() {
                    return Err(JidokaError::DeterminismFailure {
                        context: self.context.clone(),
                        first_diff_index: i,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jidoka_nan_detection() {
        // Falsifiable claim B-027
        let guard = JidokaGuard::nan_guard("test_operation");
        let output_with_nan = vec![1.0, 2.0, f32::NAN, 4.0];

        let result = guard.check_output(&output_with_nan);
        assert!(result.is_err());

        if let Err(JidokaError::NanDetected { indices, .. }) = result {
            assert_eq!(indices, vec![2]);
        } else {
            panic!("Expected NanDetected error");
        }
    }

    #[test]
    fn test_jidoka_nan_no_false_positive() {
        let guard = JidokaGuard::nan_guard("test_operation");
        let clean_output = vec![1.0, 2.0, 3.0, 4.0];

        let result = guard.check_output(&clean_output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jidoka_inf_detection() {
        // Falsifiable claim B-028
        let guard = JidokaGuard::inf_guard("test_operation");
        let output_with_inf = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY];

        let result = guard.check_output(&output_with_inf);
        assert!(result.is_err());

        if let Err(JidokaError::InfDetected { indices, .. }) = result {
            assert_eq!(indices, vec![1, 3]);
        } else {
            panic!("Expected InfDetected error");
        }
    }

    #[test]
    fn test_jidoka_divergence_detection() {
        // Falsifiable claim A-004
        let guard = JidokaGuard::divergence_guard(1e-5, "cross_backend");
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.1, 4.0]; // 0.1 diff at index 2

        let result = guard.check_divergence(&a, &b);
        assert!(result.is_err());

        if let Err(JidokaError::BackendDivergence { max_diff, .. }) = result {
            assert!((max_diff - 0.1).abs() < 1e-6);
        } else {
            panic!("Expected BackendDivergence error");
        }
    }

    #[test]
    fn test_jidoka_divergence_within_tolerance() {
        let guard = JidokaGuard::divergence_guard(1e-5, "cross_backend");
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0 + 1e-7, 4.0]; // Within tolerance

        let result = guard.check_divergence(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jidoka_determinism_check() {
        // Falsifiable claim B-017
        let guard = JidokaGuard::new(
            JidokaCondition::DeterminismFailure,
            JidokaAction::Stop,
            "determinism_test",
        );

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let result = guard.check_determinism(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jidoka_determinism_failure() {
        let guard = JidokaGuard::new(
            JidokaCondition::DeterminismFailure,
            JidokaAction::Stop,
            "determinism_test",
        );

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.000_001, 4.0]; // Different bit pattern

        // Verify they actually have different bits
        assert_ne!(a[2].to_bits(), b[2].to_bits(), "Test values must differ");

        let result = guard.check_determinism(&a, &b);
        assert!(result.is_err());

        if let Err(JidokaError::DeterminismFailure { first_diff_index, .. }) = result {
            assert_eq!(first_diff_index, 2);
        } else {
            panic!("Expected DeterminismFailure error");
        }
    }

    #[test]
    fn test_jidoka_error_display() {
        let err = JidokaError::NanDetected { context: "test".to_string(), indices: vec![0, 2] };
        let display = format!("{err}");
        assert!(display.contains("NaN"));
        assert!(display.contains("test"));

        let err2 = JidokaError::BackendDivergence {
            context: "cross".to_string(),
            max_diff: 0.01,
            tolerance: 0.001,
        };
        let display2 = format!("{err2}");
        assert!(display2.contains("divergence"));
    }

    // ================================================================
    // Coverage tests for JidokaError Display — missing variants
    // ================================================================

    #[test]
    fn test_jidoka_error_display_inf_detected() {
        let err =
            JidokaError::InfDetected { context: "matmul_output".to_string(), indices: vec![1, 3] };
        let display = format!("{err}");
        assert!(display.contains("Infinity"), "Display should contain 'Infinity', got: {display}");
        assert!(
            display.contains("matmul_output"),
            "Display should contain context, got: {display}"
        );
        assert!(display.contains("[1, 3]"), "Display should contain indices, got: {display}");
    }

    #[test]
    fn test_jidoka_error_display_performance_regression() {
        let err = JidokaError::PerformanceRegression {
            context: "avx2_dot_product".to_string(),
            regression_pct: 15.75,
            threshold_pct: 5.0,
        };
        let display = format!("{err}");
        assert!(
            display.contains("Performance regression"),
            "Display should contain 'Performance regression', got: {display}"
        );
        assert!(
            display.contains("avx2_dot_product"),
            "Display should contain context, got: {display}"
        );
        assert!(display.contains("15.75"), "Display should contain regression_pct, got: {display}");
        assert!(display.contains("5.00"), "Display should contain threshold_pct, got: {display}");
    }

    #[test]
    fn test_jidoka_error_display_determinism_failure() {
        let err = JidokaError::DeterminismFailure {
            context: "sse2_vs_avx2".to_string(),
            first_diff_index: 42,
        };
        let display = format!("{err}");
        assert!(
            display.contains("Determinism failure"),
            "Display should contain 'Determinism failure', got: {display}"
        );
        assert!(display.contains("sse2_vs_avx2"), "Display should contain context, got: {display}");
        assert!(display.contains("42"), "Display should contain first_diff_index, got: {display}");
    }

    #[test]
    fn test_jidoka_error_is_std_error() {
        // Verify the std::error::Error impl works for all variants
        let errors: Vec<Box<dyn std::error::Error>> = vec![
            Box::new(JidokaError::NanDetected { context: "a".to_string(), indices: vec![] }),
            Box::new(JidokaError::InfDetected { context: "b".to_string(), indices: vec![] }),
            Box::new(JidokaError::BackendDivergence {
                context: "c".to_string(),
                max_diff: 0.0,
                tolerance: 0.0,
            }),
            Box::new(JidokaError::PerformanceRegression {
                context: "d".to_string(),
                regression_pct: 0.0,
                threshold_pct: 0.0,
            }),
            Box::new(JidokaError::DeterminismFailure {
                context: "e".to_string(),
                first_diff_index: 0,
            }),
        ];
        // All variants should produce non-empty Display output via Error trait
        for err in &errors {
            assert!(
                !err.to_string().is_empty(),
                "Error::to_string() should produce non-empty output"
            );
        }
    }

    #[test]
    fn test_empty_output_checks() {
        let guard = JidokaGuard::nan_guard("empty_test");
        let result = guard.check_output(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_element_checks() {
        let guard = JidokaGuard::nan_guard("single_test");

        assert!(guard.check_output(&[1.0]).is_ok());
        assert!(guard.check_output(&[f32::NAN]).is_err());
    }

    #[test]
    fn test_jidoka_condition_clone() {
        let condition = JidokaCondition::BackendDivergence { tolerance: 1e-5 };
        let cloned = condition.clone();
        assert_eq!(condition, cloned);
    }

    #[test]
    fn test_jidoka_action_eq() {
        assert_eq!(JidokaAction::Stop, JidokaAction::Stop);
        assert_ne!(JidokaAction::Stop, JidokaAction::LogAndContinue);
    }
}
