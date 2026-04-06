//! Pruning-specific error types.
//!
//! Provides rich error context for pruning operations following
//! Toyota Way Jidoka (stop on defect) principles.
//!
//! # References
//! - Liker, J. K. (2004). The Toyota Way: 14 Management Principles.

use std::fmt;

/// Pruning operation errors with detailed context.
///
/// # Toyota Way: Andon
/// Errors contain actionable information for diagnosis.
/// Each variant provides specific context to help identify
/// and resolve issues quickly.
#[derive(Debug, Clone)]
pub enum PruningError {
    /// Numerical instability detected (NaN/Inf in scores).
    ///
    /// # Jidoka Principle
    /// Stop immediately when numerical issues are detected
    /// rather than propagating bad values downstream.
    NumericalInstability {
        /// Method that detected the instability
        method: String,
        /// Detailed description of what was detected
        details: String,
    },

    /// Calibration data required but not provided.
    ///
    /// Methods like Wanda require activation statistics from
    /// calibration data to compute importance scores.
    CalibrationRequired {
        /// Method requiring calibration
        method: String,
    },

    /// Tensor shape mismatch.
    ///
    /// Occurs when mask and weight tensor shapes don't align.
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape found
        got: Vec<usize>,
    },

    /// Invalid sparsity mask.
    ///
    /// Masks must contain only binary values (0.0 or 1.0).
    InvalidMask {
        /// Reason for invalidity
        reason: String,
    },

    /// Invalid sparsity pattern configuration.
    ///
    /// For N:M patterns, N must be less than M.
    InvalidPattern {
        /// Error message describing the invalid configuration
        message: String,
    },

    /// Missing activation statistics for a layer.
    ///
    /// The calibration context doesn't have stats for this layer.
    MissingActivationStats {
        /// Layer name that's missing stats
        layer: String,
    },

    /// Invalid sparsity target.
    ///
    /// Sparsity must be in range [0.0, 1.0].
    InvalidSparsity {
        /// Provided value
        value: f32,
        /// Constraint description
        constraint: String,
    },

    /// Module has no parameters to prune.
    NoParameters {
        /// Module identifier or description
        module: String,
    },
}

impl fmt::Display for PruningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PruningError::NumericalInstability { method, details } => {
                write!(f, "Numerical instability in {method}: {details}")
            }
            PruningError::CalibrationRequired { method } => {
                write!(
                    f,
                    "Method '{method}' requires calibration data but none was provided"
                )
            }
            PruningError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {got:?}")
            }
            PruningError::InvalidMask { reason } => {
                write!(f, "Invalid sparsity mask: {reason}")
            }
            PruningError::InvalidPattern { message } => {
                write!(f, "Invalid sparsity pattern: {message}")
            }
            PruningError::MissingActivationStats { layer } => {
                write!(f, "Missing activation statistics for layer '{layer}'")
            }
            PruningError::InvalidSparsity { value, constraint } => {
                write!(f, "Invalid sparsity value {value}: {constraint}")
            }
            PruningError::NoParameters { module } => {
                write!(f, "Module '{module}' has no parameters to prune")
            }
        }
    }
}

impl std::error::Error for PruningError {}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // FALSIFICATION TEST 1: Error variants exist with correct fields
    // Popper: "If PruningError does not have NumericalInstability variant,
    //          then the module does not meet Jidoka requirements"
    // ==========================================================================
    #[test]
    fn test_numerical_instability_error_has_context() {
        let err = PruningError::NumericalInstability {
            method: "MagnitudeImportance".to_string(),
            details: "NaN detected in importance scores".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("MagnitudeImportance"),
            "ERR-01 FALSIFIED: Error message must contain method name"
        );
        assert!(
            msg.contains("NaN"),
            "ERR-01 FALSIFIED: Error message must contain details"
        );
    }

    #[test]
    fn test_calibration_required_error() {
        let err = PruningError::CalibrationRequired {
            method: "Wanda".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("Wanda"),
            "ERR-02 FALSIFIED: Calibration error must contain method name"
        );
        assert!(
            msg.to_lowercase().contains("calibration"),
            "ERR-02 FALSIFIED: Error must mention calibration requirement"
        );
    }

    #[test]
    fn test_shape_mismatch_error() {
        let err = PruningError::ShapeMismatch {
            expected: vec![512, 256],
            got: vec![256, 512],
        };
        let msg = err.to_string();
        assert!(
            msg.contains("512"),
            "ERR-03 FALSIFIED: Shape mismatch must show expected dimensions"
        );
        assert!(
            msg.contains("256"),
            "ERR-03 FALSIFIED: Shape mismatch must show actual dimensions"
        );
    }

    #[test]
    fn test_invalid_mask_error() {
        let err = PruningError::InvalidMask {
            reason: "Mask contains non-binary values".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("non-binary"),
            "ERR-04 FALSIFIED: Invalid mask error must contain reason"
        );
    }

    #[test]
    fn test_invalid_pattern_error() {
        let err = PruningError::InvalidPattern {
            message: "N must be less than M".to_string(),
        };
        assert!(
            err.to_string().contains("N must be less than M"),
            "ERR-05 FALSIFIED: Pattern error must contain message"
        );
    }

    #[test]
    fn test_missing_activation_stats_error() {
        let err = PruningError::MissingActivationStats {
            layer: "model.layers.0.mlp".to_string(),
        };
        assert!(
            err.to_string().contains("model.layers.0.mlp"),
            "ERR-06 FALSIFIED: Missing stats error must contain layer name"
        );
    }

    #[test]
    fn test_invalid_sparsity_error() {
        let err = PruningError::InvalidSparsity {
            value: 1.5,
            constraint: "must be between 0.0 and 1.0".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("1.5"),
            "ERR-07 FALSIFIED: Invalid sparsity must show value"
        );
        assert!(
            msg.contains("0.0") && msg.contains("1.0"),
            "ERR-07 FALSIFIED: Invalid sparsity must show constraint"
        );
    }

    #[test]
    fn test_no_parameters_error() {
        let err = PruningError::NoParameters {
            module: "ReLU".to_string(),
        };
        assert!(
            err.to_string().contains("ReLU"),
            "ERR-08 FALSIFIED: No parameters error must contain module name"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Error implements std::error::Error
    // ==========================================================================
    #[test]
    fn test_error_implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<PruningError>();
    }

    // ==========================================================================
    // FALSIFICATION: Error has Debug impl
    // ==========================================================================
    #[test]
    fn test_error_debug_impl() {
        let err = PruningError::NumericalInstability {
            method: "test".to_string(),
            details: "test details".to_string(),
        };
        let debug_str = format!("{:?}", err);
        assert!(
            debug_str.contains("NumericalInstability"),
            "ERR-09 FALSIFIED: Debug must show variant name"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Error is Clone
    // ==========================================================================
    #[test]
    fn test_error_is_clone() {
        let err = PruningError::ShapeMismatch {
            expected: vec![10, 20],
            got: vec![20, 10],
        };
        let cloned = err.clone();
        assert_eq!(
            err.to_string(),
            cloned.to_string(),
            "ERR-10 FALSIFIED: Cloned error must be identical"
        );
    }

    // ==========================================================================
    // Edge case: Empty strings
    // ==========================================================================
    #[test]
    fn test_error_with_empty_strings() {
        let err = PruningError::NumericalInstability {
            method: String::new(),
            details: String::new(),
        };
        // Should not panic
        let _ = err.to_string();
        let _ = format!("{:?}", err);
    }

    // ==========================================================================
    // Edge case: Unicode in layer names
    // ==========================================================================
    #[test]
    fn test_error_with_unicode() {
        let err = PruningError::MissingActivationStats {
            layer: "模型.层.0".to_string(),
        };
        assert!(
            err.to_string().contains("模型.层.0"),
            "ERR-11 FALSIFIED: Unicode layer names must be preserved"
        );
    }
}
