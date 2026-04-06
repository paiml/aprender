#![allow(missing_docs)]
//! Jidoka (Autonomation) - Stop on Defect
//!
//! Runtime guards that detect and halt computation on numerical errors.
//! Part of Toyota Production System integration in BLIS.
//!
//! # Philosophy
//!
//! Jidoka (自働化) is a Toyota Production System principle meaning "automation
//! with a human touch." When a defect is detected, the process stops immediately
//! rather than propagating bad data downstream.
//!
//! # Usage
//!
//! ```
//! use trueno::blis::jidoka::{JidokaGuard, JidokaError};
//!
//! let guard = JidokaGuard::strict();
//! guard.check_input(1.0, "matrix_a")?;
//! let computed = 1.0f32;
//! let expected = 1.0f32;
//! guard.validate(computed, expected)?;
//! # Ok::<(), JidokaError>(())
//! ```

/// Jidoka error types for runtime validation
#[derive(Debug, Clone, PartialEq)]
pub enum JidokaError {
    /// Numerical deviation beyond acceptable threshold
    NumericalDeviation { computed: f32, expected: f32, relative_error: f32 },
    /// NaN detected in computation
    NaNDetected { location: &'static str },
    /// Infinity detected in computation
    InfDetected { location: &'static str },
    /// Dimension mismatch
    DimensionMismatch { expected: (usize, usize, usize), actual: (usize, usize, usize) },
}

impl std::fmt::Display for JidokaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumericalDeviation { computed, expected, relative_error } => {
                write!(
                    f,
                    "Jidoka: numerical deviation - computed={}, expected={}, error={}",
                    computed, expected, relative_error
                )
            }
            Self::NaNDetected { location } => {
                write!(f, "Jidoka: NaN detected at {}", location)
            }
            Self::InfDetected { location } => {
                write!(f, "Jidoka: Inf detected at {}", location)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Jidoka: dimension mismatch - expected {:?}, got {:?}", expected, actual)
            }
        }
    }
}

impl std::error::Error for JidokaError {}

/// Jidoka guard for runtime validation
#[derive(Debug, Clone)]
pub struct JidokaGuard {
    /// Maximum allowed relative error
    pub epsilon: f32,
    /// Whether to check for NaN/Inf
    pub check_special: bool,
    /// Sample rate (check every N outputs)
    pub sample_rate: usize,
}

impl Default for JidokaGuard {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            check_special: true,
            sample_rate: 1000, // Check every 1000th output in release
        }
    }
}

impl JidokaGuard {
    /// Create a strict guard for testing (checks every output)
    pub fn strict() -> Self {
        Self { epsilon: 1e-6, check_special: true, sample_rate: 1 }
    }

    /// Validate a computed value against expected
    #[inline]
    pub fn validate(&self, computed: f32, expected: f32) -> Result<(), JidokaError> {
        if self.check_special {
            if computed.is_nan() {
                return Err(JidokaError::NaNDetected { location: "output" });
            }
            if computed.is_infinite() {
                return Err(JidokaError::InfDetected { location: "output" });
            }
        }

        let abs_diff = (computed - expected).abs();
        let max_abs = computed.abs().max(expected.abs()).max(1e-10);
        let relative_error = abs_diff / max_abs;

        if relative_error > self.epsilon {
            return Err(JidokaError::NumericalDeviation { computed, expected, relative_error });
        }

        Ok(())
    }

    /// Check input for NaN/Inf
    #[inline]
    pub fn check_input(&self, value: f32, location: &'static str) -> Result<(), JidokaError> {
        if !self.check_special {
            return Ok(());
        }
        if value.is_nan() {
            return Err(JidokaError::NaNDetected { location });
        }
        if value.is_infinite() {
            return Err(JidokaError::InfDetected { location });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jidoka_default() {
        let guard = JidokaGuard::default();
        assert!((guard.epsilon - 1e-5).abs() < 1e-10);
        assert!(guard.check_special);
        assert_eq!(guard.sample_rate, 1000);
    }

    #[test]
    fn test_jidoka_strict() {
        let guard = JidokaGuard::strict();
        assert!((guard.epsilon - 1e-6).abs() < 1e-10);
        assert!(guard.check_special);
        assert_eq!(guard.sample_rate, 1);
    }

    #[test]
    fn test_validate_pass() {
        let guard = JidokaGuard::default();
        assert!(guard.validate(1.0, 1.0).is_ok());
        assert!(guard.validate(1.0, 1.000001).is_ok());
    }

    #[test]
    fn test_validate_nan() {
        let guard = JidokaGuard::default();
        let result = guard.validate(f32::NAN, 1.0);
        assert!(matches!(result, Err(JidokaError::NaNDetected { .. })));
    }

    #[test]
    fn test_validate_inf() {
        let guard = JidokaGuard::default();
        let result = guard.validate(f32::INFINITY, 1.0);
        assert!(matches!(result, Err(JidokaError::InfDetected { .. })));
    }

    #[test]
    fn test_validate_deviation() {
        let guard = JidokaGuard::strict();
        let result = guard.validate(1.0, 2.0);
        assert!(matches!(result, Err(JidokaError::NumericalDeviation { .. })));
    }

    #[test]
    fn test_check_input_nan() {
        let guard = JidokaGuard::default();
        let result = guard.check_input(f32::NAN, "test");
        assert!(matches!(result, Err(JidokaError::NaNDetected { .. })));
    }

    #[test]
    fn test_check_input_inf() {
        let guard = JidokaGuard::default();
        let result = guard.check_input(f32::INFINITY, "test");
        assert!(matches!(result, Err(JidokaError::InfDetected { .. })));
    }

    #[test]
    fn test_error_display() {
        let err = JidokaError::NaNDetected { location: "test" };
        assert!(format!("{}", err).contains("NaN"));

        let err = JidokaError::InfDetected { location: "test" };
        assert!(format!("{}", err).contains("Inf"));

        let err =
            JidokaError::NumericalDeviation { computed: 1.0, expected: 2.0, relative_error: 0.5 };
        assert!(format!("{}", err).contains("deviation"));

        let err = JidokaError::DimensionMismatch { expected: (1, 2, 3), actual: (4, 5, 6) };
        assert!(format!("{}", err).contains("mismatch"));
    }
}
