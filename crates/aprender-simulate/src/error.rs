//! Error types for simular.
//!
//! Implements JPL Power of 10 Rule 7: Check all return values.
//! All functions return `Result<T, SimError>` instead of panicking.

use thiserror::Error;

/// Result type alias for simular operations.
pub type SimResult<T> = Result<T, SimError>;

/// Unified error type for all simular operations.
///
/// # Design
///
/// Following Toyota's Jidoka principle, errors are:
/// 1. Immediately detectable (type-safe)
/// 2. Self-documenting (descriptive variants)
/// 3. Actionable (contain recovery hints)
#[derive(Debug, Error)]
pub enum SimError {
    // ===== Jidoka Violations =====
    /// Numerical instability detected (NaN or Inf).
    #[error("Jidoka: non-finite value detected at {location}")]
    NonFiniteValue {
        /// Location where the non-finite value was detected.
        location: String,
    },

    /// Energy conservation violated beyond tolerance.
    #[error("Jidoka: energy drift {drift:.6e} exceeds tolerance {tolerance:.6e}")]
    EnergyDrift {
        /// Relative energy drift from initial state.
        drift: f64,
        /// Configured tolerance threshold.
        tolerance: f64,
    },

    /// Constraint violation detected.
    #[error(
        "Jidoka: constraint '{name}' violated by {violation:.6e} (tolerance: {tolerance:.6e})"
    )]
    ConstraintViolation {
        /// Name of the violated constraint.
        name: String,
        /// Amount of violation.
        violation: f64,
        /// Configured tolerance.
        tolerance: f64,
    },

    // ===== Configuration Errors =====
    /// Invalid configuration parameter.
    #[error("Configuration error: {message}")]
    Config {
        /// Description of the configuration error.
        message: String,
    },

    /// YAML parsing error.
    #[error("YAML parsing error: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    /// Validation error.
    #[error("Validation error: {0}")]
    Validation(#[from] validator::ValidationErrors),

    // ===== Replay Errors =====
    /// Checkpoint integrity violation.
    #[error("Checkpoint integrity violation: hash mismatch")]
    CheckpointIntegrity,

    /// Checkpoint not found.
    #[error("Checkpoint not found for time {0:?}")]
    CheckpointNotFound(crate::engine::SimTime),

    /// Journal read error.
    #[error("Journal error: {0}")]
    Journal(String),

    // ===== I/O Errors =====
    /// File I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    // ===== Domain Errors =====
    /// Physics engine error.
    #[error("Physics error: {0}")]
    Physics(String),

    /// Monte Carlo error.
    #[error("Monte Carlo error: {0}")]
    MonteCarlo(String),

    /// Optimization error.
    #[error("Optimization error: {0}")]
    Optimization(String),

    // ===== Falsification Errors =====
    /// Hypothesis falsified.
    #[error("Hypothesis falsified: {reason} (p-value: {p_value:.6})")]
    HypothesisFalsified {
        /// Reason for falsification.
        reason: String,
        /// Statistical p-value.
        p_value: f64,
    },
}

impl SimError {
    /// Create a configuration error with a message.
    #[must_use]
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a serialization error.
    #[must_use]
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization(message.into())
    }

    /// Create a journal error.
    #[must_use]
    pub fn journal(message: impl Into<String>) -> Self {
        Self::Journal(message.into())
    }

    /// Create an optimization error.
    #[must_use]
    pub fn optimization(message: impl Into<String>) -> Self {
        Self::Optimization(message.into())
    }

    /// Create a Jidoka violation error (requires immediate stop).
    #[must_use]
    pub fn jidoka(message: impl Into<String>) -> Self {
        Self::Config {
            message: format!("Jidoka violation: {}", message.into()),
        }
    }

    /// Create an I/O error with a message (wraps in `std::io::Error`).
    #[must_use]
    pub fn io(message: impl Into<String>) -> Self {
        Self::Io(std::io::Error::other(message.into()))
    }

    /// Check if this error is a Jidoka violation (requires immediate stop).
    #[must_use]
    pub const fn is_jidoka_violation(&self) -> bool {
        matches!(
            self,
            Self::NonFiniteValue { .. }
                | Self::EnergyDrift { .. }
                | Self::ConstraintViolation { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jidoka_violation_detection() {
        let non_finite = SimError::NonFiniteValue {
            location: "position.x".to_string(),
        };
        assert!(non_finite.is_jidoka_violation());

        let energy = SimError::EnergyDrift {
            drift: 0.001,
            tolerance: 0.0001,
        };
        assert!(energy.is_jidoka_violation());

        let constraint = SimError::ConstraintViolation {
            name: "mass_positive".to_string(),
            violation: -1.0,
            tolerance: 0.0,
        };
        assert!(constraint.is_jidoka_violation());

        let config = SimError::config("invalid");
        assert!(!config.is_jidoka_violation());
    }

    #[test]
    fn test_error_display() {
        let err = SimError::EnergyDrift {
            drift: 0.001_234_567,
            tolerance: 0.000_001,
        };
        let msg = err.to_string();
        assert!(msg.contains("energy drift"));
        assert!(msg.contains("1.234567e-3"));
    }

    #[test]
    fn test_error_config() {
        let err = SimError::config("invalid parameter");
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Configuration error"));
        assert!(msg.contains("invalid parameter"));
    }

    #[test]
    fn test_error_serialization() {
        let err = SimError::serialization("failed to serialize");
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Serialization error"));
        assert!(msg.contains("failed to serialize"));
    }

    #[test]
    fn test_error_journal() {
        let err = SimError::journal("corrupted journal");
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Journal error"));
        assert!(msg.contains("corrupted journal"));
    }

    #[test]
    fn test_error_optimization() {
        let err = SimError::optimization("convergence failed");
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Optimization error"));
        assert!(msg.contains("convergence failed"));
    }

    #[test]
    fn test_error_jidoka() {
        let err = SimError::jidoka("critical failure");
        // jidoka() creates a Config error wrapping the message
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Jidoka violation"));
        assert!(msg.contains("critical failure"));
    }

    #[test]
    fn test_error_io() {
        let err = SimError::io("file not found");
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn test_error_non_finite_display() {
        let err = SimError::NonFiniteValue {
            location: "velocity.y".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("non-finite value"));
        assert!(msg.contains("velocity.y"));
    }

    #[test]
    fn test_error_constraint_violation_display() {
        let err = SimError::ConstraintViolation {
            name: "mass_positive".to_string(),
            violation: -5.0,
            tolerance: 0.001,
        };
        let msg = err.to_string();
        assert!(msg.contains("constraint"));
        assert!(msg.contains("mass_positive"));
        assert!(msg.contains("violated"));
    }

    #[test]
    fn test_error_checkpoint_integrity() {
        let err = SimError::CheckpointIntegrity;
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Checkpoint integrity"));
    }

    #[test]
    fn test_error_checkpoint_not_found() {
        let err = SimError::CheckpointNotFound(crate::engine::SimTime::from_secs(10.0));
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Checkpoint not found"));
    }

    #[test]
    fn test_error_physics() {
        let err = SimError::Physics("invalid force".to_string());
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Physics error"));
    }

    #[test]
    fn test_error_monte_carlo() {
        let err = SimError::MonteCarlo("invalid sample".to_string());
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Monte Carlo error"));
    }

    #[test]
    fn test_error_hypothesis_falsified() {
        let err = SimError::HypothesisFalsified {
            reason: "energy not conserved".to_string(),
            p_value: 0.001,
        };
        assert!(!err.is_jidoka_violation());
        let msg = err.to_string();
        assert!(msg.contains("Hypothesis falsified"));
        assert!(msg.contains("energy not conserved"));
        assert!(msg.contains("0.001"));
    }

    #[test]
    fn test_error_debug() {
        let err = SimError::config("test");
        let debug = format!("{:?}", err);
        assert!(debug.contains("Config"));
    }
}
