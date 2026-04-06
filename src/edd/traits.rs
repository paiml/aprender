//! Core EDD traits for simulation compliance.
//!
//! This module defines the mandatory traits that every EDD-compliant
//! simulation must implement:
//!
//! - `Reproducible`: Deterministic seeding and RNG state management
//! - `YamlConfigurable`: Configuration from YAML experiment specs
//! - `EddSimulation`: Supertrait combining all EDD requirements
//!
//! # EDD-03: Deterministic Reproducibility
//!
//! > **Claim:** Identical seeds produce bitwise-identical results.
//! > **Rejection Criteria:** Any non-determinism in simulation output.
//!
//! # References
//!
//! - [9] Hill, D.R.C. (2023). Numerical Reproducibility of Parallel Stochastic Simulation
//! - [10] Hinsen, K. (2015). Reproducibility in Computational Neuroscience

use super::equation::GoverningEquation;
use super::experiment::ExperimentSpec;
use super::falsifiable::FalsifiableSimulation;
use super::model_card::EquationModelCard;
use std::collections::HashMap;

/// Error type for configuration operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    /// Error message
    pub message: String,
    /// Field that caused the error (if applicable)
    pub field: Option<String>,
    /// Underlying cause (if any)
    pub cause: Option<String>,
}

impl ConfigError {
    /// Create a new configuration error.
    #[must_use]
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            field: None,
            cause: None,
        }
    }

    /// Create an error for a specific field.
    #[must_use]
    pub fn field_error(field: &str, message: &str) -> Self {
        Self {
            message: message.to_string(),
            field: Some(field.to_string()),
            cause: None,
        }
    }

    /// Add a cause to the error.
    #[must_use]
    pub fn with_cause(mut self, cause: &str) -> Self {
        self.cause = Some(cause.to_string());
        self
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref field) = self.field {
            write!(f, "Config error in '{}': {}", field, self.message)?;
        } else {
            write!(f, "Config error: {}", self.message)?;
        }
        if let Some(ref cause) = self.cause {
            write!(f, " (caused by: {cause})")?;
        }
        Ok(())
    }
}

impl std::error::Error for ConfigError {}

/// Result of validating configuration against EMC.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// List of validation errors
    pub errors: Vec<String>,
    /// List of validation warnings
    pub warnings: Vec<String>,
    /// Parameters that were validated
    pub validated_params: Vec<String>,
}

impl ValidationResult {
    /// Create a successful validation result.
    #[must_use]
    pub fn success(validated_params: Vec<String>) -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            validated_params,
        }
    }

    /// Create a failed validation result.
    #[must_use]
    pub fn failure(errors: Vec<String>) -> Self {
        Self {
            valid: false,
            errors,
            warnings: Vec::new(),
            validated_params: Vec::new(),
        }
    }

    /// Add a warning to the result.
    #[must_use]
    pub fn with_warning(mut self, warning: &str) -> Self {
        self.warnings.push(warning.to_string());
        self
    }
}

/// Result of verifying implementation against EMC test cases.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether all verification tests passed
    pub passed: bool,
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Details of each test result
    pub test_results: Vec<TestResult>,
}

impl VerificationResult {
    /// Create a new verification result from test outcomes.
    #[must_use]
    pub fn from_tests(test_results: Vec<TestResult>) -> Self {
        let passed_tests = test_results.iter().filter(|t| t.passed).count();
        let failed_tests = test_results.len() - passed_tests;

        Self {
            passed: failed_tests == 0,
            total_tests: test_results.len(),
            passed_tests,
            failed_tests,
            test_results,
        }
    }

    /// Create an empty (all passed) result.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            passed: true,
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            test_results: Vec::new(),
        }
    }
}

/// Result of a single verification test.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Expected value
    pub expected: f64,
    /// Actual value
    pub actual: f64,
    /// Tolerance used
    pub tolerance: f64,
    /// Human-readable message
    pub message: String,
}

impl TestResult {
    /// Create a passed test result.
    #[must_use]
    pub fn pass(name: &str, expected: f64, actual: f64, tolerance: f64) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            expected,
            actual,
            tolerance,
            message: format!(
                "Test '{name}' PASSED: expected {expected:.6}, got {actual:.6} (tol: {tolerance:.6})"
            ),
        }
    }

    /// Create a failed test result.
    #[must_use]
    pub fn fail(name: &str, expected: f64, actual: f64, tolerance: f64) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            expected,
            actual,
            tolerance,
            message: format!(
                "Test '{name}' FAILED: expected {expected:.6}, got {actual:.6} (tol: {tolerance:.6})"
            ),
        }
    }
}

/// Reproducibility trait for deterministic simulation.
///
/// Implements EDD-03: Deterministic Reproducibility.
///
/// # Guarantee
///
/// For all runs r1, r2 with identical seeds:
/// ```text
/// S(I, σ) → R₁ ∧ S(I, σ) → R₂ ⟹ R₁ ≡ R₂
/// ```
///
/// # Example
///
/// ```ignore
/// impl Reproducible for MySimulation {
///     fn set_seed(&mut self, seed: u64) {
///         self.rng = StdRng::seed_from_u64(seed);
///     }
///
///     fn rng_state(&self) -> [u8; 32] {
///         // Serialize RNG state for checkpointing
///     }
///
///     fn restore_rng_state(&mut self, state: &[u8; 32]) {
///         // Restore RNG state from checkpoint
///     }
/// }
/// ```
pub trait Reproducible {
    /// Set the master seed for all RNG operations.
    fn set_seed(&mut self, seed: u64);

    /// Get current RNG state for checkpointing.
    ///
    /// Returns a 32-byte array representing the complete RNG state
    /// that can be used to restore the simulation to this exact point.
    fn rng_state(&self) -> [u8; 32];

    /// Restore RNG state from a checkpoint.
    ///
    /// # Arguments
    ///
    /// * `state` - A 32-byte array previously returned by `rng_state()`
    fn restore_rng_state(&mut self, state: &[u8; 32]);

    /// Get the current seed value.
    fn current_seed(&self) -> u64;
}

/// YAML configuration trait for EDD simulations.
///
/// Allows simulations to be configured from declarative YAML
/// experiment specifications without custom code.
///
/// # Example
///
/// ```ignore
/// impl YamlConfigurable for HarmonicOscillator {
///     fn from_yaml(spec: &ExperimentSpec) -> Result<Self, ConfigError> {
///         let omega = spec.parameter("omega")
///             .ok_or_else(|| ConfigError::field_error("omega", "required"))?;
///         Ok(Self::new(omega))
///     }
///
///     fn validate_against_emc(&self, emc: &EquationModelCard) -> ValidationResult {
///         // Check parameters are within EMC domain of validity
///     }
/// }
/// ```
pub trait YamlConfigurable: Sized {
    /// Create simulation from YAML experiment specification.
    ///
    /// # Arguments
    ///
    /// * `spec` - The experiment specification from YAML
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the specification is invalid or
    /// missing required parameters.
    fn from_yaml(spec: &ExperimentSpec) -> Result<Self, ConfigError>;

    /// Validate configuration against EMC domain of validity.
    ///
    /// Checks that all parameters fall within the valid ranges
    /// specified in the Equation Model Card.
    fn validate_against_emc(&self, emc: &EquationModelCard) -> ValidationResult;

    /// Extract parameters as a hashmap for evaluation.
    fn parameters(&self) -> HashMap<String, f64>;
}

/// Core EDD trait bundle.
///
/// Every simulation in simular MUST implement this trait, which combines:
/// - `GoverningEquation`: Mathematical foundation
/// - `FalsifiableSimulation`: Active falsification search
/// - `Reproducible`: Deterministic seeding
/// - `YamlConfigurable`: Declarative configuration
///
/// # EDD Compliance
///
/// Implementing this trait ensures the simulation complies with all
/// four pillars of EDD:
///
/// 1. **Prove It**: Via `GoverningEquation` (EMC reference)
/// 2. **Fail It**: Via `FalsifiableSimulation` (falsification criteria)
/// 3. **Seed It**: Via `Reproducible` (deterministic RNG)
/// 4. **Falsify It**: Via `verify_against_emc()` (active testing)
///
/// # Example
///
/// ```ignore
/// pub struct HarmonicOscillator {
///     omega: f64,
///     emc: EquationModelCard,
///     seed: u64,
/// }
///
/// impl EddSimulation for HarmonicOscillator {
///     fn emc(&self) -> &EquationModelCard {
///         &self.emc
///     }
///
///     fn verify_against_emc(&self) -> VerificationResult {
///         // Run EMC verification tests
///     }
/// }
/// ```
pub trait EddSimulation:
    GoverningEquation + FalsifiableSimulation + Reproducible + YamlConfigurable
{
    /// Get the associated Equation Model Card.
    fn emc(&self) -> &EquationModelCard;

    /// Verify implementation against EMC test cases.
    ///
    /// Runs all verification tests defined in the EMC and returns
    /// a detailed result showing which tests passed or failed.
    fn verify_against_emc(&self) -> VerificationResult;

    /// Get simulation name from EMC.
    fn simulation_name(&self) -> &str {
        &self.emc().name
    }

    /// Get simulation version from EMC.
    fn simulation_version(&self) -> &str {
        &self.emc().version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error_new() {
        let err = ConfigError::new("test error");
        assert_eq!(err.message, "test error");
        assert!(err.field.is_none());
        assert!(err.cause.is_none());
    }

    #[test]
    fn test_config_error_field() {
        let err = ConfigError::field_error("omega", "must be positive");
        assert_eq!(err.field, Some("omega".to_string()));
        assert!(err.message.contains("positive"));
    }

    #[test]
    fn test_config_error_with_cause() {
        let err = ConfigError::new("parse failed").with_cause("invalid number");
        assert!(err.cause.is_some());
        assert!(err.cause.unwrap().contains("invalid"));
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::field_error("seed", "required").with_cause("missing key");
        let display = format!("{err}");
        assert!(display.contains("seed"));
        assert!(display.contains("required"));
        assert!(display.contains("missing key"));
    }

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success(vec!["omega".to_string(), "amplitude".to_string()]);
        assert!(result.valid);
        assert!(result.errors.is_empty());
        assert_eq!(result.validated_params.len(), 2);
    }

    #[test]
    fn test_validation_result_failure() {
        let result = ValidationResult::failure(vec!["omega out of range".to_string()]);
        assert!(!result.valid);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_validation_result_with_warning() {
        let result = ValidationResult::success(vec![]).with_warning("deprecated parameter");
        assert!(result.valid);
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_verification_result_from_tests() {
        let tests = vec![
            TestResult::pass("test1", 10.0, 10.0, 0.01),
            TestResult::fail("test2", 5.0, 6.0, 0.01),
        ];
        let result = VerificationResult::from_tests(tests);
        assert!(!result.passed);
        assert_eq!(result.total_tests, 2);
        assert_eq!(result.passed_tests, 1);
        assert_eq!(result.failed_tests, 1);
    }

    #[test]
    fn test_verification_result_empty() {
        let result = VerificationResult::empty();
        assert!(result.passed);
        assert_eq!(result.total_tests, 0);
    }

    #[test]
    fn test_test_result_pass() {
        let result = TestResult::pass("energy_conservation", 100.0, 100.001, 0.01);
        assert!(result.passed);
        assert!(result.message.contains("PASSED"));
    }

    #[test]
    fn test_test_result_fail() {
        let result = TestResult::fail("energy_conservation", 100.0, 110.0, 0.01);
        assert!(!result.passed);
        assert!(result.message.contains("FAILED"));
    }
}
