//! Compute Infrastructure Integration (trueno 0.8.8+)
//!
//! This module provides ML-specific wrappers around trueno's compute primitives
//! and simulation testing infrastructure.
//!
//! # Features
//!
//! - **Backend Selection**: Automatic CPU/GPU dispatch based on data size
//! - **Jidoka Guards**: NaN/Inf detection for training stability
//! - **Reproducibility**: Deterministic RNG for reproducible experiments
//! - **Stress Testing**: Performance validation infrastructure
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Built-in quality - stop on defect (NaN/Inf detection)
//! - **Poka-Yoke**: Mistake-proofing via type-safe backend selection
//! - **Heijunka**: Leveled testing across compute backends
//!
//! # Example
//!
//! ```rust
//! use aprender::compute::{TrainingGuard, select_backend, BackendCategory};
//!
//! // Auto-select backend based on data size (100k+ → GPU, 1k+ → SimdParallel)
//! let data_size = 50_000;
//! let gpu_available = false;
//! let backend = select_backend(data_size, gpu_available);
//! assert_eq!(backend, BackendCategory::SimdParallel);
//!
//! // Create training guard for NaN/Inf detection
//! let guard = TrainingGuard::new("gradient_update");
//!
//! // Check weights after update (detects NaN/Inf)
//! let weights = vec![1.0f32, 2.0, 3.0];
//! assert!(guard.check_weights(&weights).is_ok());
//! ```

use crate::error::{AprenderError, Result};

// Re-export trueno simulation types for advanced users
pub use trueno::simulation::{
    BackendCategory, BackendSelector, BackendTolerance, JidokaAction, JidokaCondition, JidokaError,
    JidokaGuard, StressTestConfig, StressThresholds,
};

// Re-export trueno core types
pub use trueno::{Backend, Matrix, Vector};

// =============================================================================
// BACKEND SELECTION (Poka-Yoke)
// =============================================================================

/// Default thresholds for backend selection (TRUENO-SPEC-012)
const GPU_THRESHOLD: usize = 100_000;
const PARALLEL_THRESHOLD: usize = 1_000;

/// Select optimal backend for ML operation based on data size
///
/// # Arguments
///
/// * `size` - Number of elements to process
/// * `gpu_available` - Whether GPU is available
///
/// # Returns
///
/// Recommended backend category
///
/// # Example
///
/// ```rust
/// use aprender::compute::{select_backend, BackendCategory};
///
/// let category = select_backend(50_000, false);
/// assert_eq!(category, BackendCategory::SimdParallel);
/// ```
#[must_use]
pub fn select_backend(size: usize, gpu_available: bool) -> BackendCategory {
    if size < PARALLEL_THRESHOLD {
        BackendCategory::SimdOnly
    } else if size < GPU_THRESHOLD {
        BackendCategory::SimdParallel
    } else if gpu_available {
        BackendCategory::Gpu
    } else {
        BackendCategory::SimdParallel // Graceful fallback
    }
}

/// Check if GPU would be beneficial for this operation size
#[must_use]
pub fn should_use_gpu(size: usize) -> bool {
    size >= GPU_THRESHOLD
}

/// Check if parallel execution would be beneficial
#[must_use]
pub fn should_use_parallel(size: usize) -> bool {
    size >= PARALLEL_THRESHOLD
}

// =============================================================================
// TRAINING GUARD (Jidoka)
// =============================================================================

/// ML-specific training guard for numerical stability
///
/// Wraps trueno's `JidokaGuard` with ML training-specific semantics.
/// Detects NaN and Inf values that indicate training instability.
///
/// # Example
///
/// ```rust
/// use aprender::compute::TrainingGuard;
///
/// let guard = TrainingGuard::new("epoch_1_gradients");
///
/// // After computing gradients
/// let gradients = vec![0.1, 0.2, 0.3];
/// guard.check_gradients(&gradients).expect("gradients should be finite");
///
/// // After weight update
/// let weights = vec![1.0, 2.0, 3.0];
/// guard.check_weights(&weights).expect("weights should be finite");
/// ```
#[derive(Debug, Clone)]
pub struct TrainingGuard {
    nan_guard: JidokaGuard,
    inf_guard: JidokaGuard,
    context: String,
}

impl TrainingGuard {
    /// Create a new training guard with context
    #[must_use]
    pub fn new(context: impl Into<String>) -> Self {
        let ctx = context.into();
        Self {
            nan_guard: JidokaGuard::nan_guard(format!("{ctx}:nan")),
            inf_guard: JidokaGuard::inf_guard(format!("{ctx}:inf")),
            context: ctx,
        }
    }

    /// Check gradients for NaN/Inf values
    ///
    /// # Errors
    ///
    /// Returns `AprenderError::ValidationError` if NaN or Inf detected
    pub fn check_gradients(&self, gradients: &[f32]) -> Result<()> {
        self.check_values(gradients, "gradients")
    }

    /// Check weights for NaN/Inf values
    ///
    /// # Errors
    ///
    /// Returns `AprenderError::ValidationError` if NaN or Inf detected
    pub fn check_weights(&self, weights: &[f32]) -> Result<()> {
        self.check_values(weights, "weights")
    }

    /// Check loss value for NaN/Inf
    ///
    /// # Errors
    ///
    /// Returns `AprenderError::ValidationError` if NaN or Inf detected
    pub fn check_loss(&self, loss: f32) -> Result<()> {
        if loss.is_nan() {
            return Err(AprenderError::ValidationError {
                message: format!("Jidoka: NaN loss detected at {}", self.context),
            });
        }
        if loss.is_infinite() {
            return Err(AprenderError::ValidationError {
                message: format!("Jidoka: Infinite loss detected at {}", self.context),
            });
        }
        Ok(())
    }

    /// Check any f32 slice for NaN/Inf values
    fn check_values(&self, values: &[f32], kind: &str) -> Result<()> {
        self.nan_guard
            .check_output(values)
            .map_err(|e| AprenderError::ValidationError {
                message: format!("Jidoka: NaN in {kind} at {}: {e}", self.context),
            })?;

        self.inf_guard
            .check_output(values)
            .map_err(|e| AprenderError::ValidationError {
                message: format!("Jidoka: Inf in {kind} at {}: {e}", self.context),
            })?;

        Ok(())
    }

    /// Check f64 values (converts to f32 for checking)
    ///
    /// # Errors
    ///
    /// Returns `AprenderError::ValidationError` if NaN or Inf detected
    pub fn check_f64(&self, values: &[f64], kind: &str) -> Result<()> {
        for (i, &v) in values.iter().enumerate() {
            if v.is_nan() {
                return Err(AprenderError::ValidationError {
                    message: format!("Jidoka: NaN in {kind}[{i}] at {}", self.context),
                });
            }
            if v.is_infinite() {
                return Err(AprenderError::ValidationError {
                    message: format!("Jidoka: Inf in {kind}[{i}] at {}", self.context),
                });
            }
        }
        Ok(())
    }
}

// =============================================================================
// DIVERGENCE GUARD (Cross-Backend Validation)
// =============================================================================

/// Guard for detecting cross-backend numerical divergence
///
/// Useful for validating that GPU and CPU produce consistent results.
#[derive(Debug, Clone)]
pub struct DivergenceGuard {
    guard: JidokaGuard,
}

impl DivergenceGuard {
    /// Create divergence guard with tolerance
    #[must_use]
    pub fn new(tolerance: f32, context: impl Into<String>) -> Self {
        Self {
            guard: JidokaGuard::divergence_guard(tolerance, context),
        }
    }

    /// Create guard with default ML tolerance (1e-5)
    #[must_use]
    pub fn default_tolerance(context: impl Into<String>) -> Self {
        Self::new(1e-5, context)
    }

    /// Compare two result sets for divergence
    ///
    /// # Errors
    ///
    /// Returns `AprenderError::ValidationError` if divergence exceeds tolerance
    pub fn check(&self, a: &[f32], b: &[f32]) -> Result<()> {
        self.guard
            .check_divergence(a, b)
            .map_err(|e| AprenderError::ValidationError {
                message: format!("Backend divergence: {e}"),
            })
    }
}

// =============================================================================
// REPRODUCIBILITY
// =============================================================================

/// Seed management for reproducible ML experiments
///
/// Provides deterministic seeding for all random operations in training.
#[derive(Debug, Clone, Copy)]
pub struct ExperimentSeed {
    /// Master seed for the experiment
    pub master: u64,
    /// Seed for data shuffling
    pub data_shuffle: u64,
    /// Seed for weight initialization
    pub weight_init: u64,
    /// Seed for dropout/regularization
    pub dropout: u64,
}

impl ExperimentSeed {
    /// Create experiment seeds from master seed
    ///
    /// Derives deterministic sub-seeds for different purposes
    #[must_use]
    pub fn from_master(master: u64) -> Self {
        Self {
            master,
            data_shuffle: master.wrapping_mul(6_364_136_223_846_793_005),
            weight_init: master.wrapping_mul(1_442_695_040_888_963_407),
            dropout: master.wrapping_mul(2_685_821_657_736_338_717),
        }
    }

    /// Create with explicit seeds
    #[must_use]
    pub const fn new(master: u64, data_shuffle: u64, weight_init: u64, dropout: u64) -> Self {
        Self {
            master,
            data_shuffle,
            weight_init,
            dropout,
        }
    }
}

impl Default for ExperimentSeed {
    fn default() -> Self {
        Self::from_master(42)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_backend_small() {
        let category = select_backend(100, false);
        assert_eq!(category, BackendCategory::SimdOnly);
    }

    #[test]
    fn test_select_backend_medium() {
        let category = select_backend(10_000, false);
        assert_eq!(category, BackendCategory::SimdParallel);
    }

    #[test]
    fn test_select_backend_large_no_gpu() {
        let category = select_backend(1_000_000, false);
        assert_eq!(category, BackendCategory::SimdParallel);
    }

    #[test]
    fn test_select_backend_large_with_gpu() {
        let category = select_backend(1_000_000, true);
        assert_eq!(category, BackendCategory::Gpu);
    }

    #[test]
    fn test_should_use_gpu() {
        assert!(!should_use_gpu(50_000));
        assert!(should_use_gpu(100_000));
        assert!(should_use_gpu(1_000_000));
    }

    #[test]
    fn test_should_use_parallel() {
        assert!(!should_use_parallel(500));
        assert!(should_use_parallel(1_000));
        assert!(should_use_parallel(10_000));
    }

    #[test]
    fn test_training_guard_clean_gradients() {
        let guard = TrainingGuard::new("test");
        let gradients = vec![0.1, 0.2, 0.3, -0.1];
        assert!(guard.check_gradients(&gradients).is_ok());
    }

    #[test]
    fn test_training_guard_nan_gradients() {
        let guard = TrainingGuard::new("test");
        let gradients = vec![0.1, f32::NAN, 0.3];
        assert!(guard.check_gradients(&gradients).is_err());
    }

    #[test]
    fn test_training_guard_inf_gradients() {
        let guard = TrainingGuard::new("test");
        let gradients = vec![0.1, f32::INFINITY, 0.3];
        assert!(guard.check_gradients(&gradients).is_err());
    }

    #[test]
    fn test_training_guard_loss_nan() {
        let guard = TrainingGuard::new("test");
        assert!(guard.check_loss(f32::NAN).is_err());
    }

    #[test]
    fn test_training_guard_loss_inf() {
        let guard = TrainingGuard::new("test");
        assert!(guard.check_loss(f32::INFINITY).is_err());
    }

    #[test]
    fn test_training_guard_loss_valid() {
        let guard = TrainingGuard::new("test");
        assert!(guard.check_loss(0.5).is_ok());
    }

    #[test]
    fn test_divergence_guard_within_tolerance() {
        let guard = DivergenceGuard::new(0.01, "test");
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.001, 2.002, 3.003];
        assert!(guard.check(&a, &b).is_ok());
    }

    #[test]
    fn test_divergence_guard_exceeds_tolerance() {
        let guard = DivergenceGuard::new(0.001, "test");
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.0, 3.0];
        assert!(guard.check(&a, &b).is_err());
    }

    #[test]
    fn test_experiment_seed_deterministic() {
        let seed1 = ExperimentSeed::from_master(42);
        let seed2 = ExperimentSeed::from_master(42);
        assert_eq!(seed1.master, seed2.master);
        assert_eq!(seed1.data_shuffle, seed2.data_shuffle);
        assert_eq!(seed1.weight_init, seed2.weight_init);
        assert_eq!(seed1.dropout, seed2.dropout);
    }

    #[test]
    fn test_experiment_seed_different_masters() {
        let seed1 = ExperimentSeed::from_master(42);
        let seed2 = ExperimentSeed::from_master(123);
        assert_ne!(seed1.data_shuffle, seed2.data_shuffle);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_training_guard_check_weights() {
        let guard = TrainingGuard::new("test_weights");
        let weights = vec![1.0, 2.0, 3.0, -1.0, 0.5];
        assert!(guard.check_weights(&weights).is_ok());
    }

    #[test]
    fn test_training_guard_check_weights_nan() {
        let guard = TrainingGuard::new("test_weights");
        let weights = vec![1.0, f32::NAN, 3.0];
        let result = guard.check_weights(&weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_training_guard_check_weights_inf() {
        let guard = TrainingGuard::new("test_weights");
        let weights = vec![1.0, f32::NEG_INFINITY, 3.0];
        let result = guard.check_weights(&weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_training_guard_check_f64_valid() {
        let guard = TrainingGuard::new("test_f64");
        let values = vec![1.0f64, 2.0, 3.0, -0.5];
        assert!(guard.check_f64(&values, "activations").is_ok());
    }

    #[test]
    fn test_training_guard_check_f64_nan() {
        let guard = TrainingGuard::new("test_f64");
        let values = vec![1.0f64, f64::NAN, 3.0];
        let result = guard.check_f64(&values, "activations");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_training_guard_check_f64_inf() {
        let guard = TrainingGuard::new("test_f64");
        let values = vec![1.0f64, f64::INFINITY, 3.0];
        let result = guard.check_f64(&values, "activations");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Inf"));
    }

    #[test]
    fn test_training_guard_check_f64_neg_inf() {
        let guard = TrainingGuard::new("test_f64");
        let values = vec![1.0f64, f64::NEG_INFINITY, 3.0];
        let result = guard.check_f64(&values, "outputs");
        assert!(result.is_err());
    }

    #[test]
    fn test_training_guard_debug() {
        let guard = TrainingGuard::new("debug_test");
        let debug_str = format!("{:?}", guard);
        assert!(debug_str.contains("TrainingGuard"));
    }

    #[test]
    fn test_training_guard_clone() {
        let guard = TrainingGuard::new("clone_test");
        let cloned = guard.clone();
        assert_eq!(cloned.context, guard.context);
    }

    #[test]
    fn test_training_guard_loss_neg_inf() {
        let guard = TrainingGuard::new("test");
        assert!(guard.check_loss(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn test_divergence_guard_default_tolerance() {
        let guard = DivergenceGuard::default_tolerance("default_test");
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.000001, 2.000001, 3.000001];
        assert!(guard.check(&a, &b).is_ok());
    }

    #[test]
    fn test_divergence_guard_debug() {
        let guard = DivergenceGuard::new(0.01, "debug_test");
        let debug_str = format!("{:?}", guard);
        assert!(debug_str.contains("DivergenceGuard"));
    }

    #[test]
    fn test_divergence_guard_clone() {
        let guard = DivergenceGuard::new(0.01, "clone_test");
        let _cloned = guard.clone();
        // Just verify clone works
    }

    #[test]
    fn test_experiment_seed_default() {
        let seed = ExperimentSeed::default();
        assert_eq!(seed.master, 42);
    }

    #[test]
    fn test_experiment_seed_new_const() {
        let seed = ExperimentSeed::new(100, 200, 300, 400);
        assert_eq!(seed.master, 100);
        assert_eq!(seed.data_shuffle, 200);
        assert_eq!(seed.weight_init, 300);
        assert_eq!(seed.dropout, 400);
    }

    #[test]
    fn test_experiment_seed_debug() {
        let seed = ExperimentSeed::from_master(42);
        let debug_str = format!("{:?}", seed);
        assert!(debug_str.contains("ExperimentSeed"));
        assert!(debug_str.contains("master"));
    }

    #[test]
    fn test_experiment_seed_clone() {
        let seed = ExperimentSeed::from_master(42);
        let cloned = seed; // Copy
        assert_eq!(cloned.master, seed.master);
    }

    #[test]
    fn test_experiment_seed_copy() {
        let seed = ExperimentSeed::from_master(99);
        let copied: ExperimentSeed = seed;
        assert_eq!(copied.master, 99);
        assert_eq!(seed.master, 99); // Original still accessible
    }

    #[test]
    fn test_select_backend_boundary_small() {
        // Just under parallel threshold
        let cat = select_backend(999, false);
        assert_eq!(cat, BackendCategory::SimdOnly);

        // At parallel threshold
        let cat = select_backend(1000, false);
        assert_eq!(cat, BackendCategory::SimdParallel);
    }

    #[test]
    fn test_select_backend_boundary_large() {
        // Just under GPU threshold
        let cat = select_backend(99_999, true);
        assert_eq!(cat, BackendCategory::SimdParallel);

        // At GPU threshold
        let cat = select_backend(100_000, true);
        assert_eq!(cat, BackendCategory::Gpu);
    }

    #[test]
    fn test_should_use_gpu_boundary() {
        assert!(!should_use_gpu(99_999));
        assert!(should_use_gpu(100_000));
    }

    #[test]
    fn test_should_use_parallel_boundary() {
        assert!(!should_use_parallel(999));
        assert!(should_use_parallel(1000));
    }

    #[test]
    fn test_training_guard_empty_gradients() {
        let guard = TrainingGuard::new("empty");
        let gradients: Vec<f32> = vec![];
        assert!(guard.check_gradients(&gradients).is_ok());
    }

    #[test]
    fn test_training_guard_empty_f64() {
        let guard = TrainingGuard::new("empty_f64");
        let values: Vec<f64> = vec![];
        assert!(guard.check_f64(&values, "empty").is_ok());
    }

    #[test]
    fn test_divergence_guard_empty_arrays() {
        let guard = DivergenceGuard::new(0.01, "empty");
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(guard.check(&a, &b).is_ok());
    }

    #[test]
    fn test_training_guard_error_messages() {
        let guard = TrainingGuard::new("error_context");

        let nan_result = guard.check_loss(f32::NAN);
        assert!(nan_result.is_err());
        let err_msg = nan_result.unwrap_err().to_string();
        assert!(err_msg.contains("error_context"));
        assert!(err_msg.contains("NaN"));
    }
}
