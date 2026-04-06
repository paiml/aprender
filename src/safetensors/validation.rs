//! Tensor Validation Contract (PMAT-234, PMAT-235)
//!
//! Makes it IMPOSSIBLE to load garbage data from GGUF, APR, or SafeTensors.
//!
//! ## Design Principle
//!
//! Every tensor load MUST pass semantic validation before use.
//! A tensor that parses correctly but contains garbage MUST be rejected.
//!
//! ## Compile-Time Enforcement (PMAT-235)
//!
//! This module implements the Poka-Yoke (mistake-proofing) pattern from the
//! Toyota Production System. The newtype pattern makes invalid tensor states
//! unrepresentable at the type level.
//!
//! ## Theoretical Foundation
//!
//! - Shingo, S. (1986). Zero Quality Control: Source Inspection and the
//!   Poka-Yoke System. Productivity Press.
//! - Brady, E. (2017). Type-Driven Development with Idris. Manning.
//! - Parsons, A. (2019). "Parse, Don't Validate"
//!
//! ## Validation Gates
//!
//! 1. **Density Gate**: Rejects tensors that are mostly zeros (dead weights)
//! 2. **Distribution Gate**: Rejects tensors with abnormal value distributions
//! 3. **Shape Gate**: Rejects tensors with impossible shapes for their role
//! 4. **NaN/Inf Gate**: Rejects tensors containing NaN or Inf values
//!
//! ## Contract
//!
//! See `aprender/contracts/tensor-layout-v1.yaml` for the full specification.

use crate::error::{RealizarError, Result};
use std::fmt;

/// Tensor validation statistics
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Total number of elements
    pub len: usize,
    /// Count of zero values (|x| < 1e-10)
    pub zero_count: usize,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
    /// Minimum value (excluding NaN/Inf)
    pub min: f32,
    /// Maximum value (excluding NaN/Inf)
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// L2 norm (Frobenius norm)
    pub l2_norm: f32,
}

impl TensorStats {
    /// Compute statistics for a tensor
    pub fn compute(data: &[f32]) -> Self {
        let len = data.len();
        if len == 0 {
            return Self {
                len: 0,
                zero_count: 0,
                nan_count: 0,
                inf_count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                l2_norm: 0.0,
            };
        }

        let mut zero_count = 0;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                if v.abs() < 1e-10 {
                    zero_count += 1;
                }
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                sum += v as f64;
                sum_sq += (v as f64) * (v as f64);
            }
        }

        Self {
            len,
            zero_count,
            nan_count,
            inf_count,
            min: if min == f32::INFINITY { 0.0 } else { min },
            max: if max == f32::NEG_INFINITY { 0.0 } else { max },
            mean: (sum / len as f64) as f32,
            l2_norm: (sum_sq.sqrt()) as f32,
        }
    }

    /// Percentage of zeros
    pub fn zero_pct(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        100.0 * self.zero_count as f32 / self.len as f32
    }
}

/// Validation result with detailed diagnostics
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether validation passed all gates
    pub passed: bool,
    /// Computed tensor statistics
    pub stats: TensorStats,
    /// List of failure messages (empty if passed)
    pub failures: Vec<String>,
}

/// Validate an embedding tensor
///
/// Embeddings MUST have:
/// - Less than 50% zeros (dead embeddings = broken model)
/// - No NaN or Inf values
/// - Non-zero L2 norm
/// - Reasonable value range (not all identical)
pub fn validate_embedding(
    name: &str,
    data: &[f32],
    vocab_size: usize,
    hidden_dim: usize,
) -> ValidationResult {
    let stats = TensorStats::compute(data);
    let mut failures = Vec::new();

    // Gate 1: Shape validation
    let expected_len = vocab_size * hidden_dim;
    if data.len() != expected_len {
        failures.push(format!(
            "Shape mismatch: got {} elements, expected {} ({}x{})",
            data.len(),
            expected_len,
            vocab_size,
            hidden_dim
        ));
    }

    // Gate 2: Density validation (CRITICAL - detects incorrect data offsets)
    let zero_pct = stats.zero_pct();
    if zero_pct > 50.0 {
        failures.push(format!(
            "DENSITY FAILURE: {:.1}% zeros (max 50%). Data likely loaded from wrong offset!",
            zero_pct
        ));
    }

    // Gate 3: NaN/Inf validation
    if stats.nan_count > 0 {
        failures.push(format!("Contains {} NaN values", stats.nan_count));
    }
    if stats.inf_count > 0 {
        failures.push(format!("Contains {} Inf values", stats.inf_count));
    }

    // Gate 4: Distribution validation
    if stats.l2_norm < 1e-6 {
        failures.push("L2 norm ~0: tensor is effectively empty".to_string());
    }
    if (stats.max - stats.min).abs() < 1e-10 {
        failures.push("All values identical: tensor is constant".to_string());
    }

    // Gate 5: Sample non-zero tokens (spot check)
    // Check tokens at 10%, 50%, 90% of vocab to ensure data is distributed
    for pct in [10, 50, 90] {
        let token_id = vocab_size * pct / 100;
        let start = token_id * hidden_dim;
        let end = start + hidden_dim;
        if end <= data.len() {
            let token_l2: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            if token_l2 < 1e-6 {
                failures.push(format!(
                    "Token {} ({}% of vocab) has L2=0: embedding data likely corrupted",
                    token_id, pct
                ));
            }
        }
    }

    let passed = failures.is_empty();
    if !passed {
        eprintln!("[VALIDATION FAILED] {}: {:?}", name, failures);
    }

    ValidationResult {
        passed,
        stats,
        failures,
    }
}

/// Validate a weight matrix (linear layer)
pub fn validate_weight(
    name: &str,
    data: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> ValidationResult {
    let stats = TensorStats::compute(data);
    let mut failures = Vec::new();

    // Gate 1: Shape
    let expected_len = out_dim * in_dim;
    if data.len() != expected_len {
        failures.push(format!(
            "Shape mismatch: got {} elements, expected {} ({}x{})",
            data.len(),
            expected_len,
            out_dim,
            in_dim
        ));
    }

    // Gate 2: Density (weights should be mostly non-zero)
    let zero_pct = stats.zero_pct();
    if zero_pct > 80.0 {
        failures.push(format!("DENSITY FAILURE: {:.1}% zeros (max 80%)", zero_pct));
    }

    // Gate 3: NaN/Inf
    if stats.nan_count > 0 {
        failures.push(format!("Contains {} NaN values", stats.nan_count));
    }
    if stats.inf_count > 0 {
        failures.push(format!("Contains {} Inf values", stats.inf_count));
    }

    // Gate 4: Distribution
    if stats.l2_norm < 1e-6 {
        failures.push("L2 norm ~0".to_string());
    }

    let passed = failures.is_empty();
    if !passed {
        eprintln!("[VALIDATION FAILED] {}: {:?}", name, failures);
    }

    ValidationResult {
        passed,
        stats,
        failures,
    }
}

/// Validate a 1D tensor (bias, norm weight)
pub fn validate_vector(_name: &str, data: &[f32], expected_len: usize) -> ValidationResult {
    let stats = TensorStats::compute(data);
    let mut failures = Vec::new();

    if data.len() != expected_len {
        failures.push(format!(
            "Length mismatch: got {}, expected {}",
            data.len(),
            expected_len
        ));
    }

    if stats.nan_count > 0 {
        failures.push(format!("Contains {} NaN values", stats.nan_count));
    }
    if stats.inf_count > 0 {
        failures.push(format!("Contains {} Inf values", stats.inf_count));
    }

    let passed = failures.is_empty();
    ValidationResult {
        passed,
        stats,
        failures,
    }
}

/// Enforce validation - returns error if validation fails
pub fn enforce_embedding_validation(
    name: &str,
    data: &[f32],
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<()> {
    let result = validate_embedding(name, data, vocab_size, hidden_dim);
    if !result.passed {
        return Err(RealizarError::FormatError {
            reason: format!(
                "Tensor '{}' failed validation: {}",
                name,
                result.failures.join("; ")
            ),
        });
    }
    Ok(())
}

/// Enforce weight validation
pub fn enforce_weight_validation(
    name: &str,
    data: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<()> {
    let result = validate_weight(name, data, out_dim, in_dim);
    if !result.passed {
        return Err(RealizarError::FormatError {
            reason: format!(
                "Tensor '{}' failed validation: {}",
                name,
                result.failures.join("; ")
            ),
        });
    }
    Ok(())
}

// =============================================================================
// VALIDATED NEWTYPES - Compile-Time Contract Enforcement (PMAT-235)
// =============================================================================
//
// These types implement the Poka-Yoke pattern: the inner data is private,
// so the ONLY way to construct these types is via the validated constructor.
// This makes it IMPOSSIBLE to use unvalidated tensor data at compile time.
//
// Citation: Shingo, S. (1986). Zero Quality Control: Source Inspection and
//           the Poka-Yoke System. Productivity Press.
// =============================================================================

/// Contract validation error (mirrors aprender::format::ContractValidationError)
#[derive(Debug, Clone)]
pub struct ContractValidationError {
    /// Name of the tensor that failed validation
    pub tensor_name: String,
    /// Contract rule ID that was violated (e.g., "F-DATA-QUALITY-001")
    pub rule_id: String,
    /// Human-readable error message
    pub message: String,
}

impl fmt::Display for ContractValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] Tensor '{}': {}",
            self.rule_id, self.tensor_name, self.message
        )
    }
}

impl std::error::Error for ContractValidationError {}

impl From<ContractValidationError> for RealizarError {
    fn from(e: ContractValidationError) -> Self {
        RealizarError::FormatError {
            reason: e.to_string(),
        }
    }
}

/// Validated embedding tensor - compile-time guarantee of data quality
///
/// This type can ONLY be constructed via `new()`, which enforces:
/// - Correct element count (vocab_size * hidden_dim)
/// - Density check (<50% zeros) - catches PMAT-234 bug
/// - No NaN or Inf values
/// - Non-degenerate distribution (L2 > 1e-6, values vary)
/// - Spot check at 10%/50%/90% of vocab
///
/// # Poka-Yoke Guarantee
///
/// The inner `data` field is private. There is no way to construct this type
/// without passing validation. This makes the PMAT-234 bug (94.5% zeros)
/// impossible at compile time.
#[derive(Debug, Clone)]
pub struct ValidatedEmbedding {
    // PRIVATE - cannot be accessed without going through new()
    data: Vec<f32>,
    vocab_size: usize,
    hidden_dim: usize,
    stats: TensorStats,
}

include!("validation_embedding.rs");
include!("inner.rs");
include!("falsification_tests.rs");
