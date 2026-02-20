//! Validated Tensor Types - Compile-Time Contract Enforcement (PMAT-235)
//!
//! This module implements the Poka-Yoke (mistake-proofing) pattern from the
//! Toyota Production System. It makes invalid tensor states unrepresentable
//! at the type level.
//!
//! # Theoretical Foundation
//!
//! - Shingo, S. (1986). Zero Quality Control: Source Inspection and the
//!   Poka-Yoke System. Productivity Press.
//! - Brady, E. (2017). Type-Driven Development with Idris. Manning.
//! - Parsons, A. (2019). "Parse, Don't Validate"
//!
//! # Contract
//!
//! See `contracts/tensor-layout-v1.yaml` for the full specification.
//!
//! # Compiler Guarantee
//!
//! It is IMPOSSIBLE to use unvalidated tensor data because:
//! 1. Inner `data` field is private
//! 2. `new()` is the ONLY constructor (no Default, no unsafe backdoor)
//! 3. `new()` runs ALL validation checks from the contract
//! 4. Consumer types (AprTransformer) require Validated* types, not Vec<f32>

use std::fmt;
use std::marker::PhantomData;

// =============================================================================
// LAYOUT MARKER TYPES (PMAT-248)
// =============================================================================

/// Row-major layout marker (APR convention).
///
/// APR is exclusively row-major. This marker type encodes the layout
/// in the type system via `PhantomData<RowMajor>` on `ValidatedWeight`.
///
/// There is intentionally NO `ColumnMajor` marker type — making the
/// invalid state literally unrepresentable at compile time.
///
/// # Reference
///
/// - Strom & Yemini (1986): Typestate programming concept
/// - `contracts/tensor-layout-v1.yaml` §layout_enforcement
/// - `docs/specifications/compiler-enforced-model-types-model-oracle.md` §5.3
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMajor;

/// Contract validation error
#[derive(Debug, Clone)]
pub struct ContractValidationError {
    pub tensor_name: String,
    pub rule_id: String,
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

/// Tensor statistics for validation
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub len: usize,
    pub zero_count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub l2_norm: f32,
}

impl TensorStats {
    /// Compute statistics for tensor data
    #[must_use]
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
            l2_norm: sum_sq.sqrt() as f32,
        }
    }

    /// Percentage of zeros
    #[must_use]
    pub fn zero_pct(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        100.0 * self.zero_count as f32 / self.len as f32
    }
}

// =============================================================================
// VALIDATED EMBEDDING (F-DATA-QUALITY-001, F-DATA-QUALITY-004)
// =============================================================================

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

impl ValidatedEmbedding {
    /// Contract thresholds from tensor-layout-v1.yaml
    const MAX_ZERO_PCT: f32 = 50.0;
    const MIN_L2_NORM: f32 = 1e-6;
    const MIN_TOKEN_L2: f32 = 1e-6;
    const SPOT_CHECK_PCTS: [usize; 3] = [10, 50, 90];

    /// Construct a validated embedding tensor
    ///
    /// This is the ONLY way to create a ValidatedEmbedding. All contract
    /// rules are enforced here.
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` if any validation rule fails:
    /// - Wrong element count
    /// - >50% zeros (F-DATA-QUALITY-001)
    /// - Contains NaN (F-DATA-QUALITY-002)
    /// - Contains Inf (F-DATA-QUALITY-002)
    /// - L2 norm ~0 (F-DATA-QUALITY-003)
    /// - All values identical (F-DATA-QUALITY-003)
    /// - Spot check fails (F-DATA-QUALITY-004)
    pub fn new(
        data: Vec<f32>,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Self, ContractValidationError> {
        // Gate 1: Shape validation (structural)
        let expected_len = vocab_size * hidden_dim;
        if data.len() != expected_len {
            return Err(ContractValidationError {
                tensor_name: "embedding".to_string(),
                rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
                message: format!(
                    "Shape mismatch: got {} elements, expected {} ({}x{})",
                    data.len(),
                    expected_len,
                    vocab_size,
                    hidden_dim
                ),
            });
        }

        let stats = TensorStats::compute(&data);

        // Gates 2-6: Statistical quality validation
        Self::validate_stats(&stats)?;

        // Gate 7: Spot check validation at 10%/50%/90% of vocab
        Self::validate_spot_checks(&data, vocab_size, hidden_dim)?;

        Ok(Self {
            data,
            vocab_size,
            hidden_dim,
            stats,
        })
    }

    /// Validate tensor statistics (Gates 2-6).
    fn validate_stats(stats: &TensorStats) -> Result<(), ContractValidationError> {
        if stats.zero_pct() > Self::MAX_ZERO_PCT {
            return Err(ContractValidationError {
                tensor_name: "embedding".to_string(),
                rule_id: "F-DATA-QUALITY-001".to_string(),
                message: format!(
                    "DENSITY FAILURE: {:.1}% zeros (max {}%). Data likely loaded from wrong offset!",
                    stats.zero_pct(),
                    Self::MAX_ZERO_PCT
                ),
            });
        }
        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: "embedding".to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }
        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: "embedding".to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }
        if stats.l2_norm < Self::MIN_L2_NORM {
            return Err(ContractValidationError {
                tensor_name: "embedding".to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "L2 norm ~0: tensor is effectively empty".to_string(),
            });
        }
        if (stats.max - stats.min).abs() < 1e-10 {
            return Err(ContractValidationError {
                tensor_name: "embedding".to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "All values identical: tensor is constant".to_string(),
            });
        }
        Ok(())
    }

    /// Validate spot checks at key vocab positions (Gate 7).
    fn validate_spot_checks(
        data: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<(), ContractValidationError> {
        for pct in Self::SPOT_CHECK_PCTS {
            let token_id = vocab_size * pct / 100;
            let start = token_id * hidden_dim;
            let end = start + hidden_dim;
            if end <= data.len() {
                let token_l2: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
                if token_l2 < Self::MIN_TOKEN_L2 {
                    return Err(ContractValidationError {
                        tensor_name: "embedding".to_string(),
                        rule_id: "F-DATA-QUALITY-004".to_string(),
                        message: format!(
                            "Token {} ({}% of vocab) has L2={:.2e}: embedding data likely corrupted or offset",
                            token_id, pct, token_l2
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Access the validated data
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consume and return the inner data
    #[must_use]
    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }

    /// Get vocab size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get validation statistics
    #[must_use]
    pub fn stats(&self) -> &TensorStats {
        &self.stats
    }
}

// =============================================================================
// VALIDATED WEIGHT (F-DATA-QUALITY-001 through F-DATA-QUALITY-003)
// =============================================================================

/// Validated weight matrix - compile-time guarantee of data quality and layout
///
/// This type can ONLY be constructed via `new()`, which enforces:
/// - Correct element count (out_dim * in_dim)
/// - Density check (<80% zeros)
/// - No NaN or Inf values
/// - Non-degenerate distribution
///
/// ## Layout Safety (PMAT-248)
///
/// The `PhantomData<L>` marker encodes tensor layout in the type system:
/// - `ValidatedWeight<RowMajor>` = row-major layout (APR convention)
/// - There is NO `ColumnMajor` type — making column-major unrepresentable
///
/// The default type parameter `L = RowMajor` ensures backward compatibility:
/// `ValidatedWeight` (without explicit parameter) is `ValidatedWeight<RowMajor>`.
///
/// This has zero runtime cost — `PhantomData` is a zero-sized type.
#[derive(Debug, Clone)]
pub struct ValidatedWeight<L = RowMajor> {
    data: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
    name: String,
    stats: TensorStats,
    _layout: PhantomData<L>,
}

include!("validated_vector.rs");
include!("validated_tensors_part_03.rs");
