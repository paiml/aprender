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
        let name = "embedding";

        // Gate 1: Shape validation (structural)
        let expected_len = vocab_size * hidden_dim;
        if data.len() != expected_len {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
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

        // Gate 2: Density validation (F-DATA-QUALITY-001)
        // Rejects tensors with >80% zeros (corrupt offset or uninitialized data)
        if stats.zero_pct() > Self::MAX_ZERO_PCT {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-001".to_string(),
                message: format!(
                    "DENSITY FAILURE: {:.1}% zeros (max {}%). Data likely loaded from wrong offset!",
                    stats.zero_pct(),
                    Self::MAX_ZERO_PCT
                ),
            });
        }

        // Gate 3: NaN validation (F-DATA-QUALITY-002)
        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

        // Gate 4: Inf validation (F-DATA-QUALITY-002)
        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }

        // Gate 5: L2 norm validation (F-DATA-QUALITY-003)
        if stats.l2_norm < Self::MIN_L2_NORM {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "L2 norm ~0: tensor is effectively empty".to_string(),
            });
        }

        // Gate 6: Variation validation (F-DATA-QUALITY-003)
        if (stats.max - stats.min).abs() < 1e-10 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "All values identical: tensor is constant".to_string(),
            });
        }

        // Gate 7: Spot check validation (F-DATA-QUALITY-004)
        // Check tokens at 10%, 50%, 90% of vocab to catch offset bugs
        for pct in Self::SPOT_CHECK_PCTS {
            let token_id = vocab_size * pct / 100;
            let start = token_id * hidden_dim;
            let end = start + hidden_dim;
            if end <= data.len() {
                let token_l2: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
                if token_l2 < Self::MIN_TOKEN_L2 {
                    return Err(ContractValidationError {
                        tensor_name: name.to_string(),
                        rule_id: "F-DATA-QUALITY-004".to_string(),
                        message: format!(
                            "Token {} ({}% of vocab) has L2={:.2e}: embedding data likely corrupted or offset",
                            token_id, pct, token_l2
                        ),
                    });
                }
            }
        }

        Ok(Self {
            data,
            vocab_size,
            hidden_dim,
            stats,
        })
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

impl ValidatedWeight<RowMajor> {
    const MAX_ZERO_PCT: f32 = 80.0;
    const MIN_L2_NORM: f32 = 1e-6;

    /// Construct a validated row-major weight matrix.
    ///
    /// This is the ONLY constructor. There is no way to create a
    /// `ValidatedWeight<ColumnMajor>` because `ColumnMajor` does not exist.
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` if validation fails.
    pub fn new(
        data: Vec<f32>,
        out_dim: usize,
        in_dim: usize,
        name: &str,
    ) -> Result<Self, ContractValidationError> {
        // Gate 1: Shape validation
        let expected_len = out_dim * in_dim;
        if data.len() != expected_len {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
                message: format!(
                    "Shape mismatch: got {} elements, expected {} ({}x{})",
                    data.len(),
                    expected_len,
                    out_dim,
                    in_dim
                ),
            });
        }

        let stats = TensorStats::compute(&data);

        // Gate 2: Density validation
        if stats.zero_pct() > Self::MAX_ZERO_PCT {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-001".to_string(),
                message: format!(
                    "DENSITY FAILURE: {:.1}% zeros (max {}%)",
                    stats.zero_pct(),
                    Self::MAX_ZERO_PCT
                ),
            });
        }

        // Gate 3: NaN validation
        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

        // Gate 4: Inf validation
        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }

        // Gate 5: L2 norm validation
        if stats.l2_norm < Self::MIN_L2_NORM {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-003".to_string(),
                message: "L2 norm ~0: tensor is effectively empty".to_string(),
            });
        }

        Ok(Self {
            data,
            out_dim,
            in_dim,
            name: name.to_string(),
            stats,
            _layout: PhantomData,
        })
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

    /// Get output dimension
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Get input dimension
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Get tensor name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get validation statistics
    #[must_use]
    pub fn stats(&self) -> &TensorStats {
        &self.stats
    }
}

// =============================================================================
// VALIDATED VECTOR (for 1D tensors like layer norms)
// =============================================================================

/// Validated 1D tensor (bias, norm weights)
#[derive(Debug, Clone)]
pub struct ValidatedVector {
    data: Vec<f32>,
    name: String,
    stats: TensorStats,
}

impl ValidatedVector {
    /// Construct a validated vector
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` if validation fails.
    pub fn new(
        data: Vec<f32>,
        expected_len: usize,
        name: &str,
    ) -> Result<Self, ContractValidationError> {
        // Gate 1: Length validation
        if data.len() != expected_len {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
                message: format!(
                    "Length mismatch: got {}, expected {}",
                    data.len(),
                    expected_len
                ),
            });
        }

        let stats = TensorStats::compute(&data);

        // Gate 2: NaN validation
        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

        // Gate 3: Inf validation
        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }

        Ok(Self {
            data,
            name: name.to_string(),
            stats,
        })
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

    /// Get tensor name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get validation statistics
    #[must_use]
    pub fn stats(&self) -> &TensorStats {
        &self.stats
    }
}

// =============================================================================
// POPPERIAN FALSIFICATION TESTS
// =============================================================================
//
// Per Popper (1959), these tests attempt to DISPROVE the contract works.
// If any test passes when it should fail, the contract has a logic error.

#[cfg(test)]
mod tests {
    use super::*;

    // FALSIFY-001: Embedding density check
    #[test]
    fn falsify_001_embedding_rejects_all_zeros() {
        let bad_data = vec![0.0f32; 100 * 64]; // 100% zeros
        let result = ValidatedEmbedding::new(bad_data, 100, 64);
        assert!(result.is_err(), "Should reject 100% zeros");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("DENSITY"),
            "Error should mention density: {}",
            err.message
        );
    }

    #[test]
    fn falsify_001_embedding_rejects_mostly_zeros() {
        // Simulate PMAT-234: 94.5% zeros
        let vocab_size = 1000;
        let hidden_dim = 64;
        let mut data = vec![0.0f32; vocab_size * hidden_dim];
        // Only last 5.5% non-zero
        for i in (945 * hidden_dim)..(vocab_size * hidden_dim) {
            data[i] = 0.1;
        }
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject 94.5% zeros");
    }

    #[test]
    fn falsify_001_embedding_accepts_good_data() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(
            result.is_ok(),
            "Should accept good data: {:?}",
            result.err()
        );
    }

    // FALSIFY-002: Inf rejection (Gate 4 — F-DATA-QUALITY-002)
    // §18.3: This test was missing, causing a gap in FALSIFY numbering.
    #[test]
    fn falsify_002_embedding_rejects_inf() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[7] = f32::INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject Inf");
        assert!(result.unwrap_err().message.contains("Inf"));
    }

    #[test]
    fn falsify_002_embedding_rejects_neg_inf() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[3] = f32::NEG_INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject -Inf");
        assert!(result.unwrap_err().message.contains("Inf"));
    }

    #[test]
    fn falsify_002_weight_rejects_inf() {
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        data[50] = f32::INFINITY;
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject Inf in weight");
    }

    // FALSIFY-003: NaN rejection
    #[test]
    fn falsify_003_embedding_rejects_nan() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        data[5] = f32::NAN;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject NaN");
        assert!(result.unwrap_err().message.contains("NaN"));
    }

    #[test]
    fn falsify_003_weight_rejects_nan() {
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        data[50] = f32::NAN;
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject NaN");
    }

    // FALSIFY-004: Spot check catches offset bugs
    #[test]
    fn falsify_004_spot_check_catches_offset_bug() {
        // Token at 10% of vocab has zero embedding
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at 10% (token 10)
        let token_10_start = 10 * hidden_dim;
        for i in token_10_start..(token_10_start + hidden_dim) {
            data[i] = 0.0;
        }

        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should catch zero token at 10%");
        assert!(result.unwrap_err().rule_id == "F-DATA-QUALITY-004");
    }

    // FALSIFY-005: Shape validation
    #[test]
    fn falsify_005_rejects_wrong_shape() {
        let data = vec![0.1f32; 1000];
        let result = ValidatedEmbedding::new(data, 100, 64); // expects 6400
        assert!(result.is_err(), "Should reject wrong shape");
    }

    // Weight-specific tests
    #[test]
    fn weight_rejects_all_zeros() {
        let data = vec![0.0f32; 100];
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_err());
    }

    #[test]
    fn weight_accepts_good_data() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let result = ValidatedWeight::new(data, 10, 10, "test");
        assert!(result.is_ok());
    }

    // Vector tests
    #[test]
    fn vector_rejects_wrong_length() {
        let data = vec![0.1f32; 50];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_err());
    }

    #[test]
    fn vector_accepts_good_data() {
        let data = vec![1.0f32; 100];
        let result = ValidatedVector::new(data, 100, "test");
        assert!(result.is_ok());
    }

    // PMAT-248: PhantomData<Layout> enforcement tests

    #[test]
    fn pmat_248_validated_weight_is_row_major_by_default() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let weight: ValidatedWeight = ValidatedWeight::new(data, 10, 10, "test").unwrap();
        // This compiles because ValidatedWeight == ValidatedWeight<RowMajor>
        let _explicit: ValidatedWeight<RowMajor> = weight;
    }

    #[test]
    fn pmat_248_row_major_marker_is_zero_sized() {
        assert_eq!(std::mem::size_of::<RowMajor>(), 0);
        assert_eq!(
            std::mem::size_of::<PhantomData<RowMajor>>(),
            0,
            "PhantomData<RowMajor> must be zero-sized"
        );
    }

    #[test]
    fn pmat_248_phantom_data_does_not_increase_struct_size() {
        // ValidatedWeight with PhantomData should have same layout as without
        // (PhantomData is ZST, compiler optimizes it away)
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let weight = ValidatedWeight::new(data, 10, 10, "test").unwrap();
        // Ensure all fields are accessible (compile-time check)
        let _ = weight.data();
        let _ = weight.out_dim();
        let _ = weight.in_dim();
        let _ = weight.name();
        let _ = weight.stats();
    }

    // ================================================================
    // ValidatedEmbedding - Inf rejection
    // ================================================================

    #[test]
    fn embedding_rejects_inf_values() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        data[7] = f32::INFINITY;
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject Inf values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Inf"),
            "Error should mention Inf: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedEmbedding - L2 norm ~0
    // ================================================================

    #[test]
    fn embedding_rejects_near_zero_l2_norm() {
        let vocab_size = 10;
        let hidden_dim = 8;
        // Values above the zero threshold (1e-10) but producing negligible L2 norm (< 1e-6).
        // With 80 elements at 1e-8 each: L2 = sqrt(80 * (1e-8)^2) = sqrt(80)*1e-8 ~ 8.9e-8.
        // Also make values vary slightly to pass the constant check.
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| 1e-8 + (i as f32) * 1e-12)
            .collect();
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject near-zero L2 norm");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("L2 norm"),
            "Error should mention L2 norm: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedEmbedding - Constant values
    // ================================================================

    #[test]
    fn embedding_rejects_constant_values() {
        let vocab_size = 10;
        let hidden_dim = 8;
        let data = vec![0.5f32; vocab_size * hidden_dim];
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(), "Should reject constant values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("constant"),
            "Error should mention constant: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedEmbedding accessors
    // ================================================================

    #[test]
    fn embedding_accessors_return_correct_values() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let original_data = data.clone();
        let emb = ValidatedEmbedding::new(data, vocab_size, hidden_dim)
            .expect("good data should be accepted");

        assert_eq!(emb.vocab_size(), vocab_size);
        assert_eq!(emb.hidden_dim(), hidden_dim);
        assert_eq!(emb.data().len(), vocab_size * hidden_dim);
        assert_eq!(emb.data(), original_data.as_slice());

        let stats = emb.stats();
        assert_eq!(stats.len, vocab_size * hidden_dim);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);

        // into_inner consumes
        let inner = emb.into_inner();
        assert_eq!(inner.len(), vocab_size * hidden_dim);
        assert_eq!(inner, original_data);
    }

    // ================================================================
    // ValidatedWeight - Inf rejection
    // ================================================================

    #[test]
    fn weight_rejects_inf_values() {
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        data[42] = f32::INFINITY;
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject Inf values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Inf"),
            "Error should mention Inf: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedWeight - L2 norm ~0
    // ================================================================

    #[test]
    fn weight_rejects_near_zero_l2_norm() {
        // Values above the zero threshold (1e-10) but producing negligible L2 norm (< 1e-6).
        // With 100 elements at 1e-8: L2 = sqrt(100 * (1e-8)^2) = 10 * 1e-8 = 1e-7.
        let data: Vec<f32> = (0..100).map(|i| 1e-8 + (i as f32) * 1e-12).collect();
        let result = ValidatedWeight::new(data, 10, 10, "test_weight");
        assert!(result.is_err(), "Should reject near-zero L2 norm");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("L2 norm"),
            "Error should mention L2 norm: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedWeight accessors
    // ================================================================

    #[test]
    fn weight_accessors_return_correct_values() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let original_data = data.clone();
        let weight =
            ValidatedWeight::new(data, 10, 10, "my_weight").expect("good data should be accepted");

        assert_eq!(weight.out_dim(), 10);
        assert_eq!(weight.in_dim(), 10);
        assert_eq!(weight.name(), "my_weight");
        assert_eq!(weight.data().len(), 100);
        assert_eq!(weight.data(), original_data.as_slice());

        let stats = weight.stats();
        assert_eq!(stats.len, 100);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);

        let inner = weight.into_inner();
        assert_eq!(inner, original_data);
    }

    // ================================================================
    // ValidatedVector - NaN rejection
    // ================================================================

    #[test]
    fn vector_rejects_nan_values() {
        let mut data = vec![1.0f32; 50];
        data[25] = f32::NAN;
        let result = ValidatedVector::new(data, 50, "test_vec");
        assert!(result.is_err(), "Should reject NaN values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("NaN"),
            "Error should mention NaN: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedVector - Inf rejection
    // ================================================================

    #[test]
    fn vector_rejects_inf_values() {
        let mut data = vec![1.0f32; 50];
        data[10] = f32::NEG_INFINITY;
        let result = ValidatedVector::new(data, 50, "test_vec");
        assert!(result.is_err(), "Should reject Inf values");
        let err = result.unwrap_err();
        assert!(
            err.message.contains("Inf"),
            "Error should mention Inf: {}",
            err.message
        );
    }

    // ================================================================
    // ValidatedVector accessors
    // ================================================================

    #[test]
    fn vector_accessors_return_correct_values() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let original_data = data.clone();
        let vec = ValidatedVector::new(data, 5, "my_vector").expect("good data should be accepted");

        assert_eq!(vec.name(), "my_vector");
        assert_eq!(vec.data().len(), 5);
        assert_eq!(vec.data(), original_data.as_slice());

        let stats = vec.stats();
        assert_eq!(stats.len, 5);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);

        let inner = vec.into_inner();
        assert_eq!(inner, original_data);
    }

    // ================================================================
    // TensorStats::compute edge cases
    // ================================================================

    #[test]
    fn tensor_stats_compute_empty_data() {
        let stats = TensorStats::compute(&[]);
        assert_eq!(stats.len, 0);
        assert_eq!(stats.zero_count, 0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.l2_norm, 0.0);
    }

    #[test]
    fn tensor_stats_compute_all_nan() {
        let stats = TensorStats::compute(&[f32::NAN, f32::NAN, f32::NAN]);
        assert_eq!(stats.len, 3);
        assert_eq!(stats.nan_count, 3);
        assert_eq!(stats.inf_count, 0);
        assert_eq!(stats.zero_count, 0);
        // min/max should be 0.0 since no valid values found
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn tensor_stats_compute_mixed_nan_inf_valid() {
        let stats = TensorStats::compute(&[1.0, f32::NAN, f32::INFINITY, 2.0, f32::NEG_INFINITY]);
        assert_eq!(stats.len, 5);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 2.0);
    }

    #[test]
    fn tensor_stats_zero_pct_empty() {
        let stats = TensorStats::compute(&[]);
        assert_eq!(stats.zero_pct(), 0.0);
    }

    #[test]
    fn tensor_stats_zero_pct_with_zeros() {
        let stats = TensorStats::compute(&[0.0, 0.0, 1.0, 2.0]);
        // 2 out of 4 are near-zero = 50%
        assert!((stats.zero_pct() - 50.0).abs() < 0.01);
    }

    // ================================================================
    // ContractValidationError Display and Error trait
    // ================================================================

    #[test]
    fn contract_validation_error_display_format() {
        let err = ContractValidationError {
            tensor_name: "embedding".to_string(),
            rule_id: "F-DATA-QUALITY-001".to_string(),
            message: "DENSITY FAILURE: 94.5% zeros".to_string(),
        };
        let display = format!("{err}");
        assert_eq!(
            display,
            "[F-DATA-QUALITY-001] Tensor 'embedding': DENSITY FAILURE: 94.5% zeros"
        );
    }

    #[test]
    fn contract_validation_error_implements_std_error() {
        let err = ContractValidationError {
            tensor_name: "weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
            message: "Shape mismatch".to_string(),
        };
        // Verify it implements std::error::Error
        let std_err: &dyn std::error::Error = &err;
        let display_via_error = format!("{std_err}");
        assert!(display_via_error.contains("Shape mismatch"));
        // source() should return None (no wrapped error)
        assert!(std_err.source().is_none());
    }

    #[test]
    fn contract_validation_error_clone() {
        let err = ContractValidationError {
            tensor_name: "test".to_string(),
            rule_id: "F-001".to_string(),
            message: "fail".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(cloned.tensor_name, err.tensor_name);
        assert_eq!(cloned.rule_id, err.rule_id);
        assert_eq!(cloned.message, err.message);
    }

    #[test]
    fn contract_validation_error_debug() {
        let err = ContractValidationError {
            tensor_name: "test".to_string(),
            rule_id: "F-001".to_string(),
            message: "fail".to_string(),
        };
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("ContractValidationError"));
        assert!(debug_str.contains("test"));
    }
}
