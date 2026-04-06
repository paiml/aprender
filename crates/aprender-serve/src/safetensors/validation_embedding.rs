
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
    /// Returns `ContractValidationError` if any validation rule fails.
    pub fn new(
        data: Vec<f32>,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> std::result::Result<Self, ContractValidationError> {
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

/// Validated weight matrix - compile-time guarantee of data quality
#[derive(Debug, Clone)]
pub struct ValidatedWeight {
    data: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
    name: String,
    stats: TensorStats,
}

impl ValidatedWeight {
    const MAX_ZERO_PCT: f32 = 80.0;
    const MIN_L2_NORM: f32 = 1e-6;

    /// Construct a validated weight matrix
    pub fn new(
        data: Vec<f32>,
        out_dim: usize,
        in_dim: usize,
        name: &str,
    ) -> std::result::Result<Self, ContractValidationError> {
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

        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

        if stats.inf_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} Inf values", stats.inf_count),
            });
        }

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

/// Validated 1D tensor (bias, norm weights)
#[derive(Debug, Clone)]
pub struct ValidatedVector {
    data: Vec<f32>,
    name: String,
    stats: TensorStats,
}

impl ValidatedVector {
    /// Construct a validated vector
    pub fn new(
        data: Vec<f32>,
        expected_len: usize,
        name: &str,
    ) -> std::result::Result<Self, ContractValidationError> {
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

        if stats.nan_count > 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-DATA-QUALITY-002".to_string(),
                message: format!("Contains {} NaN values", stats.nan_count),
            });
        }

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
// VALIDATED APR TRANSFORMER — Compile-Time Tensor Quality Enforcement (PMAT-235)
// =============================================================================
//
// Follows the PreparedTokens pattern from PMAT-236: the inner AprTransformer
// is PRIVATE, so construction ONLY via validate(). This makes it IMPOSSIBLE
// to use an AprTransformer with corrupt tensors in inference.
// =============================================================================

use crate::apr_transformer::{AprTransformer, AprTransformerConfig};

/// Validated APR Transformer — compile-time guarantee of tensor data quality
///
/// This wrapper validates ALL tensors in an `AprTransformer`:
/// - Embedding: density, NaN/Inf, L2 norm, spot check
/// - Layer weights: density, NaN/Inf, L2 norm, shape
/// - Norm vectors: NaN/Inf, length
///
/// The inner `AprTransformer` is private. The ONLY way to construct this type
/// is via `validate()`, which enforces all contract rules.
///
/// Access the inner transformer transparently via `Deref`.
#[derive(Debug, Clone)]
pub struct ValidatedAprTransformer {
    // PRIVATE — only constructible via validate()
    inner: AprTransformer,
}
