
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
        // Gate 0: Zero-length guard (PMAT-332)
        // A zero-length norm weight is never valid — it means the tensor is missing.
        if expected_len == 0 {
            return Err(ContractValidationError {
                tensor_name: name.to_string(),
                rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
                message: "Zero-length vector: expected_len must be > 0".to_string(),
            });
        }

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
