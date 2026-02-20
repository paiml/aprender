
impl ConversionReport {
    /// Check if conversion preserved all tensors
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.dropped_tensors.is_empty()
    }

    /// Check if tensor counts match
    #[must_use]
    pub fn tensor_counts_match(&self) -> bool {
        self.source_inspection.tensors.len() == self.target_inspection.tensors.len()
    }
}

// ============================================================================
// Verification Report
// ============================================================================

/// Report from round-trip verification
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Whether the round-trip preserved semantic equivalence
    pub is_equivalent: bool,
    /// Maximum absolute difference in any tensor value
    pub max_diff: f32,
    /// Mean absolute difference across all tensors
    pub mean_diff: f32,
    /// Tensor-wise comparison results
    pub tensor_diffs: BTreeMap<String, f32>,
    /// Metadata keys that changed
    pub changed_metadata: Vec<String>,
    /// Tensors that failed comparison
    pub failed_tensors: Vec<String>,
}

impl VerificationReport {
    /// Create a passing verification report
    #[must_use]
    pub fn passing() -> Self {
        Self {
            is_equivalent: true,
            max_diff: 0.0,
            mean_diff: 0.0,
            tensor_diffs: BTreeMap::new(),
            changed_metadata: Vec::new(),
            failed_tensors: Vec::new(),
        }
    }

    /// Check if verification passes within tolerance
    #[must_use]
    pub fn passes_with_tolerance(&self, epsilon: f32) -> bool {
        self.max_diff <= epsilon && self.failed_tensors.is_empty()
    }
}

// ============================================================================
// Validation Report (GH-175, PMAT-180)
// ============================================================================

/// Tensor validation result per APR-SPEC 10.9
#[derive(Debug, Clone)]
pub struct TensorValidation {
    /// Tensor name
    pub name: String,
    /// Whether tensor passes all checks
    pub is_valid: bool,
    /// Number of NaN values detected
    pub nan_count: usize,
    /// Number of Inf values detected
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
    /// Total element count
    pub element_count: usize,
    /// Minimum value (excluding NaN/Inf)
    pub min: f32,
    /// Maximum value (excluding NaN/Inf)
    pub max: f32,
    /// Mean value (excluding NaN/Inf)
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Specific validation failures
    pub failures: Vec<String>,
}

impl TensorValidation {
    /// Check if tensor has any NaN values
    #[must_use]
    pub fn has_nan(&self) -> bool {
        self.nan_count > 0
    }

    /// Check if tensor has any Inf values
    #[must_use]
    pub fn has_inf(&self) -> bool {
        self.inf_count > 0
    }

    /// Check if tensor is all zeros (suspicious)
    #[must_use]
    pub fn is_all_zeros(&self) -> bool {
        self.zero_count == self.element_count
    }
}

/// Complete validation report for a model file
/// Implements APR-SPEC 10.9 Physics Constraints
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Detected format
    pub format: FormatType,
    /// File path validated
    pub file_path: String,
    /// Overall pass/fail status
    pub is_valid: bool,
    /// Total tensors validated
    pub tensor_count: usize,
    /// Tensors with issues
    pub failed_tensor_count: usize,
    /// Total NaN values across all tensors
    pub total_nan_count: usize,
    /// Total Inf values across all tensors
    pub total_inf_count: usize,
    /// Tensors that are all zeros
    pub all_zero_tensors: Vec<String>,
    /// Per-tensor validation results
    pub tensors: Vec<TensorValidation>,
    /// Validation time in milliseconds
    pub duration_ms: u64,
}

impl ValidationReport {
    /// Check if validation passed (no NaN/Inf, no all-zeros)
    #[must_use]
    pub fn passed(&self) -> bool {
        self.is_valid
    }

    /// Get summary line for CLI output
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_valid {
            format!(
                "VALID: {} tensors checked, 0 contract violations (PMAT-235)",
                self.tensor_count
            )
        } else {
            let contract_failures: usize = self.tensors.iter().map(|t| t.failures.len()).sum();
            format!(
                "INVALID: {} tensors, {} contract violations, {} NaN, {} Inf, {} all-zeros",
                self.tensor_count,
                contract_failures,
                self.total_nan_count,
                self.total_inf_count,
                self.all_zero_tensors.len()
            )
        }
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Model Validation Report (APR-SPEC 10.9) ===")?;
        writeln!(f, "Format: {}", self.format)?;
        writeln!(f, "File: {}", self.file_path)?;
        writeln!(
            f,
            "Status: {}",
            if self.is_valid { "VALID" } else { "INVALID" }
        )?;
        writeln!(f, "Tensors: {}", self.tensor_count)?;
        writeln!(f, "Total NaN: {}", self.total_nan_count)?;
        writeln!(f, "Total Inf: {}", self.total_inf_count)?;
        writeln!(f, "All-Zero Tensors: {}", self.all_zero_tensors.len())?;
        writeln!(f, "Duration: {} ms", self.duration_ms)?;

        if !self.is_valid {
            writeln!(f, "\n--- Failed Tensors ---")?;
            for tv in &self.tensors {
                if !tv.is_valid {
                    writeln!(
                        f,
                        "  {}: {} NaN, {} Inf, {} zeros / {}",
                        tv.name, tv.nan_count, tv.inf_count, tv.zero_count, tv.element_count
                    )?;
                    for failure in &tv.failures {
                        writeln!(f, "    - {failure}")?;
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Conversion Options
// ============================================================================

/// Options for conversion operations
#[allow(clippy::struct_excessive_bools)] // Options structs commonly have multiple boolean flags
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Target quantization (None = preserve original)
    pub quantization: Option<String>,
    /// Verify after conversion
    pub verify: bool,
    /// Compute tensor statistics
    pub compute_stats: bool,
    /// Tolerance for floating-point comparison
    pub tolerance: f32,
    /// Preserve metadata from source
    pub preserve_metadata: bool,
    /// Add conversion provenance to metadata
    pub add_provenance: bool,
    /// External tokenizer.json for weights-only models (PMAT-232)
    pub tokenizer_path: Option<PathBuf>,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            quantization: None,
            verify: true,
            compute_stats: false,
            tolerance: 1e-6,
            preserve_metadata: true,
            add_provenance: true,
            tokenizer_path: None,
        }
    }
}

// ============================================================================
// Rosetta Stone Converter
// ============================================================================

/// Main converter implementing the Rosetta Stone pattern
#[derive(Debug)]
pub struct RosettaStone {
    /// Options for conversion
    options: ConversionOptions,
}
