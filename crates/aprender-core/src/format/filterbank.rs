
/// Extensible Poka-yoke validation trait (implement per model type)
///
/// # Example
///
/// ```rust,ignore
/// impl PokaYoke for WhisperModel {
///     fn validate(&self) -> PokaYokeResult {
///         let mut result = PokaYokeResult::new();
///
///         // Gate 1: Filterbank must be embedded
///         if self.has_filterbank() {
///             result.add_gate(Gate::pass("filterbank_present", 20));
///         } else {
///             result.add_gate(Gate::fail("filterbank_present", 20,
///                 "Fix: Embed Slaney-normalized filterbank via MelFilterbankData::mel_80()"));
///         }
///
///         // Gate 2: Filterbank must be Slaney-normalized
///         if let Some(fb) = self.filterbank() {
///             let max = fb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
///             if max < 0.1 {
///                 result.add_gate(Gate::pass("filterbank_normalized", 30));
///             } else {
///                 result.add_gate(Gate::fail("filterbank_normalized", 30,
///                     format!("Fix: Apply 2.0/bandwidth normalization (max={max:.4}, expected <0.1)")));
///             }
///         }
///
///         result
///     }
/// }
/// ```
pub trait PokaYoke {
    /// Validate model and return quality score (0-100)
    fn poka_yoke_validate(&self) -> PokaYokeResult;

    /// Get quality score (convenience method)
    fn quality_score(&self) -> u8 {
        self.poka_yoke_validate().score
    }
}

/// Create a failing result for models without `PokaYoke` implementation.
///
/// Use this when saving models that don't implement the trait.
/// Returns a result with score=0 and a single failing gate.
///
/// # Example
///
/// ```rust
/// use aprender::format::validation::fail_no_validation_rules;
///
/// let result = fail_no_validation_rules();
/// assert_eq!(result.score, 0);
/// assert_eq!(result.grade(), "F");
/// assert!(!result.passed());
/// ```
#[must_use]
pub fn fail_no_validation_rules() -> PokaYokeResult {
    let mut result = PokaYokeResult::new();
    result.add_gate(Gate::fail(
        "no_validation_rules",
        100,
        "Fix: Implement PokaYoke trait for this model type",
    ));
    result
}

/// Alias for backwards compatibility
#[deprecated(since = "0.19.0", note = "Use fail_no_validation_rules() instead")]
#[must_use]
pub fn no_validation_result() -> PokaYokeResult {
    fail_no_validation_rules()
}

// ============================================================================
// Whisper Model Poka-yoke Validation (APR-POKA-001, D11, D12)
// Toyota Way: Jidoka - Stop and fix quality issues at the source
// ============================================================================

/// Whisper model validation context
///
/// Provides poka-yoke validation for Whisper ASR models:
/// - D11: Filterbank must be embedded for mel models
/// - D12: Filterbank must be Slaney-normalized (max < 0.1)
///
/// # Example
///
/// ```rust
/// use aprender::format::validation::WhisperValidation;
///
/// // Valid Slaney-normalized filterbank (80 bins x 201 FFT bins)
/// let filterbank: Vec<f32> = vec![0.05; 80 * 201];
/// let result = WhisperValidation::validate_filterbank(Some(&filterbank));
/// assert!(result.passed());
/// assert_eq!(result.grade(), "A+");
/// ```
#[derive(Debug, Clone, Default)]
pub struct WhisperValidation;

impl WhisperValidation {
    /// Validate Whisper filterbank (D11: present, D12: Slaney-normalized)
    ///
    /// # Arguments
    /// * `filterbank` - Optional filterbank data (80 mel bins × `n_fft` bins)
    ///
    /// # Returns
    /// `PokaYokeResult` with gates:
    /// - `filterbank_present` (50 pts): Filterbank must be embedded
    /// - `filterbank_normalized` (50 pts): Max value < 0.1 (Slaney normalization)
    #[must_use]
    pub fn validate_filterbank(filterbank: Option<&[f32]>) -> PokaYokeResult {
        let mut gates = Vec::with_capacity(2);

        // D11: Filterbank must be embedded
        match filterbank {
            Some(fb) if !fb.is_empty() => {
                gates.push(Gate::pass("filterbank_present", 50));

                // D12: Filterbank must be Slaney-normalized (max < 0.1)
                let max_val = fb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if max_val < 0.1 {
                    gates.push(Gate::pass("filterbank_normalized", 50));
                } else {
                    gates.push(Gate::fail(
                        "filterbank_normalized",
                        50,
                        format!(
                            "Fix: Apply 2.0/bandwidth Slaney normalization (max={max_val:.4}, expected <0.1)"
                        ),
                    ));
                }
            }
            _ => {
                gates.push(Gate::fail(
                    "filterbank_present",
                    50,
                    "Fix: Embed mel filterbank via MelFilterbankData::mel_80()",
                ));
                gates.push(Gate::fail(
                    "filterbank_normalized",
                    50,
                    "Fix: Cannot verify normalization - filterbank missing",
                ));
            }
        }

        PokaYokeResult::from_gates(gates)
    }

    /// Validate encoder/decoder tensor statistics
    ///
    /// Checks for common conversion bugs:
    /// - `LayerNorm` weights should have mean ≈ 1.0
    /// - Linear weights should have mean ≈ 0.0
    /// - No NaN/Inf values
    #[must_use]
    pub fn validate_tensor_stats(stats: &[TensorStats]) -> PokaYokeResult {
        let mut gates = Vec::new();

        // Check for NaN values (catastrophic)
        let nan_count: usize = stats.iter().map(|s| s.nan_count).sum();
        if nan_count == 0 {
            gates.push(Gate::pass("no_nan_values", 30));
        } else {
            gates.push(Gate::fail(
                "no_nan_values",
                30,
                format!("Fix: {nan_count} NaN values found - check conversion pipeline"),
            ));
        }

        // Check for Inf values (catastrophic)
        let inf_count: usize = stats.iter().map(|s| s.inf_count).sum();
        if inf_count == 0 {
            gates.push(Gate::pass("no_inf_values", 20));
        } else {
            gates.push(Gate::fail(
                "no_inf_values",
                20,
                format!("Fix: {inf_count} Inf values found - check overflow in conversion"),
            ));
        }

        // Check LayerNorm weights
        let invalid_ln: Vec<_> = stats
            .iter()
            .filter(|s| {
                (s.name.contains("layer_norm") || s.name.contains("ln_"))
                    && (s.name.ends_with(".weight") || s.name.ends_with(".gamma"))
                    && !s.is_valid_layernorm_weight()
            })
            .collect();

        if invalid_ln.is_empty() {
            gates.push(Gate::pass("layernorm_weights_valid", 25));
        } else {
            let names: Vec<_> = invalid_ln
                .iter()
                .take(3)
                .map(|s| format!("{} (mean={:.4})", s.name, s.mean))
                .collect();
            gates.push(Gate::fail(
                "layernorm_weights_valid",
                25,
                format!(
                    "Fix: LayerNorm weights should have mean in [0.5, 3.0]: {}",
                    names.join(", ")
                ),
            ));
        }

        // Check for all-zero tensors (dead weights)
        let zero_tensors: Vec<_> = stats.iter().filter(|s| !s.is_not_all_zeros()).collect();

        if zero_tensors.is_empty() {
            gates.push(Gate::pass("no_zero_tensors", 25));
        } else {
            let names: Vec<_> = zero_tensors
                .iter()
                .take(3)
                .map(|s| s.name.clone())
                .collect();
            gates.push(Gate::fail(
                "no_zero_tensors",
                25,
                format!(
                    "Fix: All-zero tensors found (dead weights): {}",
                    names.join(", ")
                ),
            ));
        }

        PokaYokeResult::from_gates(gates)
    }

    /// Full Whisper model validation
    ///
    /// Combines filterbank and tensor validation into single result.
    #[must_use]
    pub fn validate_full(
        filterbank: Option<&[f32]>,
        tensor_stats: &[TensorStats],
    ) -> PokaYokeResult {
        let fb_result = Self::validate_filterbank(filterbank);
        let tensor_result = Self::validate_tensor_stats(tensor_stats);

        // Combine gates with weighted scoring
        let mut all_gates = fb_result.gates;
        all_gates.extend(tensor_result.gates);

        PokaYokeResult::from_gates(all_gates)
    }
}

// ============================================================================
// Poka-yoke Tests (APR-POKA-001)
// ============================================================================

// Tests extracted to validation_tests.rs (PMAT-197)
#[cfg(test)]
#[path = "validation_tests.rs"]
mod tests;
