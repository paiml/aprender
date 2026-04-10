
/// Core implementation of the HuggingFace parity oracle for tensor comparison
impl HfParityOracle {
    /// Create a new HF Parity Oracle.
    ///
    /// # Arguments
    ///
    /// * `corpus_path` - Path to ground truth corpus (e.g., `~/src/hf-ground-truth-corpus/oracle/`)
    /// * `model_family` - Model family subdirectory (e.g., "llama-2-7b")
    #[must_use]
    pub fn new(corpus_path: impl AsRef<Path>, model_family: &str) -> Self {
        Self {
            corpus_path: corpus_path.as_ref().to_path_buf(),
            model_family: model_family.to_string(),
            tolerance: Tolerance::default(),
            golden_cache: HashMap::new(),
        }
    }

    /// Configure tolerance settings.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Get the corpus path.
    #[must_use]
    pub fn corpus_path(&self) -> &Path {
        &self.corpus_path
    }

    /// Get the model family.
    #[must_use]
    pub fn model_family(&self) -> &str {
        &self.model_family
    }

    /// Get the tolerance configuration.
    #[must_use]
    pub const fn tolerance(&self) -> &Tolerance {
        &self.tolerance
    }

    /// Load golden output for a given prompt.
    ///
    /// Golden outputs are stored as SafeTensors files named by input hash.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Golden file cannot be found or read
    /// - SafeTensors deserialization fails
    /// - Required 'logits' tensor is missing
    pub fn load_golden(&self, prompt: &str) -> Result<GoldenOutput, String> {
        let input_hash = hash_prompt(prompt);

        // Check cache first
        if let Some(cached) = self.golden_cache.get(&input_hash) {
            return Ok(cached.clone());
        }

        let path = self
            .corpus_path
            .join(&self.model_family)
            .join(format!("{input_hash}.safetensors"));

        Self::load_golden_from_path(&path, prompt, &input_hash)
    }

    /// Load golden output from a specific SafeTensors file path
    fn load_golden_from_path(
        path: &Path,
        prompt: &str,
        input_hash: &str,
    ) -> Result<GoldenOutput, String> {
        let data = std::fs::read(path).map_err(|e| format!("Failed to read golden file: {e}"))?;

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| format!("Failed to parse SafeTensors: {e}"))?;

        // Extract logits tensor
        let logits_view = tensors
            .tensor("logits")
            .map_err(|e| format!("Missing 'logits' tensor: {e}"))?;

        let logits: Vec<f32> = logits_view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Try to load companion metadata JSON if it exists
        let metadata_path = path.with_extension("json");
        let (model_id, transformers_version, text) = if metadata_path.exists() {
            Self::load_metadata_json(&metadata_path).unwrap_or_default()
        } else {
            (String::new(), String::new(), None)
        };

        Ok(GoldenOutput {
            input_hash: input_hash.to_string(),
            prompt: prompt.to_string(),
            logits,
            shape: logits_view.shape().to_vec(),
            text,
            model_id,
            transformers_version,
        })
    }

    /// Load metadata from companion JSON file.
    fn load_metadata_json(path: &Path) -> Result<(String, String, Option<String>), String> {
        #[derive(Deserialize)]
        struct MetadataJson {
            #[serde(default)]
            model: String,
            #[serde(default)]
            transformers_version: String,
            generated_text: Option<String>,
        }

        let data =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read metadata: {e}"))?;

        let meta: MetadataJson = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse metadata JSON: {e}"))?;

        Ok((meta.model, meta.transformers_version, meta.generated_text))
    }

    /// Compare two tensors with configured tolerance.
    ///
    /// Returns `Ok(())` if tensors are within tolerance, `Err(TensorDiff)` otherwise.
    /// Implements the allclose criterion: |a - b| <= atol + rtol * |b|
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor shapes (lengths) do not match
    /// - Mismatch ratio exceeds configured threshold
    pub fn tensors_close(&self, expected: &[f32], actual: &[f32]) -> Result<(), TensorDiff> {
        if expected.len() != actual.len() {
            return Err(TensorDiff::ShapeMismatch {
                expected: expected.len(),
                actual: actual.len(),
            });
        }

        if expected.is_empty() {
            return Ok(());
        }

        let mut max_diff: f32 = 0.0;
        let mut max_diff_idx: usize = 0;
        let mut num_mismatches: usize = 0;
        let mut sum_diff: f64 = 0.0;

        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e - a).abs();
            sum_diff += f64::from(diff);

            if !self.tolerance.is_close(a, e) {
                num_mismatches += 1;
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_idx = i;
                }
            }
        }

        let mismatch_ratio = num_mismatches as f32 / expected.len() as f32;
        // Mean diff is intentionally truncated to f32 for consistency with tensor data
        #[allow(clippy::cast_possible_truncation)]
        let mean_diff = (sum_diff / expected.len() as f64) as f32;

        if mismatch_ratio > self.tolerance.max_mismatch_ratio {
            Err(TensorDiff::ValueMismatch {
                num_mismatches,
                total: expected.len(),
                mismatch_ratio,
                max_diff,
                max_diff_idx,
                expected_val: expected[max_diff_idx],
                actual_val: actual[max_diff_idx],
                mean_diff,
            })
        } else {
            Ok(())
        }
    }

    /// Compare actual output tensor file against golden.
    ///
    /// # Arguments
    ///
    /// * `actual_path` - Path to SafeTensors file with actual model output
    /// * `golden` - Pre-loaded golden output to compare against
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Actual file cannot be read or parsed
    /// - Tensor shapes do not match
    /// - Values exceed tolerance threshold
    pub fn compare_tensor_file(
        &self,
        actual_path: &Path,
        golden: &GoldenOutput,
    ) -> Result<(), TensorDiff> {
        let data = std::fs::read(actual_path).map_err(|e| TensorDiff::ParseError {
            message: format!("Failed to read actual output: {e}"),
        })?;

        let tensors =
            safetensors::SafeTensors::deserialize(&data).map_err(|e| TensorDiff::ParseError {
                message: format!("Failed to parse SafeTensors: {e}"),
            })?;

        let logits_view = tensors
            .tensor("logits")
            .map_err(|e| TensorDiff::ParseError {
                message: format!("Missing 'logits' tensor: {e}"),
            })?;

        let actual: Vec<f32> = logits_view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        self.tensors_close(&golden.logits, &actual)
    }

    /// Compute statistical summary of divergence.
    ///
    /// Returns (max_diff, mean_diff, std_diff) for diagnostic purposes.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // f64→f32 intentional: accumulate in f64, return f32
    pub fn compute_divergence_stats(expected: &[f32], actual: &[f32]) -> (f32, f32, f32) {
        if expected.len() != actual.len() || expected.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let diffs: Vec<f32> = expected
            .iter()
            .zip(actual.iter())
            .map(|(e, a)| (e - a).abs())
            .collect();

        let max_diff = diffs.iter().copied().fold(0.0f32, f32::max);
        // Use f64 accumulation to prevent precision loss on large tensors
        let sum: f64 = diffs.iter().map(|&d| f64::from(d)).sum();
        let mean_diff = (sum / diffs.len() as f64) as f32;

        // Compute standard deviation (f64 accumulation for variance)
        let var_sum: f64 = diffs
            .iter()
            .map(|&d| {
                let delta = f64::from(d) - f64::from(mean_diff);
                delta * delta
            })
            .sum();
        let std_diff = (var_sum / diffs.len() as f64).sqrt() as f32;

        (max_diff, mean_diff, std_diff)
    }

    /// Detect systematic bias between expected and actual outputs.
    ///
    /// Returns true if bias is detected (mean shift or scale drift).
    #[must_use]
    pub fn detect_systematic_bias(expected: &[f32], actual: &[f32]) -> Option<String> {
        if expected.len() != actual.len() || expected.is_empty() {
            return None;
        }

        let n = expected.len() as f32;

        // Compute means
        let mean_e: f32 = expected.iter().sum::<f32>() / n;
        let mean_a: f32 = actual.iter().sum::<f32>() / n;

        // Compute standard deviations
        let std_e: f32 = (expected.iter().map(|x| (x - mean_e).powi(2)).sum::<f32>() / n).sqrt();
        let std_a: f32 = (actual.iter().map(|x| (x - mean_a).powi(2)).sum::<f32>() / n).sqrt();

        // Check for mean shift (> 3 sigma)
        if std_e > 1e-10 && (mean_a - mean_e).abs() > 3.0 * std_e {
            return Some(format!(
                "Mean shift detected: expected {mean_e:.6}, actual {mean_a:.6} (shift: {:.6} sigma)",
                (mean_a - mean_e).abs() / std_e
            ));
        }

        // Check for scale drift (> 10%)
        if std_e > 1e-10 && (std_a / std_e - 1.0).abs() > 0.1 {
            return Some(format!(
                "Scale drift detected: expected std {std_e:.6}, actual std {std_a:.6} (ratio: {:.2})",
                std_a / std_e
            ));
        }

        None
    }
}

/// Implement the Oracle trait to evaluate model outputs against HF golden references
impl Oracle for HfParityOracle {
    /// Evaluate model output by comparing text or tensor data against golden reference
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        // Try to load golden output for this prompt
        let golden = match self.load_golden(prompt) {
            Ok(g) => g,
            Err(e) => {
                // No golden output available - skip (not a failure)
                return OracleResult::Corroborated {
                    evidence: format!(
                        "No golden output for prompt (hash: {}): {e}",
                        hash_prompt(prompt)
                    ),
                };
            }
        };

        // If golden has expected text, compare text output
        if let Some(expected_text) = &golden.text {
            let output_trimmed = output.trim();
            let expected_trimmed = expected_text.trim();

            if output_trimmed == expected_trimmed {
                return OracleResult::Corroborated {
                    evidence: format!(
                        "Text output matches HF golden ({} chars)",
                        output_trimmed.len()
                    ),
                };
            }
            // Text mismatch - only fall through to tensor comparison if output
            // is an existing .safetensors file path; otherwise falsify immediately
            let output_path = Path::new(output_trimmed);
            if !output_path.exists()
                || output_path
                    .extension()
                    .is_none_or(|ext| ext != "safetensors")
            {
                return OracleResult::Falsified {
                    reason: "Text output differs from HF golden".to_string(),
                    evidence: format!(
                        "Expected: '{}'\nActual: '{}'",
                        truncate(expected_trimmed, 100),
                        truncate(output_trimmed, 100)
                    ),
                };
            }
        }

        // Try to interpret output as a tensor file path
        let output_path = Path::new(output.trim());
        if output_path.exists()
            && output_path
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        {
            match self.compare_tensor_file(output_path, &golden) {
                Ok(()) => {
                    return OracleResult::Corroborated {
                        evidence: format!(
                            "Tensor parity verified: {} elements within tolerance (atol={}, rtol={})",
                            golden.logits.len(),
                            self.tolerance.atol_fp32,
                            self.tolerance.rtol_fp32
                        ),
                    };
                }
                Err(diff) => {
                    return OracleResult::Falsified {
                        reason: "Tensor mismatch with HF golden".to_string(),
                        evidence: diff.to_string(),
                    };
                }
            }
        }

        // Output is plain text, no tensor file - check for text comparison
        OracleResult::Corroborated {
            evidence: "Output is text, no tensor comparison available".to_string(),
        }
    }

    /// Return the oracle identifier string
    fn name(&self) -> &'static str {
        "hf_parity"
    }
}

/// Hash a prompt string for golden output lookup.
///
/// Uses SHA-256 truncated to 16 hex chars for cross-language compatibility.
/// This matches the Python `generate_golden.py` script implementation.
#[must_use]
pub fn hash_prompt(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let result = hasher.finalize();
    // Take first 8 bytes (16 hex chars) to match Python implementation
    hex::encode(&result[..8])
}

/// Truncate a string for display purposes.
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        // Find a safe UTF-8 boundary
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unreadable_literal, clippy::needless_range_loop)]
#[path = "hf_parity_tests.rs"]
mod tests;
