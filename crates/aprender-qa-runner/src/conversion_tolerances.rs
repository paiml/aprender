
/// Default tolerances per quantization type
pub const DEFAULT_TOLERANCES: &[ConversionTolerance] = &[
    ConversionTolerance {
        quant_type: QuantType::F32,
        atol: 1e-6,
        rtol: 1e-5,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::F16,
        atol: 1e-3,
        rtol: 1e-3,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::BF16,
        atol: 1e-2,
        rtol: 1e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q4KM,
        atol: 1e-1,
        rtol: 5e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q5KM,
        atol: 7.5e-2,
        rtol: 5e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q6K,
        atol: 5e-2,
        rtol: 5e-2,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q4_0,
        atol: 1e-1,
        rtol: 1e-1,
        expected_pygmy_fixture: String::new(),
    },
    ConversionTolerance {
        quant_type: QuantType::Q8_0,
        atol: 1e-2,
        rtol: 1e-2,
        expected_pygmy_fixture: String::new(),
    },
];

/// Get the tolerance for a given quantization type
#[must_use]
pub fn tolerance_for(qt: QuantType) -> &'static ConversionTolerance {
    DEFAULT_TOLERANCES
        .iter()
        .find(|t| t.quant_type == qt)
        .unwrap_or(&DEFAULT_TOLERANCES[0]) // F32 fallback
}

/// Simple keyword-based failure classification rules.
/// Checked in priority order; first match wins.
/// Complex rules (missing_artifact, inference) are checked separately.
const KEYWORD_FAILURE_RULES: &[(&[&str], ConversionFailureType)] = &[
    (
        &["tensor name", "name mismatch", "missing tensor", "unexpected tensor"],
        ConversionFailureType::TensorNameMismatch,
    ),
    (
        &["dequantiz", "quantiz", "nan", "infinity", "overflow"],
        ConversionFailureType::DequantizationFailure,
    ),
];

/// Config metadata keywords that indicate a metadata mismatch failure.
const CONFIG_METADATA_KEYWORDS: &[&str] = &[
    "hidden_size",
    "num_layers",
    "num_hidden_layers",
    "vocab_size",
    "metadata mismatch",
    "config mismatch",
];

/// Classify a conversion failure from stderr output and exit code.
///
/// Priority order: keyword rules -> missing artifact -> config metadata -> inference -> unknown.
#[must_use]
pub fn classify_failure(stderr: &str, exit_code: i32) -> ConversionFailureType {
    let lower = stderr.to_lowercase();

    // 1. Simple keyword-based rules (table-driven)
    for &(keywords, failure_type) in KEYWORD_FAILURE_RULES {
        if keywords.iter().any(|kw| lower.contains(kw)) {
            return failure_type;
        }
    }

    // 2. Complex rules with negation/multi-condition logic
    if is_missing_artifact(&lower) {
        ConversionFailureType::MissingArtifact
    } else if CONFIG_METADATA_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
        ConversionFailureType::ConfigMetadataMismatch
    } else if is_inference_failure(&lower, exit_code) {
        ConversionFailureType::InferenceFailure
    } else {
        ConversionFailureType::Unknown
    }
}

/// Check for missing artifact failures.
/// Uses negation logic (`missing` without `mismatch`) that cannot be
/// expressed as a simple keyword list.
fn is_missing_artifact(s: &str) -> bool {
    s.contains("not found")
        || s.contains("no such file")
        || (s.contains("config.json") && (s.contains("missing") || s.contains("error") || s.contains("fail")))
        || (s.contains("missing") && !s.contains("mismatch"))
        || (s.contains("tokenizer") && !s.contains("mismatch"))
}

fn is_inference_failure(s: &str, exit_code: i32) -> bool {
    s.contains("inference")
        || s.contains("forward pass")
        || s.contains("segfault")
        || s.contains("sigsegv")
        || exit_code == -11
}

/// Patterns that indicate specific bug types
const GARBAGE_PATTERNS: &[&str] = &[
    "PAD",
    "<pad>",
    "<|endoftext|>",
    "1. What is the difference",
    "151935", // Common garbage token ID
    "\u{0000}",
];

/// Expected patterns for arithmetic test "What is 2+2?"
const ARITHMETIC_EXPECTED: &[&str] = &["4", "four", "Four", "2+2=4", "2 + 2 = 4", "equals 4"];

/// Conversion test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionTest {
    /// Source format
    pub source_format: Format,
    /// Target format
    pub target_format: Format,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Tolerance for comparison
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    /// Binary path for apr CLI
    #[serde(skip, default = "default_binary")]
    pub binary: String,
    /// Quantization type for dtype-aware tolerance (§3.7)
    #[serde(default)]
    pub quant_type: Option<QuantType>,
    /// Output directory for conversion artifacts (ISO-OUT-001)
    #[serde(skip, default)]
    pub output_dir: Option<ConversionOutputDir>,
}

fn default_epsilon() -> f64 {
    EPSILON
}

fn default_binary() -> String {
    "apr".to_string()
}

/// Result of a conversion test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversionResult {
    /// Conversion preserved model semantics
    Corroborated {
        /// Source format
        source_format: Format,
        /// Target format
        target_format: Format,
        /// Backend used
        backend: Backend,
        /// Max tensor difference observed
        max_diff: f64,
    },
    /// Conversion introduced errors
    Falsified {
        /// Gate ID that failed
        gate_id: String,
        /// Reason for failure
        reason: String,
        /// Evidence of failure
        evidence: ConversionEvidence,
    },
}

/// Evidence collected from a failed conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionEvidence {
    /// Hash of source model output
    pub source_hash: String,
    /// Hash of converted model output
    pub converted_hash: String,
    /// Maximum difference observed
    pub max_diff: f64,
    /// Indices of differing tensors
    pub diff_indices: Vec<usize>,
    /// Source format
    pub source_format: Format,
    /// Target format
    pub target_format: Format,
    /// Backend
    pub backend: Backend,
    /// Typed failure classification (§3.4)
    #[serde(default)]
    pub failure_type: Option<ConversionFailureType>,
    /// Quantization type (§3.7)
    #[serde(default)]
    pub quant_type: Option<QuantType>,
}

impl ConversionTest {
    /// Create a new conversion test
    #[must_use]
    pub fn new(source: Format, target: Format, backend: Backend, model_id: ModelId) -> Self {
        Self {
            source_format: source,
            target_format: target,
            backend,
            model_id,
            epsilon: EPSILON,
            binary: default_binary(),
            quant_type: None,
            output_dir: None,
        }
    }

    /// Set the output directory for this test (ISO-OUT-001)
    #[must_use]
    pub fn with_output_dir(mut self, output_dir: ConversionOutputDir) -> Self {
        self.output_dir = Some(output_dir);
        self
    }

    /// Get the effective epsilon, using dtype-aware tolerance when quant_type is set
    #[must_use]
    pub fn effective_epsilon(&self) -> f64 {
        self.quant_type
            .map_or(self.epsilon, |qt| tolerance_for(qt).atol)
    }

    /// Get the gate ID for this conversion
    #[must_use]
    pub fn gate_id(&self) -> String {
        let src = format!("{:?}", self.source_format).to_uppercase();
        let tgt = format!("{:?}", self.target_format).to_uppercase();
        format!("F-CONV-{}-{}", &src[..1], &tgt[..1])
    }

    /// Resolve model path for a specific format
    ///
    /// Delegates to standalone `resolve_model_path` function.
    fn resolve_format_path(&self, base_path: &Path, format: &Format) -> Result<std::path::PathBuf> {
        resolve_model_path(base_path, *format)
    }

    /// Execute the conversion test
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion or inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve source model path based on format
        let source_path = self.resolve_format_path(model_path, &self.source_format)?;

        // 1. Run inference on source format
        let source_output = self.run_inference(&source_path, &self.source_format)?;

        // 2. Convert to target format (use resolved source path)
        let converted_path = self.convert_model(&source_path)?;

        // 3. Run inference on converted model
        // For cross-format conversions, inference may fail due to known
        // limitations (e.g., Q4K row padding in GGUF→APR). If conversion
        // succeeded but inference fails, validate at file level.
        let converted_output = match self.run_inference(&converted_path, &self.target_format) {
            Ok(output) => output,
            Err(e) if self.source_format != self.target_format && converted_path.exists() => {
                // Popperian: inference failure after conversion is falsification,
                // not corroboration. The file existing on disk proves nothing about
                // semantic correctness. (Issue #28)
                return Ok(ConversionResult::Falsified {
                    gate_id: self.gate_id(),
                    reason: format!(
                        "Conversion {:?} → {:?} produced file but inference failed: {e}",
                        self.source_format, self.target_format,
                    ),
                    evidence: ConversionEvidence {
                        source_hash: Self::hash_output(&source_output),
                        converted_hash: String::new(),
                        max_diff: f64::NAN,
                        diff_indices: vec![],
                        source_format: self.source_format,
                        target_format: self.target_format,
                        backend: self.backend,
                        failure_type: Some(ConversionFailureType::InferenceFailure),
                        quant_type: self.quant_type,
                    },
                });
            }
            Err(e) => return Err(e),
        };

        // 4. Compare outputs — cross-format conversions involve quantization
        // so text-level identity is not expected. Use garbage detection instead.
        let diff = self.compute_diff(&source_output, &converted_output);
        let is_cross_format = self.source_format != self.target_format;

        // Cross-format comparison: both outputs must be non-garbage
        // (quantization naturally produces different text, so text diff is not meaningful)
        let passes = if is_cross_format {
            let source_ok = !Self::is_garbage_output(&source_output);
            let converted_ok = !Self::is_garbage_output(&converted_output);
            source_ok && converted_ok
        } else {
            diff <= self.effective_epsilon()
        };

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: self.source_format,
                target_format: self.target_format,
                backend: self.backend,
                max_diff: diff,
            })
        } else {
            let reason = if is_cross_format {
                let source_garbage = Self::is_garbage_output(&source_output);
                let converted_garbage = Self::is_garbage_output(&converted_output);
                format!(
                    "Conversion {:?} → {:?} produced garbage output (source_garbage={source_garbage}, converted_garbage={converted_garbage}, diff: {diff:.2e})",
                    self.source_format, self.target_format,
                )
            } else {
                format!(
                    "Conversion {:?} → {:?} produced different output (diff: {:.2e}, ε: {:.2e})",
                    self.source_format,
                    self.target_format,
                    diff,
                    self.effective_epsilon()
                )
            };
            Ok(ConversionResult::Falsified {
                gate_id: self.gate_id(),
                reason,
                evidence: ConversionEvidence {
                    source_hash: Self::hash_output(&source_output),
                    converted_hash: Self::hash_output(&converted_output),
                    max_diff: diff,
                    diff_indices: self.find_diff_indices(&source_output, &converted_output),
                    source_format: self.source_format,
                    target_format: self.target_format,
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }

    /// Run inference and capture output
    fn run_inference(&self, model_path: &Path, _format: &Format) -> Result<String> {
        let backend_flag = match self.backend {
            Backend::Cpu => vec![],
            Backend::Gpu => vec!["--gpu".to_string()],
        };

        let output = Command::new(&self.binary)
            .arg("run")
            .arg(model_path)
            .arg("-p")
            .arg("What is 2+2?")
            .arg("--max-tokens")
            .arg("32")
            .args(&backend_flag)
            .output()
            .map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Execution(format!(
                "Inference failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Convert model to target format using apr rosetta
    fn convert_model(&self, source_path: &Path) -> Result<PathBuf> {
        let target_ext = match self.target_format {
            Format::Gguf => "gguf",
            Format::SafeTensors => "safetensors",
            Format::Apr => "apr",
        };

        // ISO-OUT-001: Use isolated output directory if configured
        let target_path = if let Some(ref output_dir) = self.output_dir {
            // Ensure output directory exists
            output_dir.ensure_dir("basic").map_err(Error::Io)?;

            // Get source filename without extension
            let source_name = source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            output_dir.output_path("basic", source_name, "converted", self.target_format)
        } else {
            // Legacy: write to source directory (for backward compatibility in tests).
            // PMAT-743: idempotent — never compound `.converted` on re-conversion.
            converted_output_path(source_path, target_ext)
        };

        // Use apr rosetta convert: apr rosetta convert <SOURCE> <TARGET>
        // Format is inferred from output file extension
        let output = Command::new(&self.binary)
            .arg("rosetta")
            .arg("convert")
            .arg(source_path)
            .arg(&target_path)
            .output()
            .map_err(Error::Io)?;

        if !output.status.success() {
            return Err(Error::Execution(format!(
                "Conversion failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(target_path)
    }

    /// Check if inference output is garbage (repetitive, too short, or empty).
    ///
    /// Used for cross-format conversion tests where quantization differences
    /// make text-level comparison meaningless. Instead, we verify both
    /// source and converted outputs are non-garbage.
    fn is_garbage_output(output: &str) -> bool {
        let trimmed = output.trim();
        // Empty or too short
        if trimmed.len() < 3 {
            return true;
        }
        // Check for excessive repetition (same char repeated)
        let chars: Vec<char> = trimmed.chars().collect();
        let unique_chars: std::collections::HashSet<char> = chars.iter().copied().collect();
        if unique_chars.len() < 3 {
            return true;
        }
        // Check for repeating patterns (trigram repetition)
        if chars.len() >= 9 {
            let trigrams: Vec<String> = chars.windows(3).map(|w| w.iter().collect()).collect();
            let unique_trigrams: std::collections::HashSet<&String> = trigrams.iter().collect();
            let repetition_ratio = 1.0 - (unique_trigrams.len() as f64 / trigrams.len() as f64);
            if repetition_ratio > 0.7 {
                return true;
            }
        }
        false
    }

    /// Compute difference between outputs
    fn compute_diff(&self, a: &str, b: &str) -> f64 {
        // Simple string comparison for now
        // In production, this would compare tensor values
        if a == b {
            0.0
        } else {
            // Compute character-level difference ratio
            let max_len = a.len().max(b.len());
            if max_len == 0 {
                return 0.0;
            }
            let matching: usize = a.chars().zip(b.chars()).filter(|(ca, cb)| ca == cb).count();
            1.0 - (matching as f64 / max_len as f64)
        }
    }

    /// Find indices where outputs differ (including length differences)
    fn find_diff_indices(&self, a: &str, b: &str) -> Vec<usize> {
        use std::iter;
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let max_len = a_chars.len().max(b_chars.len());
        a_chars
            .iter()
            .copied()
            .chain(iter::repeat_n('\0', max_len.saturating_sub(a_chars.len())))
            .zip(
                b_chars
                    .iter()
                    .copied()
                    .chain(iter::repeat_n('\0', max_len.saturating_sub(b_chars.len()))),
            )
            .enumerate()
            .filter(|(_, (ca, cb))| ca != cb)
            .map(|(i, _)| i)
            .collect()
    }

    /// Hash output for evidence
    fn hash_output(output: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        output.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}
