
/// Semantic conversion test that detects embedding/weight bugs (GH-187)
///
/// This test compares actual inference output between formats to detect
/// the class of bugs that have occurred 50+ times.
#[derive(Debug, Clone)]
pub struct SemanticConversionTest {
    /// Source format (SafeTensors as ground truth per spec 7.4)
    pub source_format: Format,
    /// Target format to test
    pub target_format: Format,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

/// Construction, execution, and bug classification for semantic conversion tests
impl SemanticConversionTest {
    /// Create a new semantic conversion test
    #[must_use]
    pub fn new(source: Format, target: Format, backend: Backend, model_id: ModelId) -> Self {
        Self {
            source_format: source,
            target_format: target,
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute the semantic test and classify any bug found
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<SemanticTestResult> {
        // Run inference on source (SafeTensors - ground truth per spec 7.4)
        let source_output = self.run_inference(model_path)?;

        // Convert to target format
        let converted_path = self.convert_model(model_path)?;

        // Run inference on converted model
        let target_output = self.run_inference(&converted_path)?;

        // Check for stderr containing tokenizer error
        let has_tokenizer_error = target_output.stderr.contains("PMAT-172")
            || target_output.stderr.contains("missing embedded tokenizer");

        // Classify the failure type
        let bug_type = self.classify_bug(
            &source_output.stdout,
            &target_output.stdout,
            has_tokenizer_error,
        );

        if let Some(bug) = bug_type {
            Ok(SemanticTestResult::Falsified {
                bug_type: bug,
                source_output: source_output.stdout,
                target_output: target_output.stdout,
                stderr: target_output.stderr,
            })
        } else {
            Ok(SemanticTestResult::Corroborated {
                source_output: source_output.stdout,
                target_output: target_output.stdout,
            })
        }
    }

    /// Classify the bug type based on output patterns
    fn classify_bug(
        &self,
        source: &str,
        target: &str,
        has_tokenizer_error: bool,
    ) -> Option<ConversionBugType> {
        // Check for tokenizer missing
        if has_tokenizer_error {
            return Some(ConversionBugType::TokenizerMissing);
        }

        // Check for garbage output patterns
        let has_garbage = GARBAGE_PATTERNS.iter().any(|p| target.contains(p));
        let source_has_expected = ARITHMETIC_EXPECTED.iter().any(|p| source.contains(p));
        let target_has_expected = ARITHMETIC_EXPECTED.iter().any(|p| target.contains(p));

        // Source produces correct answer but target produces garbage
        if source_has_expected && has_garbage {
            return Some(ConversionBugType::EmbeddingTransposition);
        }

        // Source correct, target wrong (but not garbage)
        if source_has_expected && !target_has_expected && !target.is_empty() {
            return Some(ConversionBugType::SemanticDrift);
        }

        // Target is empty or all whitespace
        if target.trim().is_empty() && !source.trim().is_empty() {
            return Some(ConversionBugType::WeightCorruption);
        }

        // Outputs match - no defect detected
        if source.trim() == target.trim() {
            return None;
        }

        // Outputs differ but no clear pattern
        Some(ConversionBugType::Unknown)
    }

    /// Run inference and capture both stdout and stderr
    fn run_inference(&self, model_path: &Path) -> Result<InferenceOutput> {
        let backend_flag = match self.backend {
            Backend::Cpu => vec!["--no-gpu".to_string()],
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

        Ok(InferenceOutput {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    /// Convert model to target format
    fn convert_model(&self, source_path: &Path) -> Result<std::path::PathBuf> {
        let target_ext = match self.target_format {
            Format::Gguf => "gguf",
            Format::SafeTensors => "safetensors",
            Format::Apr => "apr",
        };

        let target_path = source_path.with_extension(format!("semantic_test.{target_ext}"));

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
}

/// Output from inference command
#[derive(Debug, Clone)]
struct InferenceOutput {
    /// Standard output from the inference process
    stdout: String,
    /// Standard error from the inference process
    stderr: String,
    /// Process exit code
    #[allow(dead_code)]
    exit_code: i32,
}

/// Result of semantic conversion test
#[derive(Debug, Clone)]
pub enum SemanticTestResult {
    /// Conversion preserved semantics
    Corroborated {
        /// Source model output
        source_output: String,
        /// Target model output
        target_output: String,
    },
    /// Conversion introduced semantic errors
    Falsified {
        /// Classified bug type
        bug_type: ConversionBugType,
        /// Source model output (ground truth)
        source_output: String,
        /// Target model output (buggy)
        target_output: String,
        /// Stderr from target inference
        stderr: String,
    },
}

/// Query methods for semantic test outcomes
impl SemanticTestResult {
    /// Check if test passed
    #[must_use]
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Corroborated { .. })
    }

    /// Get the bug type if test failed
    #[must_use]
    pub fn bug_type(&self) -> Option<ConversionBugType> {
        match self {
            Self::Falsified { bug_type, .. } => Some(*bug_type),
            Self::Corroborated { .. } => None,
        }
    }
}

/// Generate all conversion test pairs
#[must_use]
pub fn all_conversion_pairs() -> Vec<(Format, Format)> {
    vec![
        (Format::Gguf, Format::Apr),
        (Format::Apr, Format::Gguf),
        (Format::Gguf, Format::SafeTensors),
        (Format::SafeTensors, Format::Gguf),
        (Format::Apr, Format::SafeTensors),
        (Format::SafeTensors, Format::Apr),
    ]
}

/// Generate all backends to test
#[must_use]
pub fn all_backends() -> Vec<Backend> {
    vec![Backend::Cpu, Backend::Gpu]
    // WASM/WGPU would be added here when supported
}

/// Generate all conversion tests for a model
#[must_use]
pub fn generate_conversion_tests(model_id: &ModelId) -> Vec<ConversionTest> {
    let mut tests = Vec::new();

    for (source, target) in all_conversion_pairs() {
        for backend in all_backends() {
            tests.push(ConversionTest::new(
                source,
                target,
                backend,
                model_id.clone(),
            ));
        }
    }

    tests
}

/// Round-trip conversion test
#[derive(Debug, Clone)]
pub struct RoundTripTest {
    /// Formats to chain through
    pub formats: Vec<Format>,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

/// Construction and execution for round-trip conversion tests
impl RoundTripTest {
    /// Create a new round-trip test
    #[must_use]
    pub fn new(formats: Vec<Format>, backend: Backend, model_id: ModelId) -> Self {
        Self {
            formats,
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute round-trip conversion test
    ///
    /// # Errors
    ///
    /// Returns an error if any conversion fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve directory to actual model file for starting format
        let resolved_path = resolve_model_path(model_path, self.formats[0])?;

        // Get original output
        let original_output = run_inference_simple(&resolved_path, self.backend, &self.binary)?;

        // Convert through chain
        let mut current_path = resolved_path;
        for i in 0..self.formats.len() {
            let next_format = self.formats[(i + 1) % self.formats.len()];
            current_path = convert_to_format(&current_path, next_format, &self.binary)?;
        }

        // Get final output
        let final_output = run_inference_simple(&current_path, self.backend, &self.binary)?;

        // Compare: round-trip through different formats involves quantization,
        // so text-level identity is not expected. Check non-garbage instead.
        let has_cross_format = self.formats.windows(2).any(|w| w[0] != w[1]);
        let passes = if has_cross_format {
            !ConversionTest::is_garbage_output(&original_output)
                && !ConversionTest::is_garbage_output(&final_output)
        } else {
            original_output == final_output
        };

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: self.formats[0],
                target_format: self.formats[0],
                backend: self.backend,
                max_diff: 0.0,
            })
        } else {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-RT-001".to_string(),
                reason: "Round-trip conversion produced different output".to_string(),
                evidence: ConversionEvidence {
                    source_hash: ConversionTest::hash_output(&original_output),
                    converted_hash: ConversionTest::hash_output(&final_output),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: self.formats[0],
                    target_format: self.formats[0],
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }
}

/// Idempotency test (MR-IDEM): convert A→B twice from same source, compare outputs
///
/// Detects non-deterministic conversion bugs. If converting the same model twice
/// produces different outputs, the converter has internal state leaks.
#[derive(Debug, Clone)]
pub struct IdempotencyTest {
    /// First format in chain
    pub format_a: Format,
    /// Second format in chain
    pub format_b: Format,
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}
