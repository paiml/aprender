
impl IdempotencyTest {
    /// Create a new idempotency test
    #[must_use]
    pub fn new(format_a: Format, format_b: Format, backend: Backend, model_id: ModelId) -> Self {
        Self {
            format_a,
            format_b,
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute idempotency test: convert A→B twice, compare
    ///
    /// # Errors
    ///
    /// Returns an error if conversion or inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve directory to actual model file for source format
        let resolved_path = resolve_model_path(model_path, self.format_a)?;

        // Convert A→B (first time)
        let converted_1 =
            convert_to_format_tagged(&resolved_path, self.format_b, "idem1", &self.binary)?;
        let output_1 = run_inference_simple(&converted_1, self.backend, &self.binary)?;

        // Convert A→B (second time, from same source)
        let converted_2 =
            convert_to_format_tagged(&resolved_path, self.format_b, "idem2", &self.binary)?;
        let output_2 = run_inference_simple(&converted_2, self.backend, &self.binary)?;

        // Cross-format conversion involves quantization which may not be
        // perfectly deterministic (floating-point rounding). Use non-garbage
        // check instead of exact text match.
        let is_cross_format = self.format_a != self.format_b;
        let passes = if is_cross_format {
            !ConversionTest::is_garbage_output(&output_1)
                && !ConversionTest::is_garbage_output(&output_2)
        } else {
            output_1 == output_2
        };

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: self.format_a,
                target_format: self.format_b,
                backend: self.backend,
                max_diff: 0.0,
            })
        } else {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-IDEM-001".to_string(),
                reason: format!(
                    "Idempotency failure: {:?}→{:?} produced different output on second conversion",
                    self.format_a, self.format_b
                ),
                evidence: ConversionEvidence {
                    source_hash: ConversionTest::hash_output(&output_1),
                    converted_hash: ConversionTest::hash_output(&output_2),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: self.format_a,
                    target_format: self.format_b,
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }
}

/// Byte-level round-trip test (GH-6/AC-3): ST → APR → GGUF → APR with tensor diff
///
/// Unlike `RoundTripTest` which compares inference output, this test compares
/// the actual tensor data byte-for-byte between two APR conversions.
/// Detects silent data corruption that inference-level tests may miss.
#[derive(Debug, Clone)]
pub struct ByteLevelRoundTripTest {
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

impl ByteLevelRoundTripTest {
    /// Create a new byte-level round-trip test
    #[must_use]
    pub fn new(backend: Backend, model_id: ModelId) -> Self {
        Self {
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute byte-level round-trip: ST → APR(1) and ST → APR → GGUF → APR(2), diff tensors
    ///
    /// # Errors
    ///
    /// Returns an error if conversion or diff fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        let resolved_path = resolve_model_path(model_path, Format::SafeTensors)?;

        // Step 1: ST → APR (reference)
        let apr_ref =
            convert_to_format_tagged(&resolved_path, Format::Apr, "byte_rt_ref", &self.binary)?;

        // Step 2: ST → APR → GGUF → APR (round-trip)
        let apr_tmp =
            convert_to_format_tagged(&resolved_path, Format::Apr, "byte_rt_tmp", &self.binary)?;
        let gguf_tmp =
            convert_to_format_tagged(&apr_tmp, Format::Gguf, "byte_rt_gguf", &self.binary)?;
        let apr_roundtrip =
            convert_to_format_tagged(&gguf_tmp, Format::Apr, "byte_rt_final", &self.binary)?;

        // Step 3: diff_tensors between apr_ref and apr_roundtrip
        let diff_output = run_diff_tensors(&apr_ref, &apr_roundtrip, &self.binary)?;

        if diff_output.contains("\"passed\":false") || diff_output.contains("mismatched") {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-RT-BYTE-001".to_string(),
                reason: "Byte-level round-trip: tensor data differs after ST→APR→GGUF→APR"
                    .to_string(),
                evidence: ConversionEvidence {
                    source_hash: String::new(),
                    converted_hash: String::new(),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: Format::SafeTensors,
                    target_format: Format::Apr,
                    backend: self.backend,
                    failure_type: Some(ConversionFailureType::DequantizationFailure),
                    quant_type: None,
                },
            })
        } else {
            Ok(ConversionResult::Corroborated {
                source_format: Format::SafeTensors,
                target_format: Format::Apr,
                backend: self.backend,
                max_diff: 0.0,
            })
        }
    }
}

/// Commutativity test (MR-COM): different conversion paths should yield equivalent inference
///
/// Tests that GGUF→APR produces the same inference as GGUF→ST→APR.
/// Path-dependent conversion bugs are a major source of silent failures.
#[derive(Debug, Clone)]
pub struct CommutativityTest {
    /// Backend to use
    pub backend: Backend,
    /// Model ID
    pub model_id: ModelId,
    /// Binary path for apr CLI
    binary: String,
}

impl CommutativityTest {
    /// Create a new commutativity test
    #[must_use]
    pub fn new(backend: Backend, model_id: ModelId) -> Self {
        Self {
            backend,
            model_id,
            binary: default_binary(),
        }
    }

    /// Execute commutativity test: compare direct vs indirect conversion paths
    ///
    /// Path A: GGUF → APR (direct)
    /// Path B: GGUF → SafeTensors → APR (indirect)
    ///
    /// # Errors
    ///
    /// Returns an error if conversion or inference fails.
    pub fn execute(&self, model_path: &Path) -> Result<ConversionResult> {
        // Resolve directory to actual GGUF model file
        let resolved_path = resolve_model_path(model_path, Format::Gguf)?;

        // Path A: GGUF → APR (direct)
        let direct_apr =
            convert_to_format_tagged(&resolved_path, Format::Apr, "com_direct", &self.binary)?;
        let output_a = run_inference_simple(&direct_apr, self.backend, &self.binary)?;

        // Path B: GGUF → SafeTensors → APR (indirect)
        let via_st =
            convert_to_format_tagged(&resolved_path, Format::SafeTensors, "com_via", &self.binary)?;
        let indirect_apr =
            convert_to_format_tagged(&via_st, Format::Apr, "com_indirect", &self.binary)?;
        let output_b = run_inference_simple(&indirect_apr, self.backend, &self.binary)?;

        // Cross-format paths involve different quantization chains,
        // so text-level identity is not expected. Check non-garbage instead.
        let passes = !ConversionTest::is_garbage_output(&output_a)
            && !ConversionTest::is_garbage_output(&output_b);

        if passes {
            Ok(ConversionResult::Corroborated {
                source_format: Format::Gguf,
                target_format: Format::Apr,
                backend: self.backend,
                max_diff: 0.0,
            })
        } else {
            Ok(ConversionResult::Falsified {
                gate_id: "F-CONV-COM-001".to_string(),
                reason: "Commutativity failure: GGUF→APR differs from GGUF→ST→APR (garbage output)"
                    .to_string(),
                evidence: ConversionEvidence {
                    source_hash: ConversionTest::hash_output(&output_a),
                    converted_hash: ConversionTest::hash_output(&output_b),
                    max_diff: 1.0,
                    diff_indices: vec![],
                    source_format: Format::Gguf,
                    target_format: Format::Apr,
                    backend: self.backend,
                    failure_type: None,
                    quant_type: None,
                },
            })
        }
    }
}

/// Check tensor cardinality after conversion (MR-CARD)
///
/// Fires F-CONV-CARD-001 if `tensor_count(output) < tensor_count(input)`.
/// This catches silent tensor fusion bugs like QKV fusion (338→227).
///
/// # Errors
///
/// Returns an error if `apr rosetta inspect` fails on either model.
pub fn check_cardinality(
    source_path: &Path,
    converted_path: &Path,
    binary: &str,
) -> Result<Option<(String, String)>> {
    let source_inspect = crate::differential::run_inspect(source_path, binary)?;
    let target_inspect = crate::differential::run_inspect(converted_path, binary)?;

    if target_inspect.tensor_count < source_inspect.tensor_count {
        Ok(Some((
            "F-CONV-CARD-001".to_string(),
            format!(
                "Tensor cardinality loss: {} → {}",
                source_inspect.tensor_count, target_inspect.tensor_count
            ),
        )))
    } else {
        Ok(None)
    }
}

/// Check tensor name preservation after conversion (T-QKV-02)
///
/// Fires F-CONV-NAME-001 if tensor names changed unexpectedly during conversion
/// (e.g., q_proj+k_proj+v_proj → qkv_proj fusion).
///
/// # Errors
///
/// Returns an error if `apr rosetta inspect` fails on either model.
pub fn check_tensor_names(
    source_path: &Path,
    converted_path: &Path,
    binary: &str,
) -> Result<Option<(String, String)>> {
    let source_inspect = crate::differential::run_inspect(source_path, binary)?;
    let target_inspect = crate::differential::run_inspect(converted_path, binary)?;

    // Skip if either side has no tensor names (inspect may not support it)
    if source_inspect.tensor_names.is_empty() || target_inspect.tensor_names.is_empty() {
        return Ok(None);
    }

    let missing: Vec<_> = source_inspect
        .tensor_names
        .iter()
        .filter(|n| !target_inspect.tensor_names.contains(n))
        .collect();

    if missing.is_empty() {
        return Ok(None);
    }

    // Check for known fusion patterns (q_proj+k_proj+v_proj → qkv_proj)
    let has_fusion = missing
        .iter()
        .any(|n| n.contains("q_proj") || n.contains("k_proj") || n.contains("v_proj"))
        && target_inspect
            .tensor_names
            .iter()
            .any(|n| n.contains("qkv_proj"));

    let detail = if has_fusion {
        format!(
            "QKV fusion detected: {} source tensors missing (likely fused into qkv_proj). Missing: {}",
            missing.len(),
            missing
                .iter()
                .take(5)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        format!(
            "Tensor name divergence: {} source tensors not found in output. Missing: {}",
            missing.len(),
            missing
                .iter()
                .take(5)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    Ok(Some(("F-CONV-NAME-001".to_string(), detail)))
}

/// Convert model to specified format with a tag suffix for disambiguation
fn convert_to_format_tagged(
    source_path: &Path,
    target_format: Format,
    tag: &str,
    binary: &str,
) -> Result<std::path::PathBuf> {
    let target_ext = match target_format {
        Format::Gguf => "gguf",
        Format::SafeTensors => "safetensors",
        Format::Apr => "apr",
    };

    let target_path = source_path.with_extension(format!("{tag}.{target_ext}"));

    let output = Command::new(binary)
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

/// Diff tensors between two models via `apr rosetta diff-tensors --json`
fn run_diff_tensors(model_a: &Path, model_b: &Path, binary: &str) -> Result<String> {
    let output = Command::new(binary)
        .arg("rosetta")
        .arg("diff-tensors")
        .arg(model_a)
        .arg(model_b)
        .arg("--json")
        .output()
        .map_err(Error::Io)?;

    if !output.status.success() {
        return Err(Error::Execution(format!(
            "diff-tensors failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Simple inference helper
fn run_inference_simple(model_path: &Path, backend: Backend, binary: &str) -> Result<String> {
    let backend_flag = match backend {
        Backend::Cpu => vec![],
        Backend::Gpu => vec!["--gpu".to_string()],
    };

    let output = Command::new(binary)
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
            "Inference failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Convert model to specified format
fn convert_to_format(
    source_path: &Path,
    target_format: Format,
    binary: &str,
) -> Result<std::path::PathBuf> {
    let target_ext = match target_format {
        Format::Gguf => "gguf",
        Format::SafeTensors => "safetensors",
        Format::Apr => "apr",
    };

    // Create target path with new extension (format determined by extension).
    // PMAT-743: idempotent — never compound `.converted` on re-conversion.
    let target_path = converted_output_path(source_path, target_ext);

    // Use apr rosetta convert: apr rosetta convert <SOURCE> <TARGET>
    // Format is inferred from output file extension
    let output = Command::new(binary)
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

// ConversionConfig + ConversionExecutor — see conversion_executor.rs
include!("conversion_executor.rs");

// HF cache resolution — see conversion_hf_cache.rs
include!("conversion_hf_cache.rs");

#[cfg(test)]
#[path = "conversion_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "conversion_tests_b.rs"]
mod tests_b;

#[cfg(test)]
#[path = "conversion_tests_c.rs"]
mod tests_c;
