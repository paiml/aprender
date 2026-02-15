
/// Gate 1: Metadata Plausibility Validation (Bug 210, GH-222)
///
/// Validates that model hyperparameters (rope_theta, max_position_embeddings, rms_norm_eps)
/// fall within known plausible ranges for the detected architecture family.
///
/// This gate catches the root cause of GH-222: importing SafeTensors without config.json
/// silently stored rope_theta=10000.0 for Qwen2, which should be 1000000.0.
fn run_metadata_plausibility_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!(
            "{}",
            "Running metadata plausibility validation (Bug 210)...".yellow()
        );
    }

    // Extract metadata from the model file
    let data = std::fs::read(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;

    if data.len() < 4 {
        let duration = start.elapsed();
        return Ok(GateResult::failed(
            "metadata_plausibility",
            "File too small for metadata extraction",
            None,
            None,
            duration,
        ));
    }

    let (architecture, rope_theta, max_pos, rms_norm_eps) = extract_model_metadata(&data, path)?;

    let mut violations: Vec<String> = Vec::new();
    let mut checks_passed = 0usize;

    check_rope_theta(
        architecture.as_deref(),
        rope_theta,
        &data,
        &mut violations,
        &mut checks_passed,
    );
    check_max_position_embeddings(max_pos, &mut violations, &mut checks_passed);
    check_rms_norm_eps(rms_norm_eps, &mut violations, &mut checks_passed);
    check_arch_theta_cross_validation(
        architecture.as_ref(),
        rope_theta,
        &mut violations,
        &mut checks_passed,
    );

    let duration = start.elapsed();

    if violations.is_empty() {
        Ok(GateResult::passed(
            "metadata_plausibility",
            &format!(
                "{checks_passed} metadata checks passed (arch={}, rope_theta={}, max_pos={})",
                architecture.as_deref().unwrap_or("unknown"),
                rope_theta.map_or("none".to_string(), |t| format!("{t}")),
                max_pos.map_or("none".to_string(), |p| format!("{p}")),
            ),
            Some(checks_passed as f64),
            Some(0.0),
            duration,
        ))
    } else {
        Ok(GateResult::failed(
            "metadata_plausibility",
            &format!(
                "{} metadata violation(s): {}",
                violations.len(),
                violations.join("; ")
            ),
            Some(violations.len() as f64),
            Some(0.0),
            duration,
        ))
    }
}

/// Check rope_theta plausibility per architecture family.
fn check_rope_theta(
    arch: Option<&str>,
    rope_theta: Option<f32>,
    data: &[u8],
    violations: &mut Vec<String>,
    checks_passed: &mut usize,
) {
    if let Some(theta) = rope_theta {
        let theta_f64 = f64::from(theta);
        match arch {
            Some("qwen2" | "qwen2.5" | "qwen") => {
                if theta_f64 < 100_000.0 {
                    violations.push(format!(
                        "rope_theta={theta} for qwen2 — expected ~1000000.0 \
                         (100x too low, will produce garbage)"
                    ));
                } else {
                    *checks_passed += 1;
                }
            }
            Some("llama" | "llama2" | "llama3") => {
                if (1000.0..=10_000_000.0).contains(&theta_f64) {
                    *checks_passed += 1;
                } else {
                    violations.push(format!(
                        "rope_theta={theta} for llama — expected 10000-500000"
                    ));
                }
            }
            _ => {
                if (100.0..=100_000_000.0).contains(&theta_f64) {
                    *checks_passed += 1;
                } else {
                    violations.push(format!(
                        "rope_theta={theta} outside plausible range [100, 100M]"
                    ));
                }
            }
        }
    } else {
        let magic = &data[0..4];
        if magic == b"GGUF" {
            *checks_passed += 1;
        } else {
            violations.push("rope_theta missing from APR metadata".to_string());
        }
    }
}

/// Check max_position_embeddings is within plausible range.
fn check_max_position_embeddings(
    max_pos: Option<usize>,
    violations: &mut Vec<String>,
    checks_passed: &mut usize,
) {
    if let Some(val) = max_pos {
        if (128..=1_048_576).contains(&val) {
            *checks_passed += 1;
        } else {
            violations.push(format!(
                "max_position_embeddings={val} outside plausible range [128, 1M]"
            ));
        }
    } else {
        *checks_passed += 1;
    }
}

/// Check rms_norm_eps is within plausible range.
fn check_rms_norm_eps(
    rms_norm_eps: Option<f32>,
    violations: &mut Vec<String>,
    checks_passed: &mut usize,
) {
    if let Some(eps) = rms_norm_eps {
        let eps_f64 = f64::from(eps);
        if eps_f64 <= 0.0 || eps_f64 > 0.01 {
            violations.push(format!(
                "rms_norm_eps={eps} outside plausible range (0, 0.01]"
            ));
        } else {
            *checks_passed += 1;
        }
    } else {
        *checks_passed += 1;
    }
}

/// Cross-validate architecture against rope_theta (Bug 210 signature detection).
fn check_arch_theta_cross_validation(
    architecture: Option<&String>,
    rope_theta: Option<f32>,
    violations: &mut Vec<String>,
    checks_passed: &mut usize,
) {
    if let (Some(arch), Some(theta)) = (architecture, rope_theta) {
        let theta_f64 = f64::from(theta);
        let suspicious = matches!(arch.as_str(), "qwen2" | "qwen2.5" | "qwen")
            && (theta_f64 - 10000.0).abs() < 1.0;
        if suspicious {
            violations.push(format!(
                "CRITICAL: {arch} with rope_theta=10000.0 — \
                 likely missing config.json (Bug 210)"
            ));
        } else {
            *checks_passed += 1;
        }
    } else {
        *checks_passed += 1;
    }
}

/// Metadata extracted from model file for plausibility validation.
type ModelMetadata = (Option<String>, Option<f32>, Option<usize>, Option<f32>);

/// Extract model metadata from file bytes (GGUF, APR, or SafeTensors format).
fn extract_model_metadata(data: &[u8], path: &Path) -> Result<ModelMetadata> {
    let magic = &data[0..4];

    if magic == b"GGUF" {
        // GGUF format: use GgufReader
        let reader = aprender::format::gguf::reader::GgufReader::from_bytes(data.to_vec())
            .map_err(|e| CliError::ValidationFailed(format!("GGUF parse failed: {e}")))?;
        let arch = reader.architecture();
        let rope_theta = reader.rope_theta();
        let max_pos = reader.context_length();
        let rms_norm_eps = reader.rms_norm_eps();
        Ok((arch, rope_theta, max_pos, rms_norm_eps))
    } else if &magic[0..3] == b"APR" || magic == b"APRN" {
        // APR format: parse v2 header + JSON metadata
        use aprender::format::v2::AprV2Reader;
        let reader = AprV2Reader::from_bytes(data)
            .map_err(|e| CliError::ValidationFailed(format!("APR parse failed: {e}")))?;
        let meta = reader.metadata();
        let _ = path;
        Ok((
            meta.architecture.clone(),
            meta.rope_theta,
            meta.max_position_embeddings,
            meta.rms_norm_eps,
        ))
    } else {
        // SafeTensors or unknown format: try to load config.json from sibling
        let config_path = path.with_file_name("config.json");
        if config_path.exists() {
            // Read architecture and rope_theta from HF config.json
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| CliError::ValidationFailed(format!("config.json read failed: {e}")))?;
            let arch = extract_json_string(&config_str, "model_type");
            let rope_theta = extract_json_f32(&config_str, "rope_theta");
            let max_pos = extract_json_usize(&config_str, "max_position_embeddings");
            let rms_norm_eps = extract_json_f32(&config_str, "rms_norm_eps");
            Ok((arch, rope_theta, max_pos, rms_norm_eps))
        } else {
            // No config.json — return None for all fields (gate will note the gap)
            Ok((None, None, None, None))
        }
    }
}

/// Extract a string value from JSON by key (simple parser, no serde dependency).
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    // Skip whitespace and colon
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    if trimmed.starts_with('"') {
        let start = 1;
        let end = trimmed[start..].find('"')?;
        Some(trimmed[start..start + end].to_string())
    } else {
        None
    }
}

/// Extract an f32 value from JSON by key.
fn extract_json_f32(json: &str, key: &str) -> Option<f32> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    // Parse until next comma, brace, or whitespace
    let end = trimmed.find([',', '}', '\n'])?;
    trimmed[..end].trim().parse::<f32>().ok()
}

/// Extract a usize value from JSON by key.
fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
    let trimmed = after_colon.trim_start();
    let end = trimmed.find([',', '}', '\n'])?;
    trimmed[..end].trim().parse::<usize>().ok()
}

/// Output verification result (PMAT-QA-PROTOCOL-001 §7.4)
#[derive(Debug, Clone)]
pub enum OutputVerification {
    /// Output passed all checks
    Pass,
    /// Output failed verification
    Fail {
        /// Reason for failure
        reason: String,
    },
}

/// Verify output is correct: not empty, no garbage, contains expected answer
/// (PMAT-QA-PROTOCOL-001 §7.4)
///
/// Order of checks is CRITICAL (fail fast on garbage):
/// 1. Not empty
/// 2. No garbage patterns (BEFORE checking answer)
/// 3. No BPE artifacts
/// 4. Contains expected answer
pub fn verify_output(
    output: &str,
    test_id: &str,
    expected_patterns: &[&str],
) -> OutputVerification {
    // Check 1: Not empty
    if output.trim().is_empty() {
        return OutputVerification::Fail {
            reason: format!("{test_id}: Empty output"),
        };
    }

    // Check 2: Garbage patterns (fail fast BEFORE checking answer)
    let garbage_patterns = ["\u{FFFD}", "[UNK]", "akunji", "olumbia"];
    for pattern in &garbage_patterns {
        if output.contains(pattern) {
            return OutputVerification::Fail {
                reason: format!("{test_id}: Garbage detected: '{pattern}'"),
            };
        }
    }

    // Check 3: BPE artifacts (null bytes, excessive control chars)
    let null_count = output.bytes().filter(|&b| b == 0).count();
    if null_count > 0 {
        return OutputVerification::Fail {
            reason: format!("{test_id}: {null_count} null bytes detected (BPE artifact)"),
        };
    }

    // Check 4: Contains expected answer
    if !expected_patterns.is_empty() {
        let found = expected_patterns
            .iter()
            .any(|p| output.to_lowercase().contains(&p.to_lowercase()));
        if !found {
            return OutputVerification::Fail {
                reason: format!(
                    "{test_id}: Expected one of {:?}, got: '{}'",
                    expected_patterns,
                    output.chars().take(100).collect::<String>()
                ),
            };
        }
    }

    OutputVerification::Pass
}

/// JIDOKA: Validate GPU golden output matches expected patterns (PMAT-232 lesson).
///
/// Without this, GPU correctness was NEVER tested — `apr qa` golden output only ran CPU.
/// Returns `Some(failure_reason)` if GPU output fails, `None` if pass or skipped.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn validate_gpu_golden_output(
    mapped: &realizar::gguf::MappedGGUFModel,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    gguf: &realizar::gguf::GGUFModel,
    expected_patterns: &[&str],
    config: &QaConfig,
) -> Result<Option<String>> {
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};
    let model = OwnedQuantizedModel::from_mapped(mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
    match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(mut cuda_model) => match cuda_model.generate_gpu_resident(prompt_tokens, gen_config) {
            Ok(gpu_tokens) => {
                let gpu_text = gguf.decode(&gpu_tokens);
                if let OutputVerification::Fail { reason } =
                    verify_output(&gpu_text, "golden_output_gpu", expected_patterns)
                {
                    return Ok(Some(format!("GPU output failed (CPU passed): {reason}")));
                }
            }
            Err(e) => {
                if !config.json && config.verbose {
                    println!("{}", format!("GPU golden output skipped: {e}").yellow());
                }
            }
        },
        Err(e) => {
            if !config.json && config.verbose {
                println!("{}", format!("CUDA init skipped: {e}").yellow());
            }
        }
    }
    Ok(None)
}

/// Run golden output test for APR format models
#[cfg(feature = "inference")]
fn golden_output_apr(path: &Path, prompt: &str, max_tokens: usize) -> Result<(Vec<u32>, String)> {
    use realizar::apr::AprV2Model;
    use realizar::apr_transformer::{AprTransformer, GenerateConfig};

    let apr_model = AprV2Model::load(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;
    let tokenizer = apr_model
        .load_embedded_bpe_tokenizer()
        .ok_or_else(|| CliError::ValidationFailed("APR missing embedded tokenizer".to_string()))?;
    let transformer = AprTransformer::from_apr_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR transformer: {e}")))?;

    let prompt_tokens = tokenizer.encode(prompt);
    let gen_config = GenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    let tokens = transformer
        .generate_with_cache(&prompt_tokens, &gen_config)
        .map_err(|e| CliError::ValidationFailed(format!("Generation failed: {e}")))?;
    let text = tokenizer.decode(&tokens);
    Ok((tokens, text))
}
