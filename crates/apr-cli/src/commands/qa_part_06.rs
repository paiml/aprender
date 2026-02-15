
/// Gate 5: Cross-Format Parity Test (F-QUAL-032)
///
/// Compares argmax output between GGUF and SafeTensors for the same model.
/// P0-QA-001: Auto-discover SafeTensors model for format parity gate.
///
/// Search strategy (in order):
/// 1. Sibling directory of GGUF file (same name but .safetensors)
/// 2. Sibling subdirectories containing .safetensors files
/// 3. HuggingFace cache (~/.cache/huggingface/hub/models--*)
///
/// Returns the first found SafeTensors path, or None.
fn auto_discover_safetensors(gguf_path: &Path) -> Option<std::path::PathBuf> {
    let parent = gguf_path.parent()?;
    let stem = gguf_path.file_stem()?.to_str()?;

    // Strategy 1: Sibling file with .safetensors extension
    let sibling = parent.join(format!("{stem}.safetensors"));
    if sibling.exists() {
        return Some(sibling);
    }

    // Strategy 2: Sibling subdirectory containing model.safetensors
    // e.g., /home/noah/models/qwen2.5-coder-7b-instruct-q4k.gguf
    //     → /home/noah/models/qwen2.5-coder-7b-instruct/model.safetensors
    // Strip quantization suffixes to find base model name
    let base_name = stem
        .trim_end_matches("-q4k")
        .trim_end_matches("-q4_k_m")
        .trim_end_matches("-q6k")
        .trim_end_matches("-q6_k")
        .trim_end_matches("-q5k")
        .trim_end_matches("-q5_k_m")
        .trim_end_matches("-q8_0")
        .trim_end_matches("-f16")
        .trim_end_matches("-f32");
    let subdir = parent.join(base_name);
    if subdir.is_dir() {
        // Check for model.safetensors (single file) or sharded index
        let single = subdir.join("model.safetensors");
        if single.exists() {
            return Some(single);
        }
        let index = subdir.join("model.safetensors.index.json");
        if index.exists() {
            // Sharded model — check if actual shard files exist
            if let Ok(entries) = std::fs::read_dir(&subdir) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.ends_with(".safetensors") && name_str != "model.safetensors" {
                        return Some(entry.path());
                    }
                }
            }
        }
    }

    // Strategy 3: HuggingFace cache
    let hf_cache = dirs::home_dir()?.join(".cache/huggingface/hub");
    if hf_cache.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&hf_cache) {
            for entry in entries.flatten() {
                let dir_name = entry.file_name();
                let dir_str = dir_name.to_string_lossy();
                if !dir_str.starts_with("models--") {
                    continue;
                }
                // Match model name case-insensitively
                let model_part = dir_str
                    .trim_start_matches("models--")
                    .replace("--", "/")
                    .to_lowercase();
                if !model_part.contains(&base_name.to_lowercase()) {
                    continue;
                }
                // Look for snapshots/*/model.safetensors
                let snapshots = entry.path().join("snapshots");
                if let Ok(snaps) = std::fs::read_dir(&snapshots) {
                    for snap in snaps.flatten() {
                        let single = snap.path().join("model.safetensors");
                        if single.exists() {
                            return Some(single);
                        }
                        // Check for sharded .safetensors files
                        if let Ok(files) = std::fs::read_dir(snap.path()) {
                            for f in files.flatten() {
                                let fname = f.file_name();
                                if fname.to_string_lossy().ends_with(".safetensors") {
                                    return Some(f.path());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Invariant: argmax(forward_gguf(M, tokens)) == argmax(forward_safetensors(M, tokens))
///
/// This is the cornerstone of the architecture's logical validity - it demonstrates
/// that independent binary format readers can reach the same logical conclusion.
fn run_format_parity_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running cross-format parity test...".yellow());
    }

    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{GGUFModel, MappedGGUFModel, OwnedQuantizedModel};
        use realizar::safetensors_infer::SafetensorsToAprConverter;

        // P0-QA-001: Never skip — auto-discover or FAIL with actionable message
        let discovered_path;
        let safetensors_path = if let Some(p) = &config.safetensors_path {
            p
        } else {
            match auto_discover_safetensors(path) {
                Some(p) => {
                    if !config.json {
                        println!(
                            "  {} Auto-discovered SafeTensors: {}",
                            "INFO".cyan(),
                            p.display()
                        );
                    }
                    discovered_path = p;
                    &discovered_path
                }
                None => {
                    return Ok(GateResult::failed(
                        "format_parity",
                        "No SafeTensors found. Provide --safetensors-path or download: \
                         huggingface-cli download <model> --include '*.safetensors'",
                        None,
                        None,
                        start.elapsed(),
                    ));
                }
            }
        };

        // Verify GGUF model
        let gguf_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read GGUF: {e}")))?;

        let gguf_format = detect_format(&gguf_bytes[..8.min(gguf_bytes.len())]).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to detect GGUF format: {e}"))
        })?;

        if gguf_format != ModelFormat::Gguf {
            return Ok(GateResult::failed(
                "format_parity",
                "Primary model must be GGUF format for cross-format parity test",
                None,
                None,
                start.elapsed(),
            ));
        }

        // Verify SafeTensors model exists
        if !safetensors_path.exists() {
            return Ok(GateResult::failed(
                "format_parity",
                &format!(
                    "SafeTensors not found: {}. Download with: huggingface-cli download <model> --include '*.safetensors'",
                    safetensors_path.display()
                ),
                None,
                None,
                start.elapsed(),
            ));
        }

        // Load GGUF model and get tokenizer
        let gguf = GGUFModel::from_bytes(&gguf_bytes)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

        // Test prompt - use simple arithmetic for deterministic output
        let prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
        let prompt_tokens: Vec<u32> = gguf.encode(prompt).unwrap_or_else(|| vec![151643, 9707]);

        // Run GGUF forward pass to get logits
        let gguf_logits = {
            let mapped = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("GGUF map failed: {e}")))?;
            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| CliError::ValidationFailed(format!("GGUF model failed: {e}")))?;
            model
                .forward(&prompt_tokens)
                .map_err(|e| CliError::ValidationFailed(format!("GGUF forward failed: {e}")))?
        };

        // Run SafeTensors forward pass to get logits
        let st_logits = {
            let transformer =
                SafetensorsToAprConverter::convert(safetensors_path).map_err(|e| {
                    CliError::ValidationFailed(format!("SafeTensors convert failed: {e}"))
                })?;
            transformer.forward(&prompt_tokens).map_err(|e| {
                CliError::ValidationFailed(format!("SafeTensors forward failed: {e}"))
            })?
        };

        let duration = start.elapsed();

        // Get argmax from logits
        let gguf_argmax = gguf_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);

        let st_argmax = st_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32);

        match (gguf_argmax, st_argmax) {
            (Some(gguf_token), Some(st_token)) => {
                if gguf_token == st_token {
                    Ok(GateResult::passed(
                        "format_parity",
                        &format!(
                            "GGUF argmax={} == SafeTensors argmax={} (Cross-format parity VERIFIED)",
                            gguf_token, st_token
                        ),
                        Some(gguf_token as f64),
                        Some(st_token as f64),
                        duration,
                    ))
                } else {
                    Ok(GateResult::failed(
                        "format_parity",
                        &format!(
                            "GGUF argmax={} != SafeTensors argmax={} (Cross-format parity BROKEN)",
                            gguf_token, st_token
                        ),
                        Some(gguf_token as f64),
                        Some(st_token as f64),
                        duration,
                    ))
                }
            }
            _ => Ok(GateResult::failed(
                "format_parity",
                "Failed to get argmax from one or both formats",
                None,
                None,
                duration,
            )),
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "format_parity",
            "Requires 'inference' feature",
        ))
    }
}

/// Check if Ollama is available by pinging the API
fn check_ollama_available() -> bool {
    // Try to connect to Ollama API
    std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://localhost:11434/api/tags",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "200")
        .unwrap_or(false)
}

/// Detect Ollama model name from GGUF filename (BUG-QA-001 fix)
/// Matches model size to avoid unfair comparison (e.g., 0.5B APR vs 1.5B Ollama)
/// Detect the matching Ollama model tag for fair like-for-like comparison.
///
/// For quantized GGUF: uses the default Ollama tag (Q4_K_M quantized).
/// For F32/F16 (SafeTensors, APR): uses the `-instruct-fp16` Ollama tag
/// so we compare unquantized vs unquantized.
///
/// Detects model size from filename, or falls back to file size heuristic
/// for hash-named pacha-cached files.
fn detect_ollama_model_from_path(path: &Path) -> String {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let filename_lower = filename.to_lowercase();

    // Detect model size from filename first
    let size = if filename_lower.contains("0.5b") || filename_lower.contains("-0_5b") {
        "0.5b"
    } else if filename_lower.contains("1.5b") || filename_lower.contains("-1_5b") {
        "1.5b"
    } else if filename_lower.contains("3b") || filename_lower.contains("-3b") {
        "3b"
    } else if filename_lower.contains("7b") || filename_lower.contains("-7b") {
        "7b"
    } else if filename_lower.contains("14b") || filename_lower.contains("-14b") {
        "14b"
    } else if filename_lower.contains("32b") || filename_lower.contains("-32b") {
        "32b"
    } else {
        // Fallback: estimate from file size (for hash-named pacha-cached files).
        // GGUF Q4_K sizes: 0.5B≈400MB, 1.5B≈1GB, 3B≈2GB, 7B≈4.5GB
        match std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) {
            0..=800_000_000 => "0.5b",
            800_000_001..=2_000_000_000 => "1.5b",
            2_000_000_001..=4_000_000_000 => "3b",
            _ => "7b",
        }
    };

    // Default Ollama tag uses Q4_K_M — fair comparison for quantized GGUF
    format!("qwen2.5-coder:{size}")
}

/// Measure Ollama throughput for comparison (GGUF only)
/// BUG-QA-002 FIX: Use Ollama's eval_duration instead of wall clock time
/// (wall clock includes HTTP overhead, making Ollama look 10x slower)
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json! macro internally uses unwrap()
fn measure_ollama_throughput(path: &Path, config: &QaConfig) -> Result<f64> {
    // Use curl to send a request to Ollama
    let prompt = "Write a hello world program in Python:";
    // BUG-QA-001 FIX: Match Ollama model to APR model size for fair comparison
    let model = detect_ollama_model_from_path(path);

    // Match parity gate: use 128 tokens minimum to amortize prefill overhead
    let parity_max_tokens = config.max_tokens.max(128);
    let request_body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_predict": parity_max_tokens,
            "temperature": 0.0
        }
    });

    let mut total_tokens = 0usize;
    let mut total_duration_ns = 0u64;

    for _ in 0..config.iterations.min(3) {
        let output = std::process::Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "http://localhost:11434/api/generate",
                "-H",
                "Content-Type: application/json",
                "-d",
                &request_body.to_string(),
            ])
            .output();

        if let Ok(output) = output {
            if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                // BUG-QA-002 FIX: Use eval_count and eval_duration from Ollama response
                // This measures actual inference time, not HTTP overhead
                if let (Some(eval_count), Some(eval_duration)) = (
                    response
                        .get("eval_count")
                        .and_then(serde_json::Value::as_u64),
                    response
                        .get("eval_duration")
                        .and_then(serde_json::Value::as_u64),
                ) {
                    total_tokens += eval_count as usize;
                    total_duration_ns += eval_duration;
                }
            }
        }
    }

    if total_tokens == 0 || total_duration_ns == 0 {
        return Ok(0.0);
    }

    // Convert nanoseconds to seconds for tok/s calculation
    let duration_s = total_duration_ns as f64 / 1_000_000_000.0;
    Ok(total_tokens as f64 / duration_s)
}

/// Print a gate result to the terminal
fn print_gate_result(result: &GateResult) {
    let badge = if result.skipped {
        output::badge_skip("SKIP")
    } else if result.passed {
        output::badge_pass("PASS")
    } else {
        output::badge_fail("FAIL")
    };

    let name = gate_display_name(&result.name);

    println!(
        "  {} {} {}",
        badge,
        name.white().bold(),
        result.message.dimmed()
    );

    if !result.skipped {
        println!(
            "       {}",
            output::duration_fmt(result.duration_ms).dimmed()
        );
    }
    println!();
}
