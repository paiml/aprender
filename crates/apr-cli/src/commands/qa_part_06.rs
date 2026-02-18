
/// Strip quantization suffixes from a GGUF stem to find the base model name.
fn strip_quant_suffix(stem: &str) -> &str {
    stem.trim_end_matches("-q4k")
        .trim_end_matches("-q4_k_m")
        .trim_end_matches("-q6k")
        .trim_end_matches("-q6_k")
        .trim_end_matches("-q5k")
        .trim_end_matches("-q5_k_m")
        .trim_end_matches("-q8_0")
        .trim_end_matches("-f16")
        .trim_end_matches("-f32")
}

/// Strategy 2 helper: find a SafeTensors entry point in a sharded model directory.
/// Returns the first shard (sorted) for sharded models. The format parity gate
/// handles converter failures for sharded models gracefully.
fn find_sharded_safetensors(dir: &Path) -> Option<std::path::PathBuf> {
    let index = dir.join("model.safetensors.index.json");
    if !index.exists() {
        return None;
    }
    // Collect all shards, sort, return first (lowest shard number = shard-00001)
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if name_str.ends_with(".safetensors") && name_str != "model.safetensors" {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    shards.into_iter().next()
}

/// Strategy 2: look in a sibling subdirectory (base model name without quant suffix)
/// for `model.safetensors` or a sharded safetensors file.
fn discover_sibling_subdir(parent: &Path, base_name: &str) -> Option<std::path::PathBuf> {
    let subdir = parent.join(base_name);
    if !subdir.is_dir() {
        return None;
    }
    let single = subdir.join("model.safetensors");
    if single.exists() {
        return Some(single);
    }
    find_sharded_safetensors(&subdir)
}

/// Find the SafeTensors entry point in a snapshot directory (single or sharded).
/// For sharded models, returns the first shard (sorted by name = shard-00001).
fn find_safetensors_in_snapshot(snap_path: &Path) -> Option<std::path::PathBuf> {
    let single = snap_path.join("model.safetensors");
    if single.exists() {
        return Some(single);
    }
    // Fallback: return first shard sorted (shard-00001)
    let mut shards: Vec<_> = std::fs::read_dir(snap_path)
        .ok()?
        .flatten()
        .filter_map(|f| {
            let fname = f.file_name();
            let name = fname.to_string_lossy();
            if name.ends_with(".safetensors") && name != "model.safetensors" {
                Some(f.path())
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    shards.into_iter().next()
}

/// Check if a HF cache directory name matches the target model.
fn hf_cache_dir_matches(dir_name: &str, base_lower: &str) -> bool {
    if !dir_name.starts_with("models--") {
        return false;
    }
    let model_part = dir_name
        .trim_start_matches("models--")
        .replace("--", "/")
        .to_lowercase();
    model_part.contains(base_lower)
}

/// Search HF cache model directory snapshots for SafeTensors files.
fn search_hf_model_snapshots(model_dir: &Path) -> Option<std::path::PathBuf> {
    let snapshots = model_dir.join("snapshots");
    for snap in std::fs::read_dir(&snapshots).ok()?.flatten() {
        if let Some(found) = find_safetensors_in_snapshot(&snap.path()) {
            return Some(found);
        }
    }
    None
}

/// Strategy 3: search HuggingFace cache (`~/.cache/huggingface/hub/models--*`)
/// for a matching model directory containing safetensors files.
fn discover_hf_cache(base_name: &str) -> Option<std::path::PathBuf> {
    let hf_cache = dirs::home_dir()?.join(".cache/huggingface/hub");
    if !hf_cache.is_dir() {
        return None;
    }
    let base_lower = base_name.to_lowercase();

    for entry in std::fs::read_dir(&hf_cache).ok()?.flatten() {
        let dir_name = entry.file_name();
        if !hf_cache_dir_matches(&dir_name.to_string_lossy(), &base_lower) {
            continue;
        }
        if let Some(found) = search_hf_model_snapshots(&entry.path()) {
            return Some(found);
        }
    }
    None
}

/// Search a single repo directory for SafeTensors files (sharded or single).
fn find_safetensors_in_repo(repo_path: &Path) -> Option<std::path::PathBuf> {
    find_sharded_safetensors(repo_path).or_else(|| {
        let single = repo_path.join("model.safetensors");
        single.exists().then_some(single)
    })
}

/// Strategy 4: search APR cache (`~/.apr/cache/hf/`) for SafeTensors files.
///
/// `apr pull` downloads sharded models to `~/.apr/cache/hf/{org}/{repo}/`.
/// This strategy searches for a repo directory whose name matches the GGUF
/// base name, then returns the first SafeTensors file found.
fn discover_apr_cache(base_name: &str) -> Option<std::path::PathBuf> {
    let apr_cache = dirs::home_dir()?.join(".apr").join("cache").join("hf");
    if !apr_cache.is_dir() {
        return None;
    }
    let base_lower = base_name.to_lowercase();
    for org_entry in std::fs::read_dir(&apr_cache).ok()?.flatten() {
        if !org_entry.path().is_dir() {
            continue;
        }
        if let Some(found) = search_org_for_model(&org_entry.path(), &base_lower) {
            return Some(found);
        }
    }
    None
}

/// Search an org directory for a repo matching `base_lower` that contains SafeTensors.
fn search_org_for_model(org_path: &Path, base_lower: &str) -> Option<std::path::PathBuf> {
    for repo_entry in std::fs::read_dir(org_path).ok()?.flatten() {
        let repo_name = repo_entry.file_name().to_string_lossy().to_lowercase();
        if repo_name.contains(base_lower) {
            if let Some(found) = find_safetensors_in_repo(&repo_entry.path()) {
                return Some(found);
            }
        }
    }
    None
}

/// Gate 5: Cross-Format Parity Test (F-QUAL-032)
///
/// Compares argmax output between GGUF and SafeTensors for the same model.
/// P0-QA-001: Auto-discover SafeTensors model for format parity gate.
///
/// Search strategy (in order):
/// 1. Sibling directory of GGUF file (same name but .safetensors)
/// 2. Sibling subdirectories containing .safetensors files
/// 3. HuggingFace cache (~/.cache/huggingface/hub/models--*)
/// 4. APR cache (~/.apr/cache/hf/) — sharded models from `apr pull` (GH-279-2)
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
    let base_name = strip_quant_suffix(stem);
    if let Some(found) = discover_sibling_subdir(parent, base_name) {
        return Some(found);
    }

    // Strategy 3: HuggingFace cache
    if let Some(found) = discover_hf_cache(base_name) {
        return Some(found);
    }

    // Strategy 4: APR cache (~/.apr/cache/hf/) — sharded models from `apr pull`
    discover_apr_cache(base_name)
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
        let st_logits = match run_safetensors_forward(safetensors_path, &prompt_tokens) {
            Ok(logits) => logits,
            Err(ForwardError::ConversionFailed(path)) => {
                return Ok(GateResult::failed(
                    "format_parity",
                    &format!("SafeTensors conversion failed: {}", path),
                    None,
                    None,
                    start.elapsed(),
                ));
            }
            Err(ForwardError::Cli(e)) => return Err(e),
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
/// Internal error type for SafeTensors forward pass (avoids early-return from parent).
#[cfg(feature = "inference")]
enum ForwardError {
    ConversionFailed(String),
    Cli(CliError),
}

/// Run SafeTensors forward pass, handling sharded and single-file models.
#[cfg(feature = "inference")]
fn run_safetensors_forward(
    safetensors_path: &Path,
    prompt_tokens: &[u32],
) -> std::result::Result<Vec<f32>, ForwardError> {
    use realizar::safetensors_infer::SafetensorsToAprConverter;
    use realizar::{SafetensorsConfig, ShardedSafeTensorsModel};

    let parent_dir = safetensors_path.parent().unwrap_or(Path::new("."));
    let index_path = parent_dir.join("model.safetensors.index.json");

    let transformer = if index_path.exists() {
        let sharded = ShardedSafeTensorsModel::load_from_index(&index_path)
            .map_err(|e| ForwardError::Cli(CliError::ValidationFailed(format!("Sharded load failed: {e}"))))?;
        let config = SafetensorsConfig::load_from_sibling(safetensors_path)
            .ok_or_else(|| ForwardError::Cli(CliError::ValidationFailed("config.json not found for sharded model".to_string())))?;
        SafetensorsToAprConverter::convert_sharded(&sharded, &config)
    } else {
        SafetensorsToAprConverter::convert(safetensors_path)
    };

    let model = match transformer {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("Tensor not found") || msg.contains("not supported") {
                return Err(ForwardError::ConversionFailed(safetensors_path.display().to_string()));
            }
            return Err(ForwardError::Cli(CliError::ValidationFailed(format!("SafeTensors convert failed: {e}"))));
        }
    };

    model.forward(prompt_tokens)
        .map_err(|e| ForwardError::Cli(CliError::ValidationFailed(format!("SafeTensors forward failed: {e}"))))
}

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

/// Detect model size label from a lowercased filename.
/// Returns None if no known size pattern is found.
fn detect_size_from_filename(filename_lower: &str) -> Option<&'static str> {
    // (primary pattern, alternate pattern, size label)
    const SIZE_PATTERNS: &[(&str, &str, &str)] = &[
        ("0.5b", "-0_5b", "0.5b"),
        ("1.5b", "-1_5b", "1.5b"),
        ("3b", "-3b", "3b"),
        ("7b", "-7b", "7b"),
        ("14b", "-14b", "14b"),
        ("32b", "-32b", "32b"),
    ];
    SIZE_PATTERNS.iter().find_map(|(primary, alt, label)| {
        let matched = filename_lower.contains(primary) || filename_lower.contains(alt);
        matched.then_some(*label)
    })
}

/// Estimate model size from file size on disk (for hash-named pacha-cached files).
/// GGUF Q4_K sizes: 0.5B~400MB, 1.5B~1GB, 3B~2GB, 7B~4.5GB
fn estimate_size_from_file(path: &Path) -> &'static str {
    match std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) {
        0..=800_000_000 => "0.5b",
        800_000_001..=2_000_000_000 => "1.5b",
        2_000_000_001..=4_000_000_000 => "3b",
        _ => "7b",
    }
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

    let size = detect_size_from_filename(&filename_lower)
        .unwrap_or_else(|| estimate_size_from_file(path));

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
