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
