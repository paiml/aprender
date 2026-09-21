
pub(super) fn calculate_stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Extract numeric field from JSON response (simple parser, no serde dependency)
/// Handles: "field_name":12345 or "field_name": 12345 (with/without space)
pub(super) fn extract_json_field(json: &str, field: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", field);
    json.find(&pattern).and_then(|start| {
        let value_start = start + pattern.len();
        let rest = &json[value_start..];
        // Skip whitespace
        let rest = rest.trim_start();
        // Extract numeric value
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(rest.len());
        rest[..end].parse::<f64>().ok()
    })
}

/// Prefix of every baseline that was NOT measured. A consumer greps for it; a
/// number never stands in for it (#3773).
pub(super) const UNMEASURED: &str = "UNMEASURED";

/// The llama.cpp baseline of `apr showcase`: never measured here, always said so.
///
/// #3773: this used to `which llama-server` (any binary on PATH, no pin, no
/// version) and then return `35.0 + jitter` tok/s — a constant with clock noise
/// added so that it looked measured — which `build_comparison` turned into a
/// "speedup vs llama.cpp". It now measures nothing and returns the reason.
///
/// Why showcase does not drive llama.cpp itself: apr's number here is an
/// in-process `generate_with_cache` loop, so a llama-server run would be a
/// different client, and its launch flags would be a second copy of the pinned
/// comparator's knob declaration (`scripts/llama_bin.sh`, #2737). The one
/// comparator measurement is the pinned parity lane, which drives both engines
/// through the same client (#2696 / #3563). No PATH lookup is made (#3740).
pub(super) fn run_llama_cpp_bench(_config: &ShowcaseConfig) -> Result<(f64, f64)> {
    Err(CliError::ValidationFailed(format!(
        "{UNMEASURED}: apr showcase does not measure llama.cpp — the comparator is \
         measured only by the pinned parity lane (`. scripts/llama_bin.sh` then \
         scripts/parity_host_receipt.sh), which runs both engines through one client; \
         no speedup is reported"
    )))
}

/// Ollama's throughput and TTFT from an `/api/generate` response body.
///
/// #3773: a response without `eval_count`/`eval_duration` used to read as
/// 200.0 tok/s and one without `prompt_eval_duration` as a 150.0 ms TTFT —
/// constants printed as Ollama's measurement. Each is now `UNMEASURED`, naming
/// the field that was missing.
pub(super) fn ollama_tps_ttft(response: &str) -> Result<(f64, f64)> {
    let unmeasured =
        |why: &str| CliError::ValidationFailed(format!("{UNMEASURED}: Ollama response {why}"));
    let count = extract_json_field(response, "eval_count")
        .ok_or_else(|| unmeasured("carried no eval_count"))?;
    let duration_ns = extract_json_field(response, "eval_duration")
        .ok_or_else(|| unmeasured("carried no eval_duration"))?;
    if duration_ns <= 0.0 {
        return Err(unmeasured("reported eval_duration 0"));
    }
    let ttft_ns = extract_json_field(response, "prompt_eval_duration")
        .ok_or_else(|| unmeasured("carried no prompt_eval_duration"))?;
    // eval_duration and prompt_eval_duration are in nanoseconds.
    Ok((count / (duration_ns / 1_000_000_000.0), ttft_ns / 1_000_000.0))
}

pub(super) fn run_ollama_bench(config: &ShowcaseConfig) -> Result<(f64, f64)> {
    // Check if ollama is available
    let ollama_available = Command::new("which")
        .arg("ollama")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !ollama_available {
        return Err(CliError::ValidationFailed("ollama not found".to_string()));
    }

    // Real benchmark against Ollama using API
    use std::process::Command;

    // Determine model to use based on config tier
    let ollama_model = match config.tier {
        ModelTier::Tiny => "qwen2.5-coder:0.5b",
        ModelTier::Small => "qwen2.5-coder:1.5b",
        ModelTier::Medium => "qwen2.5-coder:7b",
        ModelTier::Large => "qwen2.5-coder:32b",
    };

    // LESSON-001: Use Ollama HTTP API, NOT `ollama run --verbose` (hangs indefinitely)
    // See: docs/qa/benchmark-matrix-2026-01-09.md
    let prompt = "Hello, write a short function";
    let request_body = format!(
        r#"{{"model":"{}","prompt":"{}","stream":false}}"#,
        ollama_model, prompt
    );

    // Use curl with timeout to call Ollama API
    let output = Command::new("curl")
        .args([
            "-s", // Silent mode
            "--max-time",
            "60", // 60 second timeout (large models need more time)
            "-X",
            "POST",
            "http://localhost:11434/api/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            &request_body,
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| CliError::ValidationFailed(format!("curl failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::ValidationFailed(format!(
            "Ollama API failed: {}",
            stderr
        )));
    }

    // Parse JSON response from Ollama API: {"eval_count":N,"eval_duration":Dns,...}
    let response = String::from_utf8_lossy(&output.stdout);
    let (tps, ttft) = ollama_tps_ttft(&response)?;

    println!(
        "  Ollama ({}): {:.1} tok/s, TTFT: {:.1}ms",
        ollama_model, tps, ttft
    );
    Ok((tps, ttft))
}

/// Format a speedup result line with pass/fail status
fn format_speedup_line(label: &str, speedup: f64) {
    let status = if speedup >= 25.0 {
        format!("{} (target: 25%)", "PASS".green().bold())
    } else {
        format!("{} (target: 25%)", "FAIL".red().bold())
    };
    println!("Speedup vs {label}: {speedup:.1}% {status}");
}

pub(super) fn print_benchmark_results(comparison: &BenchmarkComparison) {
    println!();
    println!("{}", "═══ Benchmark Results ═══".cyan().bold());
    println!();

    println!("┌─────────────────┬────────────┬────────────┬──────────┐");
    println!("│ System          │ Tokens/sec │ TTFT (ms)  │ Runs     │");
    println!("├─────────────────┼────────────┼────────────┼──────────┤");
    println!(
        "│ {} │ {:>7.1}±{:<3.1} │ {:>10.1} │ {:>8} │",
        "APR (ours)    ".green().bold(),
        comparison.apr_tps,
        comparison.apr_tps_stddev,
        comparison.apr_ttft_ms,
        comparison.runs
    );

    // Baseline rows (both follow the same table format)
    let baselines: &[(&str, Option<f64>, Option<f64>)] = &[
        ("llama.cpp       ", comparison.llama_cpp_tps, comparison.llama_cpp_ttft_ms),
        ("Ollama          ", comparison.ollama_tps, comparison.ollama_ttft_ms),
    ];
    for &(name, tps_opt, ttft_opt) in baselines {
        if let Some(tps) = tps_opt {
            println!(
                "│ {name}│ {:>10.1} │ {:>10.1} │      N/A │",
                tps,
                ttft_opt.unwrap_or(0.0)
            );
        }
    }

    println!("└─────────────────┴────────────┴────────────┴──────────┘");
    // #3773: a requested baseline with no measurement is shown as such, with its
    // reason, rather than silently left out of the table.
    for (name, why) in &comparison.unmeasured {
        println!("{name}: {why}");
    }
    println!();

    // Speedup summary
    let speedups: &[(&str, Option<f64>)] = &[
        ("llama.cpp", comparison.speedup_vs_llama),
        ("Ollama", comparison.speedup_vs_ollama),
    ];
    for &(label, speedup_opt) in speedups {
        if let Some(speedup) = speedup_opt {
            format_speedup_line(label, speedup);
        }
    }
}
