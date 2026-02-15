
/// Detect model format from extension
fn detect_format(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("apr") => "apr",
        Some("safetensors") => "safetensors",
        Some("gguf") => "gguf",
        Some("bin") => "pytorch",
        _ => "unknown",
    }
}

/// Run profiling on the model with REAL inference
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    path: &Path,
    granular: bool,
    format: OutputFormat,
    focus: ProfileFocus,
    detect_naive: bool,
    _naive_threshold: f64,
    _compare_hf: Option<&str>,
    _energy: bool,
    perf_grade: bool,
    _callgraph: bool,
    _fail_on_naive: bool,
    output_path: Option<&Path>,
    tokens: usize,
    ollama: bool,
    no_gpu: bool,
) -> Result<(), CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let format_str = detect_format(path);

    match format {
        OutputFormat::Human => {
            output::section("apr profile (Real Per-Operation Telemetry)");
            println!();
            output::kv("Model", path.display());
            output::kv("Format", format_str);
            println!();
        }
        OutputFormat::Json => {}
        OutputFormat::Flamegraph => {}
    }

    // Profile with REAL inference — try GPU first, fall back to CPU
    let start = Instant::now();

    #[cfg(feature = "inference")]
    let mut results = if no_gpu {
        profile_real_inference_cpu(path, 3, 10)?
    } else {
        // Try GPU generation profiling first (full token generation, not just forward pass)
        match profile_gpu_generation(path, tokens, 3, 10) {
            Ok(r) => r,
            Err(_) => {
                if matches!(format, OutputFormat::Human) {
                    output::warn("GPU profiling unavailable, falling back to CPU per-op profiling");
                }
                profile_real_inference_cpu(path, 3, 10)?
            }
        }
    };

    #[cfg(not(feature = "inference"))]
    let mut results = {
        output::warn("Inference feature not enabled. Cannot run real profiling.");
        output::warn("Build with: cargo build --features inference");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    };

    let profile_time = start.elapsed();

    // Compute roofline analysis
    #[cfg(feature = "inference")]
    {
        results.roofline = Some(compute_roofline(&results));
    }

    // GH-173: Apply focus filtering to results (PMAT-182)
    let filtered_results = filter_results_by_focus(&results, focus);

    // Show focus filter if applied
    if !matches!(focus, ProfileFocus::All) {
        output::kv("Focus filter", format!("{:?}", focus));
        println!();
    }

    // Ollama comparison (if requested)
    let ollama_baseline = if ollama && matches!(format, OutputFormat::Human) {
        run_ollama_comparison(path, tokens)
    } else {
        None
    };

    print_profile_output(
        format,
        &filtered_results,
        granular,
        perf_grade,
        detect_naive,
        ollama_baseline.as_ref(),
        output_path,
        profile_time,
    )
}

#[allow(clippy::too_many_arguments)]
fn print_profile_output(
    format: OutputFormat,
    results: &RealProfileResults,
    granular: bool,
    perf_grade: bool,
    detect_naive: bool,
    ollama_baseline: Option<&OllamaBaseline>,
    output_path: Option<&Path>,
    profile_time: std::time::Duration,
) -> Result<(), CliError> {
    match format {
        OutputFormat::Human => {
            print_human_results(results, granular, perf_grade, detect_naive)?;
            if let Some(baseline) = ollama_baseline {
                print_ollama_comparison(results, baseline);
            }
            println!();
            println!(
                "{}",
                format!("Profile completed in {:.2}s", profile_time.as_secs_f64()).dimmed()
            );
        }
        OutputFormat::Json => {
            print_json_results(results)?;
        }
        OutputFormat::Flamegraph => {
            print_flamegraph(results, output_path)?;
        }
    }
    Ok(())
}

// ============================================================================
// PMAT-192: CI Assertion Mode Entry Point (GH-180)
// ============================================================================

/// Run profiling in CI mode with assertions
///
/// Returns Ok(true) if all assertions pass, Ok(false) if any fail.
/// Use the exit code to fail CI pipelines.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_ci(
    path: &Path,
    format: OutputFormat,
    assertions: &CiAssertions,
    warmup: usize,
    measure: usize,
) -> Result<bool, CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    #[cfg(feature = "inference")]
    let results = profile_real_inference_cpu(path, warmup, measure)?;

    #[cfg(not(feature = "inference"))]
    {
        output::warn("Inference feature not enabled. Cannot run CI profiling.");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    }

    // Build CI report with assertion checks
    let report = CiProfileReport::from_results(&results, assertions);

    // Output based on format
    match format {
        OutputFormat::Json => report.print_json(),
        _ => report.print_human(),
    }

    Ok(report.passed)
}

// ============================================================================
// PMAT-192 Phase 4: Differential Benchmark Mode (GH-180)
// ============================================================================

/// Differential benchmark result comparing two models
#[derive(Debug, Clone)]
pub struct DiffBenchmarkReport {
    pub model_a: String,
    pub model_b: String,
    pub throughput_a: f64,
    pub throughput_b: f64,
    pub throughput_delta_pct: f64,
    pub latency_a_ms: f64,
    pub latency_b_ms: f64,
    pub latency_delta_pct: f64,
    pub winner: String,
    pub regressions: Vec<String>,
    pub improvements: Vec<String>,
}

impl DiffBenchmarkReport {
    /// Print human-readable diff report
    pub fn print_human(&self) {
        println!();
        println!("{}", "DIFFERENTIAL BENCHMARK (PMAT-192)".white().bold());
        println!("{}", "═".repeat(70));
        println!();
        println!("  Model A: {}", self.model_a.cyan());
        println!("  Model B: {}", self.model_b.cyan());
        println!();

        // Table header
        println!("┌─────────────┬──────────────┬──────────────┬──────────────┐");
        println!("│ Metric      │ Model A      │ Model B      │ Delta        │");
        println!("├─────────────┼──────────────┼──────────────┼──────────────┤");

        // Throughput row
        let tps_delta_str = if self.throughput_delta_pct >= 0.0 {
            format!("+{:.1}% ✅", self.throughput_delta_pct)
                .green()
                .to_string()
        } else {
            format!("{:.1}% ⚠️", self.throughput_delta_pct)
                .yellow()
                .to_string()
        };
        println!(
            "│ Throughput  │ {:>10.1} t/s │ {:>10.1} t/s │ {:>12} │",
            self.throughput_a, self.throughput_b, tps_delta_str
        );

        // Latency row
        let lat_delta_str = if self.latency_delta_pct <= 0.0 {
            format!("{:.1}% ✅", self.latency_delta_pct)
                .green()
                .to_string()
        } else {
            format!("+{:.1}% ⚠️", self.latency_delta_pct)
                .yellow()
                .to_string()
        };
        println!(
            "│ Latency     │ {:>10.2} ms │ {:>10.2} ms │ {:>12} │",
            self.latency_a_ms, self.latency_b_ms, lat_delta_str
        );

        println!("└─────────────┴──────────────┴──────────────┴──────────────┘");
        println!();

        // Winner
        println!(
            "  {}: {}",
            "Winner".white().bold(),
            self.winner.green().bold()
        );
        println!();

        // Regressions
        if !self.regressions.is_empty() {
            println!("{}", "  ⚠️  REGRESSIONS:".yellow().bold());
            for r in &self.regressions {
                println!("     - {}", r);
            }
            println!();
        }

        // Improvements
        if !self.improvements.is_empty() {
            println!("{}", "  ✅ IMPROVEMENTS:".green().bold());
            for i in &self.improvements {
                println!("     - {}", i);
            }
            println!();
        }
    }

    /// Print JSON diff report
    pub fn print_json(&self) {
        let mut json = String::from("{\n");
        writeln!(json, "  \"model_a\": \"{}\",", self.model_a)
            .expect("write to String is infallible");
        writeln!(json, "  \"model_b\": \"{}\",", self.model_b)
            .expect("write to String is infallible");
        json.push_str("  \"metrics\": {\n");
        writeln!(
            json,
            "    \"throughput_a_tok_s\": {:.2},",
            self.throughput_a
        )
        .expect("write to String is infallible");
        writeln!(
            json,
            "    \"throughput_b_tok_s\": {:.2},",
            self.throughput_b
        )
        .expect("write to String is infallible");
        writeln!(
            json,
            "    \"throughput_delta_pct\": {:.2},",
            self.throughput_delta_pct
        )
        .expect("write to String is infallible");
        writeln!(json, "    \"latency_a_ms\": {:.2},", self.latency_a_ms)
            .expect("write to String is infallible");
        writeln!(json, "    \"latency_b_ms\": {:.2},", self.latency_b_ms)
            .expect("write to String is infallible");
        writeln!(
            json,
            "    \"latency_delta_pct\": {:.2}",
            self.latency_delta_pct
        )
        .expect("write to String is infallible");
        json.push_str("  },\n");
        writeln!(json, "  \"winner\": \"{}\",", self.winner)
            .expect("write to String is infallible");
        json.push_str("  \"regressions\": [");
        for (i, r) in self.regressions.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            write!(json, "\"{}\"", r).expect("write to String is infallible");
        }
        json.push_str("],\n");
        json.push_str("  \"improvements\": [");
        for (i, imp) in self.improvements.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            write!(json, "\"{}\"", imp).expect("write to String is infallible");
        }
        json.push_str("]\n");
        json.push_str("}\n");
        println!("{json}");
    }
}
