
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
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "side_effect_classification")]
pub(crate) fn run(
    path: &Path,
    granular: bool,
    format: OutputFormat,
    focus: ProfileFocus,
    detect_naive: bool,
    naive_threshold: f64,
    compare_hf: Option<&str>,
    energy: bool,
    perf_grade: bool,
    callgraph: bool,
    fail_on_naive: bool,
    output_path: Option<&Path>,
    warmup_passes: usize,
    measure_passes: usize,
    tokens: usize,
    ollama: bool,
    no_gpu: bool,
) -> Result<(), CliError> {
    // GH-517: Warn on unimplemented profiler flags (suppress unused warnings)
    if compare_hf.is_some() {
        eprintln!("Warning: --compare-hf is not yet implemented. Flag ignored.");
    }
    if energy {
        eprintln!("Warning: --energy profiling is not yet implemented. Flag ignored.");
    }
    if callgraph {
        eprintln!("Warning: --callgraph is not yet implemented. Flag ignored.");
    }
    // GH-2395: --fail-on-naive used to print "not yet implemented. Flag ignored."
    // and return 0 unconditionally, so a CI job gating on it could never fail.
    // It now runs the detection (implying --detect-naive) and exits non-zero.
    let detect_naive = detect_naive || fail_on_naive;

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
    let mut results = {
        #[allow(unused_mut)]
        let mut use_cpu = no_gpu;

        #[cfg(feature = "cuda")]
        let gpu_fallback_result: Option<RealProfileResults> =
            gpu_profile_or_none(use_cpu, path, tokens, warmup_passes, measure_passes, &format)?;
        #[cfg(not(feature = "cuda"))]
        let gpu_fallback_result: Option<RealProfileResults> = None;

        if let Some(r) = gpu_fallback_result {
            r
        } else {
            profile_real_inference_cpu(path, warmup_passes, measure_passes)?
        }
    };

    #[cfg(not(feature = "inference"))]
    let mut results = {
        let _ = (warmup_passes, measure_passes);
        output::warn("Inference feature not enabled. Cannot run real profiling.");
        output::warn("Build with: cargo build --features inference");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    };

    let profile_time = start.elapsed();

    // GH-2395: --tokens drives multi-token generation on the GPU path and the
    // ollama comparison only. The CPU per-operation path measures one forward
    // pass per measurement pass, so say so instead of discarding the value in
    // silence and reporting numbers the user did not ask for.
    #[cfg(feature = "inference")]
    if tokens != 1 && results.backend == "cpu" {
        eprintln!(
            "Warning: --tokens {tokens} has no effect on the CPU per-operation profiler, \
             which measures one forward pass per measurement pass. Use --measure to change \
             the number of samples."
        );
    }

    // Compute roofline analysis
    #[cfg(feature = "inference")]
    {
        results.roofline = Some(compute_roofline(&results));
    }

    // GH-173: Apply focus filtering to results (PMAT-182)
    let filtered_results = filter_results_by_focus(&results, focus);

    // Show focus filter if applied. GH-2395: only in Human mode — this used to
    // print unconditionally, so `--format json --focus mlp` put "  Focus filter: Mlp"
    // on stdout ahead of the JSON body.
    if !matches!(focus, ProfileFocus::All) && matches!(format, OutputFormat::Human) {
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
        naive_threshold,
        ollama_baseline.as_ref(),
        output_path,
        profile_time,
    )?;

    // GH-2395: the exit-code contract `--fail-on-naive` advertises. Evaluated on
    // the UNFILTERED results — `--focus` narrows the report, not the verdict.
    if fail_on_naive {
        let suspicions = detect_naive_implementations(&results, naive_threshold);
        if !suspicions.is_empty() {
            return Err(CliError::ValidationFailed(format!(
                "naive implementation detected ({} signal(s)): {}",
                suspicions.len(),
                suspicions
                    .iter()
                    .map(|s| s.reason.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
    }

    Ok(())
}

/// The GPU half of the benchmark (REG-15, #2971): profile on the GPU when the user did not force
/// the CPU. The load-time parity gate is NEVER disabled here — if the user set SKIP_PARITY_GATE
/// themselves it is printed as an override (every receipt of this run is INVALID-CORRECTNESS);
/// otherwise the gate runs and a failing model is refused by name.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_profile_or_none(
    use_cpu: bool,
    path: &Path,
    tokens: usize,
    warmup_passes: usize,
    measure_passes: usize,
    format: &OutputFormat,
) -> Result<Option<RealProfileResults>> {
    let r: Option<RealProfileResults> = if !use_cpu {
            // PMAT-203: Skip parity gate for profiling — known false positive on CUDA 13.1 driver.
        // The parity gate compares GPU/CPU logits and fails spuriously, but profiling
        // only needs generation throughput, not parity verification.
        if let Some(line) = crate::commands::parity_admission::override_line() {
            eprintln!("{line}");
        }

        // GH-578: Spawn GPU profiling on a thread with 16MB stack.
        // The deep call chain (profile_gpu_generation → generate_gpu_resident →
        // forward_all_layers_gpu_to_logits → transformer_layer_workspace_inner)
        // overflows the default 8MB stack on constrained hardware (RTX 4060 Yoga).
        let path_owned = path.to_path_buf();
        let gpu_result = std::thread::Builder::new()
            .name("gpu-profile".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || profile_gpu_generation(&path_owned, tokens, warmup_passes, measure_passes))
            .map_err(|e| CliError::ValidationFailed(format!("Failed to spawn profiling thread: {e}")))?
            .join()
            .map_err(|_| CliError::ValidationFailed("GPU profiling thread panicked".into()))?;

        match gpu_result {
            Ok(r) => Some(r),
            Err(e) => {
                if matches!(format, OutputFormat::Human) {
                    output::warn(&format!("GPU profiling failed: {e}, falling back to CPU per-op profiling"));
                }
                None
            }
        }
    } else {
        None
    };
    Ok(r)
}

#[allow(clippy::too_many_arguments)]
fn print_profile_output(
    format: OutputFormat,
    results: &RealProfileResults,
    granular: bool,
    perf_grade: bool,
    detect_naive: bool,
    naive_threshold: f64,
    ollama_baseline: Option<&OllamaBaseline>,
    output_path: Option<&Path>,
    profile_time: std::time::Duration,
) -> Result<(), CliError> {
    match format {
        OutputFormat::Human => {
            print_human_results(results, granular, perf_grade, detect_naive, naive_threshold)?;
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

    #[cfg(not(feature = "inference"))]
    {
        let _ = (format, assertions, warmup, measure);
        output::warn("Inference feature not enabled. Cannot run CI profiling.");
        return Err(CliError::ValidationFailed(
            "Requires --features inference".to_string(),
        ));
    }

    #[cfg(feature = "inference")]
    {
        let results = profile_real_inference_cpu(path, warmup, measure)?;

        // Build CI report with assertion checks
        let report = CiProfileReport::from_results(&results, assertions);

        // Output based on format
        match format {
            OutputFormat::Json => report.print_json(),
            _ => report.print_human(),
        }

        Ok(report.passed)
    }
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
