
/// Run differential benchmark comparing two models (Phase 4)
///
/// Returns Ok(true) if model_b is better or equal, Ok(false) if regression detected.
#[allow(dead_code)]
pub(crate) fn run_diff_benchmark(
    model_a: &Path,
    model_b: &Path,
    format: OutputFormat,
    warmup: usize,
    measure: usize,
    regression_threshold: f64, // e.g., 0.05 = 5% regression triggers failure
) -> Result<bool, CliError> {
    // Validate files exist
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    output::section("Differential Benchmark (PMAT-192 Phase 4)");
    println!();

    // Profile model A
    output::kv("Profiling Model A", model_a.display());
    #[cfg(feature = "inference")]
    let results_a = profile_real_inference_cpu(model_a, warmup, measure)?;

    #[cfg(not(feature = "inference"))]
    return Err(CliError::ValidationFailed(
        "Requires --features inference".to_string(),
    ));

    // Profile model B
    output::kv("Profiling Model B", model_b.display());
    #[cfg(feature = "inference")]
    let results_b = profile_real_inference_cpu(model_b, warmup, measure)?;

    // Calculate deltas
    let throughput_delta = if results_a.throughput_tok_s > 0.0 {
        ((results_b.throughput_tok_s - results_a.throughput_tok_s) / results_a.throughput_tok_s)
            * 100.0
    } else {
        0.0
    };

    let latency_a_ms = results_a.total_inference_us / 1000.0;
    let latency_b_ms = results_b.total_inference_us / 1000.0;
    let latency_delta = if latency_a_ms > 0.0 {
        ((latency_b_ms - latency_a_ms) / latency_a_ms) * 100.0
    } else {
        0.0
    };

    // Determine winner
    let winner = if results_b.throughput_tok_s > results_a.throughput_tok_s {
        format!("Model B ({:.1}% faster)", throughput_delta.abs())
    } else if results_a.throughput_tok_s > results_b.throughput_tok_s {
        format!("Model A ({:.1}% faster)", throughput_delta.abs())
    } else {
        "Tie".to_string()
    };

    // Detect regressions and improvements
    let mut regressions = Vec::new();
    let mut improvements = Vec::new();

    if throughput_delta < -regression_threshold * 100.0 {
        regressions.push(format!(
            "Throughput: {:.1}% slower ({:.1} → {:.1} tok/s)",
            throughput_delta.abs(),
            results_a.throughput_tok_s,
            results_b.throughput_tok_s
        ));
    } else if throughput_delta > regression_threshold * 100.0 {
        improvements.push(format!(
            "Throughput: {:.1}% faster ({:.1} → {:.1} tok/s)",
            throughput_delta, results_a.throughput_tok_s, results_b.throughput_tok_s
        ));
    }

    if latency_delta > regression_threshold * 100.0 {
        regressions.push(format!(
            "Latency: {:.1}% slower ({:.2} → {:.2} ms)",
            latency_delta, latency_a_ms, latency_b_ms
        ));
    } else if latency_delta < -regression_threshold * 100.0 {
        improvements.push(format!(
            "Latency: {:.1}% faster ({:.2} → {:.2} ms)",
            latency_delta.abs(),
            latency_a_ms,
            latency_b_ms
        ));
    }

    let report = DiffBenchmarkReport {
        model_a: model_a.display().to_string(),
        model_b: model_b.display().to_string(),
        throughput_a: results_a.throughput_tok_s,
        throughput_b: results_b.throughput_tok_s,
        throughput_delta_pct: throughput_delta,
        latency_a_ms,
        latency_b_ms,
        latency_delta_pct: latency_delta,
        winner,
        regressions: regressions.clone(),
        improvements,
    };

    // Output
    match format {
        OutputFormat::Json => report.print_json(),
        _ => report.print_human(),
    }

    // Return false if any regressions detected
    Ok(regressions.is_empty())
}

/// GH-173: Filter profile results by focus area (PMAT-182)
fn filter_results_by_focus(
    results: &RealProfileResults,
    focus: ProfileFocus,
) -> RealProfileResults {
    let filtered_hotspots = match focus {
        ProfileFocus::All => results.hotspots.clone(),
        ProfileFocus::Attention => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("attention")
                    || name_lower.contains("attn")
                    || name_lower.contains("qkv")
                    || name_lower.contains("softmax")
            })
            .cloned()
            .collect(),
        ProfileFocus::Mlp => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("mlp")
                    || name_lower.contains("ffn")
                    || name_lower.contains("gate")
                    || name_lower.contains("up_proj")
                    || name_lower.contains("down_proj")
            })
            .cloned()
            .collect(),
        ProfileFocus::Matmul => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("matmul")
                    || name_lower.contains("gemm")
                    || name_lower.contains("mm")
                    || name_lower.contains("linear")
            })
            .cloned()
            .collect(),
        ProfileFocus::Embedding => results
            .hotspots
            .iter()
            .filter(|h| {
                let name_lower = h.name.to_lowercase();
                name_lower.contains("embed")
                    || name_lower.contains("lm_head")
                    || name_lower.contains("vocab")
            })
            .cloned()
            .collect(),
    };

    RealProfileResults {
        model_path: results.model_path.clone(),
        architecture: results.architecture.clone(),
        num_layers: results.num_layers,
        vocab_size: results.vocab_size,
        hidden_dim: results.hidden_dim,
        warmup_passes: results.warmup_passes,
        measure_passes: results.measure_passes,
        total_inference_us: results.total_inference_us,
        throughput_tok_s: results.throughput_tok_s,
        tokens_per_pass: results.tokens_per_pass,
        hotspots: filtered_hotspots,
        per_layer_us: results.per_layer_us.clone(),
        is_real_data: results.is_real_data,
        roofline: results.roofline.clone(),
        category_summary: results.category_summary.clone(),
        backend: results.backend.clone(),
        latency_p50_ms: results.latency_p50_ms,
        latency_p95_ms: results.latency_p95_ms,
        latency_p99_ms: results.latency_p99_ms,
        latency_min_ms: results.latency_min_ms,
        latency_max_ms: results.latency_max_ms,
        prefill_tok_s: results.prefill_tok_s,
        decode_tok_s: results.decode_tok_s,
        total_tokens_generated: results.total_tokens_generated,
        kernel_launch_overhead_pct: results.kernel_launch_overhead_pct,
        kernel_launch_overhead_us: results.kernel_launch_overhead_us,
    }
}

/// Profile model using REAL inference passes (CPU per-operation path)
#[cfg(feature = "inference")]
fn profile_real_inference_cpu(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    let format = detect_format(path);

    match format {
        "gguf" => profile_gguf_real(path, warmup_passes, measure_passes),
        "apr" => profile_apr_real(path, warmup_passes, measure_passes),
        "safetensors" => profile_safetensors_real(path, warmup_passes, measure_passes),
        _ => Err(CliError::ValidationFailed(format!(
            "Unsupported format: {format}"
        ))),
    }
}
