
/// Compute percentage change from baseline to current, returning 0 if baseline is zero.
fn pct_change(baseline: f64, current: f64) -> f64 {
    if baseline > 0.0 {
        ((current - baseline) / baseline) * 100.0
    } else {
        0.0
    }
}

/// Classify a metric delta as regression, improvement, or neutral.
///
/// `higher_is_better`: true for throughput (positive delta = improvement),
/// false for latency (positive delta = regression).
fn classify_metric(
    label: &str,
    delta_pct: f64,
    threshold_pct: f64,
    old_val: f64,
    new_val: f64,
    unit: &str,
    higher_is_better: bool,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
) {
    let is_regression = if higher_is_better {
        delta_pct < -threshold_pct
    } else {
        delta_pct > threshold_pct
    };
    let is_improvement = if higher_is_better {
        delta_pct > threshold_pct
    } else {
        delta_pct < -threshold_pct
    };

    if is_regression {
        regressions.push(format!(
            "{label}: {:.1}% slower ({old_val:.1} → {new_val:.1} {unit})",
            delta_pct.abs()
        ));
    } else if is_improvement {
        improvements.push(format!(
            "{label}: {:.1}% faster ({old_val:.1} → {new_val:.1} {unit})",
            delta_pct.abs()
        ));
    }
}

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
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    output::section("Differential Benchmark (PMAT-192 Phase 4)");
    println!();

    output::kv("Profiling Model A", model_a.display());
    #[cfg(feature = "inference")]
    let results_a = profile_real_inference_cpu(model_a, warmup, measure)?;

    #[cfg(not(feature = "inference"))]
    return Err(CliError::ValidationFailed(
        "Requires --features inference".to_string(),
    ));

    output::kv("Profiling Model B", model_b.display());
    #[cfg(feature = "inference")]
    let results_b = profile_real_inference_cpu(model_b, warmup, measure)?;

    let throughput_delta = pct_change(results_a.throughput_tok_s, results_b.throughput_tok_s);
    let latency_a_ms = results_a.total_inference_us / 1000.0;
    let latency_b_ms = results_b.total_inference_us / 1000.0;
    let latency_delta = pct_change(latency_a_ms, latency_b_ms);

    let winner = match results_b
        .throughput_tok_s
        .partial_cmp(&results_a.throughput_tok_s)
    {
        Some(std::cmp::Ordering::Greater) => {
            format!("Model B ({:.1}% faster)", throughput_delta.abs())
        }
        Some(std::cmp::Ordering::Less) => {
            format!("Model A ({:.1}% faster)", throughput_delta.abs())
        }
        _ => "Tie".to_string(),
    };

    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    let thresh_pct = regression_threshold * 100.0;

    classify_metric(
        "Throughput",
        throughput_delta,
        thresh_pct,
        results_a.throughput_tok_s,
        results_b.throughput_tok_s,
        "tok/s",
        true,
        &mut regressions,
        &mut improvements,
    );
    classify_metric(
        "Latency",
        latency_delta,
        thresh_pct,
        latency_a_ms,
        latency_b_ms,
        "ms",
        false,
        &mut regressions,
        &mut improvements,
    );

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

    match format {
        OutputFormat::Json => report.print_json(),
        _ => report.print_human(),
    }

    Ok(regressions.is_empty())
}

/// Return focus-area keywords, or `None` for `All` (no filtering).
fn focus_keywords(focus: ProfileFocus) -> Option<&'static [&'static str]> {
    match focus {
        ProfileFocus::All => None,
        ProfileFocus::Attention => Some(&["attention", "attn", "qkv", "softmax"]),
        ProfileFocus::Mlp => Some(&["mlp", "ffn", "gate", "up_proj", "down_proj"]),
        ProfileFocus::Matmul => Some(&["matmul", "gemm", "mm", "linear"]),
        ProfileFocus::Embedding => Some(&["embed", "lm_head", "vocab"]),
    }
}

/// GH-173: Filter profile results by focus area (PMAT-182)
fn filter_results_by_focus(
    results: &RealProfileResults,
    focus: ProfileFocus,
) -> RealProfileResults {
    let filtered_hotspots = match focus_keywords(focus) {
        None => results.hotspots.clone(),
        Some(keywords) => results
            .hotspots
            .iter()
            .filter(|h| {
                let lower = h.name.to_lowercase();
                keywords.iter().any(|k| lower.contains(k))
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
