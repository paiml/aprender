
/// Profile SafeTensors model — tries sibling GGUF/APR, falls back to static analysis
#[cfg(feature = "inference")]
fn profile_safetensors_real(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    // Check sibling GGUF/APR for real inference profiling
    let gguf_path = path.with_extension("gguf");
    if gguf_path.exists() {
        output::info(&format!(
            "Found sibling GGUF: {}. Profiling that instead.",
            gguf_path.display()
        ));
        return profile_gguf_real(&gguf_path, warmup_passes, measure_passes);
    }
    let apr_path = path.with_extension("apr");
    if apr_path.exists() {
        output::info(&format!(
            "Found sibling APR: {}. Profiling that instead.",
            apr_path.display()
        ));
        return profile_apr_real(&apr_path, warmup_passes, measure_passes);
    }

    // Static analysis fallback via Rosetta Stone
    output::info("SafeTensors: running static analysis profile (no inference engine needed)");
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::InvalidFormat(format!("Inspection failed: {e}")))?;

    let num_layers = report
        .tensors
        .iter()
        .filter(|t| t.name.contains("self_attn.q_proj"))
        .count();
    let vocab_size = report
        .tensors
        .iter()
        .find(|t| t.name.contains("embed_tokens"))
        .map_or(0, |t| t.shape.first().copied().unwrap_or(0));
    let hidden_dim = report
        .tensors
        .iter()
        .find(|t| t.name.contains("embed_tokens"))
        .map_or(0, |t| t.shape.last().copied().unwrap_or(0));

    Ok(RealProfileResults {
        model_path: path.display().to_string(),
        architecture: report.architecture.unwrap_or_else(|| "unknown".to_string()),
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes: 0,
        measure_passes: 0,
        total_inference_us: 0.0,
        throughput_tok_s: 0.0,
        tokens_per_pass: 0,
        hotspots: vec![],
        per_layer_us: vec![],
        is_real_data: false,
        roofline: None,
        category_summary: None,
        backend: "static_analysis".to_string(),
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
        latency_p99_ms: 0.0,
        latency_min_ms: 0.0,
        latency_max_ms: 0.0,
        prefill_tok_s: 0.0,
        decode_tok_s: 0.0,
        total_tokens_generated: 0,
        kernel_launch_overhead_pct: 0.0,
        kernel_launch_overhead_us: 0.0,
    })
}

/// Profile GGUF model with real inference
#[cfg(feature = "inference")]
fn profile_gguf_real(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    // Load the model
    println!("{}", "Loading model...".dimmed());
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    let architecture = mapped.model.architecture().unwrap_or("unknown").to_string();

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let config = &model.config;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    // Test prompt tokens (BOS + "Hello")
    let test_tokens: Vec<u32> = vec![1, 15043]; // BOS + "Hello" for TinyLlama/Qwen
    let tokens_per_pass = test_tokens.len();

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 1, // Just one token to profile forward pass
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Warmup passes (discard timing)
    println!(
        "{}",
        format!("Running {} warmup passes...", warmup_passes).dimmed()
    );
    for _ in 0..warmup_passes {
        let _ = model.generate(&test_tokens, &gen_config);
    }

    // Measurement passes with per-operation profiler
    println!(
        "{}",
        format!(
            "Running {} measurement passes (per-op instrumented)...",
            measure_passes
        )
        .dimmed()
    );

    let mut profiler = BrickProfiler::new();
    profiler.set_num_layers(num_layers);
    profiler.set_tokens(tokens_per_pass * measure_passes);

    let mut forward_times: Vec<f64> = Vec::new();

    profiler.start_inference();

    for _ in 0..measure_passes {
        let pass_start = Instant::now();

        // Use forward_profiled() for real per-operation timing
        let logits = model.forward_profiled(&test_tokens, &mut profiler);

        let pass_time = pass_start.elapsed().as_secs_f64() * 1_000_000.0;
        forward_times.push(pass_time);

        // Validate output
        if let Ok(ref logits) = logits {
            let has_nan = logits.iter().any(|x| x.is_nan());
            let has_inf = logits.iter().any(|x| x.is_infinite());

            if has_nan || has_inf {
                output::warn(&format!(
                    "Forward pass produced invalid logits: NaN={}, Inf={}",
                    has_nan, has_inf
                ));
            }
        }
    }

    profiler.stop_inference();

    // Build results from profiler
    let report = profiler.report();

    // Compute statistics from raw timing
    let total_us: f64 = forward_times.iter().sum();
    let avg_us = total_us / measure_passes as f64;
    // min/max computed for future detailed output
    let _min_us = forward_times.iter().copied().fold(f64::INFINITY, f64::min);
    let _max_us = forward_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    // Build hotspots from profiler report with roofline classification
    let mut hotspots: Vec<Hotspot> = report
        .operations
        .iter()
        .map(|(name, stats)| {
            let category = classify_operation_category(name);
            let bottleneck = classify_operation_bottleneck(name);
            Hotspot {
                name: name.clone(),
                time_us: stats.total_us,
                percent: if report.total_inference_us > 0.0 {
                    (stats.total_us / report.total_inference_us) * 100.0
                } else {
                    0.0
                },
                count: stats.count,
                avg_us: stats.avg_us,
                min_us: stats.min_us,
                max_us: stats.max_us,
                bottleneck: Some(bottleneck),
                efficiency_pct: None, // Computed later with hardware info
                category: Some(category),
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }
        })
        .collect();

    // Sort by total time descending
    hotspots.sort_by(|a, b| {
        b.time_us
            .partial_cmp(&a.time_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build real per-layer timing from profiler report's per_layer data
    // Each operation records per_layer entries — sum across all ops per layer
    let per_layer_us = build_per_layer_timing(&report, num_layers);

    // Compute category summary
    let category_summary = compute_category_summary(&hotspots);

    // Compute real percentiles from forward times
    let mut sorted_times: Vec<f64> = forward_times.iter().map(|t| t / 1000.0).collect(); // us -> ms
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let tps = if avg_us > 0.0 {
        (tokens_per_pass as f64 / avg_us) * 1_000_000.0
    } else {
        0.0
    };

    Ok(RealProfileResults {
        model_path: path.display().to_string(),
        architecture,
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: avg_us,
        throughput_tok_s: tps,
        tokens_per_pass,
        hotspots,
        per_layer_us,
        is_real_data: report.is_real_data,
        roofline: None,
        category_summary: Some(category_summary),
        backend: "cpu".to_string(),
        latency_p50_ms: percentile(&sorted_times, 50.0),
        latency_p95_ms: percentile(&sorted_times, 95.0),
        latency_p99_ms: percentile(&sorted_times, 99.0),
        latency_min_ms: sorted_times.first().copied().unwrap_or(0.0),
        latency_max_ms: sorted_times.last().copied().unwrap_or(0.0),
        prefill_tok_s: 0.0, // CPU path doesn't separate prefill/decode
        decode_tok_s: tps,
        total_tokens_generated: tokens_per_pass * measure_passes,
        kernel_launch_overhead_pct: 0.0, // CPU path: no kernel launches
        kernel_launch_overhead_us: 0.0,
    })
}

/// Profile APR model with real inference
#[cfg(feature = "inference")]
fn profile_apr_real(
    path: &Path,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    use realizar::apr_transformer::AprTransformer;

    // Load the model using AprTransformer
    println!("{}", "Loading APR model...".dimmed());
    let model = AprTransformer::from_apr_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let config = &model.config;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;
    let hidden_dim = config.hidden_dim;

    // Test tokens
    let test_tokens: Vec<u32> = vec![1, 15043];
    let tokens_per_pass = test_tokens.len();

    // Warmup
    println!(
        "{}",
        format!("Running {} warmup passes...", warmup_passes).dimmed()
    );
    for _ in 0..warmup_passes {
        let _ = model.forward(&test_tokens);
    }

    // Measurement
    println!(
        "{}",
        format!("Running {} measurement passes...", measure_passes).dimmed()
    );

    let mut profiler = BrickProfiler::new();
    profiler.set_num_layers(num_layers);
    profiler.set_tokens(tokens_per_pass * measure_passes);

    let mut forward_times: Vec<f64> = Vec::new();

    profiler.start_inference();

    for _ in 0..measure_passes {
        profiler.start("forward_pass");
        let start = Instant::now();
        let result = model.forward(&test_tokens);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        profiler.stop("forward_pass");

        forward_times.push(elapsed);

        // Validate output
        if let Ok(ref logits) = result {
            let has_nan = logits.iter().any(|x| x.is_nan());
            let has_inf = logits.iter().any(|x| x.is_infinite());

            if has_nan || has_inf {
                output::warn(&format!(
                    "Forward pass produced invalid logits: NaN={}, Inf={}",
                    has_nan, has_inf
                ));
            }
        }
    }

    profiler.stop_inference();

    let total_us: f64 = forward_times.iter().sum();
    let avg_us = total_us / measure_passes as f64;
    let min_us = forward_times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_us = forward_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut sorted_times: Vec<f64> = forward_times.iter().map(|t| t / 1000.0).collect();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let tps = if avg_us > 0.0 {
        (tokens_per_pass as f64 / avg_us) * 1_000_000.0
    } else {
        0.0
    };

    Ok(RealProfileResults {
        model_path: path.display().to_string(),
        architecture: "apr".to_string(),
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: avg_us,
        throughput_tok_s: tps,
        tokens_per_pass,
        hotspots: vec![Hotspot {
            name: "forward_pass".to_string(),
            time_us: total_us,
            percent: 100.0,
            count: measure_passes,
            avg_us,
            min_us,
            max_us,
            bottleneck: None,
            efficiency_pct: None,
            category: Some("Other".to_string()),
            bandwidth_gbs: None,
            data_bytes_per_call: None,
        }],
        per_layer_us: vec![avg_us / num_layers as f64; num_layers],
        is_real_data: true,
        roofline: None,
        category_summary: None,
        backend: "cpu".to_string(),
        latency_p50_ms: percentile(&sorted_times, 50.0),
        latency_p95_ms: percentile(&sorted_times, 95.0),
        latency_p99_ms: percentile(&sorted_times, 99.0),
        latency_min_ms: sorted_times.first().copied().unwrap_or(0.0),
        latency_max_ms: sorted_times.last().copied().unwrap_or(0.0),
        prefill_tok_s: 0.0,
        decode_tok_s: tps,
        total_tokens_generated: tokens_per_pass * measure_passes,
        kernel_launch_overhead_pct: 0.0, // APR CPU: no kernel launches
        kernel_launch_overhead_us: 0.0,
    })
}

/// Print human-readable results with per-operation hotspots, category bars, and roofline
fn print_human_results(
    results: &RealProfileResults,
    granular: bool,
    show_perf_grade: bool,
    detect_naive: bool,
) -> Result<(), CliError> {
    print_profile_model_header(results);
    print_hotspot_table(results, granular);
    print_category_summary(results);
    print_kernel_launch_overhead(results);
    print_per_layer_timing(results, granular);
    print_roofline_section(results);
    print_perf_grade_section(results, show_perf_grade);
    print_naive_detection(results, detect_naive);
    print_generation_performance(results);
    print_latency_percentiles(results);
    print_profile_summary(results);
    Ok(())
}

fn print_profile_model_header(results: &RealProfileResults) {
    println!(
        "{}",
        output::kv_table(&[
            (
                "Architecture",
                format!(
                    "{} ({} layers, hidden={}, vocab={})",
                    results.architecture,
                    results.num_layers,
                    output::count_fmt(results.hidden_dim),
                    output::count_fmt(results.vocab_size)
                )
            ),
            ("Backend", results.backend.to_uppercase()),
            ("Warmup", format!("{} passes", results.warmup_passes)),
            ("Measure", format!("{} passes", results.measure_passes)),
        ])
    );
    println!();
    if results.is_real_data {
        println!("  {}", output::badge_pass("REAL PER-OPERATION TELEMETRY"));
    } else {
        println!(
            "  {}",
            output::badge_warn("SIMULATED DATA (inference disabled)")
        );
    }
    println!();
}
