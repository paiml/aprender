//! Handler for `probador llm` subcommands.
//!
//! Implements correctness testing, load testing, and report generation
//! for OpenAI-compatible LLM inference endpoints.

use crate::error::{CliError, CliResult};
use crate::DataAuditArgs;
use crate::ExperimentArgs;
use crate::LlmBenchArgs;
use crate::LlmGenDatasetArgs;
use crate::LlmLoadArgs;
use crate::LlmReportArgs;
use crate::LlmScoreArgs;
use crate::LlmSweepArgs;
use crate::LlmTestArgs;
use std::path::Path;
use std::time::Duration;

/// Test case loaded from YAML config.
#[derive(Debug, serde::Deserialize)]
pub struct TestConfig {
    /// List of test cases.
    pub tests: Vec<TestCase>,
}

/// A single correctness test case.
#[derive(Debug, serde::Deserialize)]
pub struct TestCase {
    /// Name of the test.
    pub name: String,
    /// Chat messages to send.
    pub messages: Vec<MessageConfig>,
    /// Assert output contains this substring.
    #[serde(default)]
    pub expect_contains: Option<String>,
    /// Assert output matches this regex pattern.
    #[serde(default)]
    pub expect_pattern: Option<String>,
    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f64>,
}

/// A message in the YAML config.
#[derive(Debug, serde::Deserialize)]
pub struct MessageConfig {
    /// Role: system, user, or assistant.
    pub role: String,
    /// Content of the message.
    pub content: String,
}

/// Result of a single correctness test.
#[derive(Debug, serde::Serialize)]
pub struct CorrectnessResult {
    /// Test name.
    pub name: String,
    /// Whether all assertions passed.
    pub passed: bool,
    /// Latency in milliseconds.
    pub latency_ms: f64,
    /// The model's output text.
    pub output: String,
    /// Details of any failures.
    pub failures: Vec<String>,
}

/// Aggregated correctness test results.
#[derive(Debug, serde::Serialize)]
pub struct CorrectnessReport {
    /// Runtime name.
    pub runtime_name: String,
    /// Timestamp.
    pub timestamp: String,
    /// Individual test results.
    pub results: Vec<CorrectnessResult>,
    /// Count of passed tests.
    pub passed: usize,
    /// Count of failed tests.
    pub failed: usize,
    /// Total tests run.
    pub total: usize,
}

/// Execute `probador llm test`.
pub async fn execute_llm_test(args: &LlmTestArgs) -> CliResult<()> {
    let config_str =
        std::fs::read_to_string(&args.config).map_err(|e| CliError::Generic(e.to_string()))?;
    let config: TestConfig =
        serde_yaml_ng::from_str(&config_str).map_err(|e| CliError::Generic(e.to_string()))?;

    let client = jugar_probar::llm::LlmClient::new(&args.url, &args.model);

    // Health check
    match client.health_check().await {
        Ok(true) => println!("Health check passed: {}", args.url),
        Ok(false) | Err(_) => {
            eprintln!(
                "Warning: health check failed for {}, proceeding anyway",
                args.url
            );
        }
    }

    let mut results = Vec::new();

    for test in &config.tests {
        print!("  {} ... ", test.name);
        let messages: Vec<jugar_probar::llm::ChatMessage> = test
            .messages
            .iter()
            .map(|m| jugar_probar::llm::ChatMessage {
                role: parse_role(&m.role),
                content: m.content.clone(),
            })
            .collect();

        match client
            .chat_completion(messages, test.temperature, test.max_tokens)
            .await
        {
            Ok(timed) => {
                let mut assertion = jugar_probar::llm::LlmAssertion::new().assert_response_valid();

                if let Some(ref s) = test.expect_contains {
                    assertion = assertion.assert_contains(s);
                }
                if let Some(ref p) = test.expect_pattern {
                    assertion = assertion.assert_matches_pattern(p);
                }

                let check_results = assertion.run(&timed);
                let failures: Vec<String> = check_results
                    .iter()
                    .filter(|r| !r.passed)
                    .filter_map(|r| r.detail.clone())
                    .collect();

                let passed = failures.is_empty();
                let output = timed
                    .response
                    .choices
                    .first()
                    .map_or_else(String::new, |c| c.message.content.clone());

                if passed {
                    println!("PASS ({:.0}ms)", timed.latency.as_secs_f64() * 1000.0);
                } else {
                    println!("FAIL");
                    for f in &failures {
                        eprintln!("    {f}");
                    }
                }

                results.push(CorrectnessResult {
                    name: test.name.clone(),
                    passed,
                    latency_ms: timed.latency.as_secs_f64() * 1000.0,
                    output,
                    failures,
                });
            }
            Err(e) => {
                println!("ERROR: {e}");
                results.push(CorrectnessResult {
                    name: test.name.clone(),
                    passed: false,
                    latency_ms: 0.0,
                    output: String::new(),
                    failures: vec![e.to_string()],
                });
            }
        }
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    let total = results.len();

    println!("\nResults: {passed}/{total} passed, {failed} failed");

    let report = CorrectnessReport {
        runtime_name: args.runtime_name.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        results,
        passed,
        failed,
        total,
    };

    if let Some(ref output_path) = args.output {
        let json =
            serde_json::to_string_pretty(&report).map_err(|e| CliError::Generic(e.to_string()))?;
        std::fs::write(output_path, json).map_err(|e| CliError::Generic(e.to_string()))?;
        println!("Results written to {}", output_path.display());
    }

    if failed > 0 {
        Err(CliError::Generic(format!("{failed} test(s) failed")))
    } else {
        Ok(())
    }
}

/// Execute `probador llm load`.
pub async fn execute_llm_load(args: &LlmLoadArgs) -> CliResult<()> {
    let duration = parse_duration(&args.duration)?;
    let warmup = parse_duration(&args.warmup)?;
    let client = jugar_probar::llm::LlmClient::new(&args.url, &args.model);

    // Load prompts: dataset > prompt-file > prompt-profile > default
    let (mut prompts, dataset_stats) = if let Some(ref dataset_path) = args.dataset {
        let (prompts, stats) = load_dataset(dataset_path)?;
        (prompts, Some(stats))
    } else {
        let prompts = resolve_prompts(args.prompt_profile.as_deref(), args.prompt_file.as_deref())?;
        (prompts, None)
    };

    // PMAT-077: Apply max_tokens override or distribution for heterogeneous traffic
    if let Some(ref dist) = args.max_tokens_distribution {
        prompts = apply_max_tokens_distribution(&prompts, dist, args.concurrency)?;
        println!(
            "Max tokens:   distribution={} ({} prompts generated)",
            dist,
            prompts.len()
        );
    } else if let Some(max_tokens) = args.max_tokens {
        for p in &mut prompts {
            p.max_tokens = Some(max_tokens);
        }
        println!("Max tokens:   {max_tokens}");
    }

    println!(
        "Load testing {} (concurrency={}, duration={:.0}s, warmup={:.0}s, runtime={})",
        args.url,
        args.concurrency,
        duration.as_secs_f64(),
        warmup.as_secs_f64(),
        args.runtime_name,
    );

    let validate = jugar_probar::llm::ValidationMode::parse(&args.validate);
    if !matches!(validate, jugar_probar::llm::ValidationMode::None) {
        println!("Validation:   {}", args.validate);
    }

    // Health gate: verify endpoint is reachable before sending load (GH-37)
    if args.skip_health_check {
        eprintln!("Warning: health check skipped (--skip-health-check)");
    } else {
        let health_timeout = Duration::from_secs(5);
        match tokio::time::timeout(health_timeout, client.health_check()).await {
            Ok(Ok(true)) => println!("Health check passed"),
            Ok(Ok(false)) => {
                return Err(CliError::Generic(format!(
                    "Health check failed: endpoint {} returned non-200 response. \
                     Use --skip-health-check to bypass.",
                    args.url
                )));
            }
            Ok(Err(e)) => {
                return Err(CliError::Generic(format!(
                    "Health check failed: endpoint {} is unreachable: {e}. \
                     Use --skip-health-check to bypass.",
                    args.url
                )));
            }
            Err(_) => {
                return Err(CliError::Generic(format!(
                    "Health check timed out after {health_timeout:?}: endpoint {} did not respond. \
                     Use --skip-health-check to bypass.",
                    args.url
                )));
            }
        }
    }

    let config = jugar_probar::llm::LoadTestConfig {
        concurrency: args.concurrency,
        duration,
        prompts,
        runtime_name: args.runtime_name.clone(),
        warmup_duration: warmup,
        stream: args.stream,
        trace_level: None,
        slo_ttft_ms: None,
        slo_tpot_ms: None,
        slo_latency_ms: None,
        rate: match args.rate {
            Some(r) if args.rate_distribution == "constant" => {
                jugar_probar::llm::RequestRate::Constant(r)
            }
            Some(r) => jugar_probar::llm::RequestRate::Poisson(r),
            None => jugar_probar::llm::RequestRate::Max,
        },
        num_layers: args.num_layers,
        validate,
        spike_threshold: args.spike_threshold,
        fail_on_quality: args.fail_on_quality,
    };

    // GPU telemetry: start collection before benchmark (GH-34: auto-detect remote host)
    let mut gpu_collector = if args.gpu_telemetry {
        let poll_interval = parse_duration(&args.gpu_poll_interval)?;
        let gpu_host = jugar_probar::llm::extract_host_from_url(&args.url);
        let mut collector = jugar_probar::llm::GpuTelemetryCollector::new(
            poll_interval.as_secs().max(1),
            args.expected_clock_mhz,
        )
        .with_host(gpu_host.clone());
        match collector.start().await {
            Ok(()) => {
                if let Some(ref host) = gpu_host {
                    println!("GPU telemetry: collecting from {host}");
                } else {
                    println!("GPU telemetry: collecting (local)");
                }
            }
            Err(e) => eprintln!("Warning: GPU telemetry failed to start: {e}"),
        }
        Some(collector)
    } else {
        None
    };

    let load_test = jugar_probar::llm::LoadTest::new(client, config);
    let mut result = load_test
        .run()
        .await
        .map_err(|e| CliError::Generic(e.to_string()))?;

    // GPU telemetry: stop and attach to result
    if let Some(ref mut collector) = gpu_collector {
        result.gpu_telemetry = collector
            .stop(result.completion_tokens_total, result.total_requests)
            .await;
    }

    // Dataset stats: attach if we loaded from dataset
    result.dataset_stats = dataset_stats;

    // Print summary
    println!("\n--- Load Test Results ---");
    println!("Runtime:      {}", result.runtime_name);
    println!(
        "Requests:     {} ({} ok, {} failed)",
        result.total_requests, result.successful, result.failed
    );
    println!("Throughput:   {:.1} req/s", result.throughput_rps);
    println!("Latency P50:  {:.1} ms", result.latency_p50_ms);
    println!("Latency P95:  {:.1} ms", result.latency_p95_ms);
    println!("Latency P99:  {:.1} ms", result.latency_p99_ms);
    println!("TTFT P50:     {:.1} ms", result.ttft_p50_ms);
    println!("Tokens/sec:   {:.1}", result.tokens_per_sec);
    println!("Avg tok/req:  {:.1}", result.avg_tok_per_req);
    if result.prefill_tok_per_sec > 0.0 {
        println!("Prefill tok/s:{:.1}", result.prefill_tok_per_sec);
    }
    if result.decode_tok_per_sec > 0.0 {
        println!("Decode tok/s: {:.1}", result.decode_tok_per_sec);
        println!("ITL P50:      {:.1} ms", result.itl_p50_ms);
        if let (Some(us_per_layer), Some(n)) = (result.decode_us_per_layer, result.num_layers) {
            println!("µs/layer:     {us_per_layer:.1} ({n} layers)");
        }
    }
    if result.tpot_p50_ms > 0.0 {
        println!("TPOT P50:     {:.1} ms", result.tpot_p50_ms);
    }
    if result.error_rate > 0.0 {
        println!("Error rate:   {:.1}%", result.error_rate * 100.0);
    }
    if let Some(dist) = &result.output_tokens_dist {
        println!(
            "Output tok:   [{:.0}, {:.0}, {:.0}, {:.0}] (min/p50/p90/max)",
            dist[0], dist[1], dist[2], dist[3]
        );
    }

    // Feature 3: Tail analysis
    if let Some(ref tail) = result.tail_analysis {
        println!("\n--- Tail Latency Analysis ---");
        println!(
            "ITL P99.9:    {:.1} ms  (tail ratio: {:.1}x)",
            tail.itl_p999_ms, tail.tail_ratio_itl
        );
        println!(
            "TTFT P99.9:   {:.1} ms  (tail ratio: {:.1}x)",
            tail.ttft_p999_ms, tail.tail_ratio_ttft
        );
        println!(
            "Lat P99.9:    {:.1} ms  (tail ratio: {:.1}x)",
            tail.latency_p999_ms, tail.tail_ratio_latency
        );
        if tail.jitter.spike_count > 0 {
            println!(
                "Spikes:       {} (threshold: {:.1}ms)",
                tail.jitter.spike_count, tail.jitter.spike_threshold_ms
            );
        }
        if tail.jitter.itl_cv > 0.0 {
            println!("ITL CV:       {:.2}", tail.jitter.itl_cv);
        }
        if tail.drift.degradation_detected {
            eprintln!(
                "Warning: Latency drift detected (ITL slope: {:.2} ms/min)",
                tail.drift.itl_slope_ms_per_min
            );
        }
    }

    // Feature 5: Quality validation
    if let Some(ref quality) = result.quality {
        println!(
            "\n--- Quality Validation ({}) ---",
            quality.validation_level
        );
        println!(
            "Validated:    {} ({} pass, {} fail, {:.1}% pass rate)",
            quality.total_validated,
            quality.passed,
            quality.failed,
            quality.pass_rate * 100.0
        );
        for failure in quality.failures.iter().take(10) {
            eprintln!(
                "  FAIL request #{}: {}",
                failure.request_idx, failure.reason
            );
        }
        if quality.failures.len() > 10 {
            eprintln!("  ... and {} more failures", quality.failures.len() - 10);
        }
    }

    // Feature 2: GPU telemetry
    if let Some(ref gpu) = result.gpu_telemetry {
        println!("\n--- GPU Telemetry ({} samples) ---", gpu.samples);
        println!(
            "GPU util:     {:.0}% avg ({:.0}% max)",
            gpu.gpu_utilization_pct.mean, gpu.gpu_utilization_pct.max
        );
        println!(
            "Memory:       {:.0} / {:.0} MB",
            gpu.memory_used_mb.mean, gpu.memory_total_mb
        );
        println!(
            "Power:        {:.1}W avg ({:.1}W max)",
            gpu.power_draw_w.mean, gpu.power_draw_w.max
        );
        println!(
            "Temperature:  {:.0}°C avg ({:.0}°C max)",
            gpu.temperature_c.mean, gpu.temperature_c.max
        );
        println!(
            "Clock:        {:.0} MHz avg ({:.0} MHz min)",
            gpu.clock_gpu_mhz.mean, gpu.clock_gpu_mhz.min
        );
        if gpu.throttle_events > 0 {
            eprintln!("Warning: {} throttle events detected", gpu.throttle_events);
        }
        if gpu.energy_per_token_mj > 0.0 {
            println!(
                "Energy:       {:.2} mJ/token, {:.2} Wh total",
                gpu.energy_per_token_mj, gpu.energy_total_wh
            );
        }
    }

    // Warnings
    if result.truncated_pct > 10.0 {
        eprintln!("Warning: {:.0}% of responses truncated by max_tokens — increase max_tokens or use longer prompts", result.truncated_pct);
    }
    if result.sse_batch_ratio > 0.0 && result.sse_batch_ratio < 0.8 {
        eprintln!("Warning: SSE batch ratio {:.2} — server batches {:.0} tokens/chunk, per-token variance unreliable",
            result.sse_batch_ratio, 1.0 / result.sse_batch_ratio);
    }

    if let Some(ref output_path) = args.output {
        let json = jugar_probar::llm::report::to_json(&result);
        std::fs::write(output_path, json).map_err(|e| CliError::Generic(e.to_string()))?;
        println!("\nResults written to {}", output_path.display());
    }

    // Feature 5: fail on quality threshold
    if let (Some(threshold), Some(ref quality)) = (args.fail_on_quality, &result.quality) {
        if quality.pass_rate < threshold {
            return Err(CliError::Generic(format!(
                "Quality pass rate {:.1}% below threshold {:.1}%",
                quality.pass_rate * 100.0,
                threshold * 100.0,
            )));
        }
    }

    Ok(())
}

/// Execute `probador llm report`.
pub fn execute_llm_report(args: &LlmReportArgs) -> CliResult<()> {
    let results = load_results_from_dir(&args.results)?;

    if results.is_empty() {
        println!("No result files found in {}", args.results.display());
        return Ok(());
    }

    println!("Loaded {} result files", results.len());

    jugar_probar::llm::report::update_performance_md(&args.output, &results)
        .map_err(|e| CliError::Generic(e.to_string()))?;
    println!("Updated {}", args.output.display());

    if let Some(ref readme_path) = args.update_readme {
        let table = jugar_probar::llm::report::to_markdown_table(&results);
        update_readme_section(readme_path, &table)?;
        println!("Updated {}", readme_path.display());
    }

    Ok(())
}

/// Execute `probador llm score`.
pub fn execute_llm_score(args: &LlmScoreArgs) -> CliResult<()> {
    let all_results = load_all_results_from_dir(&args.results)?;

    if all_results.is_empty() {
        println!("No result files found in {}", args.results.display());
        return Ok(());
    }

    // Filter by platform if specified
    let filtered: Vec<_> = all_results
        .into_iter()
        .filter(|(r, _)| {
            args.platform
                .as_ref()
                .is_none_or(|p| r.runtime_name.contains(p.as_str()))
        })
        .collect();

    // Group by concurrency
    let mut by_concurrency: std::collections::HashMap<
        usize,
        Vec<(jugar_probar::llm::LoadTestResult, String)>,
    > = std::collections::HashMap::new();
    for (result, filename) in filtered {
        by_concurrency
            .entry(result.concurrency)
            .or_default()
            .push((result, filename));
    }

    // If concurrency filter specified, only score that level
    let concurrency_levels: Vec<usize> = if let Some(c) = args.concurrency {
        vec![c]
    } else {
        let mut levels: Vec<_> = by_concurrency.keys().copied().collect();
        levels.sort_unstable();
        levels
    };

    let contract = jugar_probar::llm::ScoringContract::default();
    let c1_results = by_concurrency.get(&1);
    let mut all_output = Vec::new();
    let mut min_grade_score = f64::MAX;

    for c in &concurrency_levels {
        if let Some(results) = by_concurrency.get(c) {
            let scorecard = jugar_probar::llm::compute_scorecard(
                results,
                c1_results.map(std::vec::Vec::as_slice),
                &contract,
            );

            for rt in &scorecard.runtimes {
                if rt.composite < min_grade_score {
                    min_grade_score = rt.composite;
                }
            }

            match args.format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&scorecard)
                        .map_err(|e| CliError::Generic(e.to_string()))?;
                    all_output.push(json);
                }
                "markdown" => {
                    all_output.push(jugar_probar::llm::format_markdown(&scorecard));
                }
                _ => {
                    all_output.push(jugar_probar::llm::format_table(&scorecard));
                }
            }
        }
    }

    // Per-layer scoring (--by-layer)
    if args.by_layer {
        let all_flat: Vec<_> = by_concurrency
            .values()
            .flat_map(|v| v.iter())
            .cloned()
            .collect();
        let layer_card = jugar_probar::llm::compute_layer_scorecard(&all_flat, &contract.grades);
        if !layer_card.runtimes.is_empty() {
            match args.format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&layer_card)
                        .map_err(|e| CliError::Generic(e.to_string()))?;
                    all_output.push(json);
                }
                "markdown" => {
                    all_output.push(jugar_probar::llm::format_layer_markdown(&layer_card));
                }
                _ => {
                    all_output.push(jugar_probar::llm::format_layer_table(&layer_card));
                }
            }
        }
    }

    // Per-profile scoring (--by-profile)
    if args.by_profile {
        for c in &concurrency_levels {
            if let Some(results) = by_concurrency.get(c) {
                let profile_card = jugar_probar::llm::compute_profile_scorecard(results, &contract);
                if !profile_card.entries.is_empty() {
                    match args.format.as_str() {
                        "json" => {
                            let json = serde_json::to_string_pretty(&profile_card)
                                .map_err(|e| CliError::Generic(e.to_string()))?;
                            all_output.push(json);
                        }
                        "markdown" => {
                            all_output
                                .push(jugar_probar::llm::format_profile_markdown(&profile_card));
                        }
                        _ => {
                            all_output.push(jugar_probar::llm::format_profile_table(&profile_card));
                        }
                    }
                }
            }
        }
    }

    // Correctness scoring (--by-correctness)
    if args.by_correctness {
        let all_flat: Vec<_> = by_concurrency
            .values()
            .flat_map(|v| v.iter())
            .cloned()
            .collect();
        let card = jugar_probar::llm::compute_correctness_scorecard(&all_flat, &contract.grades);
        if !card.runtimes.is_empty() {
            match args.format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&card)
                        .map_err(|e| CliError::Generic(e.to_string()))?;
                    all_output.push(json);
                }
                "markdown" => {
                    all_output.push(jugar_probar::llm::format_correctness_markdown(&card));
                }
                _ => {
                    all_output.push(jugar_probar::llm::format_correctness_table(&card));
                }
            }
        }
    }

    // Output length profile scoring (--by-output-length)
    if args.by_output_length {
        for c in &concurrency_levels {
            if let Some(results) = by_concurrency.get(c) {
                let card = jugar_probar::llm::compute_output_length_scorecard(results, &contract);
                if !card.entries.is_empty() {
                    match args.format.as_str() {
                        "json" => {
                            let json = serde_json::to_string_pretty(&card)
                                .map_err(|e| CliError::Generic(e.to_string()))?;
                            all_output.push(json);
                        }
                        "markdown" => {
                            all_output
                                .push(jugar_probar::llm::format_output_length_markdown(&card));
                        }
                        _ => {
                            all_output.push(jugar_probar::llm::format_output_length_table(&card));
                        }
                    }
                }
            }
        }
    }

    // Memory footprint scoring (--by-memory)
    if args.by_memory {
        let all_flat: Vec<_> = by_concurrency
            .values()
            .flat_map(|v| v.iter())
            .cloned()
            .collect();
        let card = jugar_probar::llm::compute_memory_scorecard(&all_flat, &contract.grades);
        if !card.runtimes.is_empty() {
            match args.format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&card)
                        .map_err(|e| CliError::Generic(e.to_string()))?;
                    all_output.push(json);
                }
                "markdown" => {
                    all_output.push(jugar_probar::llm::format_memory_markdown(&card));
                }
                _ => {
                    all_output.push(jugar_probar::llm::format_memory_table(&card));
                }
            }
        }
    }

    // Cold start scoring (--by-cold-start)
    if args.by_cold_start {
        let all_flat: Vec<_> = by_concurrency
            .values()
            .flat_map(|v| v.iter())
            .cloned()
            .collect();
        let card = jugar_probar::llm::compute_cold_start_scorecard(&all_flat, &contract.grades);
        if !card.runtimes.is_empty() {
            match args.format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&card)
                        .map_err(|e| CliError::Generic(e.to_string()))?;
                    all_output.push(json);
                }
                "markdown" => {
                    all_output.push(jugar_probar::llm::format_cold_start_markdown(&card));
                }
                _ => {
                    all_output.push(jugar_probar::llm::format_cold_start_table(&card));
                }
            }
        }
    }

    // Power efficiency scoring (--by-power)
    if args.by_power {
        let all_flat: Vec<_> = by_concurrency
            .values()
            .flat_map(|v| v.iter())
            .cloned()
            .collect();
        let card =
            jugar_probar::llm::compute_power_efficiency_scorecard(&all_flat, &contract.grades);
        if !card.runtimes.is_empty() {
            match args.format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&card)
                        .map_err(|e| CliError::Generic(e.to_string()))?;
                    all_output.push(json);
                }
                "markdown" => {
                    all_output.push(jugar_probar::llm::format_power_markdown(&card));
                }
                _ => {
                    all_output.push(jugar_probar::llm::format_power_table(&card));
                }
            }
        }
    }

    // Concurrency scaling scoring (--by-scaling)
    if args.by_scaling {
        let all_flat: Vec<_> = by_concurrency
            .values()
            .flat_map(|v| v.iter())
            .cloned()
            .collect();
        let card =
            jugar_probar::llm::compute_concurrency_scaling_scorecard(&all_flat, &contract.grades);
        if !card.runtimes.is_empty() {
            match args.format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&card)
                        .map_err(|e| CliError::Generic(e.to_string()))?;
                    all_output.push(json);
                }
                "markdown" => {
                    all_output.push(jugar_probar::llm::format_scaling_markdown(&card));
                }
                _ => {
                    all_output.push(jugar_probar::llm::format_scaling_table(&card));
                }
            }
        }
    }

    let output_text = all_output.join("\n\n");

    if let Some(ref output_path) = args.output {
        std::fs::write(output_path, &output_text).map_err(|e| CliError::Generic(e.to_string()))?;
        println!("Scorecard written to {}", output_path.display());
    } else {
        println!("{output_text}");
    }

    // CI gate: fail if any runtime is below the specified grade
    if let Some(ref fail_grade) = args.fail_on_grade {
        let min_required = grade_to_min_score(fail_grade);
        if min_grade_score < min_required {
            return Err(CliError::Generic(format!(
                "Score gate failed: lowest score {min_grade_score:.1} < {fail_grade} ({min_required})"
            )));
        }
    }

    Ok(())
}

/// Load all JSON result files from a directory (no filename filter).
fn load_all_results_from_dir(
    dir: &Path,
) -> CliResult<Vec<(jugar_probar::llm::LoadTestResult, String)>> {
    let mut results = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| CliError::Generic(format!("Cannot read {}: {e}", dir.display())))?;

    for entry in entries {
        let entry = entry.map_err(|e| CliError::Generic(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let content =
                std::fs::read_to_string(&path).map_err(|e| CliError::Generic(e.to_string()))?;
            if let Ok(result) = serde_json::from_str::<jugar_probar::llm::LoadTestResult>(&content)
            {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                results.push((result, filename));
            }
        }
    }

    results.sort_by(|a, b| a.0.timestamp.cmp(&b.0.timestamp));
    Ok(results)
}

/// Convert a grade string to its minimum score threshold.
fn grade_to_min_score(grade: &str) -> f64 {
    match grade {
        "A+" => 95.0,
        "A" => 90.0,
        "A-" => 85.0,
        "B+" => 80.0,
        "B" => 70.0,
        "C+" => 60.0,
        "C" => 50.0,
        "D" => 40.0,
        "D-" => 30.0,
        _ => 0.0,
    }
}

/// Execute `probador llm bench`.
pub async fn execute_llm_bench(args: &LlmBenchArgs) -> CliResult<()> {
    let duration = parse_duration(&args.duration)?;
    let warmup = parse_duration(&args.warmup)?;
    let health_timeout = parse_duration(&args.health_timeout)?;
    let cooldown = parse_duration(&args.cooldown)?;
    let prompts = resolve_prompts(Some(&args.prompt_profile), args.prompt_file.as_deref())?;
    let baseline = load_baseline(args.baseline.as_deref())?;

    println!(
        "Benchmark: {} (runs={}, duration={:.0}s, warmup={:.0}s, concurrency={}, runtime={})",
        args.url,
        args.runs,
        duration.as_secs_f64(),
        warmup.as_secs_f64(),
        args.concurrency,
        args.runtime_name,
    );

    let config = jugar_probar::llm::benchmark::BenchmarkConfig {
        url: args.url.clone(),
        model: args.model.clone(),
        start_command: args.start.clone(),
        health_timeout,
        warmup,
        duration,
        concurrency: args.concurrency,
        runs: args.runs,
        cooldown,
        prompts,
        runtime_name: args.runtime_name.clone(),
        baseline,
        fail_on_regression: args.fail_on_regression,
        stream: args.stream,
        trace_level: args.trace_level.clone(),
        num_layers: args.num_layers,
    };

    let mut benchmark = jugar_probar::llm::benchmark::Benchmark::new(config);
    let report = benchmark
        .run()
        .await
        .map_err(|e| CliError::Generic(e.to_string()))?;

    print_bench_report(&report);
    write_bench_output(args.output.as_deref(), &report)?;

    let has_exceeding = report.regressions.iter().any(|r| r.exceeds_threshold);
    if has_exceeding {
        return Err(CliError::Generic(
            "Benchmark regression detected: one or more metrics exceeded threshold".to_string(),
        ));
    }

    Ok(())
}

/// Load a baseline result from a JSON file (`BenchmarkReport` or single `LoadTestResult`).
fn load_baseline(path: Option<&Path>) -> CliResult<Option<jugar_probar::llm::LoadTestResult>> {
    let Some(baseline_path) = path else {
        return Ok(None);
    };
    let content = std::fs::read_to_string(baseline_path)
        .map_err(|e| CliError::Generic(format!("Failed to read baseline: {e}")))?;
    if let Ok(report) =
        serde_json::from_str::<jugar_probar::llm::benchmark::BenchmarkReport>(&content)
    {
        return Ok(report.runs.first().cloned());
    }
    let result: jugar_probar::llm::LoadTestResult = serde_json::from_str(&content)
        .map_err(|e| CliError::Generic(format!("Failed to parse baseline: {e}")))?;
    Ok(Some(result))
}

/// Print the full benchmark report (per-run + aggregate + regressions).
fn print_bench_report(report: &jugar_probar::llm::benchmark::BenchmarkReport) {
    for (i, run) in report.runs.iter().enumerate() {
        println!("\n--- Run {}/{} ---", i + 1, report.runs.len());
        println!("  Throughput:   {:.1} req/s", run.throughput_rps);
        println!("  Latency P50:  {:.1} ms", run.latency_p50_ms);
        println!("  TTFT P50:     {:.1} ms", run.ttft_p50_ms);
        println!("  Tokens/sec:   {:.1}", run.tokens_per_sec);
        if run.tpot_p50_ms > 0.0 {
            println!("  TPOT P50:     {:.1} ms", run.tpot_p50_ms);
        }
    }

    println!("\n--- Aggregate ({} runs) ---", report.runs.len());
    print_stat("Throughput (req/s)", &report.aggregate.throughput_rps);
    print_stat("Latency P50 (ms)", &report.aggregate.latency_p50);
    print_stat("Tokens/sec", &report.aggregate.tokens_per_sec);
    print_stat("TTFT P50 (ms)", &report.aggregate.ttft_p50);
    print_stat("TPOT P50 (ms)", &report.aggregate.tpot_p50);

    // GH-114: Print brick trace summary if available
    if let Some(trace) = report
        .runs
        .last()
        .and_then(|r| r.brick_trace_summary.as_ref())
    {
        println!(
            "\n--- BrickProfiler Trace ({} ops, {} samples) ---",
            trace.len(),
            trace.first().map_or(0, |t| t.samples),
        );
        for op in trace {
            if op.name == "throughput" {
                continue; // Skip meta-op
            }
            println!(
                "  {:24} {:8.0}µs avg  ({:5.1}%)",
                op.name, op.mean_us, op.pct_of_total,
            );
        }
    }

    if !report.regressions.is_empty() {
        println!("\n--- Regressions ---");
        for r in &report.regressions {
            let tag = if r.exceeds_threshold { "EXCEEDED" } else { "" };
            println!(
                "  {}: {:.1} → {:.1} ({:+.1}%) {tag}",
                r.metric, r.baseline_value, r.current_value, r.change_pct,
            );
        }
    }
}

fn print_stat(label: &str, stat: &jugar_probar::llm::benchmark::StatSummary) {
    println!(
        "  {}: {:.1} ± {:.1} (95% CI: [{:.1}, {:.1}])",
        label, stat.mean, stat.stddev, stat.ci_95_lower, stat.ci_95_upper
    );
}

/// Write benchmark report to JSON file.
fn write_bench_output(
    output_path: Option<&Path>,
    report: &jugar_probar::llm::benchmark::BenchmarkReport,
) -> CliResult<()> {
    let Some(path) = output_path else {
        return Ok(());
    };
    let json =
        serde_json::to_string_pretty(report).map_err(|e| CliError::Generic(e.to_string()))?;
    std::fs::write(path, json).map_err(|e| CliError::Generic(e.to_string()))?;
    println!("\nResults written to {}", path.display());
    Ok(())
}

/// Resolve prompts from profile name or file path.
/// PMAT-077: Generate prompts with heterogeneous `max_tokens` from a distribution.
///
/// Supported distributions:
/// - `uniform:MIN,MAX` — uniform spread across [MIN, MAX]
/// - `fixed:N` — all requests get N (same as --max-tokens N)
///
/// Generates enough prompts (concurrency × 256) to cover long benchmarks
/// with varied `max_tokens` values for staggered slot completion.
fn apply_max_tokens_distribution(
    base_prompts: &[jugar_probar::llm::ChatRequest],
    distribution: &str,
    concurrency: usize,
) -> CliResult<Vec<jugar_probar::llm::ChatRequest>> {
    let count = concurrency * 256; // Enough for ~60s at ~4 req/s/worker

    let (min, max) = if let Some(rest) = distribution.strip_prefix("uniform:") {
        let parts: Vec<&str> = rest.split(',').collect();
        if parts.len() != 2 {
            return Err(CliError::Generic(format!(
                "Invalid distribution format: {distribution}. Expected: uniform:MIN,MAX"
            )));
        }
        let min: u32 = parts[0]
            .parse()
            .map_err(|_| CliError::Generic(format!("Invalid min value: {}", parts[0])))?;
        let max: u32 = parts[1]
            .parse()
            .map_err(|_| CliError::Generic(format!("Invalid max value: {}", parts[1])))?;
        if min > max || min == 0 {
            return Err(CliError::Generic(format!(
                "Invalid range: min={min}, max={max}. Need 0 < min <= max"
            )));
        }
        (min, max)
    } else if let Some(rest) = distribution.strip_prefix("fixed:") {
        let n: u32 = rest
            .parse()
            .map_err(|_| CliError::Generic(format!("Invalid fixed value: {rest}")))?;
        (n, n)
    } else {
        return Err(CliError::Generic(format!(
            "Unknown distribution: {distribution}. Use: uniform:MIN,MAX or fixed:N"
        )));
    };

    let range = max - min + 1;
    // Simple LCG for deterministic pseudo-random distribution (no rand crate needed).
    // Multiplier and increment from Numerical Recipes.
    let mut state: u64 = 0x517c_c1b7_2722_0a95;
    let mut prompts = Vec::with_capacity(count);
    for i in 0..count {
        let mut prompt = base_prompts[i % base_prompts.len()].clone();
        prompt.max_tokens = Some(if min == max {
            min
        } else {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            min + ((state >> 33) as u32 % range)
        });
        prompts.push(prompt);
    }
    Ok(prompts)
}

fn resolve_prompts(
    profile_name: Option<&str>,
    prompt_file: Option<&Path>,
) -> CliResult<Vec<jugar_probar::llm::ChatRequest>> {
    if let Some(file_path) = prompt_file {
        jugar_probar::llm::load_prompts_from_file(file_path).map_err(CliError::Generic)
    } else if let Some(name) = profile_name {
        let profile = jugar_probar::llm::PromptProfile::from_name(name).ok_or_else(|| {
            CliError::Generic(format!(
                "Unknown prompt profile: {name}. Use: micro, short, medium, long"
            ))
        })?;
        Ok(jugar_probar::llm::load_profile(profile))
    } else {
        // Default: use the medium profile
        Ok(jugar_probar::llm::load_profile(
            jugar_probar::llm::PromptProfile::Medium,
        ))
    }
}

/// Load all JSON result files from a directory.
fn load_results_from_dir(dir: &Path) -> CliResult<Vec<jugar_probar::llm::LoadTestResult>> {
    let mut results = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| CliError::Generic(format!("Cannot read {}: {e}", dir.display())))?;

    for entry in entries {
        let entry = entry.map_err(|e| CliError::Generic(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json")
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains("load"))
        {
            let content =
                std::fs::read_to_string(&path).map_err(|e| CliError::Generic(e.to_string()))?;
            match serde_json::from_str::<jugar_probar::llm::LoadTestResult>(&content) {
                Ok(result) => results.push(result),
                Err(e) => eprintln!("Warning: skipping {}: {e}", path.display()),
            }
        }
    }

    results.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    Ok(results)
}

/// Update a README.md with a performance section.
fn update_readme_section(path: &Path, table: &str) -> CliResult<()> {
    let existing = if path.exists() {
        std::fs::read_to_string(path).map_err(|e| CliError::Generic(e.to_string()))?
    } else {
        String::new()
    };

    let marker_start = "<!-- PERFORMANCE_START -->";
    let marker_end = "<!-- PERFORMANCE_END -->";

    let content = if existing.contains(marker_start) && existing.contains(marker_end) {
        // Replace existing section
        let before = existing.split(marker_start).next().unwrap_or("");
        let after = existing.split(marker_end).nth(1).unwrap_or("");
        format!("{before}{marker_start}\n{table}\n{marker_end}{after}")
    } else {
        // Append section
        let mut out = existing;
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("\n{marker_start}\n{table}\n{marker_end}\n"));
        out
    };

    std::fs::write(path, content).map_err(|e| CliError::Generic(e.to_string()))
}

/// Parse a role string to the LLM Role enum.
fn parse_role(s: &str) -> jugar_probar::llm::Role {
    match s.to_lowercase().as_str() {
        "system" => jugar_probar::llm::Role::System,
        "assistant" => jugar_probar::llm::Role::Assistant,
        _ => jugar_probar::llm::Role::User,
    }
}

/// Parse a duration string like "30s", "2m", "1h".
fn parse_duration(s: &str) -> CliResult<Duration> {
    let s = s.trim();
    if let Some(secs) = s.strip_suffix('s') {
        let n: u64 = secs
            .parse()
            .map_err(|_| CliError::Generic(format!("Invalid duration: {s}")))?;
        Ok(Duration::from_secs(n))
    } else if let Some(mins) = s.strip_suffix('m') {
        let n: u64 = mins
            .parse()
            .map_err(|_| CliError::Generic(format!("Invalid duration: {s}")))?;
        Ok(Duration::from_secs(n * 60))
    } else if let Some(hrs) = s.strip_suffix('h') {
        let n: u64 = hrs
            .parse()
            .map_err(|_| CliError::Generic(format!("Invalid duration: {s}")))?;
        Ok(Duration::from_secs(n * 3600))
    } else {
        // Try parsing as raw seconds
        let n: u64 = s
            .parse()
            .map_err(|_| CliError::Generic(format!("Invalid duration: {s}. Use 30s, 2m, or 1h")))?;
        Ok(Duration::from_secs(n))
    }
}

// =============================================================================
// Feature 1: Concurrency Sweep
// =============================================================================

/// Execute `probador llm sweep`.
pub async fn execute_llm_sweep(args: &LlmSweepArgs) -> CliResult<()> {
    let duration = parse_duration(&args.duration)?;
    let warmup = parse_duration(&args.warmup)?;
    let prompts = resolve_prompts(args.prompt_profile.as_deref(), args.prompt_file.as_deref())?;

    let levels: Vec<usize> = args
        .concurrency_levels
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if levels.is_empty() {
        return Err(CliError::Generic(
            "No valid concurrency levels specified".to_string(),
        ));
    }

    println!(
        "Sweep: {} (levels={:?}, duration={:.0}s, saturation={:.1}x)",
        args.url,
        levels,
        duration.as_secs_f64(),
        args.saturation_threshold
    );

    let client = jugar_probar::llm::LlmClient::new(&args.url, &args.model);

    // Health check
    match client.health_check().await {
        Ok(true) => println!("Health check passed"),
        Ok(false) | Err(_) => {
            eprintln!("Warning: health check failed, proceeding anyway");
        }
    }

    let mut sweep_levels = Vec::new();
    let mut baseline_p99: Option<f64> = None;
    let mut best_throughput = 0.0f64;
    let mut optimal_concurrency = levels[0];

    for &c in &levels {
        println!("\n--- c={c} ---");

        let config = jugar_probar::llm::LoadTestConfig {
            concurrency: c,
            duration,
            prompts: prompts.clone(),
            runtime_name: args.runtime_name.clone(),
            warmup_duration: warmup,
            stream: args.stream,
            trace_level: None,
            slo_ttft_ms: None,
            slo_tpot_ms: None,
            slo_latency_ms: None,
            rate: jugar_probar::llm::RequestRate::Max,
            num_layers: args.num_layers,
            validate: jugar_probar::llm::ValidationMode::None,
            spike_threshold: 5.0,
            fail_on_quality: None,
        };

        let load_test = jugar_probar::llm::LoadTest::new(client.clone(), config);
        let result = load_test
            .run()
            .await
            .map_err(|e| CliError::Generic(e.to_string()))?;

        let p99 = result.latency_p99_ms;
        let throughput = result.throughput_rps;
        let decode = result.decode_tok_per_sec;

        // Set baseline from first level
        if baseline_p99.is_none() {
            baseline_p99 = Some(p99);
        }

        // Saturation detection
        let (saturated, saturation_reason) = if let Some(base_p99) = baseline_p99 {
            if base_p99 > 0.0 && p99 > args.saturation_threshold * base_p99 {
                (
                    true,
                    Some(format!(
                        "latency_p99 {:.0}ms > {:.1}x baseline {:.0}ms",
                        p99, args.saturation_threshold, base_p99
                    )),
                )
            } else {
                (false, None)
            }
        } else {
            (false, None)
        };

        // GH-33: quality-aware optimal selection — disqualify zero-decode levels
        let zero_quality = decode == 0.0;
        if throughput > best_throughput && !saturated && !zero_quality {
            best_throughput = throughput;
            optimal_concurrency = c;
        }

        let status = if saturated {
            " [SATURATED]"
        } else if zero_quality {
            " [ZERO QUALITY]"
        } else {
            ""
        };
        println!(
            "  Throughput: {throughput:.1} req/s, P99: {p99:.1}ms, Decode: {decode:.1} tok/s{status}",
        );

        sweep_levels.push(jugar_probar::llm::SweepLevel {
            concurrency: c,
            throughput_rps: throughput,
            latency_p99_ms: p99,
            decode_tok_s: decode,
            saturated,
            saturation_reason,
            result,
        });

        // Early stop
        if args.early_stop && saturated {
            println!("Early stop: saturation detected at c={c}");
            break;
        }
    }

    // GH-33: Pareto frontier excludes zero-quality and saturated levels
    let pareto_frontier: Vec<usize> = {
        let mut frontier = Vec::new();
        let mut max_throughput = 0.0f64;
        for level in &sweep_levels {
            let zero_quality = level.decode_tok_s == 0.0;
            if level.throughput_rps > max_throughput && !level.saturated && !zero_quality {
                max_throughput = level.throughput_rps;
                frontier.push(level.concurrency);
            }
        }
        frontier
    };

    let sweep_result = jugar_probar::llm::SweepResult {
        levels: sweep_levels,
        optimal_concurrency,
        optimal_throughput_rps: best_throughput,
        pareto_frontier: pareto_frontier.clone(),
    };

    println!("\n--- Sweep Summary ---");
    println!("Optimal:      c={optimal_concurrency} ({best_throughput:.1} req/s)");
    println!("Pareto front: {pareto_frontier:?}");

    if let Some(ref output_path) = args.output {
        let json = serde_json::to_string_pretty(&sweep_result)
            .map_err(|e| CliError::Generic(e.to_string()))?;
        std::fs::write(output_path, json).map_err(|e| CliError::Generic(e.to_string()))?;
        println!("Results written to {}", output_path.display());
    }

    Ok(())
}

// =============================================================================
// Feature 4: Dataset loading and generation
// =============================================================================

/// Load a JSONL dataset file into `ChatRequest` prompts and compute stats.
fn load_dataset(
    path: &Path,
) -> CliResult<(
    Vec<jugar_probar::llm::ChatRequest>,
    jugar_probar::llm::DatasetStats,
)> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::Generic(format!("Cannot read dataset: {e}")))?;

    let mut prompts = Vec::new();
    let mut input_lens = Vec::new();
    let mut max_tokens_vals = Vec::new();

    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            CliError::Generic(format!("Dataset line {}: parse error: {e}", line_no + 1))
        })?;

        let messages = entry
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| {
                CliError::Generic(format!(
                    "Dataset line {}: missing 'messages' array",
                    line_no + 1
                ))
            })?;

        let chat_messages: Vec<jugar_probar::llm::ChatMessage> = messages
            .iter()
            .filter_map(|m| {
                let role_str = m.get("role")?.as_str()?;
                let content = m.get("content")?.as_str()?;
                let role = match role_str {
                    "system" => jugar_probar::llm::Role::System,
                    "assistant" => jugar_probar::llm::Role::Assistant,
                    _ => jugar_probar::llm::Role::User,
                };
                Some(jugar_probar::llm::ChatMessage {
                    role,
                    content: content.to_string(),
                })
            })
            .collect();

        let max_tokens = entry
            .get("max_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(128) as u32;

        // Estimate input tokens (words * 1.3)
        let input_tokens: usize = chat_messages
            .iter()
            .map(|m| m.content.split_whitespace().count() + 4)
            .sum();
        let estimated_tokens = (input_tokens as f64 * 1.3) as u32;

        input_lens.push(f64::from(estimated_tokens));
        max_tokens_vals.push(f64::from(max_tokens));

        prompts.push(jugar_probar::llm::ChatRequest {
            model: String::new(),
            messages: chat_messages,
            temperature: Some(0.0),
            max_tokens: Some(max_tokens),
            stream: Some(false),
        });
    }

    if prompts.is_empty() {
        return Err(CliError::Generic("Dataset is empty".to_string()));
    }

    input_lens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    max_tokens_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let stats = jugar_probar::llm::DatasetStats {
        source: path.display().to_string(),
        total_prompts: prompts.len(),
        input_tokens: dist_summary(&input_lens),
        max_tokens_requested: dist_summary(&max_tokens_vals),
    };

    println!(
        "Dataset: {} ({} prompts, input [{:.0}-{:.0}] tokens)",
        path.display(),
        stats.total_prompts,
        stats.input_tokens[0],
        stats.input_tokens[3]
    );

    Ok((prompts, stats))
}

/// Compute [min, p50, p90, max] from sorted values.
fn dist_summary(sorted: &[f64]) -> [f64; 4] {
    if sorted.is_empty() {
        return [0.0; 4];
    }
    let p = |pct: f64| -> f64 {
        let idx = ((sorted.len() as f64 - 1.0) * pct).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    [sorted[0], p(0.5), p(0.9), sorted[sorted.len() - 1]]
}

/// Execute `probador llm gen-dataset`.
pub fn execute_llm_gen_dataset(args: &LlmGenDatasetArgs) -> CliResult<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(&args.output)
        .map_err(|e| CliError::Generic(format!("Cannot create {}: {e}", args.output.display())))?;

    // Simple lognormal-ish generation using Box-Muller from xorshift
    let mut rng_state: u64 = 42;

    for _ in 0..args.count {
        // Generate lognormal-distributed input length
        let input_len =
            sample_lognormal(&mut rng_state, args.input_mean, args.input_stddev).max(4.0) as usize;

        let output_len =
            sample_lognormal(&mut rng_state, args.output_mean, args.output_stddev).max(4.0) as u32;

        // Generate a prompt of approximately input_len tokens
        let prompt = generate_synthetic_prompt(input_len);

        let entry = serde_json::json!({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len
        });

        writeln!(
            file,
            "{}",
            serde_json::to_string(&entry).unwrap_or_default()
        )
        .map_err(|e| CliError::Generic(e.to_string()))?;
    }

    println!(
        "Generated {} entries → {}",
        args.count,
        args.output.display()
    );
    Ok(())
}

/// Sample from a lognormal-ish distribution using Box-Muller.
fn sample_lognormal(state: &mut u64, mean: f64, stddev: f64) -> f64 {
    // Box-Muller transform
    let u1 = next_uniform(state);
    let u2 = next_uniform(state);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    stddev.mul_add(z, mean).max(1.0)
}

fn next_uniform(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state) as f64 / u64::MAX as f64).max(1e-10)
}

/// Generate a synthetic prompt string of approximately `target_tokens` tokens.
fn generate_synthetic_prompt(target_tokens: usize) -> String {
    // ~1.3 tokens per word, so target_words = target_tokens / 1.3
    let target_words = (target_tokens as f64 / 1.3).max(1.0) as usize;
    let words = [
        "Explain",
        "the",
        "concept",
        "of",
        "data",
        "structures",
        "and",
        "algorithms",
        "in",
        "computer",
        "science",
        "including",
        "arrays",
        "linked",
        "lists",
        "trees",
        "graphs",
        "hash",
        "tables",
        "sorting",
        "searching",
        "dynamic",
        "programming",
        "recursion",
        "iteration",
        "complexity",
        "analysis",
        "optimization",
        "performance",
        "memory",
    ];
    let mut result = String::with_capacity(target_words * 6);
    for i in 0..target_words {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(words[i % words.len()]);
    }
    result
}

// =============================================================================
// Experiment tracking (GH-32)
// =============================================================================

/// Execute `probador llm experiment` subcommands.
pub fn execute_llm_experiment(args: &ExperimentArgs) -> CliResult<()> {
    use crate::ExperimentSubcommand;
    use jugar_probar::llm::experiment::{BudgetConfig, Experiment};

    match &args.subcommand {
        ExperimentSubcommand::Init(init_args) => {
            let mut exp = Experiment::new(&init_args.name);
            exp.description = init_args.description.clone();

            if init_args.max_gpu_hours.is_some() || init_args.max_cost_usd.is_some() {
                exp.budget = Some(BudgetConfig {
                    max_gpu_hours: init_args.max_gpu_hours,
                    max_cost_usd: init_args.max_cost_usd,
                    cost_per_gpu_hour: init_args.cost_per_gpu_hour,
                });
            }

            exp.save(&init_args.output)
                .map_err(|e| CliError::Generic(format!("Failed to save experiment: {e}")))?;

            eprintln!(
                "Experiment '{}' initialized → {}",
                init_args.name,
                init_args.output.display()
            );
            if let Some(ref budget) = exp.budget {
                if let Some(h) = budget.max_gpu_hours {
                    eprintln!("  Budget: {h:.1} GPU-hours");
                }
                if let Some(c) = budget.max_cost_usd {
                    eprintln!("  Budget: ${c:.2}");
                }
            }
            Ok(())
        }
        ExperimentSubcommand::Status(status_args) => {
            let exp = Experiment::load(&status_args.file)
                .map_err(|e| CliError::Generic(format!("Failed to load experiment: {e}")))?;

            println!("Experiment: {}", exp.name);
            if let Some(ref desc) = exp.description {
                println!("  Description: {desc}");
            }
            println!("  Created: {}", exp.created);
            println!("  Runs: {}", exp.runs.len());
            println!("  Total GPU-hours: {:.2}", exp.total_gpu_hours());
            if let Some(cost) = exp.total_cost() {
                println!("  Estimated cost: ${cost:.2}");
            }

            if let Some(ref audit) = exp.data_audit {
                println!(
                    "  Data audit: {}",
                    if audit.passed { "PASS" } else { "FAIL" }
                );
                for issue in &audit.issues {
                    println!("    - {issue}");
                }
            }

            for run in &exp.runs {
                println!(
                    "  Run '{}': {:?} ({:.2} GPU-hours)",
                    run.id, run.status, run.total_gpu_hours
                );
                if let Some(ref reason) = run.stop_reason {
                    println!("    Stop reason: {reason}");
                }
                if let Some(snap) = run.snapshots.last() {
                    for (k, v) in &snap.metrics {
                        println!("    {k}: {v:.4}");
                    }
                }
            }
            Ok(())
        }
        ExperimentSubcommand::Compare(cmp_args) => {
            let exp = Experiment::load(&cmp_args.file)
                .map_err(|e| CliError::Generic(format!("Failed to load experiment: {e}")))?;

            match exp.compare_runs(
                &cmp_args.run_a,
                &cmp_args.run_b,
                &cmp_args.metric,
                cmp_args.lower_is_better,
            ) {
                Some(cmp) => {
                    println!(
                        "Comparison: {} vs {} on '{}'",
                        cmp.run_a, cmp.run_b, cmp.metric
                    );
                    println!("  {}: {:.4}", cmp.run_a, cmp.value_a);
                    println!("  {}: {:.4}", cmp.run_b, cmp.value_b);
                    println!("  Diff: {:+.4} ({:+.1}%)", cmp.diff, cmp.diff_pct);
                    Ok(())
                }
                None => Err(CliError::Generic(format!(
                    "Cannot compare: run '{}' or '{}' not found, or metric '{}' missing",
                    cmp_args.run_a, cmp_args.run_b, cmp_args.metric
                ))),
            }
        }
    }
}

/// Execute `probador llm data-audit` — pre-flight data quality check.
pub fn execute_data_audit(args: &DataAuditArgs) -> CliResult<()> {
    use jugar_probar::llm::experiment::audit_jsonl_file;

    let result = audit_jsonl_file(&args.file, args.max_imbalance)
        .map_err(|e| CliError::Generic(format!("Data audit failed: {e}")))?;

    println!("Data Audit: {}", args.file.display());
    println!("  Samples: {}", result.total_samples);
    println!("  Classes: {}", result.label_distribution.len());
    for (label, count) in &result.label_distribution {
        println!("    {label}: {count}");
    }
    println!("  Imbalance ratio: {:.1}:1", result.imbalance_ratio);
    println!("  Result: {}", if result.passed { "PASS" } else { "FAIL" });
    for issue in &result.issues {
        println!("  Issue: {issue}");
    }

    if !result.passed {
        return Err(CliError::Generic(
            "Data audit failed — fix issues before training".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("30s").unwrap(), Duration::from_secs(30));
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("2m").unwrap(), Duration::from_secs(120));
    }

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration("1h").unwrap(), Duration::from_secs(3600));
    }

    #[test]
    fn test_parse_duration_raw_number() {
        assert_eq!(parse_duration("60").unwrap(), Duration::from_secs(60));
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert!(parse_duration("abc").is_err());
    }

    #[test]
    fn test_parse_role() {
        assert_eq!(parse_role("system"), jugar_probar::llm::Role::System);
        assert_eq!(parse_role("user"), jugar_probar::llm::Role::User);
        assert_eq!(parse_role("assistant"), jugar_probar::llm::Role::Assistant);
        assert_eq!(parse_role("SYSTEM"), jugar_probar::llm::Role::System);
        assert_eq!(parse_role("unknown"), jugar_probar::llm::Role::User);
    }

    #[test]
    fn test_yaml_parsing() {
        let yaml = r#"
tests:
  - name: basic_math
    messages:
      - role: user
        content: "What is 7 * 8?"
    expect_contains: "56"
    max_tokens: 32
    temperature: 0.0
  - name: pattern_test
    messages:
      - role: user
        content: "Hello"
    expect_pattern: "(?i)hello"
"#;
        let config: TestConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.tests.len(), 2);
        assert_eq!(config.tests[0].name, "basic_math");
        assert_eq!(config.tests[0].expect_contains, Some("56".to_string()));
        assert_eq!(config.tests[0].max_tokens, Some(32));
        assert!(config.tests[1].expect_pattern.is_some());
    }

    #[test]
    fn test_update_readme_section_new() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("README.md");
        std::fs::write(&path, "# My Project\n\nSome content.\n").unwrap();

        update_readme_section(&path, "| table data |").unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<!-- PERFORMANCE_START -->"));
        assert!(content.contains("| table data |"));
        assert!(content.contains("<!-- PERFORMANCE_END -->"));
    }

    #[test]
    fn test_update_readme_section_replace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("README.md");
        std::fs::write(
            &path,
            "# My Project\n<!-- PERFORMANCE_START -->\nold data\n<!-- PERFORMANCE_END -->\n",
        )
        .unwrap();

        update_readme_section(&path, "| new data |").unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("| new data |"));
        assert!(!content.contains("old data"));
    }

    #[test]
    fn test_correctness_result_serialization() {
        let result = CorrectnessResult {
            name: "test1".to_string(),
            passed: true,
            latency_ms: 150.5,
            output: "Hello!".to_string(),
            failures: vec![],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"name\":\"test1\""));
        assert!(json.contains("\"passed\":true"));
    }
}
