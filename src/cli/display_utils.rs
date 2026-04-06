
/// Format file size in human-readable form
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Display model information based on file type
pub fn display_model_info(model_ref: &str, file_data: &[u8]) -> Result<()> {
    use crate::format::{APR_MAGIC, GGUF_MAGIC};

    if model_ref.ends_with(".gguf") || file_data.starts_with(GGUF_MAGIC) {
        use crate::gguf::GGUFModel;
        let gguf = GGUFModel::from_bytes(file_data)?;
        println!("  Format: GGUF v{}", gguf.header.version);
        println!("  Tensors: {}", gguf.header.tensor_count);
    } else if model_ref.ends_with(".safetensors") {
        use crate::safetensors::SafetensorsModel;
        let st = SafetensorsModel::from_bytes(file_data)?;
        println!("  Format: SafeTensors");
        println!("  Tensors: {}", st.tensors.len());
    } else if model_ref.ends_with(".apr") || file_data.starts_with(APR_MAGIC) {
        use crate::model_loader::read_apr_model_type;
        let model_type = read_apr_model_type(file_data).unwrap_or_else(|| "Unknown".to_string());
        println!("  Format: APR (Aprender Native)");
        println!("  Model Type: {model_type}");
    } else {
        println!("  Format: Unknown ({} bytes)", file_data.len());
    }
    Ok(())
}

/// Run visualization demo
pub fn run_visualization(use_color: bool, samples: usize) {
    use crate::viz::{
        print_benchmark_results, render_ascii_histogram, render_sparkline, BenchmarkData,
    };

    println!("Realizar Benchmark Visualization Demo");
    println!("=====================================");
    println!();

    // Generate test benchmark data (simulating inference latencies)
    let mut rng_state = 42u64;
    let latencies: Vec<f64> = (0..samples)
        .map(|_| {
            // Simple LCG for reproducible pseudo-random numbers
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let uniform = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            // Log-normal distribution (typical for latencies)
            let log_mean = 3.0; // ~20us median
            let log_std = 0.5;
            (log_mean + log_std * (2.0 * uniform - 1.0)).exp()
        })
        .collect();

    // Demo 1: Sparkline
    println!("1. Sparkline (latency trend)");
    println!("   {}", render_sparkline(&latencies, 60));
    println!();

    // Demo 2: ASCII histogram
    println!("2. ASCII Histogram (latency distribution)");
    let hist = render_ascii_histogram(&latencies, 12, 50);
    for line in hist.lines() {
        println!("   {line}");
    }
    println!();

    // Demo 3: Full benchmark report
    println!("3. Full Benchmark Report");
    let data = BenchmarkData::new("inference_latency", latencies);
    print_benchmark_results(&data, use_color);
    println!();

    // Demo 4: Multi-benchmark comparison
    println!("4. Multi-Benchmark Comparison");
    println!();

    let benchmarks = [
        ("tensor_add", 15.2, 18.1),
        ("tensor_mul", 16.8, 20.3),
        ("matmul_128", 145.3, 172.1),
        ("softmax", 23.4, 28.9),
        ("attention", 892.1, 1024.5),
    ];

    println!(
        "   {:.<20} {:>10} {:>10} {:>10}",
        "Benchmark", "p50 (us)", "p99 (us)", "Trend"
    );
    println!("   {}", "-".repeat(55));

    for (name, p50, p99) in benchmarks {
        // Generate mini trend data
        let trend: Vec<f64> = (0..20)
            .map(|i| p50 + (i as f64 / 20.0) * (p99 - p50) * 0.3)
            .collect();
        let sparkline = render_sparkline(&trend, 10);
        println!("   {:.<20} {:>10.1} {:>10.1} {}", name, p50, p99, sparkline);
    }
    println!();

    println!("Visualization powered by trueno-viz");
}

/// Print benchmark configuration info
fn print_bench_config(
    runtime_name: &str,
    model: Option<&str>,
    url: Option<&str>,
    output: Option<&str>,
) {
    println!("Benchmark Configuration:");
    println!("  Runtime: {runtime_name}");
    if let Some(m) = model {
        println!("  Model: {m}");
    }
    if let Some(u) = url {
        println!("  URL: {u}");
    }
    if let Some(o) = output {
        println!("  Output: {o}");
    }
    println!();
}

/// Write benchmark results to JSON file
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
fn write_bench_json(
    output_path: &str,
    stdout: &str,
    suite: Option<&str>,
    runtime: Option<&str>,
    model: Option<&str>,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let results = parse_cargo_bench_output(stdout, suite);

    let json_output = serde_json::json!({
        "version": "1.0",
        "timestamp": timestamp,
        "runtime": runtime.unwrap_or("realizar"),
        "suite": suite,
        "model": model,
        "results": results,
        "raw_output": stdout
    });

    let mut file = File::create(output_path).map_err(|e| RealizarError::IoError {
        message: format!("Failed to create output file {output_path}: {e}"),
    })?;

    file.write_all(
        serde_json::to_string_pretty(&json_output)
            .expect("JSON serialization of benchmark results failed")
            .as_bytes(),
    )
    .map_err(|e| RealizarError::IoError {
        message: format!("Failed to write to output file {output_path}: {e}"),
    })?;

    println!();
    println!("Benchmark results written to: {output_path}");
    Ok(())
}

/// Print benchmark usage help
fn print_bench_usage() {
    println!("Available benchmark suites:");
    println!();
    for (name, description) in BENCHMARK_SUITES {
        println!("  {name:<12} - {description}");
    }
    println!();
    println!("Usage:");
    println!("  realizar bench                        # Run all benchmarks");
    println!("  realizar bench tensor_ops             # Run specific suite");
    println!("  realizar bench --list                 # List available suites");
    println!("  realizar bench --runtime realizar     # Specify runtime");
    println!("  realizar bench --output results.json  # Save JSON results");
    println!();
    println!("External Runtime Benchmarking (REAL HTTP calls):");
    println!("  realizar bench --runtime ollama --url http://localhost:11434 --model llama3.2");
    println!("  realizar bench --runtime vllm --url http://localhost:8000 --model meta-llama/Llama-3.2-1B");
    println!("  realizar bench --runtime llama-cpp --url http://localhost:8080");
}

/// Validate benchmark suite name and print error if invalid
fn validate_suite_or_error(suite_name: &str) -> bool {
    if BENCHMARK_SUITES.iter().any(|(name, _)| *name == suite_name) {
        return true;
    }
    eprintln!("Error: Unknown benchmark suite '{suite_name}'");
    eprintln!();
    eprintln!("Available suites:");
    for (name, _) in BENCHMARK_SUITES {
        eprintln!("  {name}");
    }
    false
}

/// Execute cargo bench and capture or stream output
fn execute_cargo_bench(
    cmd: &mut std::process::Command,
    capture: bool,
) -> Result<Option<std::process::Output>> {
    if capture {
        let output = cmd
            .output()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "run_benchmarks".to_string(),
                reason: format!("Failed to execute cargo bench: {e}"),
            })?;
        return Ok(Some(output));
    }
    // Just run and show output directly
    let status = cmd
        .status()
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "run_benchmarks".to_string(),
            reason: format!("Failed to execute cargo bench: {e}"),
        })?;
    if !status.success() {
        return Err(RealizarError::UnsupportedOperation {
            operation: "run_benchmarks".to_string(),
            reason: format!("Benchmarks failed with exit code: {:?}", status.code()),
        });
    }
    Ok(None)
}

/// Run benchmarks with cargo bench or real HTTP client
pub fn run_benchmarks(
    suite: Option<String>,
    list: bool,
    runtime: Option<String>,
    model: Option<String>,
    url: Option<String>,
    output: Option<String>,
) -> Result<()> {
    if list {
        print_bench_usage();
        return Ok(());
    }

    let runtime_name = runtime.clone().unwrap_or_else(|| "realizar".to_string());
    print_bench_config(
        &runtime_name,
        model.as_deref(),
        url.as_deref(),
        output.as_deref(),
    );

    // Check if this is an external runtime benchmark (requires bench-http feature)
    if let (Some(ref rt), Some(ref server_url)) = (&runtime, &url) {
        return run_external_benchmark(rt, server_url, model.as_deref(), output.as_deref());
    }

    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("bench");

    if let Some(ref suite_name) = suite {
        if !validate_suite_or_error(suite_name) {
            std::process::exit(1);
        }
        cmd.arg("--bench").arg(suite_name);
    }

    println!("Running benchmarks...");
    println!();

    let bench_output = match execute_cargo_bench(&mut cmd, output.is_some())? {
        Some(out) => out,
        None => return Ok(()),
    };

    if !bench_output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&bench_output.stderr));
        return Err(RealizarError::UnsupportedOperation {
            operation: "run_benchmarks".to_string(),
            reason: format!(
                "Benchmarks failed with exit code: {:?}",
                bench_output.status.code()
            ),
        });
    }

    // Print benchmark output to console
    let stdout = String::from_utf8_lossy(&bench_output.stdout);
    print!("{stdout}");

    // Generate JSON output
    if let Some(ref output_path) = output {
        write_bench_json(
            output_path,
            &stdout,
            suite.as_deref(),
            runtime.as_deref(),
            model.as_deref(),
        )?;
    }

    Ok(())
}

/// Parse a single cargo bench output line into a JSON result
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
fn parse_bench_line(line: &str, suite: Option<&str>) -> Option<serde_json::Value> {
    if !line.contains("bench:") || !line.contains("ns/iter") {
        return None;
    }
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    let test_idx = parts.iter().position(|&p| p == "test")?;
    let name = parts.get(test_idx + 1)?;
    let bench_idx = parts.iter().position(|&p| p == "bench:")?;
    let time_str = parts.get(bench_idx + 1)?;
    let time_ns = time_str.replace(',', "").parse::<u64>().ok()?;
    Some(serde_json::json!({
        "name": name,
        "time_ns": time_ns,
        "suite": suite
    }))
}

/// Parse cargo bench output to extract benchmark results
fn parse_cargo_bench_output(output: &str, suite: Option<&str>) -> Vec<serde_json::Value> {
    output
        .lines()
        .filter_map(|line| parse_bench_line(line, suite))
        .collect()
}

/// Execute a single benchmark request for the given runtime
#[cfg(feature = "bench-http")]
fn execute_runtime_request(
    client: &crate::http_client::ModelHttpClient,
    runtime: &str,
    url: &str,
    model: Option<&str>,
    prompt: &str,
) -> Result<crate::http_client::InferenceTiming> {
    use crate::http_client::{CompletionRequest, OllamaOptions, OllamaRequest};

    match runtime.to_lowercase().as_str() {
        "ollama" => {
            let request = OllamaRequest {
                model: model.unwrap_or("llama3.2").to_string(),
                prompt: prompt.to_string(),
                stream: false,
                options: Some(OllamaOptions {
                    num_predict: Some(50),
                    temperature: Some(0.7),
                }),
            };
            client
                .ollama_generate(url, &request)
                .map_err(|e| RealizarError::ConnectionError(e.to_string()))
        },
        "vllm" => {
            let request = CompletionRequest {
                model: model.unwrap_or("default").to_string(),
                prompt: prompt.to_string(),
                max_tokens: 50,
                temperature: Some(0.7),
                stream: false,
            };
            client
                .openai_completion(url, &request, None)
                .map_err(|e| RealizarError::ConnectionError(e.to_string()))
        },
        "llama-cpp" => {
            let request = CompletionRequest {
                model: "default".to_string(),
                prompt: prompt.to_string(),
                max_tokens: 50,
                temperature: Some(0.7),
                stream: false,
            };
            client
                .llamacpp_completion(url, &request)
                .map_err(|e| RealizarError::ConnectionError(e.to_string()))
        },
        _ => Err(RealizarError::UnsupportedOperation {
            operation: "external_benchmark".to_string(),
            reason: format!(
                "Unknown runtime: {}. Supported: ollama, vllm, llama-cpp",
                runtime
            ),
        }),
    }
}
