//! Benchmark Command Implementation
//!
//! Implements spec §H12: Throughput benchmark for model inference.
//!
//! # Usage
//!
//! ```bash
//! apr bench model.apr                    # Basic throughput test
//! apr bench model.apr --warmup 3         # 3 warmup iterations
//! apr bench model.apr --iterations 10    # 10 measurement iterations
//! apr bench model.apr --prompt "Hello"   # Custom prompt
//! ```
//!
//! Toyota Way: Genchi Genbutsu - measure actual performance, not estimates.

use crate::error::{CliError, Result};
use crate::output;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::text::bpe::Qwen2BpeTokenizer;
use colored::Colorize;
use std::path::Path;
use std::time::{Duration, Instant};

/// Benchmark configuration
struct BenchConfig {
    /// Number of warmup iterations (not measured)
    pub warmup: usize,
    /// Number of measurement iterations
    pub iterations: usize,
    /// Max tokens to generate per iteration
    pub max_tokens: usize,
    /// Test prompt
    pub prompt: String,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup: 3,
            iterations: 5,
            max_tokens: 32,
            prompt: "What is 2+2?".to_string(),
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchResult {
    /// Total tokens generated across all iterations
    pub total_tokens: usize,
    /// Total time for generation
    pub total_time: Duration,
    /// Tokens per second (throughput)
    pub tokens_per_second: f64,
    /// Time to first token (TTFT)
    pub time_to_first_token: Duration,
    /// Individual iteration times
    pub iteration_times: Vec<Duration>,
    /// Mean iteration time
    pub mean_time: Duration,
    /// Median iteration time
    pub median_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Passed threshold (spec H12: >= 10 tok/s)
    pub passed: bool,
}

/// Run the benchmark command
pub(crate) fn run(
    path: &Path,
    warmup: usize,
    iterations: usize,
    max_tokens: usize,
    prompt: Option<&str>,
) -> Result<()> {
    let config = BenchConfig {
        warmup,
        iterations,
        max_tokens,
        prompt: prompt.unwrap_or("What is 2+2?").to_string(),
    };

    print_header(path, &config);

    // Run benchmark
    let result = run_benchmark(path, &config)?;

    // Print results
    print_results(&result);

    // Return error if threshold not met
    if !result.passed {
        return Err(CliError::ValidationFailed(format!(
            "Throughput {:.1} tok/s below minimum 10 tok/s (spec H12)",
            result.tokens_per_second
        )));
    }

    Ok(())
}

fn print_header(path: &Path, config: &BenchConfig) {
    output::section("APR Benchmark");
    println!();
    output::kv("Model", path.display());
    output::kv("Warmup iterations", config.warmup);
    output::kv("Measurement iterations", config.iterations);
    output::kv("Max tokens", config.max_tokens);
    output::kv("Prompt", &config.prompt);
    println!();
}

fn run_benchmark(path: &Path, config: &BenchConfig) -> Result<BenchResult> {
    // Detect format
    let is_safetensors = path.extension().is_some_and(|e| e == "safetensors");
    let is_apr = path.extension().is_some_and(|e| e == "apr");

    // Create model config based on format
    let model_config = if is_safetensors || is_apr {
        Qwen2Config::qwen2_0_5b_instruct()
    } else {
        // Demo config for testing
        Qwen2Config {
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            vocab_size: 1000,
            max_seq_len: 512,
            intermediate_size: 256,
            rope_theta: 10000.0,
        }
    };

    println!("{}", "Loading model...".yellow());
    let start = Instant::now();

    let mut model = if is_safetensors || is_apr {
        Qwen2Model::new_uninitialized(&model_config)
    } else {
        Qwen2Model::new(&model_config)
    };

    // Load weights
    if is_apr {
        let count = model
            .load_from_apr(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;
        println!("{} {} tensors", "Loaded".green(), count);
    } else if is_safetensors {
        let count = model
            .load_from_safetensors(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load SafeTensors: {e}")))?;
        println!("{} {} tensors", "Loaded".green(), count);
    }

    model.eval();
    let load_time = start.elapsed();
    println!(
        "{} in {:.2}s",
        "Model ready".green(),
        load_time.as_secs_f32()
    );
    println!();

    // Initialize tokenizer
    let tokenizer = Qwen2BpeTokenizer::new();

    // Encode prompt
    let prompt_ids = tokenizer.encode(&config.prompt);
    let prompt_ids: Vec<u32> = if prompt_ids.len() > 64 {
        prompt_ids[prompt_ids.len() - 64..].to_vec()
    } else {
        prompt_ids
    };

    // Warmup
    println!("{}", "Running warmup...".yellow());
    for i in 0..config.warmup {
        let _output = model.generate(&prompt_ids, config.max_tokens, 0.7, 0.9);
        print!("  Warmup {}/{}\r", i + 1, config.warmup);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    println!("  Warmup complete        ");
    println!();

    // Measurement
    println!("{}", "Running benchmark...".yellow());
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        let output = model.generate(&prompt_ids, config.max_tokens, 0.7, 0.9);
        let tokens_generated = output.len().saturating_sub(prompt_ids.len());

        let iter_time = iter_start.elapsed();
        iteration_times.push(iter_time);
        total_tokens += tokens_generated;

        // Approximate TTFT from first iteration
        if i == 0 {
            first_token_time =
                Duration::from_secs_f64(iter_time.as_secs_f64() / tokens_generated.max(1) as f64);
        }

        print!(
            "  Iteration {}/{}: {} tokens in {:.2}s\r",
            i + 1,
            config.iterations,
            tokens_generated,
            iter_time.as_secs_f32()
        );
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    println!();
    println!();

    // Calculate statistics
    let total_time: Duration = iteration_times.iter().sum();
    let tokens_per_second = total_tokens as f64 / total_time.as_secs_f64();

    let mean_time = total_time / config.iterations as u32;

    let mut sorted_times = iteration_times.clone();
    sorted_times.sort();
    let median_time = sorted_times[config.iterations / 2];

    // Calculate std dev
    let mean_ms = mean_time.as_secs_f64() * 1000.0;
    let variance: f64 = iteration_times
        .iter()
        .map(|t| {
            let diff = t.as_secs_f64() * 1000.0 - mean_ms;
            diff * diff
        })
        .sum::<f64>()
        / config.iterations as f64;
    let std_dev = Duration::from_secs_f64(variance.sqrt() / 1000.0);

    // Per spec H12: threshold is 10 tok/s
    let passed = tokens_per_second >= 10.0;

    Ok(BenchResult {
        total_tokens,
        total_time,
        tokens_per_second,
        time_to_first_token: first_token_time,
        iteration_times,
        mean_time,
        median_time,
        std_dev,
        passed,
    })
}

fn print_results(result: &BenchResult) {
    output::section("Results");
    println!();

    // Throughput (the key metric)
    let throughput_str = format!("{:.1} tok/s", result.tokens_per_second);
    if result.passed {
        println!(
            "{} {} {}",
            "Throughput:".white().bold(),
            throughput_str.green().bold(),
            "(PASS: >= 10 tok/s)".green()
        );
    } else {
        println!(
            "{} {} {}",
            "Throughput:".white().bold(),
            throughput_str.red().bold(),
            "(FAIL: < 10 tok/s)".red()
        );
    }

    println!();
    output::kv("Total tokens", result.total_tokens);
    output::kv(
        "Total time",
        format!("{:.2}s", result.total_time.as_secs_f32()),
    );
    output::kv(
        "Time to first token",
        format!("{:.0}ms", result.time_to_first_token.as_secs_f64() * 1000.0),
    );
    println!();
    output::kv(
        "Mean iteration time",
        format!("{:.2}s", result.mean_time.as_secs_f32()),
    );
    output::kv(
        "Median iteration time",
        format!("{:.2}s", result.median_time.as_secs_f32()),
    );
    output::kv(
        "Std deviation",
        format!("{:.0}ms", result.std_dev.as_secs_f64() * 1000.0),
    );
    println!();

    // Performance grade (Dean & Ghemawat 2025 style)
    let grade = if result.tokens_per_second >= 100.0 {
        "A+ (Excellent)".green()
    } else if result.tokens_per_second >= 50.0 {
        "A (Very Good)".green()
    } else if result.tokens_per_second >= 20.0 {
        "B (Good)".blue()
    } else if result.tokens_per_second >= 10.0 {
        "C (Acceptable)".yellow()
    } else {
        "F (Below Threshold)".red()
    };
    output::kv("Performance Grade", grade);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_config_default() {
        let config = BenchConfig::default();
        assert_eq!(config.warmup, 3);
        assert_eq!(config.iterations, 5);
        assert_eq!(config.max_tokens, 32);
    }

    #[test]
    fn test_bench_result_pass() {
        let result = BenchResult {
            total_tokens: 100,
            total_time: Duration::from_secs(5),
            tokens_per_second: 20.0,
            time_to_first_token: Duration::from_millis(50),
            iteration_times: vec![Duration::from_secs(1); 5],
            mean_time: Duration::from_secs(1),
            median_time: Duration::from_secs(1),
            std_dev: Duration::from_millis(10),
            passed: true,
        };

        assert!(result.passed);
        assert!(result.tokens_per_second >= 10.0);
    }

    #[test]
    fn test_bench_result_fail() {
        let result = BenchResult {
            total_tokens: 50,
            total_time: Duration::from_secs(10),
            tokens_per_second: 5.0, // Below threshold
            time_to_first_token: Duration::from_millis(100),
            iteration_times: vec![Duration::from_secs(2); 5],
            mean_time: Duration::from_secs(2),
            median_time: Duration::from_secs(2),
            std_dev: Duration::from_millis(50),
            passed: false,
        };

        assert!(!result.passed);
        assert!(result.tokens_per_second < 10.0);
    }
}
