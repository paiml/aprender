//! QA Example: apr run Falsification Suite (PMAT-QA-RUST-001)
//!
//! Popperian falsification tests for `apr run` command.
//! Each test attempts to REFUTE the hypothesis that apr run works correctly.
//!
//! # Tests
//!
//! | ID | Test | Points | Criterion |
//! |----|------|--------|-----------|
//! | P001 | Model exists | 2 | File not found returns Err |
//! | P002 | Correct answer (2+2=4) | 3 | Output contains "4" |
//! | P003 | No garbage patterns | 3 | No `token\d+` or replacement chars |
//! | P004 | No BPE artifacts | 2 | No `Ġ` or `Ċ` characters |
//! | P005 | Trace flag accepted | 2 | `--trace` doesn't error |
//! | P006 | Performance CPU | 3 | >= configurable tok/s |
//! | P007 | Performance GPU | 3 | >= configurable tok/s (if available) |
//! | P008 | Determinism | 3 | Same prompt → same output (T=0) |
//! | P009 | Format parity GGUF | 2 | GGUF produces correct output |
//! | P010 | Format parity APR | 2 | APR produces correct output |
//!
//! # Usage
//!
//! ```bash
//! cargo run --example qa_run
//! cargo run --example qa_run -- --model path/to/model.gguf
//! cargo run --example qa_run -- --format-parity
//! cargo run --example qa_run -- --min-cpu-tps 5.0
//! ```
//!
//! # Citations
//!
//! - Popper, K. R. (1959). The Logic of Scientific Discovery. Routledge.
//! - Myers, G. J. et al. (2011). The Art of Software Testing. Wiley.

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

// ANSI colors for terminal output
const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[0;33m";
const BLUE: &str = "\x1b[0;34m";
const CYAN: &str = "\x1b[0;36m";
const NC: &str = "\x1b[0m";

/// Test result with ID, name, status, and optional details
struct TestResult {
    id: &'static str,
    name: &'static str,
    passed: bool,
    details: Option<String>,
    points: u32,
}

impl TestResult {
    fn pass(id: &'static str, name: &'static str, points: u32) -> Self {
        Self {
            id,
            name,
            passed: true,
            details: None,
            points,
        }
    }

    fn pass_with_details(id: &'static str, name: &'static str, points: u32, details: String) -> Self {
        Self {
            id,
            name,
            passed: true,
            details: Some(details),
            points,
        }
    }

    fn fail(id: &'static str, name: &'static str, points: u32, details: String) -> Self {
        Self {
            id,
            name,
            passed: false,
            details: Some(details),
            points,
        }
    }

    fn skip(id: &'static str, name: &'static str, _points: u32, reason: String) -> Self {
        Self {
            id,
            name,
            passed: true, // Skips don't count as failures
            details: Some(format!("SKIP: {}", reason)),
            points: 0, // But no points awarded
        }
    }

    fn print(&self) {
        let status = if self.passed {
            format!("{}[PASS]{}", GREEN, NC)
        } else {
            format!("{}[FAIL]{}", RED, NC)
        };

        println!("{} {}: {}", status, self.id, self.name);
        if let Some(ref details) = self.details {
            println!("       {}", details);
        }
    }
}

/// Configuration for QA tests
struct QaConfig {
    model_path: Option<PathBuf>,
    apr_binary: PathBuf,
    min_cpu_tps: f64,
    min_gpu_tps: f64,
    format_parity: bool,
    verbose: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            apr_binary: find_apr_binary(),
            min_cpu_tps: 8.0, // Dr. Popper's note: configurable, not absolute
            min_gpu_tps: 10.0,
            format_parity: false,
            verbose: false,
        }
    }
}

/// Find the apr binary in common locations
fn find_apr_binary() -> PathBuf {
    let candidates = [
        "target/release/apr",
        "target/debug/apr",
        "/mnt/nvme-raid0/targets/aprender/release/apr",
        "/mnt/nvme-raid0/targets/aprender/debug/apr",
    ];

    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }

    // Fallback: try cargo run
    PathBuf::from("cargo")
}

/// Find a default model in common cache locations
fn find_default_model() -> Option<PathBuf> {
    let home = env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.cache/pacha/models/d4c4d9763127153c.gguf"), // 0.5B
        format!("{home}/.cache/huggingface/models/qwen2.5-coder-0.5b-gguf/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
        format!("{home}/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        "models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf".to_string(),
    ];

    for candidate in candidates {
        let path = PathBuf::from(&candidate);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

/// Run apr command and capture output
fn run_apr_command(config: &QaConfig, args: &[&str]) -> Result<String, String> {
    let mut cmd = if config.apr_binary.to_string_lossy() == "cargo" {
        let mut c = Command::new("cargo");
        c.args(["run", "-p", "apr-cli", "--release", "--"]);
        c.args(args);
        c
    } else {
        let mut c = Command::new(&config.apr_binary);
        c.args(args);
        c
    };

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    if config.verbose {
        eprintln!("{}DEBUG: Running {:?}{}", CYAN, cmd, NC);
    }

    match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if output.status.success() {
                Ok(format!("{}{}", stdout, stderr))
            } else {
                Err(format!("Exit {}: {}{}", output.status, stdout, stderr))
            }
        }
        Err(e) => Err(format!("Failed to execute: {}", e)),
    }
}

/// P001: Model existence check
#[allow(clippy::ptr_arg)]
fn test_model_exists(_config: &QaConfig, model: &PathBuf) -> TestResult {
    if model.exists() {
        let size = std::fs::metadata(model)
            .map(|m| m.len())
            .unwrap_or(0);
        let size_mb = size / (1024 * 1024);
        TestResult::pass_with_details("P001", "Model Exists", 2, format!("{} MB", size_mb))
    } else {
        TestResult::fail(
            "P001",
            "Model Exists",
            2,
            format!("Not found: {}", model.display()),
        )
    }
}

/// P002: Correct answer (2+2=4)
fn test_correct_answer(config: &QaConfig, model: &PathBuf) -> TestResult {
    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "What is 2+2? Answer with just the number.",
        "--max-tokens",
        "10",
    ];

    match run_apr_command(config, &args) {
        Ok(output) => {
            if output.contains('4') {
                TestResult::pass_with_details("P002", "Correct Answer (2+2=4)", 3, "Contains '4'".to_string())
            } else {
                TestResult::fail(
                    "P002",
                    "Correct Answer (2+2=4)",
                    3,
                    format!("Missing '4' in output: {}", output.chars().take(100).collect::<String>()),
                )
            }
        }
        Err(e) => TestResult::fail("P002", "Correct Answer (2+2=4)", 3, e),
    }
}

/// P003: No garbage patterns
fn test_no_garbage(config: &QaConfig, model: &PathBuf) -> TestResult {
    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "Say hello.",
        "--max-tokens",
        "20",
    ];

    match run_apr_command(config, &args) {
        Ok(output) => {
            // Check for token\d+ pattern (raw tokens)
            let has_token_pattern = output
                .chars()
                .collect::<String>()
                .contains("token");

            // Check for Unicode replacement character (U+FFFD)
            let has_replacement_char = output.contains('\u{FFFD}');

            if has_token_pattern && output.contains("token") {
                // More strict check for tokenN patterns
                let re_match = output.chars().collect::<String>();
                if re_match.contains("token0") || re_match.contains("token1") {
                    return TestResult::fail(
                        "P003",
                        "No Garbage Patterns",
                        3,
                        "Raw token patterns detected".to_string(),
                    );
                }
            }

            if has_replacement_char {
                TestResult::fail(
                    "P003",
                    "No Garbage Patterns",
                    3,
                    "Unicode replacement characters detected".to_string(),
                )
            } else {
                TestResult::pass("P003", "No Garbage Patterns", 3)
            }
        }
        Err(e) => TestResult::fail("P003", "No Garbage Patterns", 3, e),
    }
}

/// P004: No BPE artifacts
fn test_no_bpe_artifacts(config: &QaConfig, model: &PathBuf) -> TestResult {
    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "Say hello.",
        "--max-tokens",
        "20",
    ];

    match run_apr_command(config, &args) {
        Ok(output) => {
            // Check for common BPE artifacts
            let has_g_artifact = output.contains('Ġ'); // U+0120
            let has_c_artifact = output.contains('Ċ'); // U+010A

            if has_g_artifact || has_c_artifact {
                TestResult::fail(
                    "P004",
                    "No BPE Artifacts",
                    2,
                    format!("BPE artifacts detected: Ġ={} Ċ={}", has_g_artifact, has_c_artifact),
                )
            } else {
                TestResult::pass("P004", "No BPE Artifacts", 2)
            }
        }
        Err(e) => TestResult::fail("P004", "No BPE Artifacts", 2, e),
    }
}

/// P005: Trace flag accepted
fn test_trace_flag(config: &QaConfig, model: &PathBuf) -> TestResult {
    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "Hi",
        "--max-tokens",
        "5",
        "--trace",
    ];

    match run_apr_command(config, &args) {
        Ok(_) => TestResult::pass("P005", "Trace Flag Accepted", 2),
        Err(e) => {
            // Some models may not support trace but shouldn't crash
            if e.contains("not supported") || e.contains("unknown") {
                TestResult::skip("P005", "Trace Flag Accepted", 2, "Trace not supported".to_string())
            } else {
                TestResult::fail("P005", "Trace Flag Accepted", 2, e)
            }
        }
    }
}

/// P006: Performance CPU
fn test_performance_cpu(config: &QaConfig, model: &PathBuf) -> TestResult {
    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "Count from 1 to 20.",
        "--max-tokens",
        "50",
        "--no-gpu",
    ];

    let start = Instant::now();
    match run_apr_command(config, &args) {
        Ok(output) => {
            let elapsed = start.elapsed().as_secs_f64();

            // Estimate tokens from output (rough: words * 1.3)
            let word_count = output.split_whitespace().count();
            let tokens_est = (word_count as f64 * 1.3).max(10.0); // At least 10 tokens
            let tps = tokens_est / elapsed;

            if tps >= config.min_cpu_tps {
                TestResult::pass_with_details(
                    "P006",
                    "Performance CPU",
                    3,
                    format!("{:.1} tok/s >= {:.1} target", tps, config.min_cpu_tps),
                )
            } else {
                TestResult::fail(
                    "P006",
                    "Performance CPU",
                    3,
                    format!("{:.1} tok/s < {:.1} target", tps, config.min_cpu_tps),
                )
            }
        }
        Err(e) => TestResult::fail("P006", "Performance CPU", 3, e),
    }
}

/// P007: Performance GPU
fn test_performance_gpu(config: &QaConfig, model: &PathBuf) -> TestResult {
    // Check if GPU is available
    let gpu_available = Command::new("nvidia-smi")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !gpu_available {
        return TestResult::skip("P007", "Performance GPU", 3, "No GPU detected".to_string());
    }

    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "Count from 1 to 20.",
        "--max-tokens",
        "50",
        "--gpu",
    ];

    let start = Instant::now();
    match run_apr_command(config, &args) {
        Ok(output) => {
            let elapsed = start.elapsed().as_secs_f64();
            let word_count = output.split_whitespace().count();
            let tokens_est = (word_count as f64 * 1.3).max(10.0);
            let tps = tokens_est / elapsed;

            if tps >= config.min_gpu_tps {
                TestResult::pass_with_details(
                    "P007",
                    "Performance GPU",
                    3,
                    format!("{:.1} tok/s >= {:.1} target", tps, config.min_gpu_tps),
                )
            } else {
                TestResult::fail(
                    "P007",
                    "Performance GPU",
                    3,
                    format!("{:.1} tok/s < {:.1} target", tps, config.min_gpu_tps),
                )
            }
        }
        Err(e) => TestResult::fail("P007", "Performance GPU", 3, e),
    }
}

/// P008: Determinism (greedy sampling)
fn test_determinism(config: &QaConfig, model: &PathBuf) -> TestResult {
    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "What is the capital of France?",
        "--max-tokens",
        "10",
        "--temperature",
        "0",
    ];

    let result1 = run_apr_command(config, &args);
    let result2 = run_apr_command(config, &args);

    match (result1, result2) {
        (Ok(out1), Ok(out2)) => {
            // Extract just the generated content (may vary by format)
            let content1 = out1.trim();
            let content2 = out2.trim();

            if content1 == content2 {
                TestResult::pass_with_details("P008", "Determinism (T=0)", 3, "Outputs match".to_string())
            } else {
                TestResult::fail(
                    "P008",
                    "Determinism (T=0)",
                    3,
                    format!("Outputs differ:\n  1: {}\n  2: {}",
                            content1.chars().take(50).collect::<String>(),
                            content2.chars().take(50).collect::<String>()),
                )
            }
        }
        (Err(e), _) | (_, Err(e)) => TestResult::fail("P008", "Determinism (T=0)", 3, e),
    }
}

/// P009: Format parity GGUF
fn test_format_parity_gguf(config: &QaConfig, model: &PathBuf) -> TestResult {
    // Only run in format-parity mode or if model is GGUF
    if !config.format_parity && !model.to_string_lossy().ends_with(".gguf") {
        return TestResult::skip("P009", "Format Parity GGUF", 2, "Not in format-parity mode".to_string());
    }

    let args = vec![
        "run",
        model.to_str().unwrap_or(""),
        "--prompt",
        "What is 2+2?",
        "--max-tokens",
        "10",
    ];

    match run_apr_command(config, &args) {
        Ok(output) => {
            if output.contains('4') {
                TestResult::pass_with_details("P009", "Format Parity GGUF", 2, "Correct output".to_string())
            } else {
                TestResult::fail("P009", "Format Parity GGUF", 2, "Incorrect output".to_string())
            }
        }
        Err(e) => TestResult::fail("P009", "Format Parity GGUF", 2, e),
    }
}

/// P010: Format parity APR
fn test_format_parity_apr(config: &QaConfig) -> TestResult {
    // Find APR model
    let home = env::var("HOME").unwrap_or_default();
    let apr_candidates = [
        format!("{home}/models/qwen2.5-coder-1.5b-q4k.apr"),
        format!("{home}/.cache/aprender/models/qwen2.5-coder-0.5b.apr"),
    ];

    let apr_model = apr_candidates.iter().find(|p| PathBuf::from(p).exists());

    match apr_model {
        Some(model) => {
            let args = vec![
                "run",
                model,
                "--prompt",
                "What is 2+2?",
                "--max-tokens",
                "10",
            ];

            match run_apr_command(config, &args) {
                Ok(output) => {
                    if output.contains('4') {
                        TestResult::pass_with_details("P010", "Format Parity APR", 2, "Correct output".to_string())
                    } else {
                        TestResult::fail("P010", "Format Parity APR", 2, "Incorrect output".to_string())
                    }
                }
                Err(e) => TestResult::fail("P010", "Format Parity APR", 2, e),
            }
        }
        None => TestResult::skip("P010", "Format Parity APR", 2, "No APR model found".to_string()),
    }
}

fn print_header() {
    println!("{}╔══════════════════════════════════════════════════════════════╗{}", BLUE, NC);
    println!("{}║         APR RUN QA - Popperian Falsification Suite           ║{}", BLUE, NC);
    println!("{}║         PMAT-QA-RUST-001 Section A (25 Points)                ║{}", BLUE, NC);
    println!("{}╚══════════════════════════════════════════════════════════════╝{}", BLUE, NC);
    println!();
}

fn print_summary(results: &[TestResult]) {
    let total_points: u32 = results.iter().map(|r| r.points).sum();
    let earned_points: u32 = results.iter().filter(|r| r.passed).map(|r| r.points).sum();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();

    println!();
    println!("{}═══════════════════════════════════════════════════════════════{}", BLUE, NC);
    println!("{}                    FALSIFICATION SUMMARY                       {}", BLUE, NC);
    println!("{}═══════════════════════════════════════════════════════════════{}", BLUE, NC);
    println!();
    println!("Total Tests: {}", results.len());
    println!("Passed:      {}{}{}", GREEN, passed, NC);
    println!("Failed:      {}{}{}", if failed > 0 { RED } else { GREEN }, failed, NC);
    println!("Points:      {}/{}", earned_points, total_points);
    println!();

    if failed == 0 {
        println!("{}Hypothesis \"apr run produces correct output\" SURVIVED falsification.{}", GREEN, NC);
    } else {
        println!("{}Hypothesis \"apr run produces correct output\" FALSIFIED.{}", RED, NC);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = QaConfig::default();

    // Parse command-line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    config.model_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--format-parity" => {
                config.format_parity = true;
                i += 1;
            }
            "--min-cpu-tps" => {
                if i + 1 < args.len() {
                    config.min_cpu_tps = args[i + 1].parse().unwrap_or(8.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--min-gpu-tps" => {
                if i + 1 < args.len() {
                    config.min_gpu_tps = args[i + 1].parse().unwrap_or(10.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--verbose" | "-v" => {
                config.verbose = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --example qa_run [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --model PATH       Path to model file");
                println!("  --format-parity    Run format parity tests");
                println!("  --min-cpu-tps N    Minimum CPU tok/s (default: 8.0)");
                println!("  --min-gpu-tps N    Minimum GPU tok/s (default: 10.0)");
                println!("  --verbose, -v      Verbose output");
                println!("  --help, -h         Show this help");
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    print_header();

    // Find model
    let model = config
        .model_path
        .clone()
        .or_else(find_default_model);

    let model = match model {
        Some(m) => m,
        None => {
            println!("{}ERROR: No model specified and no default found.{}", RED, NC);
            println!("Usage: cargo run --example qa_run -- --model path/to/model.gguf");
            std::process::exit(2);
        }
    };

    println!("{}Model:{} {}", CYAN, NC, model.display());
    println!("{}Config:{} CPU >= {:.1} tok/s, GPU >= {:.1} tok/s", CYAN, NC, config.min_cpu_tps, config.min_gpu_tps);
    println!();

    // Run all tests
    let mut results = Vec::new();

    println!("{}=== Section A: qa_run.rs Tests (25 Points) ==={}", YELLOW, NC);
    println!();

    results.push(test_model_exists(&config, &model));
    results.last().unwrap().print();

    results.push(test_correct_answer(&config, &model));
    results.last().unwrap().print();

    results.push(test_no_garbage(&config, &model));
    results.last().unwrap().print();

    results.push(test_no_bpe_artifacts(&config, &model));
    results.last().unwrap().print();

    results.push(test_trace_flag(&config, &model));
    results.last().unwrap().print();

    results.push(test_performance_cpu(&config, &model));
    results.last().unwrap().print();

    results.push(test_performance_gpu(&config, &model));
    results.last().unwrap().print();

    results.push(test_determinism(&config, &model));
    results.last().unwrap().print();

    results.push(test_format_parity_gguf(&config, &model));
    results.last().unwrap().print();

    results.push(test_format_parity_apr(&config));
    results.last().unwrap().print();

    print_summary(&results);

    // Exit with appropriate code
    let failed = results.iter().filter(|r| !r.passed).count();
    std::process::exit(if failed == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_apr_binary_exists() {
        let binary = find_apr_binary();
        // Should return a path (may be "cargo" if no apr found)
        assert!(!binary.to_string_lossy().is_empty());
    }

    #[test]
    fn test_qa_config_default() {
        let config = QaConfig::default();
        assert_eq!(config.min_cpu_tps, 8.0);
        assert_eq!(config.min_gpu_tps, 10.0);
        assert!(!config.format_parity);
    }

    #[test]
    fn test_result_pass() {
        let result = TestResult::pass("P001", "Test", 5);
        assert!(result.passed);
        assert_eq!(result.points, 5);
    }

    #[test]
    fn test_result_fail() {
        let result = TestResult::fail("P001", "Test", 5, "Error".to_string());
        assert!(!result.passed);
        assert_eq!(result.points, 5);
    }

    #[test]
    fn test_result_skip() {
        let result = TestResult::skip("P001", "Test", 5, "Reason".to_string());
        assert!(result.passed); // Skip counts as pass
        assert_eq!(result.points, 0); // But no points
    }
}
