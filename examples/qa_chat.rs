//! QA Example: apr chat Falsification Suite (PMAT-QA-RUST-001)
//!
//! Popperian falsification tests for `apr chat` command.
//!
//! # Tests
//!
//! | ID | Test | Points | Criterion |
//! |----|------|--------|-----------|
//! | P011 | Model exists | 2 | File not found returns Err |
//! | P012 | Correct answer (2+2=4) | 3 | Output contains "4" |
//! | P013 | No garbage patterns | 3 | No raw tokens or replacement chars |
//! | P014 | No BPE artifacts | 2 | No `Ġ` or `Ċ` characters |
//! | P015 | Performance CPU | 5 | >= configurable tok/s |
//! | P016 | Performance GPU | 5 | >= configurable tok/s (if available) |
//!
//! # Usage
//!
//! ```bash
//! cargo run --example qa_chat
//! cargo run --example qa_chat -- --model path/to/model.gguf
//! cargo run --example qa_chat -- --format-parity
//! ```
//!
//! # Citations
//!
//! - Popper, K. R. (1959). The Logic of Scientific Discovery. Routledge.

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

// ANSI colors
const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[0;33m";
const BLUE: &str = "\x1b[0;34m";
const CYAN: &str = "\x1b[0;36m";
const NC: &str = "\x1b[0m";

/// Test result
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

    fn pass_with_details(
        id: &'static str,
        name: &'static str,
        points: u32,
        details: String,
    ) -> Self {
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
            passed: true,
            details: Some(format!("SKIP: {}", reason)),
            points: 0,
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

/// Configuration
struct QaConfig {
    model_path: Option<PathBuf>,
    apr_binary: PathBuf,
    min_cpu_tps: f64,
    min_gpu_tps: f64,
    format_parity: bool,
    verbose: bool,
    #[allow(dead_code)]
    timeout_secs: u64,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            apr_binary: find_apr_binary(),
            // Conservative defaults for 1.5B Q4_K model (PMAT-SHOWCASE-METHODOLOGY-001)
            // Word-based tok/s estimation has high variance (~30%)
            // Observed: 4-20 tok/s CPU, 7-40 tok/s GPU depending on prompt/output
            min_cpu_tps: 3.0, // Conservative: observed 3.9-20 tok/s
            min_gpu_tps: 5.0, // Conservative: observed 7-40 tok/s
            format_parity: false,
            verbose: false,
            timeout_secs: 60,
        }
    }
}

fn find_apr_binary() -> PathBuf {
    // Check custom target directory FIRST (common dev setup)
    let candidates = [
        "/mnt/nvme-raid0/targets/aprender/release/apr",
        "/mnt/nvme-raid0/targets/aprender/debug/apr",
        "target/release/apr",
        "target/debug/apr",
    ];
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("cargo")
}

fn find_default_model() -> Option<PathBuf> {
    let home = env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        format!("{home}/.cache/pacha/models/d4c4d9763127153c.gguf"),
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

/// Run apr chat with piped input
fn run_chat_command(
    config: &QaConfig,
    model: &PathBuf,
    input: &str,
    extra_args: &[&str],
) -> Result<(String, f64), String> {
    let mut args = vec!["chat", model.to_str().unwrap_or("")];
    args.extend(extra_args);

    let mut cmd = if config.apr_binary.to_string_lossy() == "cargo" {
        let mut c = Command::new("cargo");
        c.args(["run", "-p", "apr-cli", "--release", "--"]);
        c.args(&args);
        c
    } else {
        let mut c = Command::new(&config.apr_binary);
        c.args(&args);
        c
    };

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if config.verbose {
        eprintln!(
            "{}DEBUG: Running {:?} with input '{}'{}",
            CYAN, cmd, input, NC
        );
    }

    let start = Instant::now();

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn: {}", e))?;

    // Write input to stdin
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(input.as_bytes());
        let _ = stdin.write_all(b"\n");
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait: {}", e))?;
    let elapsed = start.elapsed().as_secs_f64();

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok((format!("{}{}", stdout, stderr), elapsed))
}

/// P011: Model existence check
fn test_model_exists(model: &PathBuf) -> TestResult {
    if model.exists() {
        let size = std::fs::metadata(model).map(|m| m.len()).unwrap_or(0);
        let size_mb = size / (1024 * 1024);
        TestResult::pass_with_details("P011", "Model Exists", 2, format!("{} MB", size_mb))
    } else {
        TestResult::fail(
            "P011",
            "Model Exists",
            2,
            format!("Not found: {}", model.display()),
        )
    }
}

/// P012: Correct answer
fn test_correct_answer(config: &QaConfig, model: &PathBuf) -> TestResult {
    let extra_args = ["--max-tokens", "10"];
    let input = "What is 2+2? Answer with just the number.";

    match run_chat_command(config, model, input, &extra_args) {
        Ok((output, _)) => {
            if output.contains('4') {
                TestResult::pass_with_details(
                    "P012",
                    "Correct Answer",
                    3,
                    "Contains '4'".to_string(),
                )
            } else {
                TestResult::fail(
                    "P012",
                    "Correct Answer",
                    3,
                    format!(
                        "Missing '4': {}",
                        output.chars().take(100).collect::<String>()
                    ),
                )
            }
        }
        Err(e) => TestResult::fail("P012", "Correct Answer", 3, e),
    }
}

/// P013: No garbage patterns
fn test_no_garbage(config: &QaConfig, model: &PathBuf) -> TestResult {
    let extra_args = ["--max-tokens", "20"];
    let input = "Say hello.";

    match run_chat_command(config, model, input, &extra_args) {
        Ok((output, _)) => {
            let has_replacement = output.contains('\u{FFFD}');
            let has_raw_tokens = output.contains("token0") || output.contains("token1");

            if has_replacement || has_raw_tokens {
                TestResult::fail(
                    "P013",
                    "No Garbage Patterns",
                    3,
                    "Garbage detected".to_string(),
                )
            } else {
                TestResult::pass("P013", "No Garbage Patterns", 3)
            }
        }
        Err(e) => TestResult::fail("P013", "No Garbage Patterns", 3, e),
    }
}

/// P014: No BPE artifacts
fn test_no_bpe_artifacts(config: &QaConfig, model: &PathBuf) -> TestResult {
    let extra_args = ["--max-tokens", "20"];
    let input = "Say hello.";

    match run_chat_command(config, model, input, &extra_args) {
        Ok((output, _)) => {
            if output.contains('Ġ') || output.contains('Ċ') {
                TestResult::fail(
                    "P014",
                    "No BPE Artifacts",
                    2,
                    "BPE artifacts detected".to_string(),
                )
            } else {
                TestResult::pass("P014", "No BPE Artifacts", 2)
            }
        }
        Err(e) => TestResult::fail("P014", "No BPE Artifacts", 2, e),
    }
}

/// P015: Performance CPU
fn test_performance_cpu(config: &QaConfig, model: &PathBuf) -> TestResult {
    let extra_args = ["--max-tokens", "50", "--no-gpu"];
    let input = "Write a short poem about coding.";

    match run_chat_command(config, model, input, &extra_args) {
        Ok((output, elapsed)) => {
            let word_count = output.split_whitespace().count();
            let tokens_est = (word_count as f64 * 1.3).max(10.0);
            let tps = tokens_est / elapsed;

            if tps >= config.min_cpu_tps {
                TestResult::pass_with_details(
                    "P015",
                    "Performance CPU",
                    5,
                    format!("{:.1} tok/s >= {:.1}", tps, config.min_cpu_tps),
                )
            } else {
                TestResult::fail(
                    "P015",
                    "Performance CPU",
                    5,
                    format!("{:.1} tok/s < {:.1}", tps, config.min_cpu_tps),
                )
            }
        }
        Err(e) => TestResult::fail("P015", "Performance CPU", 5, e),
    }
}

/// P016: Performance GPU
fn test_performance_gpu(config: &QaConfig, model: &PathBuf) -> TestResult {
    let gpu_available = Command::new("nvidia-smi")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !gpu_available {
        return TestResult::skip("P016", "Performance GPU", 5, "No GPU detected".to_string());
    }

    let extra_args = ["--max-tokens", "50", "--gpu"];
    let input = "Write a short poem about coding.";

    match run_chat_command(config, model, input, &extra_args) {
        Ok((output, elapsed)) => {
            let word_count = output.split_whitespace().count();
            let tokens_est = (word_count as f64 * 1.3).max(10.0);
            let tps = tokens_est / elapsed;

            if tps >= config.min_gpu_tps {
                TestResult::pass_with_details(
                    "P016",
                    "Performance GPU",
                    5,
                    format!("{:.1} tok/s >= {:.1}", tps, config.min_gpu_tps),
                )
            } else {
                TestResult::fail(
                    "P016",
                    "Performance GPU",
                    5,
                    format!("{:.1} tok/s < {:.1}", tps, config.min_gpu_tps),
                )
            }
        }
        Err(e) => TestResult::fail("P016", "Performance GPU", 5, e),
    }
}

fn print_header() {
    println!(
        "{}╔══════════════════════════════════════════════════════════════╗{}",
        BLUE, NC
    );
    println!(
        "{}║         APR CHAT QA - Popperian Falsification Suite          ║{}",
        BLUE, NC
    );
    println!(
        "{}║         PMAT-QA-RUST-001 Section B (20 Points)                ║{}",
        BLUE, NC
    );
    println!(
        "{}╚══════════════════════════════════════════════════════════════╝{}",
        BLUE, NC
    );
    println!();
}

fn print_summary(results: &[TestResult]) {
    let earned: u32 = results.iter().filter(|r| r.passed).map(|r| r.points).sum();
    let total: u32 = results.iter().map(|r| r.points).sum();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();

    println!();
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        BLUE, NC
    );
    println!(
        "Total: {}, Passed: {}{}{}, Failed: {}{}{}",
        results.len(),
        GREEN,
        passed,
        NC,
        if failed > 0 { RED } else { GREEN },
        failed,
        NC
    );
    println!("Points: {}/{}", earned, total);

    if failed == 0 {
        println!(
            "{}Hypothesis \"apr chat produces correct output\" SURVIVED.{}",
            GREEN, NC
        );
    } else {
        println!(
            "{}Hypothesis \"apr chat produces correct output\" FALSIFIED.{}",
            RED, NC
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = QaConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                config.model_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--format-parity" => {
                config.format_parity = true;
                i += 1;
            }
            "--min-cpu-tps" if i + 1 < args.len() => {
                config.min_cpu_tps = args[i + 1].parse().unwrap_or(30.0);
                i += 2;
            }
            "--verbose" | "-v" => {
                config.verbose = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --example qa_chat [OPTIONS]");
                println!("  --model PATH       Path to model file");
                println!("  --min-cpu-tps N    Minimum CPU tok/s (default: 30.0)");
                println!("  --verbose          Verbose output");
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    print_header();

    let model = config.model_path.clone().or_else(find_default_model);
    let model = match model {
        Some(m) => m,
        None => {
            println!("{}ERROR: No model found.{}", RED, NC);
            std::process::exit(2);
        }
    };

    println!("{}Model:{} {}", CYAN, NC, model.display());
    println!();

    let mut results = Vec::new();
    println!(
        "{}=== Section B: qa_chat.rs Tests (20 Points) ==={}",
        YELLOW, NC
    );
    println!();

    results.push(test_model_exists(&model));
    results.last().unwrap().print();

    results.push(test_correct_answer(&config, &model));
    results.last().unwrap().print();

    results.push(test_no_garbage(&config, &model));
    results.last().unwrap().print();

    results.push(test_no_bpe_artifacts(&config, &model));
    results.last().unwrap().print();

    results.push(test_performance_cpu(&config, &model));
    results.last().unwrap().print();

    results.push(test_performance_gpu(&config, &model));
    results.last().unwrap().print();

    print_summary(&results);

    let failed = results.iter().filter(|r| !r.passed).count();
    std::process::exit(if failed == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QaConfig::default();
        // Conservative defaults for 1.5B Q4_K model (PMAT-SHOWCASE-METHODOLOGY-001)
        assert!((config.min_cpu_tps - 3.0).abs() < 0.01);
        assert!((config.min_gpu_tps - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_result_types() {
        let pass = TestResult::pass("P011", "Test", 5);
        assert!(pass.passed);

        let fail = TestResult::fail("P011", "Test", 5, "err".to_string());
        assert!(!fail.passed);

        let skip = TestResult::skip("P011", "Test", 5, "reason".to_string());
        assert!(skip.passed);
        assert_eq!(skip.points, 0);
    }
}
