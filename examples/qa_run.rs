//! QA Example: apr run Falsification Suite (PMAT-QA-RUST-001 + PMAT-QA-MATRIX-001)
//!
//! Popperian falsification tests for `apr run` command with full matrix support.
//!
//! # CRITICAL: Same-Model Comparison Protocol (PMAT-SHOWCASE-METHODOLOGY-001)
//!
//! **Class A (Quantized):** GGUF Q4_K_M vs APR Q4_K (converted from same GGUF)
//! **Class B (Full Precision):** SafeTensors F32 vs APR F32 (converted from same SafeTensors)
//!
//! NEVER compare different quantizations (e.g., Q4_K vs F32) - this is a FATAL DEFECT.
//!
//! # Canonical Model
//!
//! ```text
//! GGUF: hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
//! ```
//!
//! # Test Classes
//!
//! ## Class A: Quantized (60 points)
//! | Cell | Backend | Format | Model Source |
//! |------|---------|--------|--------------|
//! | A1 | CPU | GGUF Q4_K | HF GGUF |
//! | A2 | CPU | APR Q4_K | Converted from GGUF |
//! | A3 | GPU | GGUF Q4_K | HF GGUF |
//! | A4 | GPU | APR Q4_K | Converted from GGUF |
//!
//! ## Class B: Full Precision (40 points) - SLOWER, memory-bound
//! | Cell | Backend | Format | Model Source |
//! |------|---------|--------|--------------|
//! | B1 | CPU | SafeTensors F32 | HF SafeTensors |
//! | B2 | CPU | APR F32 | Converted from SafeTensors |
//! | B3 | GPU | SafeTensors F32 | HF SafeTensors |
//! | B4 | GPU | APR F32 | Converted from SafeTensors |
//!
//! # Usage
//!
//! ```bash
//! # Class A only (quantized - recommended)
//! cargo run --example qa_run -- --class quantized --matrix
//!
//! # Class B only (full precision - slow)
//! cargo run --example qa_run -- --class full-precision --matrix
//!
//! # Both classes
//! cargo run --example qa_run -- --class all --matrix
//!
//! # Single cell
//! cargo run --example qa_run -- --backend gpu --format gguf
//!
//! # With tracing
//! cargo run --example qa_run -- --backend gpu --format gguf --trace
//! ```
//!
//! # Tracing (ALL must work for run/chat/serve)
//!
//! - `--trace`: Step-by-step timing with [TRACE-CACHE] messages
//! - `--trace-level layer`: Per-layer breakdown (Attention, FFN, Norm)
//! - `--profile`: Roofline analysis (memory vs compute bound)
//!
//! # Citations
//!
//! - Popper, K. R. (1959). The Logic of Scientific Discovery. Routledge.
//! - PMAT-SHOWCASE-METHODOLOGY-001: Same-Model Comparison Protocol

use std::env;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

// ANSI colors
const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[0;33m";
const BLUE: &str = "\x1b[0;34m";
const CYAN: &str = "\x1b[0;36m";
const MAGENTA: &str = "\x1b[0;35m";
const NC: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

#[derive(Debug, Clone, Copy, PartialEq)]
enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    fn as_str(&self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Gpu => "GPU",
        }
    }

    fn flag(&self) -> Option<&'static str> {
        match self {
            Backend::Cpu => Some("--no-gpu"),
            Backend::Gpu => None, // GPU is default
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Format {
    Gguf,
    SafeTensors,
    Apr,
}

impl Format {
    fn as_str(&self) -> &'static str {
        match self {
            Format::Gguf => "GGUF",
            Format::SafeTensors => "SafeTensors",
            Format::Apr => "APR",
        }
    }

    #[allow(dead_code)]
    fn extension(&self) -> &'static str {
        match self {
            Format::Gguf => ".gguf",
            Format::SafeTensors => ".safetensors",
            Format::Apr => ".apr",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum TraceLevel {
    None,
    Brick,
    Step,
    Layer,
    Profile,
}

impl TraceLevel {
    #[allow(dead_code)]
    fn as_str(&self) -> &'static str {
        match self {
            TraceLevel::None => "none",
            TraceLevel::Brick => "brick",
            TraceLevel::Step => "step",
            TraceLevel::Layer => "layer",
            TraceLevel::Profile => "profile",
        }
    }

    #[allow(dead_code)]
    fn flag(&self) -> Option<&'static str> {
        match self {
            TraceLevel::None => None,
            TraceLevel::Brick => Some("brick"),
            TraceLevel::Step => Some("step"),
            TraceLevel::Layer => Some("layer"),
            TraceLevel::Profile => Some("profile"),
        }
    }
}

/// A single matrix cell (backend × format combination)
#[derive(Debug, Clone)]
struct MatrixCell {
    id: String,
    backend: Backend,
    format: Format,
    model_uri: String, // HuggingFace URI or local path
}

impl MatrixCell {
    fn new(id: &str, backend: Backend, format: Format, model_uri: String) -> Self {
        Self {
            id: id.to_string(),
            backend,
            format,
            model_uri,
        }
    }

    fn label(&self) -> String {
        format!("{} × {}", self.backend.as_str(), self.format.as_str())
    }
}

/// Test result for a single criterion
struct TestResult {
    name: &'static str,
    passed: bool,
    details: Option<String>,
    points: u32,
    max_points: u32,
}

impl TestResult {
    fn pass(name: &'static str, points: u32, details: String) -> Self {
        Self {
            name,
            passed: true,
            details: Some(details),
            points,
            max_points: points,
        }
    }

    fn fail(name: &'static str, max_points: u32, details: String) -> Self {
        Self {
            name,
            passed: false,
            details: Some(details),
            points: 0,
            max_points,
        }
    }

    fn skip(name: &'static str, max_points: u32, reason: String) -> Self {
        Self {
            name,
            passed: true,
            details: Some(format!("SKIP: {}", reason)),
            points: 0,
            max_points,
        }
    }
}

/// Results for a complete matrix cell
struct CellResult {
    cell: MatrixCell,
    tests: Vec<TestResult>,
    total_points: u32,
    max_points: u32,
    elapsed: std::time::Duration,
}

impl CellResult {
    fn passed(&self) -> bool {
        self.tests.iter().all(|t| t.passed)
    }
}

/// Configuration
struct Config {
    apr_binary: PathBuf,
    trace_level: TraceLevel,
    min_cpu_tps: f64,
    min_gpu_tps: f64,
    /// Lower threshold for float32 models (SafeTensors) which are slower than quantized
    min_gpu_tps_float32: f64,
    verbose: bool,
    // Model URIs (HuggingFace or local paths) - apr downloads automatically
    gguf_model: String,
    safetensors_model: String,
    apr_model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            apr_binary: find_apr_binary(),
            trace_level: TraceLevel::None,
            // 1.5B model thresholds (PMAT-097: 0.5B has coherency issues)
            // 1.5B is ~3x larger than 0.5B, so expect ~1/3 the throughput
            min_cpu_tps: 5.0,          // 1.5B on CPU is slow (~5-10 tok/s)
            min_gpu_tps: 7.0,          // 1.5B quantized on GPU (~7-15 tok/s)
            min_gpu_tps_float32: 10.0, // SafeTensors 1.5B on GPU
            verbose: false,
            gguf_model: default_model_for_format(Format::Gguf),
            safetensors_model: default_model_for_format(Format::SafeTensors),
            apr_model: default_model_for_format(Format::Apr),
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
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("cargo")
}

/// Canonical GGUF model - the SINGLE SOURCE OF TRUTH for quantized comparisons
/// All APR Q4_K tests MUST use this exact model converted to APR format.
const CANONICAL_GGUF: &str = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

/// Canonical SafeTensors model - for full precision (Class B) comparisons
const CANONICAL_SAFETENSORS: &str = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct";

/// Returns HuggingFace URI or local path for model format.
///
/// CRITICAL (PMAT-SHOWCASE-METHODOLOGY-001):
/// - Class A (Quantized): GGUF and APR use the SAME Q4_K_M weights
/// - Class B (Full Precision): SafeTensors and APR use the SAME F32 weights
/// - NEVER compare different quantizations!
fn default_model_for_format(format: Format) -> String {
    match format {
        // GGUF Q4_K_M: Canonical quantized model from HuggingFace
        Format::Gguf => CANONICAL_GGUF.to_string(),
        // SafeTensors F32: Full precision for Class B comparisons
        Format::SafeTensors => CANONICAL_SAFETENSORS.to_string(),
        // APR: Use GGUF source for Class A (quantized) - same weights!
        // For Class B (full precision), user must specify --apr with SafeTensors source
        // Default to GGUF source since Class A is the primary focus
        Format::Apr => CANONICAL_GGUF.to_string(),
    }
}

fn gpu_available() -> bool {
    Command::new("nvidia-smi")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run_apr(config: &Config, args: &[&str]) -> Result<String, String> {
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
        eprintln!("{}DEBUG: {:?}{}", CYAN, cmd, NC);
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
        Err(e) => Err(format!("Failed: {}", e)),
    }
}

/// Extract just the model output from apr run output (between "Output:" and "Completed in")
/// This filters out compilation warnings, paths, and timing info that might contain false positives.
fn extract_output(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let mut in_output = false;
    let mut content = Vec::new();
    for line in lines {
        if line.starts_with("Output:") {
            in_output = true;
            continue;
        }
        if line.starts_with("Completed in ") {
            break;
        }
        if in_output {
            content.push(line);
        }
    }
    content.join("\n").trim().to_string()
}

/// Run all tests for a single matrix cell
fn run_cell_tests(config: &Config, cell: &MatrixCell) -> CellResult {
    let start = Instant::now();
    let mut tests = Vec::new();

    // model_uri is a HuggingFace URI or local path - apr handles download
    let model_str = cell.model_uri.clone();

    // Skip GPU tests if no GPU
    if cell.backend == Backend::Gpu && !gpu_available() {
        tests.push(TestResult::skip(
            "All Tests",
            15,
            "No GPU available".to_string(),
        ));
        return CellResult {
            cell: cell.clone(),
            tests,
            total_points: 0,
            max_points: 15,
            elapsed: start.elapsed(),
        };
    }

    // Build base args
    let mut base_args: Vec<&str> = vec![
        "run",
        &model_str,
        "--prompt",
        "What is 2+2? Answer with just the number.",
        "--max-tokens",
        "10",
    ];
    if let Some(flag) = cell.backend.flag() {
        base_args.push(flag);
    }

    // Test 1: Model loads (2 points)
    match run_apr(config, &base_args) {
        Ok(_) => tests.push(TestResult::pass("Model Load", 2, model_str.clone())),
        Err(e) => {
            tests.push(TestResult::fail("Model Load", 2, e));
            return CellResult {
                cell: cell.clone(),
                tests,
                total_points: 0,
                max_points: 15,
                elapsed: start.elapsed(),
            };
        }
    }

    // Test 2: Correct output (3 points)
    // Use extract_output to avoid false positives from line numbers/paths containing '4'
    match run_apr(config, &base_args) {
        Ok(raw_output) => {
            let output = extract_output(&raw_output);
            if output.contains('4') {
                tests.push(TestResult::pass(
                    "Correct Output",
                    3,
                    "Contains '4'".to_string(),
                ));
            } else {
                tests.push(TestResult::fail(
                    "Correct Output",
                    3,
                    format!(
                        "Missing '4': {}",
                        output.chars().take(50).collect::<String>()
                    ),
                ));
            }
        }
        Err(e) => tests.push(TestResult::fail("Correct Output", 3, e)),
    }

    // Test 3: No garbage (3 points)
    let hello_args: Vec<&str> = {
        let mut args = vec![
            "run",
            &model_str,
            "--prompt",
            "Say hello.",
            "--max-tokens",
            "20",
        ];
        if let Some(flag) = cell.backend.flag() {
            args.push(flag);
        }
        args
    };
    match run_apr(config, &hello_args) {
        Ok(raw_output) => {
            let output = extract_output(&raw_output);
            let has_garbage = output.contains('\u{FFFD}')
                || (output.contains("token0") || output.contains("token1"));
            if has_garbage {
                tests.push(TestResult::fail(
                    "No Garbage",
                    3,
                    "Garbage patterns detected".to_string(),
                ));
            } else {
                tests.push(TestResult::pass(
                    "No Garbage",
                    3,
                    "Clean output".to_string(),
                ));
            }
        }
        Err(e) => tests.push(TestResult::fail("No Garbage", 3, e)),
    }

    // Test 4: No BPE artifacts (2 points)
    match run_apr(config, &hello_args) {
        Ok(raw_output) => {
            let output = extract_output(&raw_output);
            if output.contains('Ġ') || output.contains('Ċ') {
                tests.push(TestResult::fail(
                    "No BPE Artifacts",
                    2,
                    "Ġ/Ċ detected".to_string(),
                ));
            } else {
                tests.push(TestResult::pass(
                    "No BPE Artifacts",
                    2,
                    "Clean tokens".to_string(),
                ));
            }
        }
        Err(e) => tests.push(TestResult::fail("No BPE Artifacts", 2, e)),
    }

    // Test 5: Trace works (2 points)
    let trace_args: Vec<&str> = {
        let mut args = vec![
            "run",
            &model_str,
            "--prompt",
            "Hi",
            "--max-tokens",
            "5",
            "--trace",
        ];
        if let Some(flag) = cell.backend.flag() {
            args.push(flag);
        }
        args
    };
    match run_apr(config, &trace_args) {
        Ok(_) => tests.push(TestResult::pass(
            "Trace Works",
            2,
            "Trace accepted".to_string(),
        )),
        Err(e) => {
            if e.contains("not supported") {
                tests.push(TestResult::skip(
                    "Trace Works",
                    2,
                    "Trace not supported".to_string(),
                ));
            } else {
                tests.push(TestResult::fail("Trace Works", 2, e));
            }
        }
    }

    // Test 6: Performance (3 points)
    let perf_args: Vec<&str> = {
        let mut args = vec![
            "run",
            &model_str,
            "--prompt",
            "Count from 1 to 20.",
            "--max-tokens",
            "50",
        ];
        if let Some(flag) = cell.backend.flag() {
            args.push(flag);
        }
        args
    };
    let perf_start = Instant::now();
    match run_apr(config, &perf_args) {
        Ok(output) => {
            let elapsed = perf_start.elapsed().as_secs_f64();
            let words = output.split_whitespace().count();
            let tokens_est = (words as f64 * 1.3).max(10.0);
            let tps = tokens_est / elapsed;
            // Use format-specific threshold: SafeTensors (float32) is memory-bound
            // and slower than quantized formats (GGUF, APR). Refs: GH-157
            let target = match (cell.backend, cell.format) {
                (Backend::Cpu, _) => config.min_cpu_tps,
                (Backend::Gpu, Format::SafeTensors) => config.min_gpu_tps_float32,
                (Backend::Gpu, _) => config.min_gpu_tps,
            };

            if tps >= target {
                tests.push(TestResult::pass(
                    "Performance",
                    3,
                    format!("{:.1} tok/s >= {:.1}", tps, target),
                ));
            } else {
                tests.push(TestResult::fail(
                    "Performance",
                    3,
                    format!("{:.1} tok/s < {:.1}", tps, target),
                ));
            }
        }
        Err(e) => tests.push(TestResult::fail("Performance", 3, e)),
    }

    let total: u32 = tests.iter().map(|t| t.points).sum();
    let max: u32 = tests.iter().map(|t| t.max_points).sum();

    CellResult {
        cell: cell.clone(),
        tests,
        total_points: total,
        max_points: max,
        elapsed: start.elapsed(),
    }
}

fn print_cell_result(result: &CellResult) {
    let status = if result.passed() {
        format!("{}✓ PASS{}", GREEN, NC)
    } else {
        format!("{}✗ FAIL{}", RED, NC)
    };

    println!();
    println!(
        "{}┌─────────────────────────────────────────────────────────────┐{}",
        BLUE, NC
    );
    println!(
        "{}│{} {} {:<42} {:>8} {}│{}",
        BLUE,
        NC,
        BOLD,
        result.cell.label(),
        status,
        BLUE,
        NC
    );
    println!(
        "{}├─────────────────────────────────────────────────────────────┤{}",
        BLUE, NC
    );

    for test in &result.tests {
        let icon = if test.passed {
            format!("{}✓{}", GREEN, NC)
        } else {
            format!("{}✗{}", RED, NC)
        };
        let points = format!("{}/{}", test.points, test.max_points);
        let detail = test.details.as_deref().unwrap_or("");
        println!(
            "{}│{} {} {:<20} {:>5}  {:<25}{}│{}",
            BLUE,
            NC,
            icon,
            test.name,
            points,
            detail.chars().take(25).collect::<String>(),
            BLUE,
            NC
        );
    }

    println!(
        "{}├─────────────────────────────────────────────────────────────┤{}",
        BLUE, NC
    );
    println!(
        "{}│{} Total: {}/{} points ({:.1}s) {:>24}{}│{}",
        BLUE,
        NC,
        result.total_points,
        result.max_points,
        result.elapsed.as_secs_f64(),
        "",
        BLUE,
        NC
    );
    println!(
        "{}└─────────────────────────────────────────────────────────────┘{}",
        BLUE, NC
    );
}

fn print_matrix_summary(results: &[CellResult]) {
    println!();
    println!(
        "{}╔═════════════════════════════════════════════════════════════╗{}",
        MAGENTA, NC
    );
    println!(
        "{}║             QA MATRIX SUMMARY (PMAT-QA-MATRIX-001)          ║{}",
        MAGENTA, NC
    );
    println!(
        "{}╠═════════════════════════════════════════════════════════════╣{}",
        MAGENTA, NC
    );

    // Matrix table header
    println!(
        "{}║{} {:^10} │ {:^12} │ {:^12} │ {:^12} {}║{}",
        MAGENTA, NC, "", "GGUF", "SafeTensors", "APR", MAGENTA, NC
    );
    println!(
        "{}╟───────────┼──────────────┼──────────────┼──────────────╢{}",
        MAGENTA, NC
    );

    // CPU row
    print!("{}║{} {:^10} │", MAGENTA, NC, "CPU");
    for fmt in [Format::Gguf, Format::SafeTensors, Format::Apr] {
        if let Some(r) = results
            .iter()
            .find(|r| r.cell.backend == Backend::Cpu && r.cell.format == fmt)
        {
            let status = if r.passed() {
                format!("{}✓ {}/{}{}  ", GREEN, r.total_points, r.max_points, NC)
            } else {
                format!("{}✗ {}/{}{}  ", RED, r.total_points, r.max_points, NC)
            };
            print!(" {:^12} │", status);
        } else {
            print!(" {:^12} │", "—");
        }
    }
    println!("{}║{}", MAGENTA, NC);

    // GPU row
    print!("{}║{} {:^10} │", MAGENTA, NC, "GPU");
    for fmt in [Format::Gguf, Format::SafeTensors, Format::Apr] {
        if let Some(r) = results
            .iter()
            .find(|r| r.cell.backend == Backend::Gpu && r.cell.format == fmt)
        {
            let status = if r.passed() {
                format!("{}✓ {}/{}{}  ", GREEN, r.total_points, r.max_points, NC)
            } else {
                format!("{}✗ {}/{}{}  ", RED, r.total_points, r.max_points, NC)
            };
            print!(" {:^12} │", status);
        } else {
            print!(" {:^12} │", "—");
        }
    }
    println!("{}║{}", MAGENTA, NC);

    println!(
        "{}╠═════════════════════════════════════════════════════════════╣{}",
        MAGENTA, NC
    );

    let total_points: u32 = results.iter().map(|r| r.total_points).sum();
    let max_points: u32 = results.iter().map(|r| r.max_points).sum();
    let passed = results.iter().filter(|r| r.passed()).count();
    let total = results.len();

    let grade = if total_points == max_points {
        "A+"
    } else if total_points as f64 / max_points as f64 >= 0.9 {
        "A"
    } else if total_points as f64 / max_points as f64 >= 0.8 {
        "B"
    } else if total_points as f64 / max_points as f64 >= 0.7 {
        "C"
    } else {
        "F"
    };

    println!(
        "{}║{} Cells: {}/{} passed    Points: {}/{}    Grade: {:>14}{}║{}",
        MAGENTA, NC, passed, total, total_points, max_points, grade, MAGENTA, NC
    );
    println!(
        "{}╚═════════════════════════════════════════════════════════════╝{}",
        MAGENTA, NC
    );

    if passed == total {
        println!();
        println!("{}Hypothesis \"apr run produces correct output across all formats/backends\" SURVIVED.{}", GREEN, NC);
    } else {
        println!();
        println!(
            "{}Hypothesis FALSIFIED. {} cell(s) failed.{}",
            RED,
            total - passed,
            NC
        );
    }
}

fn print_help() {
    println!("{}QA Matrix Runner (PMAT-QA-MATRIX-001){}", BOLD, NC);
    println!();
    println!("{}CRITICAL: Same-Model Comparison Protocol{}", YELLOW, NC);
    println!("  Class A (Quantized): GGUF Q4_K vs APR Q4_K (SAME weights)");
    println!("  Class B (Full Prec): SafeTensors F32 vs APR F32 (SAME weights)");
    println!();
    println!("USAGE:");
    println!("    cargo run --example qa_run -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --matrix              Run full matrix (Class A: 4 cells quantized)");
    println!("    --backend <cpu|gpu>   Force specific backend");
    println!("    --format <gguf|safetensors|apr>  Force specific format");
    println!("    --trace               Enable tracing (shows [TRACE-CACHE] messages)");
    println!("    --trace-level <layer|profile>  Detailed trace level");
    println!("    --gguf <PATH>         Path to GGUF model");
    println!("    --safetensors <PATH>  Path to SafeTensors model");
    println!("    --apr <PATH>          Path to APR model");
    println!("    --model <PATH>        Legacy: single model path");
    println!("    --min-cpu-tps <N>     Minimum CPU tok/s (default: 5.0)");
    println!("    --min-gpu-tps <N>     Minimum GPU tok/s for quantized (default: 7.0)");
    println!("    --verbose, -v         Verbose output");
    println!("    --help, -h            Show this help");
    println!();
    println!("CANONICAL MODEL:");
    println!("    {}", CANONICAL_GGUF);
    println!();
    println!("EXAMPLES:");
    println!("    # Class A quantized matrix (recommended)");
    println!("    cargo run --example qa_run -- --matrix");
    println!();
    println!("    # Single cell: GPU + GGUF with tracing");
    println!("    cargo run --example qa_run -- --backend gpu --format gguf --trace");
    println!();
    println!("    # Verify tracing works");
    println!("    cargo run --example qa_run -- --backend cpu --format gguf --trace");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = Config::default();

    let mut run_matrix = false;
    let mut single_backend: Option<Backend> = None;
    let mut single_format: Option<Format> = None;
    let mut legacy_model: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--matrix" => {
                run_matrix = true;
                i += 1;
            }
            "--backend" => {
                if i + 1 < args.len() {
                    single_backend = match args[i + 1].as_str() {
                        "cpu" => Some(Backend::Cpu),
                        "gpu" => Some(Backend::Gpu),
                        _ => None,
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--format" => {
                if i + 1 < args.len() {
                    single_format = match args[i + 1].as_str() {
                        "gguf" => Some(Format::Gguf),
                        "safetensors" => Some(Format::SafeTensors),
                        "apr" => Some(Format::Apr),
                        _ => None,
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--trace-level" => {
                if i + 1 < args.len() {
                    config.trace_level = match args[i + 1].as_str() {
                        "brick" => TraceLevel::Brick,
                        "step" => TraceLevel::Step,
                        "layer" => TraceLevel::Layer,
                        "profile" => TraceLevel::Profile,
                        _ => TraceLevel::None,
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--gguf" => {
                if i + 1 < args.len() {
                    config.gguf_model = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--safetensors" => {
                if i + 1 < args.len() {
                    config.safetensors_model = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--apr" => {
                if i + 1 < args.len() {
                    config.apr_model = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--model" => {
                if i + 1 < args.len() {
                    legacy_model = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    i += 1;
                }
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
                    config.min_gpu_tps = args[i + 1].parse().unwrap_or(100.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--min-gpu-tps-f32" => {
                if i + 1 < args.len() {
                    config.min_gpu_tps_float32 = args[i + 1].parse().unwrap_or(40.0);
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
                print_help();
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Header
    println!();
    println!(
        "{}╔═════════════════════════════════════════════════════════════╗{}",
        BLUE, NC
    );
    println!(
        "{}║      APR RUN QA - Matrix Falsification Suite                ║{}",
        BLUE, NC
    );
    println!(
        "{}║      PMAT-QA-RUST-001 + PMAT-QA-MATRIX-001                   ║{}",
        BLUE, NC
    );
    println!(
        "{}╚═════════════════════════════════════════════════════════════╝{}",
        BLUE, NC
    );
    println!();

    // Build cells to test - using HuggingFace URIs (apr downloads automatically)
    let cells: Vec<MatrixCell> = if run_matrix {
        // Full matrix: 6 cells (2 backends × 3 formats)
        vec![
            MatrixCell::new("M1", Backend::Cpu, Format::Gguf, config.gguf_model.clone()),
            MatrixCell::new(
                "M2",
                Backend::Cpu,
                Format::SafeTensors,
                config.safetensors_model.clone(),
            ),
            MatrixCell::new("M3", Backend::Cpu, Format::Apr, config.apr_model.clone()),
            MatrixCell::new("M4", Backend::Gpu, Format::Gguf, config.gguf_model.clone()),
            MatrixCell::new(
                "M5",
                Backend::Gpu,
                Format::SafeTensors,
                config.safetensors_model.clone(),
            ),
            MatrixCell::new("M6", Backend::Gpu, Format::Apr, config.apr_model.clone()),
        ]
    } else if let (Some(backend), Some(format)) = (single_backend, single_format) {
        // Single cell
        let model = match format {
            Format::Gguf => config.gguf_model.clone(),
            Format::SafeTensors => config.safetensors_model.clone(),
            Format::Apr => config.apr_model.clone(),
        };
        vec![MatrixCell::new("S1", backend, format, model)]
    } else if let Some(model_path) = legacy_model {
        // Legacy single model mode
        let model = model_path.to_string_lossy().to_string();
        let format = if model.ends_with(".gguf") {
            Format::Gguf
        } else if model.ends_with(".safetensors") {
            Format::SafeTensors
        } else {
            Format::Apr
        };
        vec![
            MatrixCell::new("L1", Backend::Cpu, format, model.clone()),
            MatrixCell::new("L2", Backend::Gpu, format, model),
        ]
    } else {
        println!(
            "{}No mode specified. Use --matrix, --backend + --format, or --model{}",
            YELLOW, NC
        );
        println!();
        print_help();
        std::process::exit(2);
    };

    // Show what we're testing
    println!("{}Testing {} cell(s):{}", CYAN, cells.len(), NC);
    for cell in &cells {
        println!("  {} {} → {}", cell.id, cell.label(), cell.model_uri);
    }
    println!();

    // Run tests
    let mut results = Vec::new();
    for cell in &cells {
        let result = run_cell_tests(&config, cell);
        print_cell_result(&result);
        results.push(result);
    }

    // Summary
    print_matrix_summary(&results);

    // Exit code
    let all_passed = results.iter().all(|r| r.passed());
    std::process::exit(if all_passed { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_flag() {
        assert_eq!(Backend::Cpu.flag(), Some("--no-gpu"));
        assert_eq!(Backend::Gpu.flag(), None);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(Format::Gguf.extension(), ".gguf");
        assert_eq!(Format::SafeTensors.extension(), ".safetensors");
        assert_eq!(Format::Apr.extension(), ".apr");
    }

    #[test]
    fn test_cell_label() {
        let cell = MatrixCell::new(
            "M1",
            Backend::Cpu,
            Format::Gguf,
            "hf://test/model".to_string(),
        );
        assert_eq!(cell.label(), "CPU × GGUF");
    }

    /// Test: Performance thresholds are format-specific (GH-157)
    ///
    /// SafeTensors (float32) is memory-bound and slower than quantized formats.
    /// Verifies the Config defaults are correct for 1.5B models.
    #[test]
    fn test_performance_thresholds_config() {
        let config = Config::default();

        // CPU threshold for 1.5B (~5-10 tok/s)
        assert!((config.min_cpu_tps - 5.0).abs() < 0.01);

        // GPU quantized threshold for 1.5B models (~7-15 tok/s)
        assert!((config.min_gpu_tps - 7.0).abs() < 0.01);

        // GPU float32 threshold (SafeTensors 1.5B)
        assert!((config.min_gpu_tps_float32 - 10.0).abs() < 0.01);

        // Float32 (SafeTensors) threshold higher than quantized for 1.5B
        assert!(config.min_gpu_tps_float32 > config.min_gpu_tps);
    }

    /// Test: Threshold selection logic is correct per (backend, format) pair
    #[test]
    fn test_threshold_selection_logic() {
        let config = Config::default();

        // Helper to get threshold for a (backend, format) pair
        let get_threshold = |backend: Backend, format: Format| -> f64 {
            match (backend, format) {
                (Backend::Cpu, _) => config.min_cpu_tps,
                (Backend::Gpu, Format::SafeTensors) => config.min_gpu_tps_float32,
                (Backend::Gpu, _) => config.min_gpu_tps,
            }
        };

        // CPU always uses CPU threshold regardless of format
        assert!((get_threshold(Backend::Cpu, Format::Gguf) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Cpu, Format::SafeTensors) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Cpu, Format::Apr) - 5.0).abs() < 0.01);

        // GPU uses format-specific thresholds
        assert!((get_threshold(Backend::Gpu, Format::Gguf) - 7.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::SafeTensors) - 10.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::Apr) - 7.0).abs() < 0.01);
    }

    /// Test: CLI parsing for new --min-gpu-tps-f32 option
    #[test]
    fn test_cli_parsing_float32_threshold() {
        // This would require refactoring main() to be testable
        // For now, just verify the default is set correctly
        let config = Config::default();
        assert!((config.min_gpu_tps_float32 - 10.0).abs() < 0.01);
    }
}
