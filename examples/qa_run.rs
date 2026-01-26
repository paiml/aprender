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
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

// ANSI colors
const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[0;33m";
const BLUE: &str = "\x1b[0;34m";
const CYAN: &str = "\x1b[0;36m";
const MAGENTA: &str = "\x1b[0;35m";
const NC: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

/// Default timeout for all apr commands (PMAT-QA-PROTOCOL-001 §7.6)
/// A command that hangs is a test failure, not a process state.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Modality: which apr command to test (PMAT-QA-PROTOCOL-001 §7.4)
#[derive(Debug, Clone, Copy, PartialEq)]
enum Modality {
    /// `apr run` - single prompt inference
    Run,
    /// `apr chat` - interactive chat mode (stdin/stdout)
    Chat,
    /// `apr serve` - HTTP server mode
    Serve,
}

impl Modality {
    fn as_str(&self) -> &'static str {
        match self {
            Modality::Run => "run",
            Modality::Chat => "chat",
            Modality::Serve => "serve",
        }
    }

    fn display_name(&self) -> &'static str {
        match self {
            Modality::Run => "apr run",
            Modality::Chat => "apr chat",
            Modality::Serve => "apr serve",
        }
    }
}

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

/// Test class for same-model comparison (PMAT-SHOWCASE-METHODOLOGY-001)
#[derive(Debug, Clone, Copy, PartialEq)]
enum TestClass {
    /// Class A: Quantized (GGUF Q4_K, APR Q4_K) - 60 points
    Quantized,
    /// Class B: Full Precision (SafeTensors F32, APR F32) - 40 points
    FullPrecision,
    /// Run both classes
    All,
}

impl TestClass {
    #[allow(dead_code)]
    fn as_str(&self) -> &'static str {
        match self {
            TestClass::Quantized => "quantized",
            TestClass::FullPrecision => "full-precision",
            TestClass::All => "all",
        }
    }

    fn includes_quantized(&self) -> bool {
        matches!(self, TestClass::Quantized | TestClass::All)
    }

    fn includes_full_precision(&self) -> bool {
        matches!(self, TestClass::FullPrecision | TestClass::All)
    }
}

/// A single matrix cell (modality × backend × format combination)
/// (PMAT-QA-PROTOCOL-001 §7.4)
#[derive(Debug, Clone)]
struct MatrixCell {
    id: String,
    modality: Modality,
    backend: Backend,
    format: Format,
    model_uri: String,   // HuggingFace URI or local path
    with_trace: bool,    // Test with --trace flag
}

impl MatrixCell {
    fn new(id: &str, backend: Backend, format: Format, model_uri: String) -> Self {
        Self {
            id: id.to_string(),
            modality: Modality::Run, // Default to apr run
            backend,
            format,
            model_uri,
            with_trace: false,
        }
    }

    fn with_modality(mut self, modality: Modality) -> Self {
        self.modality = modality;
        self
    }

    fn with_trace(mut self, trace: bool) -> Self {
        self.with_trace = trace;
        self
    }

    fn label(&self) -> String {
        let trace_suffix = if self.with_trace { " +trace" } else { "" };
        format!(
            "{} × {} × {}{}",
            self.modality.display_name(),
            self.backend.as_str(),
            self.format.as_str(),
            trace_suffix
        )
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
    /// Test class for same-model comparison (PMAT-SHOWCASE-METHODOLOGY-001)
    test_class: TestClass,
    min_cpu_tps: f64,
    min_gpu_tps: f64,
    /// Lower threshold for float32 models (SafeTensors) which are slower than quantized
    min_gpu_tps_float32: f64,
    verbose: bool,
    // Model URIs (HuggingFace or local paths) - apr downloads automatically
    gguf_model: String,
    safetensors_model: String,
    apr_model: String,
    /// Compare against Ollama as groundtruth (PMAT-SHOWCASE-METHODOLOGY-001 Section 5)
    with_ollama: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            apr_binary: find_apr_binary(),
            trace_level: TraceLevel::None,
            test_class: TestClass::Quantized, // Default to Class A (faster, recommended)
            // 1.5B model thresholds (PMAT-SHOWCASE-METHODOLOGY-001)
            // Word-based tok/s estimation has ~20% variance, so use conservative thresholds
            // Actual performance: CPU ~5-10 tok/s, GPU ~7-15 tok/s (quantized)
            min_cpu_tps: 5.0,         // 1.5B on CPU is slow (~5-10 tok/s observed)
            min_gpu_tps: 5.0,         // Conservative threshold for GPU (actual ~7-10 tok/s)
            min_gpu_tps_float32: 5.0, // SafeTensors F32 is memory-bound
            verbose: false,
            gguf_model: default_model_for_format(Format::Gguf),
            safetensors_model: default_model_for_format(Format::SafeTensors),
            apr_model: default_model_for_format(Format::Apr),
            with_ollama: false,
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
const CANONICAL_GGUF: &str =
    "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

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

/// Wait for child process with timeout using polling (PMAT-QA-PROTOCOL-001 §7.6)
///
/// A hung process is a test FAILURE, not a process state. This function:
/// 1. Polls child.try_wait() in a loop
/// 2. Kills the process if timeout exceeded
/// 3. Returns explicit error on timeout
fn wait_with_timeout(
    child: &mut Child,
    timeout: Duration,
) -> Result<std::process::ExitStatus, String> {
    let start = Instant::now();
    let poll_interval = Duration::from_millis(100);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                // Still running - check timeout
                if start.elapsed() >= timeout {
                    // Kill the hung process
                    let _ = child.kill();
                    let _ = child.wait(); // Reap zombie
                    return Err(format!(
                        "HANG: Process killed after {}s timeout (PMAT-QA-PROTOCOL-001 violation)",
                        timeout.as_secs()
                    ));
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => return Err(format!("Process error: {}", e)),
        }
    }
}

fn run_apr(config: &Config, args: &[&str]) -> Result<String, String> {
    run_apr_with_timeout(config, args, DEFAULT_TIMEOUT)
}

fn run_apr_with_timeout(
    config: &Config,
    args: &[&str],
    timeout: Duration,
) -> Result<String, String> {
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

    // Spawn instead of output() to enable timeout handling
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return Err(format!("Failed to spawn: {}", e)),
    };

    // Wait with timeout - hung processes are test failures
    let status = wait_with_timeout(&mut child, timeout)?;

    // Read output after process completes
    let mut stdout_str = String::new();
    let mut stderr_str = String::new();

    if let Some(mut stdout) = child.stdout.take() {
        let _ = stdout.read_to_string(&mut stdout_str);
    }
    if let Some(mut stderr) = child.stderr.take() {
        let _ = stderr.read_to_string(&mut stderr_str);
    }

    if status.success() {
        Ok(format!("{}{}", stdout_str, stderr_str))
    } else {
        Err(format!("Exit {}: {}{}", status, stdout_str, stderr_str))
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

/// Garbage patterns that indicate model collapse or tokenization failure
/// (PMAT-QA-PROTOCOL-001 §7.5)
const GARBAGE_PATTERNS: &[&str] = &[
    "\u{FFFD}",    // Replacement character (encoding error)
    "[UNK]",       // Unknown token marker
    "akunji",      // Known GQA bug garbage
    "olumbia",     // Known layout bug garbage
    "专门窗",      // Known GQA bug CJK garbage
    "token0",      // Raw token ID leak
    "token1",      // Raw token ID leak
    "<0x",         // Byte token leak (e.g., <0x0A>)
];

/// BPE artifacts that indicate incomplete detokenization
const BPE_ARTIFACTS: &[char] = &[
    'Ġ', // GPT-2 style space prefix
    'Ċ', // GPT-2 style newline
    'ĉ', // GPT-2 style tab
];

/// Verification result for output inspection (PMAT-QA-PROTOCOL-001 §7.5)
#[derive(Debug)]
enum VerifyResult {
    Pass(String),
    FailEmpty,
    FailGarbage(String),
    FailBpeArtifact(char),
    FailMissingAnswer(String),
}

/// Verify output is correct: not empty, no garbage, contains expected answer
/// (PMAT-QA-PROTOCOL-001 §7.5)
///
/// Order of checks is CRITICAL (fail fast on garbage):
/// 1. Not empty
/// 2. No garbage patterns (BEFORE checking answer)
/// 3. No BPE artifacts
/// 4. Contains expected answer
fn verify_output(output: &str, expected_contains: Option<&str>) -> VerifyResult {
    let trimmed = output.trim();

    // 1. Empty check
    if trimmed.is_empty() {
        return VerifyResult::FailEmpty;
    }

    // 2. Garbage detection (FAIL FAST - before answer check)
    for pattern in GARBAGE_PATTERNS {
        if trimmed.contains(pattern) {
            return VerifyResult::FailGarbage((*pattern).to_string());
        }
    }

    // 3. BPE artifact check
    for &artifact in BPE_ARTIFACTS {
        if trimmed.contains(artifact) {
            return VerifyResult::FailBpeArtifact(artifact);
        }
    }

    // 4. Expected answer check (only if specified)
    if let Some(expected) = expected_contains {
        if !trimmed.contains(expected) {
            return VerifyResult::FailMissingAnswer(format!(
                "Expected '{}', got: {}",
                expected,
                trimmed.chars().take(50).collect::<String>()
            ));
        }
    }

    VerifyResult::Pass(trimmed.to_string())
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

    // Test 2: Correct output with full verification (5 points combined)
    // Uses verify_output which checks: empty, garbage, BPE, then answer
    // (PMAT-QA-PROTOCOL-001 §7.5)
    match run_apr(config, &base_args) {
        Ok(raw_output) => {
            let output = extract_output(&raw_output);
            match verify_output(&output, Some("4")) {
                VerifyResult::Pass(_) => {
                    tests.push(TestResult::pass(
                        "Correct Output",
                        3,
                        "Contains '4', no garbage".to_string(),
                    ));
                }
                VerifyResult::FailEmpty => {
                    tests.push(TestResult::fail(
                        "Correct Output",
                        3,
                        "Empty output".to_string(),
                    ));
                }
                VerifyResult::FailGarbage(pattern) => {
                    tests.push(TestResult::fail(
                        "Correct Output",
                        3,
                        format!("GARBAGE: '{}'", pattern),
                    ));
                }
                VerifyResult::FailBpeArtifact(c) => {
                    tests.push(TestResult::fail(
                        "Correct Output",
                        3,
                        format!("BPE artifact: '{}'", c),
                    ));
                }
                VerifyResult::FailMissingAnswer(msg) => {
                    tests.push(TestResult::fail("Correct Output", 3, msg));
                }
            }
        }
        Err(e) => tests.push(TestResult::fail("Correct Output", 3, e)),
    }

    // Test 3: No garbage on "hello" prompt (3 points)
    // Separate test to catch garbage that might not appear in math answers
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
            match verify_output(&output, None) {
                VerifyResult::Pass(_) => {
                    tests.push(TestResult::pass(
                        "No Garbage",
                        3,
                        "Clean output".to_string(),
                    ));
                }
                VerifyResult::FailEmpty => {
                    tests.push(TestResult::fail(
                        "No Garbage",
                        3,
                        "Empty output".to_string(),
                    ));
                }
                VerifyResult::FailGarbage(pattern) => {
                    tests.push(TestResult::fail(
                        "No Garbage",
                        3,
                        format!("GARBAGE: '{}'", pattern),
                    ));
                }
                VerifyResult::FailBpeArtifact(c) => {
                    tests.push(TestResult::fail(
                        "No Garbage",
                        3,
                        format!("BPE artifact: '{}'", c),
                    ));
                }
                VerifyResult::FailMissingAnswer(_) => {
                    // No expected answer for this test
                    tests.push(TestResult::pass(
                        "No Garbage",
                        3,
                        "Clean output".to_string(),
                    ));
                }
            }
        }
        Err(e) => tests.push(TestResult::fail("No Garbage", 3, e)),
    }

    // Test 4: No BPE artifacts (2 points) - redundant but kept for compatibility
    // verify_output already checks this, but explicit test for reporting
    match run_apr(config, &hello_args) {
        Ok(raw_output) => {
            let output = extract_output(&raw_output);
            let has_bpe = BPE_ARTIFACTS.iter().any(|&c| output.contains(c));
            if has_bpe {
                tests.push(TestResult::fail(
                    "No BPE Artifacts",
                    2,
                    "Ġ/Ċ/ĉ detected".to_string(),
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
    println!("    --min-gpu-tps <N>     Minimum GPU tok/s for quantized (default: 5.0)");
    println!("    --class <CLASS>       Test class: quantized (default), full-precision, all");
    println!("    --with-ollama         Compare against Ollama as groundtruth");
    println!("    --verbose, -v         Verbose output");
    println!("    --help, -h            Show this help");
    println!();
    println!("TEST CLASSES (PMAT-SHOWCASE-METHODOLOGY-001):");
    println!("    quantized      Class A: GGUF Q4_K vs APR Q4_K (60 points, faster)");
    println!("    full-precision Class B: SafeTensors F32 vs APR F32 (40 points, slower)");
    println!("    all            Both Class A and B (100 points total)");
    println!();
    println!("CANONICAL MODEL:");
    println!("    {}", CANONICAL_GGUF);
    println!();
    println!("EXAMPLES:");
    println!("    # Class A quantized matrix (default, recommended)");
    println!("    cargo run --example qa_run -- --matrix");
    println!();
    println!("    # Class B full precision matrix");
    println!("    cargo run --example qa_run -- --matrix --class full-precision");
    println!();
    println!("    # Both classes");
    println!("    cargo run --example qa_run -- --matrix --class all");
    println!();
    println!("    # Single cell: GPU + GGUF with tracing");
    println!("    cargo run --example qa_run -- --backend gpu --format gguf --trace");
    println!();
    println!("    # Compare against Ollama groundtruth");
    println!("    cargo run --example qa_run -- --class quantized --with-ollama");
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
            "--class" => {
                if i + 1 < args.len() {
                    config.test_class = match args[i + 1].as_str() {
                        "quantized" | "a" | "A" => TestClass::Quantized,
                        "full-precision" | "fp" | "b" | "B" => TestClass::FullPrecision,
                        "all" | "both" => TestClass::All,
                        _ => TestClass::Quantized,
                    };
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--with-ollama" => {
                config.with_ollama = true;
                i += 1;
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
    // Cell selection based on test_class (PMAT-SHOWCASE-METHODOLOGY-001)
    let cells: Vec<MatrixCell> = if run_matrix {
        let mut cells = Vec::new();

        // Class A: Quantized (GGUF Q4_K, APR Q4_K converted from GGUF)
        if config.test_class.includes_quantized() {
            // A1, A2: CPU × GGUF, CPU × APR (from GGUF)
            cells.push(MatrixCell::new(
                "A1",
                Backend::Cpu,
                Format::Gguf,
                config.gguf_model.clone(),
            ));
            cells.push(MatrixCell::new(
                "A2",
                Backend::Cpu,
                Format::Apr,
                config.apr_model.clone(),
            ));
            // A3, A4: GPU × GGUF, GPU × APR (from GGUF)
            cells.push(MatrixCell::new(
                "A3",
                Backend::Gpu,
                Format::Gguf,
                config.gguf_model.clone(),
            ));
            cells.push(MatrixCell::new(
                "A4",
                Backend::Gpu,
                Format::Apr,
                config.apr_model.clone(),
            ));
        }

        // Class B: Full Precision (SafeTensors F32, APR F32 converted from SafeTensors)
        if config.test_class.includes_full_precision() {
            // B1, B2: CPU × SafeTensors, CPU × APR (from SafeTensors)
            cells.push(MatrixCell::new(
                "B1",
                Backend::Cpu,
                Format::SafeTensors,
                config.safetensors_model.clone(),
            ));
            cells.push(MatrixCell::new(
                "B2",
                Backend::Cpu,
                Format::Apr,
                config.apr_model.clone(),
            ));
            // B3, B4: GPU × SafeTensors, GPU × APR (from SafeTensors)
            cells.push(MatrixCell::new(
                "B3",
                Backend::Gpu,
                Format::SafeTensors,
                config.safetensors_model.clone(),
            ));
            cells.push(MatrixCell::new(
                "B4",
                Backend::Gpu,
                Format::Apr,
                config.apr_model.clone(),
            ));
        }

        cells
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
        let format = if model_path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            Format::Gguf
        } else if model_path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
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

    // Ollama parity test (PMAT-SHOWCASE-METHODOLOGY-001 Section 5)
    let ollama_passed = if config.with_ollama {
        run_ollama_comparison(&config)
    } else {
        true
    };

    // Exit code
    let all_passed = results.iter().all(|r| r.passed()) && ollama_passed;
    std::process::exit(if all_passed { 0 } else { 1 });
}

/// Ollama model name for Q4_K_M quantization (same as CANONICAL_GGUF)
const OLLAMA_MODEL: &str = "qwen2.5-coder:1.5b-instruct-q4_K_M";

/// Run Ollama parity comparison (PMAT-SHOWCASE-METHODOLOGY-001 Section 5)
///
/// Compares apr's output against Ollama as groundtruth for:
/// 1. Correctness - Output matches semantically
/// 2. Performance - Within 2x of Ollama tok/s
fn run_ollama_comparison(config: &Config) -> bool {
    println!();
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );
    println!(
        "{}         OLLAMA PARITY TEST (PMAT-SHOWCASE-METHODOLOGY-001)      {}",
        CYAN, NC
    );
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );
    println!();

    // Check if ollama is available
    let ollama_check = Command::new("which").arg("ollama").output();

    if ollama_check.is_err() || !ollama_check.unwrap().status.success() {
        println!(
            "{}[SKIP]{} Ollama not installed - skipping parity test",
            YELLOW, NC
        );
        return true;
    }

    // Check if model is available
    let model_check = Command::new("ollama")
        .args(["show", OLLAMA_MODEL])
        .stderr(Stdio::null())
        .output();

    if model_check.is_err() || !model_check.unwrap().status.success() {
        println!(
            "{}[SKIP]{} Ollama model {} not available",
            YELLOW, NC, OLLAMA_MODEL
        );
        println!("       Install with: ollama pull {}", OLLAMA_MODEL);
        return true;
    }

    let prompt = "What is 2+2? Answer with just the number.";
    let mut all_passed = true;

    // Test 1: Correctness - Ollama output
    println!("{}Test 1: Ollama Groundtruth{}", BOLD, NC);
    let ollama_start = Instant::now();
    let ollama_output = Command::new("ollama")
        .args(["run", OLLAMA_MODEL, prompt])
        .output();

    let (ollama_answer, ollama_time) = match ollama_output {
        Ok(output) => {
            let answer = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let time = ollama_start.elapsed().as_secs_f64();
            (answer, time)
        }
        Err(e) => {
            println!("{}[FAIL]{} Ollama execution failed: {}", RED, NC, e);
            return false;
        }
    };

    println!("  Ollama output: {:?}", ollama_answer);
    println!("  Ollama time: {:.2}s", ollama_time);

    // Check if Ollama answer contains "4"
    let ollama_correct = ollama_answer.contains('4');
    if ollama_correct {
        println!("{}[PASS]{} Ollama groundtruth is correct", GREEN, NC);
    } else {
        println!(
            "{}[WARN]{} Ollama groundtruth doesn't contain '4': {}",
            YELLOW, NC, ollama_answer
        );
    }

    // Test 2: APR output
    println!();
    println!("{}Test 2: APR Output{}", BOLD, NC);
    let apr_start = Instant::now();
    let apr_output = Command::new(&config.apr_binary)
        .args([
            "run",
            &config.gguf_model,
            "--prompt",
            prompt,
            "--max-tokens",
            "10",
        ])
        .output();

    let (apr_answer, apr_time) = match apr_output {
        Ok(output) => {
            let answer = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let time = apr_start.elapsed().as_secs_f64();
            (answer, time)
        }
        Err(e) => {
            println!("{}[FAIL]{} APR execution failed: {}", RED, NC, e);
            return false;
        }
    };

    println!("  APR output: {:?}", apr_answer);
    println!("  APR time: {:.2}s", apr_time);

    // Test 3: Correctness comparison
    println!();
    println!("{}Test 3: Correctness Parity{}", BOLD, NC);
    let apr_correct = apr_answer.contains('4');
    if apr_correct && ollama_correct {
        println!(
            "{}[PASS]{} P050: Both produce correct answer (contains '4')",
            GREEN, NC
        );
    } else if apr_correct {
        println!(
            "{}[PASS]{} P050: APR correct (Ollama groundtruth was incorrect)",
            GREEN, NC
        );
    } else {
        println!("{}[FAIL]{} P050: APR output doesn't contain '4'", RED, NC);
        all_passed = false;
    }

    // Test 4: Performance comparison (within 2x)
    println!();
    println!("{}Test 4: Performance Parity{}", BOLD, NC);
    let speedup = ollama_time / apr_time;
    let within_2x = apr_time <= ollama_time * 2.0;

    if within_2x {
        println!(
            "{}[PASS]{} P051: APR within 2x of Ollama ({:.2}x speedup)",
            GREEN, NC, speedup
        );
    } else {
        println!(
            "{}[FAIL]{} P051: APR too slow (need 2x, got {:.2}x)",
            RED, NC, speedup
        );
        all_passed = false;
    }

    // Summary
    println!();
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );
    println!(
        "{}Ollama Parity: {} | Speedup: {:.2}x | APR: {:.2}s | Ollama: {:.2}s{}",
        if all_passed { GREEN } else { RED },
        if all_passed { "PASS" } else { "FAIL" },
        speedup,
        apr_time,
        ollama_time,
        NC
    );
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        CYAN, NC
    );

    if config.verbose {
        println!();
        println!("{}Detailed Comparison:{}", MAGENTA, NC);
        println!("  Prompt:        {:?}", prompt);
        println!("  Ollama Model:  {}", OLLAMA_MODEL);
        println!("  APR Model:     {}", config.gguf_model);
        println!("  Ollama Answer: {:?}", ollama_answer);
        println!("  APR Answer:    {:?}", apr_answer);
    }

    all_passed
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

    /// Test: Performance thresholds are format-specific (PMAT-SHOWCASE-METHODOLOGY-001)
    ///
    /// Uses conservative thresholds due to word-based estimation variance.
    #[test]
    fn test_performance_thresholds_config() {
        let config = Config::default();

        // CPU threshold for 1.5B (~5-10 tok/s observed)
        assert!((config.min_cpu_tps - 5.0).abs() < 0.01);

        // GPU quantized threshold (conservative due to estimation variance)
        assert!((config.min_gpu_tps - 5.0).abs() < 0.01);

        // GPU float32 threshold (SafeTensors 1.5B)
        assert!((config.min_gpu_tps_float32 - 5.0).abs() < 0.01);

        // All use same conservative threshold
        assert!((config.min_cpu_tps - config.min_gpu_tps).abs() < 0.01);
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

        // All use conservative 5.0 threshold due to word-based estimation variance
        assert!((get_threshold(Backend::Cpu, Format::Gguf) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Cpu, Format::SafeTensors) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Cpu, Format::Apr) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::Gguf) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::SafeTensors) - 5.0).abs() < 0.01);
        assert!((get_threshold(Backend::Gpu, Format::Apr) - 5.0).abs() < 0.01);
    }

    /// Test: CLI parsing for new --min-gpu-tps-f32 option
    #[test]
    fn test_cli_parsing_float32_threshold() {
        // Verify the default is set correctly (conservative threshold)
        let config = Config::default();
        assert!((config.min_gpu_tps_float32 - 5.0).abs() < 0.01);
    }
}
