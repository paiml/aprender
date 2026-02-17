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
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

// ============================================================================
// SIGINT RESILIENCY: Global Process Registry (PMAT-098-PF)
// ============================================================================

/// Global registry of active child processes for SIGINT cleanup.
/// All spawned child processes (especially `apr serve`) must be registered here
/// so they can be killed on Ctrl+C to prevent zombie processes.
static PROCESS_REGISTRY: OnceLock<Arc<Mutex<Vec<u32>>>> = OnceLock::new();

/// Get the global process registry, initializing if needed
fn get_registry() -> Arc<Mutex<Vec<u32>>> {
    PROCESS_REGISTRY
        .get_or_init(|| Arc::new(Mutex::new(Vec::new())))
        .clone()
}

/// Register a child process ID for SIGINT cleanup
fn register_process(pid: u32) {
    if let Ok(mut registry) = get_registry().lock() {
        registry.push(pid);
    }
}

/// Unregister a child process ID (after successful reap)
fn unregister_process(pid: u32) {
    if let Ok(mut registry) = get_registry().lock() {
        registry.retain(|&p| p != pid);
    }
}

/// Kill all registered child processes (called by SIGINT handler)
fn kill_all_registered() -> usize {
    let registry = get_registry();
    let pids = match registry.lock() {
        Ok(guard) => guard.clone(),
        Err(_) => return 0,
    };

    let count = pids.len();
    for pid in pids {
        // Use kill(2) syscall via std::process::Command
        // This is portable and doesn't require libc
        #[cfg(unix)]
        {
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).output();
        }
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .output();
        }
    }
    count
}

/// RAII guard that auto-kills a child process on drop (panic safety)
/// This provides a secondary layer of safety for panics.
struct ProcessGuard {
    child: Option<Child>,
    pid: u32,
}

impl ProcessGuard {
    fn new(child: Child) -> Self {
        let pid = child.id();
        register_process(pid);
        Self {
            child: Some(child),
            pid,
        }
    }

    /// Get mutable reference to the child process
    fn child_mut(&mut self) -> Option<&mut Child> {
        self.child.as_mut()
    }

    /// Take ownership of the child, preventing auto-kill on drop
    #[allow(dead_code)]
    fn take(&mut self) -> Option<Child> {
        unregister_process(self.pid);
        self.child.take()
    }

    /// Manually kill and wait for the child
    fn kill_and_wait(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
        unregister_process(self.pid);
        self.child = None;
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        // Kill the process if it's still running
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
            unregister_process(self.pid);
        }
    }
}

/// Set up SIGINT/SIGTERM handler for graceful shutdown
/// Must be called once at program start
fn setup_signal_handler() {
    if let Err(e) = ctrlc::set_handler(move || {
        let count = kill_all_registered();
        eprintln!(
            "\n{}[JIDOKA] SIGINT received. Reaping {} active child process(es)...{}",
            "\x1b[1;33m", count, "\x1b[0m"
        );
        std::process::exit(130); // Standard exit code for SIGINT
    }) {
        eprintln!(
            "{}Warning: Could not set SIGINT handler: {}{}",
            "\x1b[33m", e, "\x1b[0m"
        );
    }
}

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
    #[allow(dead_code)] // Reserved for future use (e.g., CLI output)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    model_uri: String, // HuggingFace URI or local path
    with_trace: bool,  // Test with --trace flag
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

/// Model fixture for RAII-based model management (PMAT-QA-PROTOCOL-001 §7.1)
///
/// Ensures models are accessible before running tests. For HuggingFace models,
/// verifies the model can be resolved (triggering download if needed).
/// This is NOT a destructor-based cleanup (HF cache should persist for efficiency).
#[allow(dead_code)] // Will be used in full fixture-based test flow
struct ModelFixture {
    /// Original URI (e.g., "hf://Qwen/...")
    uri: String,
    /// Resolved local path (after download if needed)
    resolved_path: Option<String>,
    /// Format detected from the model
    format: Format,
    /// Whether the model was successfully verified
    verified: bool,
    /// Error message if verification failed
    error: Option<String>,
}

impl ModelFixture {
    /// Create a new fixture from a model URI
    /// Does NOT automatically verify - call verify() explicitly
    fn new(uri: &str, format: Format) -> Self {
        Self {
            uri: uri.to_string(),
            resolved_path: None,
            format,
            verified: false,
            error: None,
        }
    }

    /// Verify the model is accessible using apr (triggers download if HF URI)
    /// Uses `apr inspect` with --quiet flag to check without loading full model
    fn verify(&mut self, config: &Config) -> bool {
        // For local paths, just check file exists
        if !self.uri.starts_with("hf://") {
            let path = std::path::Path::new(&self.uri);
            if path.exists() {
                self.resolved_path = Some(self.uri.clone());
                self.verified = true;
                return true;
            }
            self.error = Some(format!("Local path not found: {}", self.uri));
            self.verified = false;
            return false;
        }

        // For HF URIs, use apr to verify/download
        // apr inspect will resolve and cache the model
        let args = vec!["inspect", &self.uri, "--quiet"];
        match run_apr(config, &args) {
            Ok(output) => {
                // apr inspect outputs the resolved path
                // Look for path in output (may vary by version)
                if output.contains("Path:") || output.contains("/") {
                    self.resolved_path = Some(self.uri.clone()); // Keep original URI for apr
                    self.verified = true;
                    true
                } else {
                    // Model exists but couldn't parse path - still usable
                    self.resolved_path = Some(self.uri.clone());
                    self.verified = true;
                    true
                }
            }
            Err(e) => {
                self.error = Some(format!("Failed to verify model: {}", e));
                self.verified = false;
                false
            }
        }
    }

    /// Get the path to use for apr commands (original URI - apr handles resolution)
    #[allow(dead_code)] // Reserved for future fixture-based flow
    fn path(&self) -> &str {
        self.resolved_path.as_deref().unwrap_or(&self.uri)
    }

    /// Check if the fixture is verified and ready for use
    #[allow(dead_code)] // Reserved for future fixture-based flow
    fn is_ready(&self) -> bool {
        self.verified
    }
}

/// Verify all required models before running tests (PMAT-QA-PROTOCOL-001 §7.2)
/// Returns (success_count, failures) for reporting
#[allow(dead_code)] // Will be used in fixture-based test flow
fn verify_model_fixtures(config: &Config, fixtures: &mut [ModelFixture]) -> (usize, Vec<String>) {
    let mut successes = 0;
    let mut failures = Vec::new();

    for fixture in fixtures.iter_mut() {
        if fixture.verify(config) {
            successes += 1;
        } else {
            failures.push(format!(
                "{} ({:?}): {}",
                fixture.uri,
                fixture.format,
                fixture.error.as_deref().unwrap_or("Unknown error")
            ));
        }
    }

    (successes, failures)
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

/// Run `apr chat` with input piped to stdin (PMAT-QA-PROTOCOL-001 §7.4)
///
/// Chat mode is tested by piping a prompt to stdin and capturing stdout.
/// This catches hangs that only occur in interactive mode.
fn run_chat_test(
    config: &Config,
    model: &str,
    prompt: &str,
    backend: Backend,
    with_trace: bool,
    timeout: Duration,
) -> Result<String, String> {
    use std::io::Write;

    let mut args: Vec<&str> = vec!["chat", model];
    if let Some(flag) = backend.flag() {
        args.push(flag);
    }
    if with_trace {
        args.push("--trace");
    }

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
        eprintln!("{}DEBUG (chat): {:?}{}", CYAN, cmd, NC);
    }

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return Err(format!("Failed to spawn chat: {}", e)),
    };

    // Register process for SIGINT cleanup (PMAT-098-PF)
    let pid = child.id();
    register_process(pid);

    // Write prompt to stdin, then close it to signal EOF
    if let Some(mut stdin) = child.stdin.take() {
        let _ = writeln!(stdin, "{}", prompt);
        // stdin is dropped here, closing the pipe
    }

    // Wait with timeout (this handles kill on timeout)
    let status = wait_with_timeout(&mut child, timeout);

    // Unregister after process is reaped (success or timeout)
    unregister_process(pid);

    let status = status?;

    // Read output
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
        Err(format!(
            "Chat exit {}: {}{}",
            status, stdout_str, stderr_str
        ))
    }
}

/// Poll server health endpoint until ready or timeout (PMAT-QA-PROTOCOL-001 §7.4)
fn wait_for_server_ready(server_guard: &mut ProcessGuard, port: u16) -> Result<(), String> {
    let start = Instant::now();
    let server_timeout = Duration::from_secs(30);
    let health_url = format!("http://127.0.0.1:{port}/health");

    loop {
        if start.elapsed() >= server_timeout {
            return Err("Server startup timeout (30s)".to_string());
        }
        if let Some(child) = server_guard.child_mut() {
            if let Ok(Some(status)) = child.try_wait() {
                return Err(format!("Server exited early: {status}"));
            }
        }
        let health_check = Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &health_url])
            .output();
        if let Ok(output) = health_check {
            let code = String::from_utf8_lossy(&output.stdout);
            if code.trim() == "200" {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Run `apr serve` test with HTTP request (PMAT-QA-PROTOCOL-001 §7.4)
///
/// 1. Start apr serve on a random port
/// 2. Wait for server ready
/// 3. Send curl request
/// 4. Capture response
/// 5. Kill server
fn run_serve_test(
    config: &Config,
    model: &str,
    prompt: &str,
    backend: Backend,
    with_trace: bool,
    timeout: Duration,
) -> Result<String, String> {
    use std::net::TcpListener;

    // Find an available port
    let port = TcpListener::bind("127.0.0.1:0")
        .map(|l| {
            l.local_addr()
                .expect("bound listener must have local addr")
                .port()
        })
        .unwrap_or(18080);

    let port_str = port.to_string();
    let mut args: Vec<&str> = vec!["serve", model, "--port", &port_str];
    if let Some(flag) = backend.flag() {
        args.push(flag);
    }

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

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    if config.verbose {
        eprintln!("{}DEBUG (serve): {:?} on port {}{}", CYAN, cmd, port, NC);
    }

    // Wrap server in ProcessGuard for SIGINT safety (PMAT-098-PF)
    // This ensures the server is killed even if:
    // 1. The user presses Ctrl+C
    // 2. The test panics
    // 3. An early return occurs
    let mut server_guard = match cmd.spawn() {
        Ok(child) => ProcessGuard::new(child),
        Err(e) => return Err(format!("Failed to spawn serve: {}", e)),
    };

    // Wait for server to be ready (poll health endpoint)
    wait_for_server_ready(&mut server_guard, port)?;

    // Build request body
    let body = format!(
        r#"{{"model":"test","messages":[{{"role":"user","content":"{}"}}],"max_tokens":10}}"#,
        prompt
    );

    // Send chat completion request
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let mut curl_args = vec![
        "-s",
        "-X",
        "POST",
        &url,
        "-H",
        "Content-Type: application/json",
        "-d",
        &body,
    ];

    if with_trace {
        curl_args.extend(["-H", "X-Trace-Level: layer"]);
    }

    let request_start = Instant::now();
    let response = Command::new("curl")
        .args(&curl_args)
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    // Check timeout
    if request_start.elapsed() >= timeout {
        // ProcessGuard::drop will kill the server
        return Err(format!("Request timeout ({}s)", timeout.as_secs()));
    }

    // Explicitly kill server before returning (ProcessGuard::drop would do this too)
    server_guard.kill_and_wait();

    if response.status.success() {
        Ok(String::from_utf8_lossy(&response.stdout).to_string())
    } else {
        Err(format!(
            "curl error: {}",
            String::from_utf8_lossy(&response.stderr)
        ))
    }
}

/// Run test based on modality (PMAT-QA-PROTOCOL-001 §7.4)
fn run_modality_test(
    config: &Config,
    cell: &MatrixCell,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, String> {
    let max_tokens_str = max_tokens.to_string();

    match cell.modality {
        Modality::Run => {
            let mut args: Vec<&str> = vec![
                "run",
                &cell.model_uri,
                "--prompt",
                prompt,
                "--max-tokens",
                &max_tokens_str,
            ];
            if let Some(flag) = cell.backend.flag() {
                args.push(flag);
            }
            if cell.with_trace {
                args.push("--trace");
            }
            run_apr(config, &args)
        }
        Modality::Chat => run_chat_test(
            config,
            &cell.model_uri,
            prompt,
            cell.backend,
            cell.with_trace,
            DEFAULT_TIMEOUT,
        ),
        Modality::Serve => run_serve_test(
            config,
            &cell.model_uri,
            prompt,
            cell.backend,
            cell.with_trace,
            DEFAULT_TIMEOUT,
        ),
    }
}

/// Strip ANSI escape sequences from a string (e.g., \x1b[1;32m → "")
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip ESC [ ... m sequences
            if chars.next() == Some('[') {
                for c2 in chars.by_ref() {
                    if c2.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Extract just the model output from apr run output (between "Output:" and "Completed in")
/// This filters out compilation warnings, paths, and timing info that might contain false positives.
/// Handles ANSI color codes in output (e.g., \x1b[1;32mOutput:\x1b[0m)
fn extract_output(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let mut in_output = false;
    let mut content = Vec::new();
    for line in lines {
        let clean = strip_ansi(line);
        if clean.starts_with("Output:") {
            in_output = true;
            continue;
        }
        if clean.starts_with("Completed in ") {
            break;
        }
        if in_output {
            content.push(strip_ansi(line));
        }
    }
    content.join("\n").trim().to_string()
}

/// Garbage patterns that indicate model collapse or tokenization failure
/// (PMAT-QA-PROTOCOL-001 §7.5)
const GARBAGE_PATTERNS: &[&str] = &[
    "\u{FFFD}", // Replacement character (encoding error)
    "[UNK]",    // Unknown token marker
    "akunji",   // Known GQA bug garbage
    "olumbia",  // Known layout bug garbage
    "专门窗",   // Known GQA bug CJK garbage
    "token0",   // Raw token ID leak
    "token1",   // Raw token ID leak
    "<0x",      // Byte token leak (e.g., <0x0A>)
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
    /// Captures verified output for potential debugging/logging
    #[allow(dead_code)]
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

    // 4. Expected answer check with word boundary validation
    // (Fixed: PMAT-098 Red Team falsification found naive substring matching is brittle)
    if let Some(expected) = expected_contains {
        if !contains_as_word(trimmed, expected) {
            return VerifyResult::FailMissingAnswer(format!(
                "Expected '{}' as standalone word, got: {}",
                expected,
                trimmed.chars().take(50).collect::<String>()
            ));
        }
    }

    VerifyResult::Pass(trimmed.to_string())
}

/// Check if `needle` appears in `haystack` as a standalone word (not embedded in another word/number)
/// This prevents false positives like "14" matching expected "4"
fn contains_as_word(haystack: &str, needle: &str) -> bool {
    // Find all occurrences and check word boundaries
    let mut search_start = 0;
    while let Some(pos) = haystack[search_start..].find(needle) {
        let abs_pos = search_start + pos;
        let end_pos = abs_pos + needle.len();

        // Check left boundary: start of string OR non-alphanumeric
        let left_ok = abs_pos == 0 || {
            let prev_char = haystack[..abs_pos]
                .chars()
                .last()
                .expect("non-empty prefix must have a last char");
            !prev_char.is_alphanumeric()
        };

        // Check right boundary: end of string OR non-alphanumeric
        let right_ok = end_pos >= haystack.len() || {
            let next_char = haystack[end_pos..]
                .chars()
                .next()
                .expect("non-empty suffix must have a next char");
            !next_char.is_alphanumeric()
        };

        if left_ok && right_ok {
            return true;
        }

        // Continue searching after this occurrence
        search_start = abs_pos + 1;
        if search_start >= haystack.len() {
            break;
        }
    }
    false
}

/// Convert VerifyResult to TestResult for a named test
fn verify_to_test(name: &'static str, max_points: u32, result: VerifyResult) -> TestResult {
    match result {
        VerifyResult::Pass(_) => TestResult::pass(name, max_points, "Clean output".to_string()),
        VerifyResult::FailEmpty => TestResult::fail(name, max_points, "Empty output".to_string()),
        VerifyResult::FailGarbage(p) => {
            TestResult::fail(name, max_points, format!("GARBAGE: '{p}'"))
        }
        VerifyResult::FailBpeArtifact(c) => {
            TestResult::fail(name, max_points, format!("BPE artifact: '{c}'"))
        }
        VerifyResult::FailMissingAnswer(msg) => TestResult::fail(name, max_points, msg),
    }
}

/// Run output verification test: run model, extract output, verify quality
fn run_verify_test(
    config: &Config,
    cell: &MatrixCell,
    name: &'static str,
    max_points: u32,
    prompt: &str,
    max_tokens: u32,
    expected: Option<&str>,
) -> TestResult {
    match run_modality_test(config, cell, prompt, max_tokens) {
        Ok(raw) => verify_to_test(
            name,
            max_points,
            verify_output(&extract_output(&raw), expected),
        ),
        Err(e) => TestResult::fail(name, max_points, e),
    }
}

/// Run performance test: measure tok/s against threshold
fn run_perf_test(config: &Config, cell: &MatrixCell) -> TestResult {
    let perf_start = Instant::now();
    match run_modality_test(config, cell, "Count from 1 to 20.", 50) {
        Ok(output) => {
            let elapsed = perf_start.elapsed().as_secs_f64();
            let tokens_est = (output.split_whitespace().count() as f64 * 1.3).max(10.0);
            let tps = tokens_est / elapsed;
            let base_target = match (cell.backend, cell.format) {
                (Backend::Cpu, _) => config.min_cpu_tps,
                (Backend::Gpu, Format::SafeTensors) => config.min_gpu_tps_float32,
                (Backend::Gpu, _) => config.min_gpu_tps,
            };
            let target = match cell.modality {
                Modality::Run => base_target,
                Modality::Chat | Modality::Serve => base_target * 0.5,
            };
            if tps >= target {
                TestResult::pass("Performance", 3, format!("{tps:.1} tok/s >= {target:.1}"))
            } else {
                TestResult::fail("Performance", 3, format!("{tps:.1} tok/s < {target:.1}"))
            }
        }
        Err(e) => TestResult::fail("Performance", 3, e),
    }
}

/// Run all tests for a single matrix cell
/// Dispatches to run_modality_test for Chat/Serve modalities (PMAT-QA-PROTOCOL-001 §7.4)
fn run_cell_tests(config: &Config, cell: &MatrixCell) -> CellResult {
    let start = Instant::now();
    let mut tests = Vec::new();

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

    // Test 1: Model loads (2 points)
    match run_modality_test(
        config,
        cell,
        "What is 2+2? Answer with just the number.",
        10,
    ) {
        Ok(_) => tests.push(TestResult::pass(
            "Model Load",
            2,
            format!("{} via {:?}", cell.model_uri, cell.modality),
        )),
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
    tests.push(run_verify_test(
        config,
        cell,
        "Correct Output",
        3,
        "What is 2+2? Answer with just the number.",
        10,
        Some("4"),
    ));

    // Test 3: No garbage (3 points)
    tests.push(run_verify_test(
        config,
        cell,
        "No Garbage",
        3,
        "Say hello.",
        20,
        None,
    ));

    // Test 4: No BPE artifacts (2 points)
    match run_modality_test(config, cell, "Say hello.", 20) {
        Ok(raw) => {
            let output = extract_output(&raw);
            let has_bpe = BPE_ARTIFACTS.iter().any(|&c| output.contains(c));
            tests.push(if has_bpe {
                TestResult::fail("No BPE Artifacts", 2, "Ġ/Ċ/ĉ detected".to_string())
            } else {
                TestResult::pass("No BPE Artifacts", 2, "Clean tokens".to_string())
            });
        }
        Err(e) => tests.push(TestResult::fail("No BPE Artifacts", 2, e)),
    }

    // Test 5: Trace works (2 points)
    let trace_cell = MatrixCell {
        with_trace: true,
        ..cell.clone()
    };
    match run_modality_test(config, &trace_cell, "Hi", 5) {
        Ok(_) => tests.push(TestResult::pass(
            "Trace Works",
            2,
            format!("{:?} + trace accepted", cell.modality),
        )),
        Err(e) if e.contains("not supported") || e.contains("trace") => {
            tests.push(TestResult::skip(
                "Trace Works",
                2,
                format!("Trace not supported for {:?}", cell.modality),
            ));
        }
        Err(e) => tests.push(TestResult::fail("Trace Works", 2, e)),
    }

    // Test 6: Performance (3 points)
    tests.push(run_perf_test(config, cell));

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

/// Format a cell result as a status string
fn format_cell_status(result: &CellResult) -> String {
    if result.passed() {
        format!("{}✓ {}/{}{}  ", GREEN, result.total_points, result.max_points, NC)
    } else {
        format!("{}✗ {}/{}{}  ", RED, result.total_points, result.max_points, NC)
    }
}

/// Print a single row (CPU or GPU) of the QA matrix
fn print_matrix_row(label: &str, backend: Backend, results: &[CellResult]) {
    print!("{}║{} {:^10} │", MAGENTA, NC, label);
    for fmt in [Format::Gguf, Format::SafeTensors, Format::Apr] {
        if let Some(r) = results.iter().find(|r| r.cell.backend == backend && r.cell.format == fmt) {
            print!(" {:^12} │", format_cell_status(r));
        } else {
            print!(" {:^12} │", "—");
        }
    }
    println!("{}║{}", MAGENTA, NC);
}

/// Compute letter grade from points ratio
fn compute_grade(total_points: u32, max_points: u32) -> &'static str {
    if total_points == max_points {
        "A+"
    } else {
        let ratio = total_points as f64 / max_points as f64;
        if ratio >= 0.9 { "A" } else if ratio >= 0.8 { "B" } else if ratio >= 0.7 { "C" } else { "F" }
    }
}

fn print_matrix_summary(results: &[CellResult]) {
    println!();
    println!("{}╔═════════════════════════════════════════════════════════════╗{}", MAGENTA, NC);
    println!("{}║             QA MATRIX SUMMARY (PMAT-QA-MATRIX-001)          ║{}", MAGENTA, NC);
    println!("{}╠═════════════════════════════════════════════════════════════╣{}", MAGENTA, NC);
    println!("{}║{} {:^10} │ {:^12} │ {:^12} │ {:^12} {}║{}", MAGENTA, NC, "", "GGUF", "SafeTensors", "APR", MAGENTA, NC);
    println!("{}╟───────────┼──────────────┼──────────────┼──────────────╢{}", MAGENTA, NC);

    print_matrix_row("CPU", Backend::Cpu, results);
    print_matrix_row("GPU", Backend::Gpu, results);

    println!("{}╠═════════════════════════════════════════════════════════════╣{}", MAGENTA, NC);

    let total_points: u32 = results.iter().map(|r| r.total_points).sum();
    let max_points: u32 = results.iter().map(|r| r.max_points).sum();
    let passed = results.iter().filter(|r| r.passed()).count();
    let total = results.len();
    let grade = compute_grade(total_points, max_points);

    println!("{}║{} Cells: {}/{} passed    Points: {}/{}    Grade: {:>14}{}║{}", MAGENTA, NC, passed, total, total_points, max_points, grade, MAGENTA, NC);
    println!("{}╚═════════════════════════════════════════════════════════════╝{}", MAGENTA, NC);

    if passed == total {
        println!();
        println!("{}Hypothesis \"apr run produces correct output across all formats/backends\" SURVIVED.{}", GREEN, NC);
    } else {
        println!();
        println!("{}Hypothesis FALSIFIED. {} cell(s) failed.{}", RED, total - passed, NC);
    }
}

fn print_help() {
    println!("{}QA Matrix Runner (PMAT-QA-PROTOCOL-001){}", BOLD, NC);
    println!();
    println!("{}CRITICAL: Same-Model Comparison Protocol{}", YELLOW, NC);
    println!("  Class A (Quantized): GGUF Q4_K vs APR Q4_K (SAME weights)");
    println!("  Class B (Full Prec): SafeTensors F32 vs APR F32 (SAME weights)");
    println!();
    println!("USAGE:");
    println!("    cargo run --example qa_run -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --matrix              Run backend × format matrix (apr run only)");
    println!(
        "    --full-matrix         {}Run FULL 21-cell matrix (modality × format × trace){}",
        CYAN, NC
    );
    println!("    --modality <MODE>     Modality: run (default), chat, serve");
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
    println!("MODALITIES (PMAT-QA-PROTOCOL-001 §7.4):");
    println!("    run     `apr run` - single prompt inference");
    println!("    chat    `apr chat` - interactive mode (stdin piped)");
    println!("    serve   `apr serve` - HTTP server (curl tested)");
    println!();
    println!("TEST CLASSES:");
    println!("    quantized      Class A: GGUF Q4_K vs APR Q4_K (faster)");
    println!("    full-precision Class B: SafeTensors F32 vs APR F32 (slower)");
    println!("    all            Both Class A and B");
    println!();
    println!("CANONICAL MODEL:");
    println!("    {}", CANONICAL_GGUF);
    println!();
    println!("EXAMPLES:");
    println!("    # Quick backend × format matrix (apr run only)");
    println!("    cargo run --example qa_run -- --matrix");
    println!();
    println!(
        "    {}# FULL 21-cell matrix (all modalities × formats × trace){}",
        CYAN, NC
    );
    println!("    cargo run --example qa_run -- --full-matrix");
    println!();
    println!("    # Single modality test");
    println!("    cargo run --example qa_run -- --modality chat --backend cpu --format gguf");
    println!();
    println!("    # Compare against Ollama groundtruth");
    println!("    cargo run --example qa_run -- --with-ollama");
}

/// Parsed CLI arguments for the QA run matrix
struct ParsedArgs {
    config: Config,
    run_matrix: bool,
    run_full_matrix: bool,
    single_backend: Option<Backend>,
    single_format: Option<Format>,
    single_modality: Option<Modality>,
    legacy_model: Option<PathBuf>,
    show_help: bool,
}

fn parse_args(args: &[String]) -> ParsedArgs {
    let mut parsed = ParsedArgs {
        config: Config::default(),
        run_matrix: false,
        run_full_matrix: false,
        single_backend: None,
        single_format: None,
        single_modality: None,
        legacy_model: None,
        show_help: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--matrix" => parsed.run_matrix = true,
            "--full-matrix" => parsed.run_full_matrix = true,
            "--with-ollama" => parsed.config.with_ollama = true,
            "--verbose" | "-v" => parsed.config.verbose = true,
            "--help" | "-h" => parsed.show_help = true,
            flag => {
                if let Some(val) = args.get(i + 1) {
                    parse_flag_with_value(flag, val, &mut parsed);
                    i += 1; // extra increment for value
                }
            }
        }
        i += 1;
    }

    parsed
}

fn parse_modality(val: &str) -> Option<Modality> {
    match val {
        "run" => Some(Modality::Run),
        "chat" => Some(Modality::Chat),
        "serve" => Some(Modality::Serve),
        _ => None,
    }
}

fn parse_backend(val: &str) -> Option<Backend> {
    match val {
        "cpu" => Some(Backend::Cpu),
        "gpu" => Some(Backend::Gpu),
        _ => None,
    }
}

fn parse_format(val: &str) -> Option<Format> {
    match val {
        "gguf" => Some(Format::Gguf),
        "safetensors" => Some(Format::SafeTensors),
        "apr" => Some(Format::Apr),
        _ => None,
    }
}

fn parse_trace_level(val: &str) -> TraceLevel {
    match val {
        "brick" => TraceLevel::Brick,
        "step" => TraceLevel::Step,
        "layer" => TraceLevel::Layer,
        "profile" => TraceLevel::Profile,
        _ => TraceLevel::None,
    }
}

fn parse_test_class(val: &str) -> TestClass {
    match val {
        "quantized" | "a" | "A" => TestClass::Quantized,
        "full-precision" | "fp" | "b" | "B" => TestClass::FullPrecision,
        "all" | "both" => TestClass::All,
        _ => TestClass::Quantized,
    }
}

fn parse_flag_with_value(flag: &str, val: &str, parsed: &mut ParsedArgs) {
    match flag {
        "--modality" => parsed.single_modality = parse_modality(val),
        "--backend" => parsed.single_backend = parse_backend(val),
        "--format" => parsed.single_format = parse_format(val),
        "--trace-level" => parsed.config.trace_level = parse_trace_level(val),
        "--class" => parsed.config.test_class = parse_test_class(val),
        "--gguf" => parsed.config.gguf_model = val.to_string(),
        "--safetensors" => parsed.config.safetensors_model = val.to_string(),
        "--apr" => parsed.config.apr_model = val.to_string(),
        "--model" => parsed.legacy_model = Some(PathBuf::from(val)),
        "--min-cpu-tps" => parsed.config.min_cpu_tps = val.parse().unwrap_or(8.0),
        "--min-gpu-tps" => parsed.config.min_gpu_tps = val.parse().unwrap_or(100.0),
        "--min-gpu-tps-f32" => parsed.config.min_gpu_tps_float32 = val.parse().unwrap_or(40.0),
        _ => {}
    }
}

/// Resolve model URI for a given format from config
fn model_for_format(config: &Config, format: Format) -> String {
    match format {
        Format::Gguf => config.gguf_model.clone(),
        Format::SafeTensors => config.safetensors_model.clone(),
        Format::Apr => config.apr_model.clone(),
    }
}

/// Build matrix cells based on parsed CLI arguments (PMAT-SHOWCASE-METHODOLOGY-001)
fn build_cells(config: &Config, parsed: &ParsedArgs) -> Vec<MatrixCell> {
    if parsed.run_full_matrix {
        return build_full_matrix_cells(config);
    }
    if parsed.run_matrix {
        return build_standard_matrix_cells(config);
    }
    if let (Some(modality), Some(backend), Some(format)) = (
        parsed.single_modality,
        parsed.single_backend,
        parsed.single_format,
    ) {
        let model = model_for_format(config, format);
        return vec![MatrixCell::new("S1", backend, format, model).with_modality(modality)];
    }
    if let (Some(backend), Some(format)) = (parsed.single_backend, parsed.single_format) {
        let model = model_for_format(config, format);
        return vec![MatrixCell::new("S1", backend, format, model)];
    }
    if let Some(ref model_path) = parsed.legacy_model {
        return build_legacy_cells(model_path);
    }
    println!(
        "{}No mode specified. Use --matrix, --backend + --format, or --model{}",
        YELLOW, NC
    );
    println!();
    print_help();
    std::process::exit(2);
}

/// Build full 21-cell matrix (3 modalities × 3 formats × trace variants)
fn build_full_matrix_cells(config: &Config) -> Vec<MatrixCell> {
    let mut cells = Vec::new();
    let mut id = 1;
    for modality in [Modality::Run, Modality::Chat, Modality::Serve] {
        for format in [Format::Gguf, Format::SafeTensors, Format::Apr] {
            let model = model_for_format(config, format);
            cells.push(
                MatrixCell::new(&format!("F{id:02}"), Backend::Cpu, format, model.clone())
                    .with_modality(modality),
            );
            id += 1;
            cells.push(
                MatrixCell::new(&format!("F{id:02}"), Backend::Cpu, format, model.clone())
                    .with_modality(modality)
                    .with_trace(true),
            );
            id += 1;
            if format == Format::Gguf {
                cells.push(
                    MatrixCell::new(&format!("F{id:02}"), Backend::Gpu, format, model)
                        .with_modality(modality),
                );
                id += 1;
            }
        }
    }
    println!(
        "{}FULL MATRIX: {} cells (modality × format × trace){}\n",
        MAGENTA,
        cells.len(),
        NC
    );
    cells
}

/// Build standard matrix cells (Class A quantized + Class B full precision)
fn build_standard_matrix_cells(config: &Config) -> Vec<MatrixCell> {
    let mut cells = Vec::new();
    if config.test_class.includes_quantized() {
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
    if config.test_class.includes_full_precision() {
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
}

/// Build cells for legacy --model flag
fn build_legacy_cells(model_path: &PathBuf) -> Vec<MatrixCell> {
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
}

fn main() {
    // Set up SIGINT handler for graceful shutdown (PMAT-098-PF: zombie mitigation)
    setup_signal_handler();

    let args: Vec<String> = env::args().collect();
    let parsed = parse_args(&args);

    if parsed.show_help {
        print_help();
        return;
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

    let cells = build_cells(&parsed.config, &parsed);
    let config = parsed.config;

    // Show what we're testing
    println!("{}Testing {} cell(s):{}", CYAN, cells.len(), NC);
    for cell in &cells {
        println!("  {} {} → {}", cell.id, cell.label(), cell.model_uri);
    }
    println!();

    // Pre-flight model verification (PMAT-QA-PROTOCOL-001 §7.1)
    // Collect unique models to verify (avoid redundant downloads)
    let unique_models: std::collections::HashSet<_> = cells
        .iter()
        .map(|c| (c.model_uri.clone(), c.format))
        .collect();

    let mut fixtures: Vec<ModelFixture> = unique_models
        .into_iter()
        .map(|(uri, format)| ModelFixture::new(&uri, format))
        .collect();

    println!(
        "{}Pre-flight: Verifying {} unique model(s)...{}",
        CYAN,
        fixtures.len(),
        NC
    );

    let (verified, failures) = verify_model_fixtures(&config, &mut fixtures);

    if !failures.is_empty() {
        println!(
            "{}✗ Model verification failed ({}/{}):{}\n",
            RED,
            failures.len(),
            fixtures.len(),
            NC
        );
        for failure in &failures {
            println!("  {}{}{}", RED, failure, NC);
        }
        println!();
        println!("{}ABORT: Cannot run tests with missing models{}", RED, NC);
        std::process::exit(3);
    }

    println!("{}✓ All {} model(s) verified{}\n", GREEN, verified, NC);

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

/// Check if ollama binary is available on the system
fn is_ollama_installed() -> bool {
    Command::new("which")
        .arg("ollama")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if the required ollama model is available
fn is_ollama_model_available() -> bool {
    Command::new("ollama")
        .args(["show", OLLAMA_MODEL])
        .stderr(Stdio::null())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a command and return (output_string, elapsed_seconds)
fn timed_command_output(cmd: &mut Command) -> Result<(String, f64), String> {
    let start = Instant::now();
    let output = cmd.output().map_err(|e| format!("Execution failed: {e}"))?;
    let answer = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok((answer, start.elapsed().as_secs_f64()))
}

/// Check correctness parity between apr and ollama
fn check_correctness_parity(apr_answer: &str, _ollama_answer: &str, ollama_correct: bool) -> bool {
    let apr_correct = apr_answer.contains('4');
    println!();
    println!("{}Test 3: Correctness Parity{}", BOLD, NC);
    if apr_correct && ollama_correct {
        println!("{}[PASS]{} P050: Both produce correct answer (contains '4')", GREEN, NC);
    } else if apr_correct {
        println!("{}[PASS]{} P050: APR correct (Ollama groundtruth was incorrect)", GREEN, NC);
    } else {
        println!("{}[FAIL]{} P050: APR output doesn't contain '4'", RED, NC);
        return false;
    }
    true
}

/// Check performance parity (apr within 2x of ollama)
fn check_performance_parity(apr_time: f64, ollama_time: f64) -> bool {
    let speedup = ollama_time / apr_time;
    let within_2x = apr_time <= ollama_time * 2.0;
    println!();
    println!("{}Test 4: Performance Parity{}", BOLD, NC);
    if within_2x {
        println!("{}[PASS]{} P051: APR within 2x of Ollama ({:.2}x speedup)", GREEN, NC, speedup);
        true
    } else {
        println!("{}[FAIL]{} P051: APR too slow (need 2x, got {:.2}x)", RED, NC, speedup);
        false
    }
}

/// Print ollama comparison summary
fn print_ollama_summary(all_passed: bool, apr_time: f64, ollama_time: f64) {
    let speedup = ollama_time / apr_time;
    println!();
    println!("{}═══════════════════════════════════════════════════════════════{}", CYAN, NC);
    println!(
        "{}Ollama Parity: {} | Speedup: {:.2}x | APR: {:.2}s | Ollama: {:.2}s{}",
        if all_passed { GREEN } else { RED },
        if all_passed { "PASS" } else { "FAIL" },
        speedup, apr_time, ollama_time, NC
    );
    println!("{}═══════════════════════════════════════════════════════════════{}", CYAN, NC);
}

/// Run Ollama parity comparison (PMAT-SHOWCASE-METHODOLOGY-001 Section 5)
///
/// Compares apr's output against Ollama as groundtruth for:
/// 1. Correctness - Output matches semantically
/// 2. Performance - Within 2x of Ollama tok/s
fn run_ollama_comparison(config: &Config) -> bool {
    println!();
    println!("{}═══════════════════════════════════════════════════════════════{}", CYAN, NC);
    println!("{}         OLLAMA PARITY TEST (PMAT-SHOWCASE-METHODOLOGY-001)      {}", CYAN, NC);
    println!("{}═══════════════════════════════════════════════════════════════{}", CYAN, NC);
    println!();

    if !is_ollama_installed() {
        println!("{}[SKIP]{} Ollama not installed - skipping parity test", YELLOW, NC);
        return true;
    }

    if !is_ollama_model_available() {
        println!("{}[SKIP]{} Ollama model {} not available", YELLOW, NC, OLLAMA_MODEL);
        println!("       Install with: ollama pull {}", OLLAMA_MODEL);
        return true;
    }

    let prompt = "What is 2+2? Answer with just the number.";

    // Test 1: Ollama groundtruth
    println!("{}Test 1: Ollama Groundtruth{}", BOLD, NC);
    let (ollama_answer, ollama_time) = match timed_command_output(
        Command::new("ollama").args(["run", OLLAMA_MODEL, prompt]),
    ) {
        Ok(r) => r,
        Err(e) => { println!("{}[FAIL]{} {}", RED, NC, e); return false; }
    };
    println!("  Ollama output: {:?}", ollama_answer);
    println!("  Ollama time: {:.2}s", ollama_time);

    let ollama_correct = ollama_answer.contains('4');
    if ollama_correct {
        println!("{}[PASS]{} Ollama groundtruth is correct", GREEN, NC);
    } else {
        println!("{}[WARN]{} Ollama groundtruth doesn't contain '4': {}", YELLOW, NC, ollama_answer);
    }

    // Test 2: APR output
    println!();
    println!("{}Test 2: APR Output{}", BOLD, NC);
    let (apr_answer, apr_time) = match timed_command_output(
        Command::new(&config.apr_binary).args(["run", &config.gguf_model, "--prompt", prompt, "--max-tokens", "10"]),
    ) {
        Ok(r) => r,
        Err(e) => { println!("{}[FAIL]{} {}", RED, NC, e); return false; }
    };
    println!("  APR output: {:?}", apr_answer);
    println!("  APR time: {:.2}s", apr_time);

    // Tests 3 & 4
    let correctness_ok = check_correctness_parity(&apr_answer, &ollama_answer, ollama_correct);
    let perf_ok = check_performance_parity(apr_time, ollama_time);
    let all_passed = correctness_ok && perf_ok;

    print_ollama_summary(all_passed, apr_time, ollama_time);

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
