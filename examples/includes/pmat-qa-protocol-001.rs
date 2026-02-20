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

