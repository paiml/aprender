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

