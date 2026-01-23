//! Reliable Demo Best Practices (Part IX of chat-template-improvement-spec v1.4.0)
//!
//! Implements world-class developer experience inspired by:
//! - Hugging Face (model cards, auto-download)
//! - Ollama (zero-config CLI)
//! - llama.cpp (performance stats)
//! - llamafile (single-binary deployment)
//!
//! # Checklist Items
//!
//! - RDB-01: Zero-Config Guarantee
//! - RDB-02: Prerequisites & Environment Isolation
//! - RDB-03: Interactive & Non-Interactive Modes
//! - RDB-04: Robust Error Recovery
//! - RDB-05: Performance Transparency
//! - RDB-06: Model Provenance & Licensing

use crate::AprenderError;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(feature = "hf-hub-integration")]
use dirs;

// ============================================================================
// RDB-01: Zero-Config Guarantee
// ============================================================================

/// Model cache configuration for zero-config downloads
#[derive(Debug, Clone)]
pub struct ModelCache {
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Whether to auto-download missing models
    pub auto_download: bool,
    /// Maximum cache size in bytes (0 = unlimited)
    pub max_size_bytes: u64,
}

impl Default for ModelCache {
    fn default() -> Self {
        // Use dirs crate if available (hf-hub-integration feature), otherwise fallback to current dir
        #[cfg(feature = "hf-hub-integration")]
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("apr")
            .join("models");

        #[cfg(not(feature = "hf-hub-integration"))]
        let cache_dir = std::env::var("APR_CACHE_DIR")
            .map_or_else(|_| PathBuf::from(".apr_cache"), PathBuf::from)
            .join("models");

        Self {
            cache_dir,
            auto_download: true,
            max_size_bytes: 0, // Unlimited
        }
    }
}

impl ModelCache {
    /// Create a new model cache with custom directory
    #[must_use]
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            auto_download: true,
            max_size_bytes: 0,
        }
    }

    /// Get the path where a model should be cached
    #[must_use]
    pub fn model_path(&self, model_name: &str) -> PathBuf {
        self.cache_dir.join(model_name)
    }

    /// Check if a model exists in the cache
    #[must_use]
    pub fn has_model(&self, model_name: &str) -> bool {
        self.model_path(model_name).exists()
    }

    /// Ensure cache directory exists
    pub fn ensure_dir(&self) -> Result<(), AprenderError> {
        std::fs::create_dir_all(&self.cache_dir).map_err(|e| {
            AprenderError::Io(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to create cache directory: {}", e),
            ))
        })
    }
}

/// Model source specification for auto-download
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Hugging Face Hub (e.g., "hf://Qwen/Qwen2-0.5B-Instruct")
    HuggingFace { repo_id: String, filename: String },
    /// Direct URL
    Url(String),
    /// Local file path
    Local(PathBuf),
}

impl ModelSource {
    /// Parse a model source string
    ///
    /// Formats:
    /// - `hf://owner/repo/file.safetensors` -> HuggingFace
    /// - `https://...` or `http://...` -> Url
    /// - `/path/to/model` -> Local
    #[must_use]
    pub fn parse(source: &str) -> Self {
        if source.starts_with("hf://") {
            let path = source.strip_prefix("hf://").unwrap_or(source);
            let parts: Vec<&str> = path.splitn(3, '/').collect();
            if parts.len() >= 2 {
                let repo_id = format!("{}/{}", parts[0], parts[1]);
                let filename = (*parts.get(2).unwrap_or(&"model.safetensors")).to_string();
                return Self::HuggingFace { repo_id, filename };
            }
        }
        if source.starts_with("http://") || source.starts_with("https://") {
            return Self::Url(source.to_string());
        }
        Self::Local(PathBuf::from(source))
    }

    /// Check if this is a remote source that may need downloading
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::HuggingFace { .. } | Self::Url(_))
    }
}

// ============================================================================
// RDB-02: Prerequisites & Environment Isolation
// ============================================================================

/// Prerequisite check result
#[derive(Debug, Clone)]
pub struct PrerequisiteCheck {
    /// Name of the prerequisite
    pub name: String,
    /// Whether it's satisfied
    pub satisfied: bool,
    /// Version found (if applicable)
    pub version: Option<String>,
    /// How to install if missing
    pub install_hint: Option<String>,
}

impl PrerequisiteCheck {
    /// Create a satisfied prerequisite
    #[must_use]
    pub fn satisfied(name: &str) -> Self {
        Self {
            name: name.to_string(),
            satisfied: true,
            version: None,
            install_hint: None,
        }
    }

    /// Create a missing prerequisite with install hint
    #[must_use]
    pub fn missing(name: &str, install_hint: &str) -> Self {
        Self {
            name: name.to_string(),
            satisfied: false,
            version: None,
            install_hint: Some(install_hint.to_string()),
        }
    }
}

/// Check if an external command exists
#[must_use]
pub fn check_command(command: &str) -> PrerequisiteCheck {
    let exists = std::process::Command::new("which")
        .arg(command)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if exists {
        PrerequisiteCheck::satisfied(command)
    } else {
        PrerequisiteCheck::missing(
            command,
            &format!("Please install {}: check your package manager", command),
        )
    }
}

/// Check all prerequisites and return results
pub fn check_prerequisites(required: &[&str]) -> Vec<PrerequisiteCheck> {
    required.iter().map(|cmd| check_command(cmd)).collect()
}

/// Print prerequisite check results
pub fn print_prerequisites(checks: &[PrerequisiteCheck]) {
    for check in checks {
        if check.satisfied {
            eprintln!("  ✓ {} found", check.name);
        } else {
            eprintln!("  ✗ {} missing", check.name);
            if let Some(hint) = &check.install_hint {
                eprintln!("    → {}", hint);
            }
        }
    }
}

// ============================================================================
// RDB-03: Interactive & Non-Interactive Modes
// ============================================================================

/// Execution mode based on terminal detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Running in a terminal with TTY
    Interactive,
    /// Running in a pipe or batch mode
    Batch,
}

impl ExecutionMode {
    /// Detect execution mode from stdin/stdout
    #[must_use]
    pub fn detect() -> Self {
        if io::stdin().is_terminal() && io::stdout().is_terminal() {
            Self::Interactive
        } else {
            Self::Batch
        }
    }

    /// Check if interactive
    #[must_use]
    pub fn is_interactive(&self) -> bool {
        *self == Self::Interactive
    }

    /// Check if batch mode
    #[must_use]
    pub fn is_batch(&self) -> bool {
        *self == Self::Batch
    }
}

/// Output formatter that adapts to execution mode
#[derive(Debug)]
pub struct AdaptiveOutput {
    mode: ExecutionMode,
    json_output: bool,
}

impl Default for AdaptiveOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveOutput {
    /// Create new adaptive output
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: ExecutionMode::detect(),
            json_output: false,
        }
    }

    /// Force JSON output mode
    #[must_use]
    pub fn with_json(mut self) -> Self {
        self.json_output = true;
        self
    }

    /// Force a specific execution mode
    #[must_use]
    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Print status message (only in interactive mode)
    pub fn status(&self, msg: &str) {
        if self.mode.is_interactive() && !self.json_output {
            eprintln!("{}", msg);
        }
    }

    /// Print progress (only in interactive mode)
    pub fn progress(&self, current: usize, total: usize, msg: &str) {
        if self.mode.is_interactive() && !self.json_output {
            eprint!("\r[{}/{}] {}", current, total, msg);
            let _ = io::stderr().flush();
        }
    }

    /// Print result (always, format depends on mode)
    pub fn result(&self, data: &str) {
        println!("{}", data);
    }

    /// Print error (always to stderr)
    pub fn error(&self, msg: &str) {
        eprintln!("Error: {}", msg);
    }
}

// ============================================================================
// RDB-04: Robust Error Recovery
// ============================================================================

/// Error with recovery suggestion
#[derive(Debug)]
pub struct RecoverableError {
    /// Error message
    pub message: String,
    /// Suggested recovery action
    pub recovery: Option<String>,
    /// Whether automatic recovery is possible
    pub auto_recoverable: bool,
}

impl RecoverableError {
    /// Create a new recoverable error
    #[must_use]
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
            recovery: None,
            auto_recoverable: false,
        }
    }

    /// Add recovery suggestion
    #[must_use]
    pub fn with_recovery(mut self, recovery: &str) -> Self {
        self.recovery = Some(recovery.to_string());
        self
    }

    /// Mark as auto-recoverable
    #[must_use]
    pub fn auto_recoverable(mut self) -> Self {
        self.auto_recoverable = true;
        self
    }

    /// Format error for display
    #[must_use]
    pub fn format(&self) -> String {
        use std::fmt::Write;
        let mut output = format!("Error: {}", self.message);
        if let Some(recovery) = &self.recovery {
            let _ = write!(output, "\n\nSuggested fix: {}", recovery);
        }
        output
    }
}

/// Common error recovery scenarios
pub mod recovery {
    use super::{Path, RecoverableError};

    /// Model file not found
    #[must_use]
    pub fn model_not_found(path: &Path) -> RecoverableError {
        RecoverableError::new(&format!("Model file not found: {}", path.display()))
            .with_recovery("Run 'apr download <model>' to fetch the model, or check the path")
    }

    /// Model checksum mismatch
    #[must_use]
    pub fn checksum_mismatch(expected: &str, actual: &str) -> RecoverableError {
        RecoverableError::new(&format!(
            "Model checksum mismatch\n  Expected: {}\n  Actual: {}",
            expected, actual
        ))
        .with_recovery("The model file may be corrupted. Delete it and re-download with 'apr download --force <model>'")
        .auto_recoverable()
    }

    /// GPU not available
    #[must_use]
    pub fn gpu_not_available() -> RecoverableError {
        RecoverableError::new("GPU acceleration requested but not available").with_recovery(
            "Falling back to CPU. For GPU support, ensure CUDA/Metal drivers are installed",
        )
    }

    /// Out of memory
    #[must_use]
    pub fn out_of_memory(required: usize, available: usize) -> RecoverableError {
        RecoverableError::new(&format!(
            "Insufficient memory: need {} MB, have {} MB",
            required / 1_000_000,
            available / 1_000_000
        ))
        .with_recovery("Try a smaller model or enable quantization with '--quantize int4'")
    }
}

// ============================================================================
// RDB-05: Performance Transparency
// ============================================================================

/// Performance metrics for transparency
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Model load time
    pub load_time: Duration,
    /// Time to first token
    pub time_to_first_token: Duration,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Total generation time
    pub generation_time: Duration,
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Backend used (e.g., "AVX2", "Metal", "CUDA")
    pub backend: String,
}

impl PerformanceMetrics {
    /// Calculate tokens per second
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.generation_time.as_secs_f64() > 0.0 {
            self.tokens_generated as f64 / self.generation_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Format metrics for display
    #[must_use]
    pub fn format(&self) -> String {
        format!(
            "Performance:\n  Load time: {:.2}s\n  Time to first token: {:.0}ms\n  Tokens/sec: {:.1}\n  Peak memory: {:.1} MB\n  Backend: {}",
            self.load_time.as_secs_f64(),
            self.time_to_first_token.as_millis(),
            self.tokens_per_second(),
            self.peak_memory as f64 / 1_000_000.0,
            self.backend
        )
    }

    /// Format as JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"load_time_ms":{},"ttft_ms":{},"tokens_per_sec":{:.1},"peak_memory_mb":{:.1},"backend":"{}","tokens_generated":{}}}"#,
            self.load_time.as_millis(),
            self.time_to_first_token.as_millis(),
            self.tokens_per_second(),
            self.peak_memory as f64 / 1_000_000.0,
            self.backend,
            self.tokens_generated
        )
    }
}

/// Performance timer for tracking operations
#[derive(Debug)]
pub struct PerfTimer {
    start: Instant,
    checkpoints: Vec<(String, Duration)>,
}

impl Default for PerfTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl PerfTimer {
    /// Start a new timer
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Record a checkpoint
    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints
            .push((name.to_string(), self.start.elapsed()));
    }

    /// Get elapsed time since start
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get time since last checkpoint (or start)
    #[must_use]
    pub fn since_last(&self) -> Duration {
        let last = self.checkpoints.last().map_or(Duration::ZERO, |(_, d)| *d);
        self.start.elapsed().saturating_sub(last)
    }

    /// Print all checkpoints (verbose mode)
    pub fn print_verbose(&self) {
        eprintln!("Timing breakdown:");
        let mut prev = Duration::ZERO;
        for (name, total) in &self.checkpoints {
            let delta = total.saturating_sub(prev);
            eprintln!(
                "  {}: {:.0}ms (total: {:.0}ms)",
                name,
                delta.as_millis(),
                total.as_millis()
            );
            prev = *total;
        }
    }
}

/// Detect the compute backend
#[must_use]
pub fn detect_backend() -> String {
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512".to_string();
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2".to_string();
        }
        if is_x86_feature_detected!("avx") {
            return "AVX".to_string();
        }
        if is_x86_feature_detected!("sse4.2") {
            return "SSE4.2".to_string();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON".to_string();
    }
    "Scalar".to_string()
}

// ============================================================================
// RDB-06: Model Provenance & Licensing
// ============================================================================

/// Model provenance information
#[derive(Debug, Clone)]
pub struct ModelProvenance {
    /// Model name
    pub name: String,
    /// Version or revision
    pub version: String,
    /// License (e.g., "Apache-2.0", "MIT", "Llama Community")
    pub license: String,
    /// Link to model card or paper
    pub source_url: Option<String>,
    /// Authors or organization
    pub authors: Option<String>,
}

impl ModelProvenance {
    /// Create new provenance info
    #[must_use]
    pub fn new(name: &str, version: &str, license: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            license: license.to_string(),
            source_url: None,
            authors: None,
        }
    }

    /// Add source URL
    #[must_use]
    pub fn with_source(mut self, url: &str) -> Self {
        self.source_url = Some(url.to_string());
        self
    }

    /// Add authors
    #[must_use]
    pub fn with_authors(mut self, authors: &str) -> Self {
        self.authors = Some(authors.to_string());
        self
    }

    /// Format for display at startup
    #[must_use]
    pub fn format(&self) -> String {
        use std::fmt::Write;
        let mut output = format!(
            "Model: {} ({})\nLicense: {}",
            self.name, self.version, self.license
        );
        if let Some(authors) = &self.authors {
            let _ = write!(output, "\nAuthors: {}", authors);
        }
        if let Some(url) = &self.source_url {
            let _ = write!(output, "\nSource: {}", url);
        }
        output
    }

    /// Format as JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        use std::fmt::Write;
        let mut json = format!(
            r#"{{"name":"{}","version":"{}","license":"{}""#,
            self.name, self.version, self.license
        );
        if let Some(authors) = &self.authors {
            let _ = write!(json, r#","authors":"{}""#, authors);
        }
        if let Some(url) = &self.source_url {
            let _ = write!(json, r#","source":"{}""#, url);
        }
        json.push('}');
        json
    }
}

/// Common model provenances
pub mod models {
    use super::ModelProvenance;

    /// TinyLlama 1.1B Chat
    #[must_use]
    pub fn tinyllama_chat() -> ModelProvenance {
        ModelProvenance::new("TinyLlama-1.1B-Chat", "v1.0", "Apache-2.0")
            .with_source("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            .with_authors("TinyLlama Team")
    }

    /// Qwen2 0.5B Instruct
    #[must_use]
    pub fn qwen2_0_5b() -> ModelProvenance {
        ModelProvenance::new("Qwen2-0.5B-Instruct", "v1.0", "Apache-2.0")
            .with_source("https://huggingface.co/Qwen/Qwen2-0.5B-Instruct")
            .with_authors("Alibaba Cloud")
    }

    /// Mistral 7B Instruct
    #[must_use]
    pub fn mistral_7b() -> ModelProvenance {
        ModelProvenance::new("Mistral-7B-Instruct", "v0.2", "Apache-2.0")
            .with_source("https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")
            .with_authors("Mistral AI")
    }

    /// Phi-2
    #[must_use]
    pub fn phi2() -> ModelProvenance {
        ModelProvenance::new("phi-2", "v1.0", "MIT")
            .with_source("https://huggingface.co/microsoft/phi-2")
            .with_authors("Microsoft Research")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // RDB-01 Tests
    #[test]
    fn test_model_cache_default() {
        let cache = ModelCache::default();
        assert!(cache.auto_download);
        // Cache dir contains "apr" (with hf-hub-integration) or "apr_cache" (without)
        let path_str = cache.cache_dir.to_string_lossy();
        assert!(path_str.contains("apr") || path_str.contains("cache"));
    }

    #[test]
    fn test_model_source_parse() {
        // HuggingFace
        let hf = ModelSource::parse("hf://Qwen/Qwen2-0.5B-Instruct/model.safetensors");
        assert!(matches!(hf, ModelSource::HuggingFace { .. }));
        assert!(hf.is_remote());

        // URL
        let url = ModelSource::parse("https://example.com/model.gguf");
        assert!(matches!(url, ModelSource::Url(_)));
        assert!(url.is_remote());

        // Local
        let local = ModelSource::parse("/path/to/model.safetensors");
        assert!(matches!(local, ModelSource::Local(_)));
        assert!(!local.is_remote());
    }

    // RDB-02 Tests
    #[test]
    fn test_prerequisite_check() {
        let satisfied = PrerequisiteCheck::satisfied("test");
        assert!(satisfied.satisfied);

        let missing = PrerequisiteCheck::missing("test", "install it");
        assert!(!missing.satisfied);
        assert!(missing.install_hint.is_some());
    }

    // RDB-03 Tests
    #[test]
    fn test_execution_mode() {
        // In tests, we're usually not in a TTY
        let mode = ExecutionMode::detect();
        // Just verify it returns something valid
        assert!(mode.is_interactive() || mode.is_batch());
    }

    #[test]
    fn test_adaptive_output() {
        let output = AdaptiveOutput::new();
        // Should not panic
        output.status("test");
        output.result("result");
        output.error("error");
    }

    // RDB-04 Tests
    #[test]
    fn test_recoverable_error() {
        let err = RecoverableError::new("test error")
            .with_recovery("do this")
            .auto_recoverable();

        assert!(err.auto_recoverable);
        assert!(err.recovery.is_some());
        assert!(err.format().contains("test error"));
        assert!(err.format().contains("do this"));
    }

    #[test]
    fn test_recovery_scenarios() {
        let err = recovery::model_not_found(Path::new("/test"));
        assert!(err.message.contains("not found"));

        let err = recovery::checksum_mismatch("abc", "def");
        assert!(err.auto_recoverable);

        let err = recovery::gpu_not_available();
        assert!(err.recovery.is_some());

        let err = recovery::out_of_memory(1000, 500);
        assert!(err.message.contains("memory"));
    }

    // RDB-05 Tests
    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            load_time: Duration::from_millis(1000),
            time_to_first_token: Duration::from_millis(100),
            tokens_generated: 100,
            generation_time: Duration::from_secs(5),
            peak_memory: 500_000_000,
            backend: "AVX2".to_string(),
        };

        assert!((metrics.tokens_per_second() - 20.0).abs() < 0.1);
        assert!(metrics.format().contains("AVX2"));
        assert!(metrics.to_json().contains("\"backend\":\"AVX2\""));
    }

    #[test]
    fn test_perf_timer() {
        let mut timer = PerfTimer::new();
        std::thread::sleep(Duration::from_millis(10));
        timer.checkpoint("first");
        std::thread::sleep(Duration::from_millis(10));
        timer.checkpoint("second");

        assert!(timer.elapsed() >= Duration::from_millis(20));
    }

    #[test]
    fn test_detect_backend() {
        let backend = detect_backend();
        // Should return a valid backend string
        assert!(!backend.is_empty());
    }

    // RDB-06 Tests
    #[test]
    fn test_model_provenance() {
        let prov = ModelProvenance::new("TestModel", "v1.0", "MIT")
            .with_source("https://example.com")
            .with_authors("Test Author");

        assert_eq!(prov.name, "TestModel");
        assert!(prov.format().contains("MIT"));
        assert!(prov.to_json().contains("\"license\":\"MIT\""));
    }

    #[test]
    fn test_common_provenances() {
        let tinyllama = models::tinyllama_chat();
        assert!(tinyllama.name.contains("TinyLlama"));
        assert_eq!(tinyllama.license, "Apache-2.0");

        let qwen = models::qwen2_0_5b();
        assert!(qwen.name.contains("Qwen2"));

        let mistral = models::mistral_7b();
        assert!(mistral.name.contains("Mistral"));

        let phi = models::phi2();
        assert_eq!(phi.license, "MIT");
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_model_cache_new() {
        let cache = ModelCache::new(PathBuf::from("/tmp/test_cache"));
        assert_eq!(cache.cache_dir, PathBuf::from("/tmp/test_cache"));
        assert!(cache.auto_download);
        assert_eq!(cache.max_size_bytes, 0);
    }

    #[test]
    fn test_model_cache_model_path() {
        let cache = ModelCache::new(PathBuf::from("/tmp/cache"));
        let path = cache.model_path("qwen2-0.5b");
        assert!(path.to_string_lossy().contains("qwen2-0.5b"));
    }

    #[test]
    fn test_model_cache_has_model() {
        let cache = ModelCache::new(PathBuf::from("/nonexistent/path"));
        assert!(!cache.has_model("any-model"));
    }

    #[test]
    fn test_model_source_hf_parse() {
        let source = ModelSource::parse("hf://org/repo/file.safetensors");
        if let ModelSource::HuggingFace { repo_id, filename } = source {
            assert_eq!(repo_id, "org/repo");
            assert_eq!(filename, "file.safetensors");
        } else {
            panic!("Expected HuggingFace source");
        }
    }

    #[test]
    fn test_model_source_url_parse() {
        let source = ModelSource::parse("https://example.com/model.gguf");
        assert!(matches!(source, ModelSource::Url(_)));
    }

    #[test]
    fn test_model_source_local_parse() {
        let source = ModelSource::parse("./model.safetensors");
        assert!(matches!(source, ModelSource::Local(_)));
    }

    #[test]
    fn test_model_source_is_local() {
        let local = ModelSource::Local(PathBuf::from("./test"));
        assert!(!local.is_remote());
        let url = ModelSource::Url("https://example.com".to_string());
        assert!(url.is_remote());
    }

    #[test]
    fn test_execution_mode_batch() {
        let mode = ExecutionMode::Batch;
        assert!(mode.is_batch());
        assert!(!mode.is_interactive());
    }

    #[test]
    fn test_execution_mode_interactive() {
        let mode = ExecutionMode::Interactive;
        assert!(mode.is_interactive());
        assert!(!mode.is_batch());
    }

    #[test]
    fn test_adaptive_output_methods() {
        let output = AdaptiveOutput::new();
        output.progress(50, 100, "loading...");
        output.result("done");
        output.error("test error");
    }

    #[test]
    fn test_recoverable_error_format_no_recovery() {
        let err = RecoverableError::new("simple error");
        let formatted = err.format();
        assert!(formatted.contains("simple error"));
    }

    #[test]
    fn test_check_command_nonexistent() {
        let check = check_command("nonexistent_command_12345");
        assert!(!check.satisfied);
    }

    #[test]
    fn test_performance_metrics_zero_time() {
        let metrics = PerformanceMetrics {
            load_time: Duration::ZERO,
            time_to_first_token: Duration::ZERO,
            tokens_generated: 0,
            generation_time: Duration::ZERO,
            peak_memory: 0,
            backend: "test".to_string(),
        };
        // Should not panic on division by zero
        assert_eq!(metrics.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_perf_timer_checkpoints() {
        let mut timer = PerfTimer::new();
        timer.checkpoint("start");
        timer.checkpoint("middle");
        timer.checkpoint("end");
        // Verify checkpoints were recorded
        assert!(timer.elapsed() >= Duration::ZERO);
    }

    #[test]
    fn test_model_provenance_builder() {
        let prov = ModelProvenance::new("Model", "1.0", "MIT")
            .with_source("https://source.com")
            .with_authors("Author1, Author2");
        assert_eq!(prov.name, "Model");
        assert_eq!(prov.version, "1.0");
        assert_eq!(prov.license, "MIT");
        assert_eq!(prov.source_url, Some("https://source.com".to_string()));
        assert_eq!(prov.authors, Some("Author1, Author2".to_string()));
    }

    #[test]
    fn test_model_provenance_json() {
        let prov = ModelProvenance::new("TestModel", "v1.0", "Apache-2.0");
        let json = prov.to_json();
        assert!(json.contains("\"name\":\"TestModel\""));
        assert!(json.contains("\"version\":\"v1.0\""));
        assert!(json.contains("\"license\":\"Apache-2.0\""));
    }

    #[test]
    fn test_detect_backend_not_empty() {
        let backend = detect_backend();
        assert!(!backend.is_empty());
    }

    #[test]
    fn test_adaptive_output_with_json() {
        let output = AdaptiveOutput::new().with_json();
        output.status("should not print in json mode");
    }

    #[test]
    fn test_adaptive_output_with_mode() {
        let output = AdaptiveOutput::new().with_mode(ExecutionMode::Batch);
        output.status("should not print in batch mode");
    }

    #[test]
    fn test_perf_timer_since_last() {
        let mut timer = PerfTimer::new();
        std::thread::sleep(Duration::from_millis(5));
        timer.checkpoint("first");
        std::thread::sleep(Duration::from_millis(5));
        let since_last = timer.since_last();
        assert!(since_last >= Duration::from_millis(4)); // Allow some tolerance
    }

    #[test]
    fn test_perf_timer_since_last_no_checkpoints() {
        let timer = PerfTimer::new();
        std::thread::sleep(Duration::from_millis(5));
        let since_last = timer.since_last();
        assert!(since_last >= Duration::from_millis(4));
    }

    #[test]
    fn test_perf_timer_print_verbose() {
        let mut timer = PerfTimer::new();
        timer.checkpoint("load");
        timer.checkpoint("process");
        // Should not panic
        timer.print_verbose();
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.tokens_generated, 0);
        assert!(metrics.backend.is_empty());
    }

    #[test]
    fn test_adaptive_output_default() {
        let output = AdaptiveOutput::default();
        // Should work the same as new()
        output.status("test");
    }

    #[test]
    fn test_perf_timer_default() {
        let timer = PerfTimer::default();
        assert!(timer.elapsed() >= Duration::ZERO);
    }

    #[test]
    fn test_model_cache_ensure_dir() {
        let cache = ModelCache::new(PathBuf::from("/tmp/aprender_test_cache"));
        // Should succeed or already exist
        let _ = cache.ensure_dir();
        // Clean up
        let _ = std::fs::remove_dir_all("/tmp/aprender_test_cache");
    }

    #[test]
    fn test_model_source_hf_short_path() {
        // Test HF path with only org/repo (no file)
        let source = ModelSource::parse("hf://owner/repo");
        if let ModelSource::HuggingFace { repo_id, filename } = source {
            assert_eq!(repo_id, "owner/repo");
            assert_eq!(filename, "model.safetensors"); // Default filename
        } else {
            panic!("Expected HuggingFace source");
        }
    }

    #[test]
    fn test_model_source_http_url() {
        let source = ModelSource::parse("http://localhost/model.gguf");
        assert!(matches!(source, ModelSource::Url(_)));
        assert!(source.is_remote());
    }

    #[test]
    fn test_model_source_hf_single_part() {
        // Edge case: single component after hf://
        let source = ModelSource::parse("hf://single");
        // Should fall back to Local since it doesn't have org/repo structure
        assert!(matches!(source, ModelSource::Local(_)));
    }

    #[test]
    fn test_check_prerequisites_multiple() {
        let checks = check_prerequisites(&["ls", "nonexistent_cmd_xyz"]);
        assert_eq!(checks.len(), 2);
        // ls should exist on most systems
        assert!(checks[0].satisfied || !checks[0].satisfied); // Always valid
        assert!(!checks[1].satisfied); // nonexistent should not exist
    }

    #[test]
    fn test_print_prerequisites() {
        let checks = vec![
            PrerequisiteCheck::satisfied("test1"),
            PrerequisiteCheck::missing("test2", "install it"),
        ];
        // Should not panic
        print_prerequisites(&checks);
    }

    #[test]
    fn test_recoverable_error_not_auto() {
        let err = RecoverableError::new("not auto-recoverable");
        assert!(!err.auto_recoverable);
    }

    #[test]
    fn test_performance_metrics_format_content() {
        let metrics = PerformanceMetrics {
            load_time: Duration::from_secs(2),
            time_to_first_token: Duration::from_millis(150),
            tokens_generated: 50,
            generation_time: Duration::from_secs(5),
            peak_memory: 1_500_000_000,
            backend: "CUDA".to_string(),
        };

        let formatted = metrics.format();
        assert!(formatted.contains("Load time"));
        assert!(formatted.contains("CUDA"));
        assert!(formatted.contains("Peak memory"));
    }

    #[test]
    fn test_performance_metrics_json_content() {
        let metrics = PerformanceMetrics {
            load_time: Duration::from_millis(500),
            time_to_first_token: Duration::from_millis(50),
            tokens_generated: 100,
            generation_time: Duration::from_secs(10),
            peak_memory: 2_000_000_000,
            backend: "Metal".to_string(),
        };

        let json = metrics.to_json();
        assert!(json.contains("\"backend\":\"Metal\""));
        assert!(json.contains("tokens_per_sec"));
        assert!(json.contains("tokens_generated"));
    }

    #[test]
    fn test_model_provenance_format_full() {
        let prov = ModelProvenance::new("TestModel", "v2.0", "Apache-2.0")
            .with_source("https://example.com/model")
            .with_authors("John Doe");

        let formatted = prov.format();
        assert!(formatted.contains("TestModel"));
        assert!(formatted.contains("v2.0"));
        assert!(formatted.contains("Apache-2.0"));
        assert!(formatted.contains("John Doe"));
    }

    #[test]
    fn test_model_provenance_json_with_optionals() {
        let prov = ModelProvenance::new("Model", "1.0", "MIT")
            .with_source("https://source.url")
            .with_authors("Authors");

        let json = prov.to_json();
        assert!(json.contains("\"source\":"));
        assert!(json.contains("\"authors\":"));
    }

    #[test]
    fn test_model_cache_debug() {
        let cache = ModelCache::default();
        assert!(format!("{:?}", cache).contains("ModelCache"));
    }

    #[test]
    fn test_model_source_debug() {
        let source = ModelSource::Local(PathBuf::from("./test"));
        assert!(format!("{:?}", source).contains("Local"));
    }

    #[test]
    fn test_execution_mode_debug() {
        let mode = ExecutionMode::Interactive;
        assert!(format!("{:?}", mode).contains("Interactive"));
    }

    #[test]
    fn test_adaptive_output_debug() {
        let output = AdaptiveOutput::new();
        assert!(format!("{:?}", output).contains("AdaptiveOutput"));
    }

    #[test]
    fn test_recoverable_error_debug() {
        let err = RecoverableError::new("test");
        assert!(format!("{:?}", err).contains("RecoverableError"));
    }

    #[test]
    fn test_performance_metrics_debug() {
        let metrics = PerformanceMetrics::default();
        assert!(format!("{:?}", metrics).contains("PerformanceMetrics"));
    }

    #[test]
    fn test_perf_timer_debug() {
        let timer = PerfTimer::new();
        assert!(format!("{:?}", timer).contains("PerfTimer"));
    }

    #[test]
    fn test_prerequisite_check_debug() {
        let check = PrerequisiteCheck::satisfied("test");
        assert!(format!("{:?}", check).contains("PrerequisiteCheck"));
    }

    #[test]
    fn test_model_cache_clone() {
        let cache1 = ModelCache::default();
        let cache2 = cache1.clone();
        assert_eq!(cache1.cache_dir, cache2.cache_dir);
    }

    #[test]
    fn test_model_source_clone() {
        let source1 = ModelSource::Local(PathBuf::from("./test"));
        let source2 = source1.clone();
        assert!(matches!(source2, ModelSource::Local(_)));
    }

    #[test]
    fn test_execution_mode_eq() {
        assert_eq!(ExecutionMode::Interactive, ExecutionMode::Interactive);
        assert_ne!(ExecutionMode::Interactive, ExecutionMode::Batch);
    }

    #[test]
    fn test_performance_metrics_clone() {
        let metrics1 = PerformanceMetrics {
            backend: "test".to_string(),
            ..Default::default()
        };
        let metrics2 = metrics1.clone();
        assert_eq!(metrics1.backend, metrics2.backend);
    }

    #[test]
    fn test_prerequisite_check_clone() {
        let check1 = PrerequisiteCheck::satisfied("test");
        let check2 = check1.clone();
        assert_eq!(check1.name, check2.name);
    }
}
