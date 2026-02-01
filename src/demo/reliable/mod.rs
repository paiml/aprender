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
mod tests;
