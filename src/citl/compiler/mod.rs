//! Compiler interface abstraction.
//!
//! Per the CITL specification, the compiler interface must support
//! diverse compiler backends uniformly, enabling cross-language
//! learning transfer.

use super::diagnostic::{
    CompilerDiagnostic, CompilerSuggestion, DiagnosticSeverity, SourceSpan, TypeInfo,
};
use super::error::{CITLError, CITLResult};
use super::ErrorCode;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Universal compiler interface supporting any language toolchain.
///
/// Implements NASA NPR 7150.2 fault tolerance via timeout and fallback mechanisms.
///
/// # Example
///
/// ```ignore
/// use aprender::citl::{CompilerInterface, RustCompiler, CompilationResult};
///
/// let compiler = RustCompiler::new()
///     .edition(RustEdition::E2021)
///     .timeout(Duration::from_secs(30));
///
/// let result = compiler.compile(code, &CompileOptions::default())?;
/// match result {
///     CompilationResult::Success { warnings, .. } => { /* learn from success */ }
///     CompilationResult::Failure { errors, .. } => { /* learn from errors */ }
/// }
/// ```
pub trait CompilerInterface: Send + Sync {
    /// Compile source code and return structured feedback.
    ///
    /// # Arguments
    /// * `source` - Source code to compile
    /// * `options` - Compilation options
    ///
    /// # Errors
    ///
    /// Returns `CITLError::CompilerTimeout` if compilation exceeds configured timeout.
    /// Returns `CITLError::CompilerNotFound` if compiler binary not found.
    fn compile(&self, source: &str, options: &CompileOptions) -> CITLResult<CompilationResult>;

    /// Parse a raw compiler diagnostic into structured form.
    ///
    /// Per Mesbah et al. (2019), structured parsing enables
    /// 50% repair accuracy vs 23% with raw text matching.
    fn parse_diagnostic(&self, raw: &str) -> Option<CompilerDiagnostic>;

    /// Return compiler version for reproducibility.
    fn version(&self) -> CITLResult<CompilerVersion>;

    /// Compiler name for identification.
    fn name(&self) -> &'static str;

    /// Check if the compiler is available.
    fn is_available(&self) -> bool;
}

/// Compilation options.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Optimization level
    pub opt_level: OptLevel,
    /// Additional compiler flags
    pub extra_flags: Vec<String>,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Working directory
    pub working_dir: Option<PathBuf>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::Debug,
            extra_flags: Vec::new(),
            env: HashMap::new(),
            working_dir: None,
        }
    }
}

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// Debug build (fast compilation)
    Debug,
    /// Release build (optimized)
    Release,
}

/// Structured compilation result.
///
/// Following Wang et al. (2022), we separate warnings from errors
/// to enable compilability reinforcement learning.
#[derive(Debug, Clone)]
pub enum CompilationResult {
    /// Compilation succeeded (may include warnings)
    Success {
        /// Compiled artifact (if any)
        artifact: Option<CompiledArtifact>,
        /// Non-fatal diagnostics
        warnings: Vec<CompilerDiagnostic>,
        /// Compilation metrics
        metrics: CompilationMetrics,
    },
    /// Compilation failed with errors
    Failure {
        /// Fatal diagnostics preventing compilation
        errors: Vec<CompilerDiagnostic>,
        /// Non-fatal diagnostics
        warnings: Vec<CompilerDiagnostic>,
        /// Raw compiler output for debugging
        raw_output: String,
    },
}

impl CompilationResult {
    /// Check if compilation was successful.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, CompilationResult::Success { .. })
    }

    /// Get error count.
    #[must_use]
    pub fn error_count(&self) -> usize {
        match self {
            CompilationResult::Success { .. } => 0,
            CompilationResult::Failure { errors, .. } => errors.len(),
        }
    }

    /// Get warning count.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings().len()
    }

    /// Get all errors (empty for success).
    #[must_use]
    pub fn errors(&self) -> &[CompilerDiagnostic] {
        match self {
            CompilationResult::Success { .. } => &[],
            CompilationResult::Failure { errors, .. } => errors,
        }
    }

    /// Get all warnings.
    #[must_use]
    pub fn warnings(&self) -> &[CompilerDiagnostic] {
        match self {
            CompilationResult::Success { warnings, .. }
            | CompilationResult::Failure { warnings, .. } => warnings,
        }
    }
}

/// Compiled artifact information.
#[derive(Debug, Clone)]
pub struct CompiledArtifact {
    /// Artifact type
    pub artifact_type: ArtifactType,
    /// Path to artifact (if written to disk)
    pub path: Option<PathBuf>,
    /// Size in bytes
    pub size: usize,
}

/// Type of compiled artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactType {
    /// Static library
    StaticLib,
    /// Dynamic library
    DynamicLib,
    /// Executable binary
    Binary,
    /// Object file
    Object,
    /// WASM module
    Wasm,
}

/// Compilation metrics for analysis.
#[derive(Debug, Clone)]
pub struct CompilationMetrics {
    /// Total compilation time
    pub duration: Duration,
    /// Peak memory usage (if available)
    pub memory_bytes: Option<usize>,
    /// Number of compilation units
    pub units: usize,
}

impl Default for CompilationMetrics {
    fn default() -> Self {
        Self {
            duration: Duration::ZERO,
            memory_bytes: None,
            units: 1,
        }
    }
}

/// Compiler version information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerVersion {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Full version string
    pub full: String,
    /// Commit hash (if available)
    pub commit: Option<String>,
}

impl CompilerVersion {
    /// Parse a version string like "1.75.0".
    #[must_use]
    pub fn parse(version_str: &str) -> Option<Self> {
        let parts: Vec<&str> = version_str.trim().split('.').collect();
        if parts.len() >= 3 {
            let major = parts[0].parse().ok()?;
            let minor = parts[1].parse().ok()?;
            let patch_str = parts[2].split('-').next()?;
            let patch = patch_str.parse().ok()?;
            Some(Self {
                major,
                minor,
                patch,
                full: version_str.to_string(),
                commit: None,
            })
        } else {
            None
        }
    }
}

impl std::fmt::Display for CompilerVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Compilation mode for Rust compiler.
///
/// Per Appendix C of the CITL spec, standalone rustc cannot resolve
/// crate dependencies, so we need Cargo mode for real-world code.
#[derive(Debug, Clone, Default)]
pub enum CompilationMode {
    /// Fast standalone rustc (no external crates)
    #[default]
    Standalone,
    /// Full cargo build (resolves dependencies)
    Cargo {
        /// Path to Cargo.toml
        manifest_path: PathBuf,
    },
    /// Cargo check (faster than build, still resolves deps)
    CargoCheck {
        /// Path to Cargo.toml
        manifest_path: PathBuf,
    },
}

/// Rust edition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RustEdition {
    /// Rust 2015
    E2015,
    /// Rust 2018
    E2018,
    /// Rust 2021
    #[default]
    E2021,
    /// Rust 2024
    E2024,
}

impl RustEdition {
    /// Get the edition string for rustc.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            RustEdition::E2015 => "2015",
            RustEdition::E2018 => "2018",
            RustEdition::E2021 => "2021",
            RustEdition::E2024 => "2024",
        }
    }
}

/// Rust compiler interface.
///
/// Supports both direct rustc invocation and cargo-based compilation.
///
/// # NASA Fault Tolerance (NPR 7150.2)
/// - Configurable timeout with graceful process termination
/// - Fallback to cached diagnostics on compiler unavailability
/// - Redundant validation via multiple rustc versions
#[derive(Debug, Clone)]
pub struct RustCompiler {
    /// Path to rustc binary
    rustc_path: PathBuf,
    /// Path to cargo binary (reserved for Cargo mode)
    #[allow(dead_code)]
    cargo_path: PathBuf,
    /// Rust edition
    edition: RustEdition,
    /// Target triple
    target: Option<String>,
    /// Compilation timeout
    timeout: Duration,
    /// Compilation mode
    mode: CompilationMode,
    /// Additional flags
    extra_flags: Vec<String>,
}

include!("cargo_project.rs");
include!("json.rs");
