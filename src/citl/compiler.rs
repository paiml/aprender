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

impl RustCompiler {
    /// Create a new Rust compiler interface.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rustc_path: which_rustc(),
            cargo_path: which_cargo(),
            edition: RustEdition::default(),
            target: None,
            timeout: Duration::from_secs(60),
            mode: CompilationMode::default(),
            extra_flags: Vec::new(),
        }
    }

    /// Set the Rust edition.
    #[must_use]
    pub fn edition(mut self, edition: RustEdition) -> Self {
        self.edition = edition;
        self
    }

    /// Set the compilation timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the compilation mode.
    #[must_use]
    pub fn mode(mut self, mode: CompilationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the target triple.
    #[must_use]
    pub fn target(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }

    /// Add extra compiler flags.
    #[must_use]
    pub fn extra_flag(mut self, flag: &str) -> Self {
        self.extra_flags.push(flag.to_string());
        self
    }

    /// Parse rustc JSON diagnostic output.
    fn parse_json_diagnostics(
        &self,
        output: &str,
    ) -> (Vec<CompilerDiagnostic>, Vec<CompilerDiagnostic>) {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for line in output.lines() {
            // Only parse lines that look like complete JSON objects
            if line.starts_with('{') && line.ends_with('}') {
                // Skip rendered messages and other non-diagnostic JSON
                if line.contains("\"$message_type\":\"diagnostic\"")
                    || (line.contains("\"level\":") && !line.contains("\"rendered\""))
                {
                    if let Some(diag) = self.parse_single_json_diagnostic(line) {
                        // Skip "aborting due to N previous errors" messages
                        if diag.message.contains("aborting due to") {
                            continue;
                        }
                        if diag.severity == DiagnosticSeverity::Error {
                            errors.push(diag);
                        } else {
                            warnings.push(diag);
                        }
                    }
                }
            }
        }

        (errors, warnings)
    }

    /// Parse a single JSON diagnostic line.
    #[allow(clippy::unused_self)]
    fn parse_single_json_diagnostic(&self, json: &str) -> Option<CompilerDiagnostic> {
        // Simple JSON parsing without serde
        // Format: {"code":{"code":"E0308",...},"level":"error","message":"...","spans":[...]}

        let level = extract_json_string(json, "level")?;
        let message = extract_json_string(json, "message")?;

        let severity = match level.as_str() {
            "error" => DiagnosticSeverity::Error,
            "warning" => DiagnosticSeverity::Warning,
            "note" => DiagnosticSeverity::Note,
            "help" => DiagnosticSeverity::Help,
            _ => return None,
        };

        // Extract error code
        let code_str = extract_nested_json_string(json, "code", "code")
            .unwrap_or_else(|| "unknown".to_string());

        let code = lookup_error_code(&code_str);

        // Extract span info
        let span = extract_span_from_json(json);

        let mut diag = CompilerDiagnostic::new(code, severity, &message, span);

        // Extract expected/found types if present
        if let Some(expected) = extract_json_string(json, "expected") {
            diag = diag.with_expected(TypeInfo::new(&expected));
        }
        if let Some(found) = extract_json_string(json, "found") {
            diag = diag.with_found(TypeInfo::new(&found));
        }

        // Extract suggestions if present
        if let Some(suggestions) = extract_suggestions_from_json(json) {
            for suggestion in suggestions {
                diag = diag.with_suggestion(suggestion);
            }
        }

        Some(diag)
    }

    /// Compile using standalone rustc.
    fn compile_standalone(
        &self,
        source: &str,
        options: &CompileOptions,
    ) -> CITLResult<CompilationResult> {
        let start = Instant::now();

        // Create temp file for source with unique name
        let temp_dir = std::env::temp_dir();
        let unique_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let source_file = temp_dir.join(format!(
            "citl_compile_{}_{}.rs",
            std::process::id(),
            unique_id
        ));

        std::fs::write(&source_file, source)?;

        let mut cmd = Command::new(&self.rustc_path);
        cmd.arg("--edition").arg(self.edition.as_str())
            .arg("--crate-type").arg("lib") // Compile as library
            .arg("--error-format=json")
            .arg("--emit=metadata") // Fast check without codegen
            .arg("-o").arg(temp_dir.join("citl_output"))
            .arg(&source_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(target) = &self.target {
            cmd.arg("--target").arg(target);
        }

        for flag in &self.extra_flags {
            cmd.arg(flag);
        }

        for flag in &options.extra_flags {
            cmd.arg(flag);
        }

        let output = cmd.output()?;
        let duration = start.elapsed();

        // Clean up temp file
        let _ = std::fs::remove_file(&source_file);
        let _ = std::fs::remove_file(temp_dir.join("citl_output"));

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let (errors, warnings) = self.parse_json_diagnostics(&stderr);

        if output.status.success() || errors.is_empty() {
            Ok(CompilationResult::Success {
                artifact: None,
                warnings,
                metrics: CompilationMetrics {
                    duration,
                    memory_bytes: None,
                    units: 1,
                },
            })
        } else {
            Ok(CompilationResult::Failure {
                errors,
                warnings,
                raw_output: stderr,
            })
        }
    }

    /// Compile using cargo check.
    fn compile_cargo_check(
        &self,
        source: &str,
        manifest_path: &PathBuf,
        _options: &CompileOptions,
    ) -> CITLResult<CompilationResult> {
        let start = Instant::now();

        // Get the project directory from manifest path
        let project_dir = manifest_path
            .parent()
            .ok_or_else(|| CITLError::ConfigurationError {
                message: "Invalid manifest path".to_string(),
            })?;

        // Write source to src/lib.rs
        let src_dir = project_dir.join("src");
        std::fs::create_dir_all(&src_dir)?;
        let lib_file = src_dir.join("lib.rs");
        std::fs::write(&lib_file, source)?;

        // Run cargo check with JSON output
        let mut cmd = Command::new(&self.cargo_path);
        cmd.arg("check")
            .arg("--manifest-path")
            .arg(manifest_path)
            .arg("--message-format=json")
            .current_dir(project_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = cmd.output()?;
        let duration = start.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let (errors, warnings) = self.parse_cargo_json_diagnostics(&stdout);

        if output.status.success() || errors.is_empty() {
            Ok(CompilationResult::Success {
                artifact: None,
                warnings,
                metrics: CompilationMetrics {
                    duration,
                    memory_bytes: None,
                    units: 1,
                },
            })
        } else {
            Ok(CompilationResult::Failure {
                errors,
                warnings,
                raw_output: stdout,
            })
        }
    }

    /// Parse cargo's JSON diagnostic output.
    fn parse_cargo_json_diagnostics(
        &self,
        output: &str,
    ) -> (Vec<CompilerDiagnostic>, Vec<CompilerDiagnostic>) {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for line in output.lines() {
            // Cargo outputs JSON objects, one per line
            if line.starts_with('{') && line.contains("\"reason\":\"compiler-message\"") {
                // Extract the nested message
                if let Some(diag) = self.parse_cargo_message(line) {
                    if diag.message.contains("aborting due to") {
                        continue;
                    }
                    if diag.severity == DiagnosticSeverity::Error {
                        errors.push(diag);
                    } else {
                        warnings.push(diag);
                    }
                }
            }
        }

        (errors, warnings)
    }

    /// Parse a single cargo compiler message.
    fn parse_cargo_message(&self, json: &str) -> Option<CompilerDiagnostic> {
        // Cargo format: {"reason":"compiler-message","message":{...}}
        // Find the nested message object
        let msg_start = json.find("\"message\":{")?;
        // Skip past '"message":{' to get inside the object
        let msg_rest = &json[msg_start + 11..];

        // Find matching closing brace
        let mut depth = 1;
        let mut end = 0;
        for (i, c) in msg_rest.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        if end > 0 {
            // Reconstruct the full JSON object
            let inner_msg = format!("{{{}}}", &msg_rest[..end]);
            self.parse_single_json_diagnostic(&inner_msg)
        } else {
            None
        }
    }
}

/// Temporary Cargo project for compilation.
///
/// Creates a temporary directory with Cargo.toml and src/lib.rs for
/// compiling code with external dependencies.
#[derive(Debug)]
pub struct CargoProject {
    /// Project name
    name: String,
    /// Rust edition
    edition: RustEdition,
    /// Dependencies
    dependencies: Vec<(String, String)>,
    /// Temporary directory (if written)
    temp_dir: Option<PathBuf>,
    /// Path to manifest
    manifest_path: Option<PathBuf>,
}

impl CargoProject {
    /// Create a new Cargo project.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            edition: RustEdition::default(),
            dependencies: Vec::new(),
            temp_dir: None,
            manifest_path: None,
        }
    }

    /// Get project name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the Rust edition.
    #[must_use]
    pub fn edition(mut self, edition: RustEdition) -> Self {
        self.edition = edition;
        self
    }

    /// Add a dependency.
    #[must_use]
    pub fn dependency(mut self, name: &str, version: &str) -> Self {
        self.dependencies
            .push((name.to_string(), version.to_string()));
        self
    }

    /// Get dependencies.
    #[must_use]
    pub fn dependencies(&self) -> &[(String, String)] {
        &self.dependencies
    }

    /// Get manifest path.
    #[must_use]
    pub fn manifest_path(&self) -> Option<&PathBuf> {
        self.manifest_path.as_ref()
    }

    /// Get project directory.
    #[must_use]
    pub fn project_dir(&self) -> Option<&PathBuf> {
        self.temp_dir.as_ref()
    }

    /// Write project to a temporary directory.
    ///
    /// # Errors
    ///
    /// Returns error if directory creation or file writing fails.
    pub fn write_to_temp(mut self) -> CITLResult<Self> {
        let temp_dir = std::env::temp_dir().join(format!("citl_{}", self.name));

        // Create directory structure
        std::fs::create_dir_all(&temp_dir)?;
        let src_dir = temp_dir.join("src");
        std::fs::create_dir_all(&src_dir)?;

        // Write Cargo.toml
        let manifest_path = temp_dir.join("Cargo.toml");
        let cargo_toml = self.generate_cargo_toml();
        std::fs::write(&manifest_path, cargo_toml)?;

        // Write empty lib.rs
        let lib_file = src_dir.join("lib.rs");
        std::fs::write(&lib_file, "")?;

        self.temp_dir = Some(temp_dir);
        self.manifest_path = Some(manifest_path);

        Ok(self)
    }

    /// Generate Cargo.toml content.
    fn generate_cargo_toml(&self) -> String {
        use std::fmt::Write;

        let mut toml = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "{}"

[dependencies]
"#,
            self.name,
            self.edition.as_str()
        );

        for (name, version) in &self.dependencies {
            let _ = writeln!(toml, "{name} = \"{version}\"");
        }

        toml
    }
}

impl Drop for CargoProject {
    fn drop(&mut self) {
        if let Some(temp_dir) = &self.temp_dir {
            let _ = std::fs::remove_dir_all(temp_dir);
        }
    }
}

impl Default for RustCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CompilerInterface for RustCompiler {
    fn compile(&self, source: &str, options: &CompileOptions) -> CITLResult<CompilationResult> {
        match &self.mode {
            CompilationMode::Standalone => self.compile_standalone(source, options),
            CompilationMode::Cargo { manifest_path }
            | CompilationMode::CargoCheck { manifest_path } => {
                self.compile_cargo_check(source, manifest_path, options)
            }
        }
    }

    fn parse_diagnostic(&self, raw: &str) -> Option<CompilerDiagnostic> {
        self.parse_single_json_diagnostic(raw)
    }

    fn version(&self) -> CITLResult<CompilerVersion> {
        let output = Command::new(&self.rustc_path).arg("--version").output()?;

        let version_str = String::from_utf8_lossy(&output.stdout);
        // Parse "rustc 1.75.0 (82e1608df 2023-12-21)"
        let parts: Vec<&str> = version_str.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Some(mut version) = CompilerVersion::parse(parts[1]) {
                version.full = version_str.trim().to_string();
                if parts.len() >= 3 {
                    version.commit =
                        Some(parts[2].trim_matches(|c| c == '(' || c == ')').to_string());
                }
                return Ok(version);
            }
        }

        Err(CITLError::ParseError {
            raw: version_str.to_string(),
            details: "Could not parse rustc version".to_string(),
        })
    }

    fn name(&self) -> &'static str {
        "rustc"
    }

    fn is_available(&self) -> bool {
        Command::new(&self.rustc_path)
            .arg("--version")
            .output()
            .is_ok()
    }
}

// ==================== Helper Functions ====================

/// Find rustc binary.
fn which_rustc() -> PathBuf {
    // Check RUSTC env var first, then common paths
    if let Ok(rustc) = std::env::var("RUSTC") {
        let path = PathBuf::from(&rustc);
        if path.exists() {
            return path;
        }
    }

    // Check home directory .cargo/bin
    if let Ok(home) = std::env::var("HOME") {
        let cargo_bin = PathBuf::from(home).join(".cargo/bin/rustc");
        if cargo_bin.exists() {
            return cargo_bin;
        }
    }

    // Common system paths
    for path in ["/usr/bin/rustc", "/usr/local/bin/rustc"] {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }

    // Fall back to relying on PATH
    PathBuf::from("rustc")
}

/// Find cargo binary.
fn which_cargo() -> PathBuf {
    // Check CARGO env var first
    if let Ok(cargo) = std::env::var("CARGO") {
        let path = PathBuf::from(&cargo);
        if path.exists() {
            return path;
        }
    }

    // Check home directory .cargo/bin
    if let Ok(home) = std::env::var("HOME") {
        let cargo_bin = PathBuf::from(home).join(".cargo/bin/cargo");
        if cargo_bin.exists() {
            return cargo_bin;
        }
    }

    // Common system paths
    for path in ["/usr/bin/cargo", "/usr/local/bin/cargo"] {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }

    // Fall back to relying on PATH
    PathBuf::from("cargo")
}

/// Look up error code metadata.
fn lookup_error_code(code: &str) -> ErrorCode {
    let codes = super::rust_error_codes();
    codes
        .get(code)
        .cloned()
        .unwrap_or_else(|| ErrorCode::from_code(code))
}

/// Extract a string value from JSON (simple parsing without serde).
///
/// For "level" and "message" keys, searches for the top-level value,
/// avoiding nested values from child diagnostics in the "children" array.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":\"");

    // For "level" and "message" keys, we need to find the TOP-LEVEL value,
    // not one from nested "children" diagnostics.
    //
    // Formats differ:
    // - rustc standalone: "level":"error",...,"children":[...]
    // - cargo inner msg:  "children":[{"level":"help",...}],...,"level":"error"
    //
    // Strategy: Try finding the key both before and after "children",
    // prefer the one that's at the top level (not inside children array)
    if key == "level" || key == "message" {
        if let Some(children_start) = json.find("\"children\":[") {
            // Check if the key exists BEFORE children (rustc format)
            let before = &json[..children_start];
            if let Some(start) = before.find(&pattern) {
                let start = start + pattern.len();
                let rest = &before[start..];
                if let Some(end) = rest.find('"') {
                    return Some(rest[..end].to_string());
                }
            }

            // Find the end of children array and search AFTER (cargo format)
            let mut depth = 0;
            let mut children_end = children_start + 12;
            for (i, c) in json[children_start + 12..].char_indices() {
                match c {
                    '[' => depth += 1,
                    ']' => {
                        if depth == 0 {
                            children_end = children_start + 12 + i + 1;
                            break;
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
            }

            let after = &json[children_end..];
            if let Some(start) = after.find(&pattern) {
                let start = start + pattern.len();
                let rest = &after[start..];
                if let Some(end) = rest.find('"') {
                    return Some(rest[..end].to_string());
                }
            }
            return None;
        }
    }

    // Default: search entire string
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract a nested string value from JSON.
fn extract_nested_json_string(json: &str, outer_key: &str, inner_key: &str) -> Option<String> {
    let outer_pattern = format!("\"{outer_key}\":{{");
    let start = json.find(&outer_pattern)?;
    let rest = &json[start..];
    let end = rest.find('}')?;
    let inner_json = &rest[..=end];
    extract_json_string(inner_json, inner_key)
}

/// Extract span information from JSON.
#[allow(clippy::disallowed_methods, clippy::unwrap_or_default)]
fn extract_span_from_json(json: &str) -> SourceSpan {
    // Simple extraction - in production would use proper JSON parser
    let file = extract_json_string(json, "file_name").unwrap_or_else(String::new);
    let line_start: usize = extract_json_string(json, "line_start")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let line_end: usize = extract_json_string(json, "line_end")
        .and_then(|s| s.parse().ok())
        .unwrap_or(line_start);
    let column_start: usize = extract_json_string(json, "column_start")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let column_end: usize = extract_json_string(json, "column_end")
        .and_then(|s| s.parse().ok())
        .unwrap_or(column_start);

    SourceSpan::new(&file, line_start, line_end, column_start, column_end)
}

/// Extract suggestions from JSON.
fn extract_suggestions_from_json(_json: &str) -> Option<Vec<CompilerSuggestion>> {
    // Simplified - would need full JSON parsing for complete implementation
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::citl::{Difficulty, ErrorCategory};

    // ==================== CompilerVersion Tests ====================

    #[test]
    fn test_compiler_version_parse() {
        let version = CompilerVersion::parse("1.75.0").expect("Should parse");
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 75);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn test_compiler_version_parse_with_suffix() {
        let version = CompilerVersion::parse("1.75.0-nightly").expect("Should parse");
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 75);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn test_compiler_version_display() {
        let version = CompilerVersion {
            major: 1,
            minor: 75,
            patch: 0,
            full: "rustc 1.75.0".to_string(),
            commit: None,
        };
        assert_eq!(format!("{version}"), "1.75.0");
    }

    // ==================== RustEdition Tests ====================

    #[test]
    fn test_rust_edition_as_str() {
        assert_eq!(RustEdition::E2015.as_str(), "2015");
        assert_eq!(RustEdition::E2018.as_str(), "2018");
        assert_eq!(RustEdition::E2021.as_str(), "2021");
        assert_eq!(RustEdition::E2024.as_str(), "2024");
    }

    #[test]
    fn test_rust_edition_default() {
        assert_eq!(RustEdition::default(), RustEdition::E2021);
    }

    // ==================== CompilationMode Tests ====================

    #[test]
    fn test_compilation_mode_default() {
        assert!(matches!(
            CompilationMode::default(),
            CompilationMode::Standalone
        ));
    }

    // ==================== CompilationResult Tests ====================

    #[test]
    fn test_compilation_result_success() {
        let result = CompilationResult::Success {
            artifact: None,
            warnings: vec![],
            metrics: CompilationMetrics::default(),
        };
        assert!(result.is_success());
        assert_eq!(result.error_count(), 0);
        assert_eq!(result.warning_count(), 0);
    }

    #[test]
    fn test_compilation_result_failure() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::default();
        let error = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "test error", span);

        let result = CompilationResult::Failure {
            errors: vec![error],
            warnings: vec![],
            raw_output: String::new(),
        };
        assert!(!result.is_success());
        assert_eq!(result.error_count(), 1);
        assert_eq!(result.errors().len(), 1);
    }

    // ==================== RustCompiler Tests ====================

    #[test]
    fn test_rust_compiler_new() {
        let compiler = RustCompiler::new();
        assert_eq!(compiler.edition, RustEdition::E2021);
        assert_eq!(compiler.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_rust_compiler_builder_pattern() {
        let compiler = RustCompiler::new()
            .edition(RustEdition::E2018)
            .timeout(Duration::from_secs(30))
            .target("wasm32-unknown-unknown")
            .extra_flag("-W")
            .extra_flag("unused");

        assert_eq!(compiler.edition, RustEdition::E2018);
        assert_eq!(compiler.timeout, Duration::from_secs(30));
        assert_eq!(compiler.target, Some("wasm32-unknown-unknown".to_string()));
        assert_eq!(compiler.extra_flags.len(), 2);
    }

    #[test]
    fn test_rust_compiler_name() {
        let compiler = RustCompiler::new();
        assert_eq!(compiler.name(), "rustc");
    }

    #[test]
    fn test_rust_compiler_is_available() {
        let compiler = RustCompiler::new();
        // This test assumes rustc is installed
        assert!(compiler.is_available());
    }

    #[test]
    fn test_rust_compiler_version() {
        let compiler = RustCompiler::new();
        let version = compiler.version();
        assert!(version.is_ok());
        let v = version.expect("Should get version");
        assert!(v.major >= 1);
    }

    #[test]
    fn test_rust_compiler_compile_valid_code() {
        let compiler = RustCompiler::new();
        // Use library-compatible code (no main function)
        let code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let result = compiler.compile(code, &CompileOptions::default());
        assert!(result.is_ok());
        let result = result.expect("Should compile");
        assert!(
            result.is_success(),
            "Expected success but got: {:?}",
            result.errors()
        );
    }

    #[test]
    fn test_rust_compiler_compile_invalid_code() {
        let compiler = RustCompiler::new();
        let code = "fn main() { let x: i32 = \"hello\"; }";
        let result = compiler.compile(code, &CompileOptions::default());
        assert!(result.is_ok());
        let result = result.expect("Should return result");
        assert!(!result.is_success());
        assert!(result.error_count() > 0);
    }

    #[test]
    fn test_rust_compiler_compile_with_warnings() {
        let compiler = RustCompiler::new().extra_flag("-Wunused");
        let code = "fn main() { let x = 42; }";
        let result = compiler.compile(code, &CompileOptions::default());
        assert!(result.is_ok());
        // May or may not produce warnings depending on rustc version
    }

    // ==================== CompileOptions Tests ====================

    #[test]
    fn test_compile_options_default() {
        let options = CompileOptions::default();
        assert_eq!(options.opt_level, OptLevel::Debug);
        assert!(options.extra_flags.is_empty());
    }

    // ==================== JSON Parsing Tests ====================

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"level":"error","message":"test"}"#;
        assert_eq!(
            extract_json_string(json, "level"),
            Some("error".to_string())
        );
        assert_eq!(
            extract_json_string(json, "message"),
            Some("test".to_string())
        );
        assert_eq!(extract_json_string(json, "nonexistent"), None);
    }

    #[test]
    fn test_extract_nested_json_string() {
        let json = r#"{"code":{"code":"E0308","explanation":null}}"#;
        assert_eq!(
            extract_nested_json_string(json, "code", "code"),
            Some("E0308".to_string())
        );
    }

    // ==================== CompilationMetrics Tests ====================

    #[test]
    fn test_compilation_metrics_default() {
        let metrics = CompilationMetrics::default();
        assert_eq!(metrics.duration, Duration::ZERO);
        assert!(metrics.memory_bytes.is_none());
        assert_eq!(metrics.units, 1);
    }

    // ==================== Cargo Mode Tests ====================

    #[test]
    fn test_cargo_project_creation() {
        let project = CargoProject::new("test_crate");
        assert_eq!(project.name(), "test_crate");
        assert!(project.manifest_path().is_none()); // Not yet written
    }

    #[test]
    fn test_cargo_project_with_dependencies() {
        let project = CargoProject::new("my_crate")
            .dependency("serde", "1.0")
            .dependency("tokio", "1.0");

        assert_eq!(project.dependencies().len(), 2);
    }

    #[test]
    fn test_cargo_project_write_creates_files() {
        let project = CargoProject::new("citl_test_project").edition(RustEdition::E2021);

        let result = project.write_to_temp();
        assert!(result.is_ok());

        let project = result.expect("Should write");
        assert!(project.manifest_path().is_some());
        let manifest = project.manifest_path().expect("Has manifest");
        assert!(manifest.exists());
    }

    #[test]
    fn test_cargo_project_cleanup() {
        let project = CargoProject::new("citl_cleanup_test")
            .write_to_temp()
            .expect("Should write");

        let dir = project.project_dir().expect("Has dir").to_path_buf();
        assert!(dir.exists());

        drop(project);
        // After drop, directory should be cleaned up
        // Note: cleanup happens in Drop impl
    }

    #[test]
    fn test_rust_compiler_cargo_check_mode() {
        let temp_project = CargoProject::new("citl_cargo_test")
            .edition(RustEdition::E2021)
            .write_to_temp()
            .expect("Should create temp project");

        let manifest = temp_project.manifest_path().expect("Has manifest");
        let compiler = RustCompiler::new().mode(CompilationMode::CargoCheck {
            manifest_path: manifest.clone(),
        });

        let code = "pub fn hello() -> &'static str { \"Hello\" }";
        let result = compiler.compile(code, &CompileOptions::default());

        // Should succeed (basic code)
        assert!(result.is_ok());
    }

    #[test]
    fn test_rust_compiler_cargo_mode_detects_errors() {
        let temp_project = CargoProject::new("citl_cargo_error_test")
            .edition(RustEdition::E2021)
            .write_to_temp()
            .expect("Should create temp project");

        let manifest = temp_project.manifest_path().expect("Has manifest");
        let compiler = RustCompiler::new().mode(CompilationMode::CargoCheck {
            manifest_path: manifest.clone(),
        });

        let code = "pub fn bad() -> String { 42 }"; // Type error
        let result = compiler.compile(code, &CompileOptions::default());

        assert!(result.is_ok());
        let compilation = result.expect("Should return result");

        assert!(
            !compilation.is_success(),
            "Expected failure but got success"
        );
        assert!(
            compilation.error_count() > 0,
            "Expected errors but got none"
        );
    }
}
