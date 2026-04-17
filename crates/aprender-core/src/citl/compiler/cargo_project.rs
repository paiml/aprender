
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

        std::fs::remove_file(&source_file).ok();
        std::fs::remove_file(temp_dir.join("citl_output")).ok();

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
