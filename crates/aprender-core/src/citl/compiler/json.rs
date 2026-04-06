
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

/// Check if an environment variable points to an existing binary.
fn check_env_binary(env_var: &str) -> Option<PathBuf> {
    std::env::var(env_var)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Check for a binary in the user's cargo bin directory.
fn check_cargo_bin(binary_name: &str) -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cargo/bin").join(binary_name))
        .filter(|p| p.exists())
}

/// Check common system paths for a binary.
fn check_system_paths(paths: &[&str]) -> Option<PathBuf> {
    paths.iter().map(PathBuf::from).find(|p| p.exists())
}

/// Find rustc binary.
fn which_rustc() -> PathBuf {
    check_env_binary("RUSTC")
        .or_else(|| check_cargo_bin("rustc"))
        .or_else(|| check_system_paths(&["/usr/bin/rustc", "/usr/local/bin/rustc"]))
        .unwrap_or_else(|| PathBuf::from("rustc"))
}

/// Find cargo binary.
fn which_cargo() -> PathBuf {
    check_env_binary("CARGO")
        .or_else(|| check_cargo_bin("cargo"))
        .or_else(|| check_system_paths(&["/usr/bin/cargo", "/usr/local/bin/cargo"]))
        .unwrap_or_else(|| PathBuf::from("cargo"))
}

/// Look up error code metadata.
fn lookup_error_code(code: &str) -> ErrorCode {
    let codes = super::rust_error_codes();
    codes
        .get(code)
        .cloned()
        .unwrap_or_else(|| ErrorCode::from_code(code))
}

/// Extract a quoted string value from a JSON segment given a pattern.
fn extract_value_from_segment(segment: &str, pattern: &str) -> Option<String> {
    let start = segment.find(pattern)? + pattern.len();
    let rest = &segment[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Find the end index of the children array in JSON.
fn find_children_array_end(json: &str, children_start: usize) -> usize {
    let offset = children_start + 12; // length of "\"children\":["
    let mut depth = 0;
    for (i, c) in json[offset..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                if depth == 0 {
                    return offset + i + 1;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    json.len()
}

/// Extract a top-level JSON value, avoiding nested children values.
fn extract_top_level_json_value(
    json: &str,
    pattern: &str,
    children_start: usize,
) -> Option<String> {
    // Check if the key exists BEFORE children (rustc format)
    let before = &json[..children_start];
    if let Some(value) = extract_value_from_segment(before, pattern) {
        return Some(value);
    }

    // Find the end of children array and search AFTER (cargo format)
    let children_end = find_children_array_end(json, children_start);
    let after = &json[children_end..];
    extract_value_from_segment(after, pattern)
}

/// Extract a string value from JSON (simple parsing without serde).
///
/// For "level" and "message" keys, searches for the top-level value,
/// avoiding nested values from child diagnostics in the "children" array.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":\"");

    // For "level" and "message" keys, we need to find the TOP-LEVEL value,
    // not one from nested "children" diagnostics.
    if key == "level" || key == "message" {
        if let Some(children_start) = json.find("\"children\":[") {
            return extract_top_level_json_value(json, &pattern, children_start);
        }
    }

    // Default: search entire string
    extract_value_from_segment(json, &pattern)
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
mod tests;
