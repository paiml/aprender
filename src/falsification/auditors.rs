//! Audit Utilities for Falsification Checks
//!
//! Provides reusable audit functions for the checklist evaluation.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Result of a file audit.
#[derive(Debug, Clone)]
pub struct FileAuditResult {
    /// Files matching the pattern
    pub matches: Vec<PathBuf>,
    /// Total files scanned
    pub scanned: usize,
    /// Patterns used
    pub patterns: Vec<String>,
}

impl FileAuditResult {
    /// Check if any violations were found.
    pub fn has_violations(&self) -> bool {
        !self.matches.is_empty()
    }

    /// Get violation count.
    pub fn violation_count(&self) -> usize {
        self.matches.len()
    }
}

/// Non-production paths to exclude from scripting audits
const EXCLUDED_SCRIPT_DIRS: &[&str] = &[
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "/target/",
    "/dist/",
    "/examples/",
    "/migrations/",
    "/book/",
    "/docs/",
    "/fixtures/",
    "/testdata/",
];

/// Check if a path should be excluded from scripting audit
fn is_excluded_script_path(path_str: &str) -> bool {
    EXCLUDED_SCRIPT_DIRS.iter().any(|ex| path_str.contains(ex))
}

/// Audit for scripting language files.
pub fn audit_scripting_files(project_path: &Path) -> FileAuditResult {
    let patterns = vec![
        "**/*.py".to_string(),
        "**/*.js".to_string(),
        "**/*.ts".to_string(),
        "**/*.lua".to_string(),
        "**/*.rb".to_string(),
    ];

    let mut matches = Vec::new();
    let mut scanned = 0;

    for pattern in &patterns {
        let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) else {
            continue;
        };
        for entry in entries.flatten() {
            scanned += 1;
            if !is_excluded_script_path(&entry.to_string_lossy()) {
                matches.push(entry);
            }
        }
    }

    FileAuditResult { matches, scanned, patterns }
}

/// Audit for test framework files.
pub fn audit_test_frameworks(project_path: &Path) -> FileAuditResult {
    let patterns = vec![
        // JavaScript test files
        "**/*.test.js".to_string(),
        "**/*.spec.js".to_string(),
        "**/*.test.ts".to_string(),
        "**/*.spec.ts".to_string(),
        "**/jest.config.*".to_string(),
        "**/vitest.config.*".to_string(),
        // Python test files
        "**/test_*.py".to_string(),
        "**/*_test.py".to_string(),
        "**/conftest.py".to_string(),
        "**/pytest.ini".to_string(),
    ];

    let mut matches = Vec::new();
    let mut scanned = 0;

    for pattern in &patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                scanned += 1;
                let path_str = entry.to_string_lossy();

                if !path_str.contains("node_modules") && !path_str.contains("venv") {
                    matches.push(entry);
                }
            }
        }
    }

    FileAuditResult { matches, scanned, patterns }
}

/// Audit for YAML configuration files.
pub fn audit_yaml_configs(project_path: &Path) -> FileAuditResult {
    let patterns = vec![
        "*.yaml".to_string(),
        "*.yml".to_string(),
        "config/**/*.yaml".to_string(),
        "config/**/*.yml".to_string(),
        "examples/**/*.yaml".to_string(),
        "examples/**/*.yml".to_string(),
    ];

    let mut matches = Vec::new();
    let mut scanned = 0;

    for pattern in &patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                scanned += 1;
                matches.push(entry);
            }
        }
    }

    FileAuditResult { matches, scanned, patterns }
}

/// Result of a dependency audit.
#[derive(Debug, Clone)]
pub struct DependencyAuditResult {
    /// Forbidden dependencies found
    pub forbidden: Vec<String>,
    /// All dependencies checked
    pub checked: Vec<String>,
    /// Raw cargo tree output (if available)
    pub cargo_tree_output: Option<String>,
}

impl DependencyAuditResult {
    /// Check if any forbidden dependencies were found.
    pub fn has_violations(&self) -> bool {
        !self.forbidden.is_empty()
    }
}

/// Audit Cargo.toml for forbidden dependencies.
pub fn audit_cargo_dependencies(project_path: &Path, forbidden: &[&str]) -> DependencyAuditResult {
    let cargo_toml = project_path.join("Cargo.toml");
    let mut found_forbidden = Vec::new();
    let mut checked = Vec::new();

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            for dep in forbidden {
                checked.push((*dep).to_string());

                // Simple check - a proper implementation would parse TOML
                if content.contains(&format!("{} =", dep))
                    || content.contains(&format!("{}=", dep))
                    || content.contains(&format!("\"{}\"", dep))
                {
                    found_forbidden.push((*dep).to_string());
                }
            }
        }
    }

    // Try to get cargo tree output for more detailed analysis
    let cargo_tree_output = Command::new("cargo")
        .args(["tree", "--edges", "no-dev", "-p"])
        .current_dir(project_path)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        });

    DependencyAuditResult { forbidden: found_forbidden, checked, cargo_tree_output }
}

/// Check if a Rust project has tests.
pub fn has_rust_tests(project_path: &Path) -> bool {
    project_path.join("tests").exists()
        || super::helpers::files_contain_pattern(
            project_path,
            &["src/**/*.rs"],
            &["#[test]", "#[cfg(test)]"],
        )
}

/// Check if a project has WASM support.
pub fn has_wasm_support(project_path: &Path) -> WasmSupport {
    let cargo_toml = project_path.join("Cargo.toml");
    let mut support = WasmSupport::default();

    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            support.has_wasm_feature = content.contains("[features]")
                && (content.contains("wasm") || content.contains("web"));

            support.has_wasm_bindgen = content.contains("wasm-bindgen");
            support.has_web_sys = content.contains("web-sys");
            support.has_wasm_pack = content.contains("wasm-pack");
        }
    }

    // Check for wasm.rs module
    support.has_wasm_module = project_path.join("src/wasm.rs").exists();

    // Check for wasm target in .cargo/config.toml
    let cargo_config = project_path.join(".cargo/config.toml");
    if cargo_config.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_config) {
            support.has_wasm_target = content.contains("wasm32");
        }
    }

    support
}

/// WASM support detection result.
#[derive(Debug, Clone, Default)]
pub struct WasmSupport {
    pub has_wasm_feature: bool,
    pub has_wasm_bindgen: bool,
    pub has_web_sys: bool,
    pub has_wasm_pack: bool,
    pub has_wasm_module: bool,
    pub has_wasm_target: bool,
}

impl WasmSupport {
    /// Check if any WASM support is present.
    pub fn is_supported(&self) -> bool {
        self.has_wasm_feature
            || self.has_wasm_bindgen
            || self.has_web_sys
            || self.has_wasm_pack
            || self.has_wasm_module
    }

    /// Get support level description.
    pub fn level(&self) -> &'static str {
        if self.has_wasm_bindgen && self.has_wasm_module {
            "Full"
        } else if self.has_wasm_bindgen || self.has_wasm_feature {
            "Partial"
        } else if self.has_wasm_module {
            "Basic"
        } else {
            "None"
        }
    }
}

/// Scan source files for Deserialize structs and config patterns.
fn scan_deserialize_structs(project_path: &Path) -> (bool, bool) {
    let mut has_deserialize = false;
    let mut has_config = false;
    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return (false, false);
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        if content.contains("#[derive") && content.contains("Deserialize") {
            has_deserialize = true;
            if content.to_lowercase().contains("config") {
                has_config = true;
            }
        }
    }
    (has_deserialize, has_config)
}

/// Check for serde-based config validation.
pub fn has_serde_config(project_path: &Path) -> SerdeConfigSupport {
    let cargo_toml = project_path.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).unwrap_or_default();
    let (has_deserialize_structs, has_config_struct) = scan_deserialize_structs(project_path);

    SerdeConfigSupport {
        has_serde: content.contains("serde"),
        has_serde_yaml: content.contains("serde_yaml")
            || content.contains("serde_yml")
            || content.contains("serde_yaml_ng"),
        has_serde_json: content.contains("serde_json"),
        has_toml: content.contains("toml"),
        has_validator: content.contains("validator") || content.contains("garde"),
        has_deserialize_structs,
        has_config_struct,
    }
}

/// Serde config support detection result.
#[derive(Debug, Clone, Default)]
pub struct SerdeConfigSupport {
    pub has_serde: bool,
    pub has_serde_yaml: bool,
    pub has_serde_json: bool,
    pub has_toml: bool,
    pub has_validator: bool,
    pub has_deserialize_structs: bool,
    pub has_config_struct: bool,
}

impl SerdeConfigSupport {
    /// Check if typed config is supported.
    pub fn has_typed_config(&self) -> bool {
        self.has_serde && self.has_config_struct
    }

    /// Check if validation is supported.
    pub fn has_validation(&self) -> bool {
        self.has_validator || self.has_serde // serde itself provides some validation
    }
}

#[cfg(test)]
#[path = "auditors_tests.rs"]
mod tests;
