//! Architectural Invariants - CRITICAL Checks
//!
//! Section 10 of the Popperian Falsification Checklist.
//! Any failure in this section = Project FAIL.
//!
//! - AI-01: Declarative YAML Configuration
//! - AI-02: Zero Scripting in Production
//! - AI-03: Pure Rust Testing (No Jest/Pytest)
//! - AI-04: WASM-First Browser Support
//! - AI-05: Declarative Schema Validation

use crate::falsification::helpers::{apply_check_outcome, CheckOutcome};
use crate::falsification::types::*;
use std::path::Path;
use std::time::Instant;

/// Evaluate all architectural invariants.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_declarative_yaml(project_path),
        check_zero_scripting(project_path),
        check_pure_rust_testing(project_path),
        check_wasm_first(project_path),
        check_schema_validation(project_path),
    ]
}

/// AI-01: Declarative YAML Configuration
///
/// **Claim:** Project offers full functionality via declarative YAML without code.
///
/// **Rejection Criteria (CRITICAL):**
/// - Any core feature unavailable via YAML config
/// - User must write Rust/code to use basic functionality
pub fn check_declarative_yaml(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-01",
        "Declarative YAML Configuration",
        "Project offers full functionality via declarative YAML without code",
    )
    .with_severity(Severity::Critical)
    .with_tps("Poka-Yoke — enable non-developers");

    // Check for YAML config files
    let yaml_patterns = ["*.yaml", "*.yml"];
    let mut yaml_files = Vec::new();

    for pattern in yaml_patterns {
        if let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                yaml_files.push(entry);
            }
        }
        // Also check config directories (both singular and plural)
        for dir in ["config", "configs"] {
            if let Ok(entries) =
                glob::glob(&format!("{}/{}/**/{}", project_path.display(), dir, pattern))
            {
                for entry in entries.flatten() {
                    yaml_files.push(entry);
                }
            }
        }
    }

    // Check for schema definitions (serde structs)
    let has_config_module = project_path.join("src/config.rs").exists()
        || project_path.join("src/config/mod.rs").exists();

    // Check for example configs
    let has_examples = project_path.join("examples").exists() || !yaml_files.is_empty();

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} YAML config files", yaml_files.len()),
        yaml_files.clone(),
    ));

    item = apply_check_outcome(
        item,
        &[
            (has_config_module && has_examples, CheckOutcome::Pass),
            (
                has_config_module || !yaml_files.is_empty(),
                CheckOutcome::Partial("Config module exists but examples incomplete"),
            ),
            (true, CheckOutcome::Fail("No declarative YAML configuration found")),
        ],
    );

    item.finish_timed(start)
}

/// AI-02: Zero Scripting in Production
///
/// **Claim:** No Python/JavaScript/Lua in production runtime paths.
///
/// **Rejection Criteria (CRITICAL):**
/// - Any `.py`, `.js`, `.lua` in src/ or runtime
/// - pyo3, napi-rs, mlua in non-dev dependencies
/// - Interpreter embedded in binary
pub fn check_zero_scripting(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-02",
        "Zero Scripting in Production",
        "No Python/JavaScript/Lua in production runtime paths",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — type safety, determinism");

    // Check for scripting language files in src/
    let violations = find_glob_violations(
        project_path,
        &["src/**/*.py", "src/**/*.js", "src/**/*.ts", "src/**/*.lua", "src/**/*.rb"],
        &["/target/", "/pkg/", ".wasm"],
    );

    // Check Cargo.toml for scripting runtime dependencies
    let scripting_deps = super::helpers::find_scripting_deps(project_path);

    item = item.with_evidence(Evidence::dependency_audit(
        "Checked Cargo.toml for scripting runtimes".to_string(),
        format!("Found: {:?}", scripting_deps),
    ));

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} scripting files in src/", violations.len()),
        violations.clone(),
    ));

    let fail_reasons = {
        let mut r = Vec::new();
        if !violations.is_empty() {
            r.push(format!("{} scripting files in src/", violations.len()));
        }
        if !scripting_deps.is_empty() {
            r.push(format!("Scripting deps: {:?}", scripting_deps));
        }
        r.join("; ")
    };
    item = apply_check_outcome(
        item,
        &[
            (violations.is_empty() && scripting_deps.is_empty(), CheckOutcome::Pass),
            (true, CheckOutcome::Fail(&fail_reasons)),
        ],
    );

    item.finish_timed(start)
}

/// Find files matching glob patterns, excluding paths containing any exclude string
fn find_glob_violations(
    project_path: &Path,
    patterns: &[&str],
    excludes: &[&str],
) -> Vec<std::path::PathBuf> {
    let mut results = Vec::new();
    for pattern in patterns {
        let Ok(entries) = glob::glob(&format!("{}/{}", project_path.display(), pattern)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path_str = entry.to_string_lossy();
            if !excludes.iter().any(|ex| path_str.contains(ex)) {
                results.push(entry);
            }
        }
    }
    results
}

/// Check if Rust tests exist in the project
fn has_rust_tests(project_path: &Path) -> bool {
    project_path.join("tests").exists()
        || glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
            .ok()
            .map(|entries| {
                entries.flatten().any(|p| {
                    std::fs::read_to_string(&p)
                        .ok()
                        .is_some_and(|c| c.contains("#[test]") || c.contains("#[cfg(test)]"))
                })
            })
            .unwrap_or(false)
}

/// AI-03: Pure Rust Testing (No Jest/Pytest)
///
/// **Claim:** All tests written in Rust, no external test frameworks.
///
/// **Rejection Criteria (CRITICAL):**
/// - Any Jest, Mocha, Pytest, unittest files
/// - package.json with test scripts
/// - requirements-dev.txt with pytest
pub fn check_pure_rust_testing(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-03",
        "Pure Rust Testing",
        "All tests written in Rust, no external test frameworks",
    )
    .with_severity(Severity::Critical)
    .with_tps("Zero scripting policy");

    let mut violations = find_glob_violations(
        project_path,
        &[
            "**/*.test.js",
            "**/*.spec.js",
            "**/*.test.ts",
            "**/*.spec.ts",
            "**/jest.config.*",
            "**/vitest.config.*",
        ],
        &["node_modules"],
    );

    violations.extend(find_glob_violations(
        project_path,
        &["**/test_*.py", "**/*_test.py", "**/conftest.py", "**/pytest.ini", "**/pyproject.toml"],
        &["venv", ".venv"],
    ));

    let package_json = project_path.join("package.json");
    if let Ok(content) = std::fs::read_to_string(&package_json) {
        if content.contains("\"test\"") || content.contains("\"jest\"") {
            violations.push(package_json);
        }
    }

    let node_modules = project_path.join("node_modules");
    if node_modules.exists() {
        violations.push(node_modules);
    }

    violations.extend(find_glob_violations(project_path, &["**/__pycache__"], &[]));

    item = item.with_evidence(Evidence::file_audit(
        format!("Found {} non-Rust test artifacts", violations.len()),
        violations.clone(),
    ));

    let has_tests = has_rust_tests(project_path);
    let fail_msg = format!(
        "Found {} non-Rust test artifacts: {:?}",
        violations.len(),
        violations.iter().take(5).collect::<Vec<_>>()
    );
    item = apply_check_outcome(
        item,
        &[
            (violations.is_empty() && has_tests, CheckOutcome::Pass),
            (
                violations.is_empty(),
                CheckOutcome::Partial("No violations but no Rust tests detected"),
            ),
            (true, CheckOutcome::Fail(&fail_msg)),
        ],
    );

    item.finish_timed(start)
}

/// Check if a JS file path should be excluded from analysis
fn is_excluded_js_path(path_str: &str) -> bool {
    const EXCLUDED_DIRS: &[&str] =
        &["node_modules", "/pkg/", "/dist/", "/target/", "/book/", "/book-output/", "/docs/"];
    const EXCLUDED_PREFIXES: &[&str] =
        &["target/", "pkg/", "dist/", "book/", "book-output/", "docs/"];

    EXCLUDED_DIRS.iter().any(|d| path_str.contains(d))
        || EXCLUDED_PREFIXES.iter().any(|p| path_str.starts_with(p))
}

/// Find non-excluded JS files in project
fn find_js_files(project_path: &Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = glob::glob(&format!("{}/**/*.js", project_path.display())) else {
        return Vec::new();
    };
    entries.flatten().filter(|entry| !is_excluded_js_path(&entry.to_string_lossy())).collect()
}

/// Detect JS framework in package.json
fn detect_js_framework(project_path: &Path) -> bool {
    let package_json = project_path.join("package.json");
    let Ok(content) = std::fs::read_to_string(package_json) else {
        return false;
    };
    ["react", "vue", "svelte", "angular", "next", "nuxt"].iter().any(|fw| content.contains(fw))
}

/// Parse WASM feature/bindgen info from Cargo.toml
fn parse_wasm_cargo_info(project_path: &Path) -> (bool, bool) {
    let cargo_toml = project_path.join("Cargo.toml");
    let Ok(content) = std::fs::read_to_string(cargo_toml) else {
        return (false, false);
    };
    let has_feature = content.contains("wasm") || content.contains("web");
    let has_bindgen = content.contains("wasm-bindgen")
        || content.contains("wasm-pack")
        || content.contains("web-sys");
    (has_feature, has_bindgen)
}

/// AI-04: WASM-First Browser Support
///
/// **Claim:** Browser functionality via WASM, not JavaScript.
///
/// **Rejection Criteria (CRITICAL):**
/// - JS files beyond minimal WASM glue
/// - npm dependencies for core functionality
/// - React/Vue/Svelte instead of WASM UI
pub fn check_wasm_first(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-04",
        "WASM-First Browser Support",
        "Browser functionality via WASM, not JavaScript",
    )
    .with_severity(Severity::Critical)
    .with_tps("Zero scripting, sovereignty");

    let (has_wasm_feature, has_wasm_bindgen) = parse_wasm_cargo_info(project_path);
    let has_wasm_module =
        project_path.join("src/wasm.rs").exists() || project_path.join("src/lib.rs").exists();
    let js_files = find_js_files(project_path);
    let has_js_framework = detect_js_framework(project_path);

    item = item.with_evidence(Evidence::file_audit(
        format!(
            "WASM: feature={}, bindgen={}, JS files={}",
            has_wasm_feature,
            has_wasm_bindgen,
            js_files.len()
        ),
        js_files.clone(),
    ));

    let too_many_js_msg = format!("Too many JS files ({}) beyond WASM glue", js_files.len());
    let wasm_partial_msg = format!("WASM support exists but {} JS files found", js_files.len());
    let has_wasm_support = has_wasm_bindgen || has_wasm_feature;
    item = apply_check_outcome(
        item,
        &[
            (
                has_js_framework,
                CheckOutcome::Fail("JavaScript framework detected (React/Vue/Svelte)"),
            ),
            (js_files.len() > 5, CheckOutcome::Fail(&too_many_js_msg)),
            (has_wasm_support && js_files.is_empty(), CheckOutcome::Pass),
            (has_wasm_support, CheckOutcome::Partial(&wasm_partial_msg)),
            (
                has_wasm_module && js_files.is_empty(),
                CheckOutcome::Partial("No explicit WASM feature but no JS violations"),
            ),
            (true, CheckOutcome::Fail("No WASM support detected")),
        ],
    );

    item.finish_timed(start)
}

/// AI-05: Declarative Schema Validation
///
/// **Claim:** YAML configs validated against typed schema.
///
/// **Rejection Criteria (CRITICAL):**
/// - Invalid YAML silently accepted
/// - No JSON Schema or serde validation
/// - Runtime panics on bad config
pub fn check_schema_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "AI-05",
        "Declarative Schema Validation",
        "YAML configs validated against typed schema",
    )
    .with_severity(Severity::Critical)
    .with_tps("Poka-Yoke — prevent config errors");

    // Check for serde in Cargo.toml
    let schema = super::helpers::detect_schema_deps(project_path);
    let has_serde = schema.has_serde;
    let has_serde_yaml = schema.has_serde_yaml;
    let has_validator = schema.has_validator;

    // Check for config struct with Deserialize
    let has_config_struct = super::helpers::has_deserialize_config_struct(project_path);

    // Check for JSON Schema files
    let has_json_schema = glob::glob(&format!("{}/**/*.schema.json", project_path.display()))
        .ok()
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);

    item = item.with_evidence(Evidence::schema_validation(
        format!(
            "serde={}, yaml={}, validator={}, config_struct={}, json_schema={}",
            has_serde, has_serde_yaml, has_validator, has_config_struct, has_json_schema
        ),
        format!(
            "Schema validation: {}",
            if has_config_struct || has_json_schema { "PRESENT" } else { "MISSING" }
        ),
    ));

    let has_full_serde = has_serde && has_serde_yaml && has_config_struct;
    item = apply_check_outcome(
        item,
        &[
            (has_full_serde && (has_validator || has_json_schema), CheckOutcome::Pass),
            (
                has_full_serde,
                CheckOutcome::Partial("Basic serde validation but no explicit validator"),
            ),
            (
                has_serde && has_config_struct,
                CheckOutcome::Partial("Config struct exists but YAML support unclear"),
            ),
            (true, CheckOutcome::Fail("No typed schema validation for configs")),
        ],
    );

    item.finish_timed(start)
}

#[cfg(test)]
#[path = "invariants_tests.rs"]
mod tests;
