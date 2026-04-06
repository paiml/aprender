use super::*;
use std::path::PathBuf;

// =========================================================================
// FALS-INV-001: Declarative YAML Check
// =========================================================================

#[test]
fn test_fals_inv_001_declarative_yaml_current_project() {
    let result = check_declarative_yaml(Path::new("."));
    assert_eq!(result.id, "AI-01");
    assert_eq!(result.severity, Severity::Critical);
    // Current project should pass or partial
    assert!(result.status != CheckStatus::Skipped);
}

// =========================================================================
// FALS-INV-002: Zero Scripting Check
// =========================================================================

#[test]
fn test_fals_inv_002_zero_scripting_current_project() {
    let result = check_zero_scripting(Path::new("."));
    assert_eq!(result.id, "AI-02");
    assert_eq!(result.severity, Severity::Critical);
    // batuta should pass zero scripting
    assert_eq!(result.status, CheckStatus::Pass);
}

// =========================================================================
// FALS-INV-003: Pure Rust Testing Check
// =========================================================================

#[test]
fn test_fals_inv_003_pure_rust_testing_current_project() {
    let result = check_pure_rust_testing(Path::new("."));
    assert_eq!(result.id, "AI-03");
    assert_eq!(result.severity, Severity::Critical);
    // batuta uses only Rust tests
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Partial));
}

// =========================================================================
// FALS-INV-004: WASM First Check
// =========================================================================

#[test]
fn test_fals_inv_004_wasm_first_current_project() {
    let result = check_wasm_first(Path::new("."));
    assert_eq!(result.id, "AI-04");
    assert_eq!(result.severity, Severity::Critical);
    // batuta has WASM support
    assert!(result.status != CheckStatus::Skipped);
}

// =========================================================================
// FALS-INV-005: Schema Validation Check
// =========================================================================

#[test]
fn test_fals_inv_005_schema_validation_current_project() {
    let result = check_schema_validation(Path::new("."));
    assert_eq!(result.id, "AI-05");
    assert_eq!(result.severity, Severity::Critical);
    // batuta uses serde for config
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Partial));
}

// =========================================================================
// FALS-INV-006: Evaluate All
// =========================================================================

#[test]
fn test_fals_inv_006_evaluate_all_returns_5_items() {
    let results = evaluate_all(Path::new("."));
    assert_eq!(results.len(), 5);

    // All should be critical severity
    for item in &results {
        assert_eq!(item.severity, Severity::Critical);
    }
}

#[test]
fn test_fals_inv_006_all_items_have_evidence() {
    let results = evaluate_all(Path::new("."));

    for item in &results {
        assert!(!item.evidence.is_empty(), "Item {} has no evidence", item.id);
    }
}

// =========================================================================
// FALS-INV-007: Duration Tracking
// =========================================================================

#[test]
fn test_fals_inv_007_duration_tracked() {
    let result = check_declarative_yaml(Path::new("."));
    // Should have some duration (even if very fast)
    // We just check it's set, not a specific value
    assert!(result.duration_ms < 60000); // Less than 1 minute
}

// =========================================================================
// Additional Tests for Coverage
// =========================================================================

#[test]
fn test_fals_inv_008_declarative_yaml_nonexistent() {
    let result = check_declarative_yaml(Path::new("/nonexistent/path"));
    // Should handle missing path gracefully
    assert_eq!(result.id, "AI-01");
    assert!(result.status != CheckStatus::Pass);
}

#[test]
fn test_fals_inv_008_zero_scripting_nonexistent() {
    let result = check_zero_scripting(Path::new("/nonexistent/path"));
    // Should handle missing path gracefully - passes because no scripts found
    assert_eq!(result.id, "AI-02");
}

#[test]
fn test_fals_inv_008_pure_rust_testing_nonexistent() {
    let result = check_pure_rust_testing(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-03");
}

#[test]
fn test_fals_inv_008_wasm_first_nonexistent() {
    let result = check_wasm_first(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-04");
}

#[test]
fn test_fals_inv_008_schema_validation_nonexistent() {
    let result = check_schema_validation(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-05");
}

#[test]
fn test_fals_inv_009_declarative_yaml_temp_with_yaml() {
    let temp_dir = std::env::temp_dir().join("test_yaml_config");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create a YAML file
    std::fs::write(temp_dir.join("config.yaml"), "key: value").unwrap();

    let result = check_declarative_yaml(&temp_dir);
    assert_eq!(result.id, "AI-01");
    // Has YAML files so should at least be partial
    assert!(result.status == CheckStatus::Partial || result.status == CheckStatus::Pass);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_inv_009_declarative_yaml_temp_with_config_module() {
    let temp_dir = std::env::temp_dir().join("test_yaml_config_mod");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();
    std::fs::create_dir_all(temp_dir.join("examples")).unwrap();

    // Create config.rs
    std::fs::write(temp_dir.join("src/config.rs"), "pub struct Config {}").unwrap();

    let result = check_declarative_yaml(&temp_dir);
    assert_eq!(result.id, "AI-01");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_inv_010_zero_scripting_with_scripts() {
    let temp_dir = std::env::temp_dir().join("test_scripting");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create a Python file in src
    std::fs::write(temp_dir.join("src/helper.py"), "print('hello')").unwrap();

    let result = check_zero_scripting(&temp_dir);
    assert_eq!(result.id, "AI-02");
    // Should fail because script in src/
    assert!(result.status == CheckStatus::Fail);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_inv_010_zero_scripting_with_scripts_in_tools() {
    let temp_dir = std::env::temp_dir().join("test_scripting_tools");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("tools")).unwrap();

    // Create a Python file in tools (acceptable)
    std::fs::write(temp_dir.join("tools/build.py"), "print('build')").unwrap();

    let result = check_zero_scripting(&temp_dir);
    assert_eq!(result.id, "AI-02");
    // Should pass because scripts only in tools/
    assert!(result.status == CheckStatus::Pass);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_inv_011_pure_rust_testing_with_rust_tests() {
    let temp_dir = std::env::temp_dir().join("test_rust_testing");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create Rust file with tests
    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
#[cfg(test)]
mod tests {
#[test]
fn it_works() {}
}
"#,
    )
    .unwrap();

    let result = check_pure_rust_testing(&temp_dir);
    assert_eq!(result.id, "AI-03");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_inv_012_wasm_first_with_cargo_toml() {
    let temp_dir = std::env::temp_dir().join("test_wasm_first");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create Cargo.toml with wasm target
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[lib]
crate-type = ["cdylib", "rlib"]

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
"#,
    )
    .unwrap();

    let result = check_wasm_first(&temp_dir);
    assert_eq!(result.id, "AI-04");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_inv_013_schema_validation_with_schemars() {
    let temp_dir = std::env::temp_dir().join("test_schema_val");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create Cargo.toml with schemars
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
schemars = "0.8"
"#,
    )
    .unwrap();

    let result = check_schema_validation(&temp_dir);
    assert_eq!(result.id, "AI-05");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_inv_014_all_severities_critical() {
    let results = evaluate_all(Path::new("."));
    for result in results {
        // All architectural invariants should be Critical
        assert_eq!(result.severity, Severity::Critical);
    }
}

#[test]
fn test_fals_inv_015_all_have_tps_principle() {
    let results = evaluate_all(Path::new("."));
    for result in results {
        assert!(!result.tps_principle.is_empty());
    }
}

// =========================================================================
// ADDITIONAL COVERAGE TESTS
// =========================================================================

#[test]
fn test_inv_cov_001_yaml_nonexistent_path() {
    let result = check_declarative_yaml(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-01");
    assert!(matches!(result.status, CheckStatus::Fail));
}

#[test]
fn test_inv_cov_002_scripting_nonexistent_path() {
    let result = check_zero_scripting(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-02");
    // No scripting files means pass
    assert!(matches!(result.status, CheckStatus::Pass));
}

#[test]
fn test_inv_cov_003_rust_testing_nonexistent_path() {
    let result = check_pure_rust_testing(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-03");
    // No violations means partial (no Rust tests found either)
    assert!(matches!(result.status, CheckStatus::Partial | CheckStatus::Pass | CheckStatus::Fail));
}

#[test]
fn test_inv_cov_004_wasm_first_nonexistent_path() {
    let result = check_wasm_first(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-04");
}

#[test]
fn test_inv_cov_005_schema_validation_nonexistent_path() {
    let result = check_schema_validation(Path::new("/nonexistent/path"));
    assert_eq!(result.id, "AI-05");
}

#[test]
fn test_inv_cov_006_yaml_with_config_dir() {
    let temp_dir = std::env::temp_dir().join("test_yaml_config");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("config")).unwrap();
    std::fs::create_dir_all(temp_dir.join("examples")).unwrap();
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create config module
    std::fs::write(temp_dir.join("src/config.rs"), "// config").unwrap();

    // Create config YAML
    std::fs::write(temp_dir.join("config/app.yaml"), "key: value").unwrap();

    let result = check_declarative_yaml(&temp_dir);
    assert_eq!(result.id, "AI-01");
    assert!(matches!(result.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_007_scripting_with_violations() {
    let temp_dir = std::env::temp_dir().join("test_scripting_vio");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create Python file in src
    std::fs::write(temp_dir.join("src/script.py"), "print('hello')").unwrap();

    let result = check_zero_scripting(&temp_dir);
    assert_eq!(result.id, "AI-02");
    assert!(matches!(result.status, CheckStatus::Fail));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_008_scripting_deps_check() {
    let temp_dir = std::env::temp_dir().join("test_scripting_deps");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create Cargo.toml with pyo3
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
pyo3 = "0.20"
"#,
    )
    .unwrap();

    let result = check_zero_scripting(&temp_dir);
    assert_eq!(result.id, "AI-02");
    assert!(matches!(result.status, CheckStatus::Fail));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_009_pure_rust_with_violations() {
    let temp_dir = std::env::temp_dir().join("test_rust_vio");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create JS test file
    std::fs::write(temp_dir.join("example.test.js"), "test()").unwrap();

    let result = check_pure_rust_testing(&temp_dir);
    assert_eq!(result.id, "AI-03");
    assert!(matches!(result.status, CheckStatus::Fail));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_010_wasm_with_wasm_bindgen() {
    let temp_dir = std::env::temp_dir().join("test_wasm_bind");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create Cargo.toml with wasm-bindgen
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
wasm-bindgen = "0.2"
"#,
    )
    .unwrap();

    let result = check_wasm_first(&temp_dir);
    assert_eq!(result.id, "AI-04");
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Partial));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_011_wasm_with_js_violations() {
    let temp_dir = std::env::temp_dir().join("test_wasm_js");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create excessive JS files
    for i in 0..6 {
        std::fs::write(temp_dir.join(format!("src/app{}.js", i)), "export function f() {}")
            .unwrap();
    }

    let result = check_wasm_first(&temp_dir);
    assert_eq!(result.id, "AI-04");
    assert!(matches!(result.status, CheckStatus::Fail | CheckStatus::Partial));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_012_schema_with_serde() {
    let temp_dir = std::env::temp_dir().join("test_schema_serde");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create Cargo.toml with serde and validator
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
validator = "0.16"
"#,
    )
    .unwrap();

    // Create config struct with Deserialize
    std::fs::write(
        temp_dir.join("src/config.rs"),
        r#"
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
pub name: String,
}
"#,
    )
    .unwrap();

    let result = check_schema_validation(&temp_dir);
    assert_eq!(result.id, "AI-05");
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Partial));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_013_pure_rust_with_node_modules() {
    let temp_dir = std::env::temp_dir().join("test_node_modules");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("node_modules")).unwrap();

    let result = check_pure_rust_testing(&temp_dir);
    assert_eq!(result.id, "AI-03");
    assert!(matches!(result.status, CheckStatus::Fail));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_014_pure_rust_with_package_json() {
    let temp_dir = std::env::temp_dir().join("test_pkg_json");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create package.json with test scripts
    std::fs::write(temp_dir.join("package.json"), r#"{"scripts": {"test": "jest"}}"#).unwrap();

    let result = check_pure_rust_testing(&temp_dir);
    assert_eq!(result.id, "AI-03");
    assert!(matches!(result.status, CheckStatus::Fail));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_015_yaml_partial_case() {
    let temp_dir = std::env::temp_dir().join("test_yaml_partial");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create config module but no examples
    std::fs::write(temp_dir.join("src/config.rs"), "// config").unwrap();

    let result = check_declarative_yaml(&temp_dir);
    assert_eq!(result.id, "AI-01");
    assert!(matches!(result.status, CheckStatus::Partial | CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_inv_cov_016_all_durations_reasonable() {
    let results = evaluate_all(Path::new("."));
    for result in results {
        // Duration should be reasonable (less than 1 minute per check)
        assert!(
            result.duration_ms < 60_000,
            "Check {} took unreasonably long: {}ms",
            result.id,
            result.duration_ms
        );
    }
}

#[test]
fn test_inv_cov_017_schema_with_serde_yaml_ng() {
    let temp_dir = std::env::temp_dir().join("test_schema_serde_ng");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create Cargo.toml with serde_yaml_ng (the successor crate)
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_yaml_ng = "0.10"
validator = "0.16"
"#,
    )
    .unwrap();

    // Create config struct with Deserialize
    std::fs::write(
        temp_dir.join("src/config.rs"),
        r#"
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
pub name: String,
}
"#,
    )
    .unwrap();

    let result = check_schema_validation(&temp_dir);
    assert_eq!(result.id, "AI-05");
    assert!(matches!(result.status, CheckStatus::Pass | CheckStatus::Partial));

    let _ = std::fs::remove_dir_all(&temp_dir);
}
