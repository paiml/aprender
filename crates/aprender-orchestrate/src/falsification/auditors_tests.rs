use super::*;

// =========================================================================
// FALS-AUD-001: Scripting File Audit
// =========================================================================

#[test]
fn test_fals_aud_001_scripting_audit_current_project() {
    let result = audit_scripting_files(Path::new("."));
    // batuta should have no scripting files in src
    assert!(
        !result.has_violations()
            || result.matches.iter().all(|p| {
                let s = p.to_string_lossy();
                s.contains("scripts/") || s.contains("tests/") || s.contains("target/")
            }),
        "Found scripting violations: {:?}",
        result.matches
    );
}

// =========================================================================
// FALS-AUD-002: Test Framework Audit
// =========================================================================

#[test]
fn test_fals_aud_002_test_framework_audit() {
    let result = audit_test_frameworks(Path::new("."));
    // batuta should not have Jest/Pytest
    assert!(!result.has_violations(), "Found test framework violations: {:?}", result.matches);
}

// =========================================================================
// FALS-AUD-003: YAML Config Audit
// =========================================================================

#[test]
fn test_fals_aud_003_yaml_config_audit() {
    let _result = audit_yaml_configs(Path::new("."));
    // Just verify it runs without panic
}

// =========================================================================
// FALS-AUD-004: Dependency Audit
// =========================================================================

#[test]
fn test_fals_aud_004_dependency_audit_no_pyo3() {
    let result = audit_cargo_dependencies(Path::new("."), &["pyo3", "napi", "mlua"]);
    assert!(!result.has_violations(), "Found forbidden deps: {:?}", result.forbidden);
}

// =========================================================================
// FALS-AUD-005: Rust Tests Detection
// =========================================================================

#[test]
fn test_fals_aud_005_has_rust_tests() {
    assert!(has_rust_tests(Path::new(".")));
}

// =========================================================================
// FALS-AUD-006: WASM Support Detection
// =========================================================================

#[test]
fn test_fals_aud_006_wasm_support() {
    let support = has_wasm_support(Path::new("."));
    // batuta has WASM support
    assert!(support.is_supported(), "Expected WASM support");
}

// =========================================================================
// FALS-AUD-007: Serde Config Detection
// =========================================================================

#[test]
fn test_fals_aud_007_serde_config() {
    let support = has_serde_config(Path::new("."));
    assert!(support.has_serde, "Expected serde");
    assert!(support.has_config_struct, "Expected config struct");
}

// =========================================================================
// Additional Tests for Coverage
// =========================================================================

#[test]
fn test_fals_aud_008_file_audit_result_methods() {
    let result = FileAuditResult {
        matches: vec![PathBuf::from("test.py")],
        scanned: 10,
        patterns: vec!["*.py".to_string()],
    };

    assert!(result.has_violations());
    assert_eq!(result.violation_count(), 1);

    let empty_result =
        FileAuditResult { matches: vec![], scanned: 10, patterns: vec!["*.py".to_string()] };

    assert!(!empty_result.has_violations());
    assert_eq!(empty_result.violation_count(), 0);
}

#[test]
fn test_fals_aud_009_dependency_audit_result_methods() {
    let result = DependencyAuditResult {
        forbidden: vec!["pyo3".to_string()],
        checked: vec!["pyo3".to_string()],
        cargo_tree_output: Some("tree output".to_string()),
    };

    assert!(result.has_violations());

    let clean_result = DependencyAuditResult {
        forbidden: vec![],
        checked: vec!["pyo3".to_string()],
        cargo_tree_output: None,
    };

    assert!(!clean_result.has_violations());
}

#[test]
fn test_fals_aud_010_audit_nonexistent_path() {
    let result = audit_scripting_files(Path::new("/nonexistent/path"));
    assert_eq!(result.matches.len(), 0);

    let yaml_result = audit_yaml_configs(Path::new("/nonexistent/path"));
    assert_eq!(yaml_result.matches.len(), 0);

    let test_result = audit_test_frameworks(Path::new("/nonexistent/path"));
    assert_eq!(test_result.matches.len(), 0);
}

#[test]
fn test_fals_aud_011_has_rust_tests_no_tests() {
    // Test with a path that has no tests
    let temp_dir = std::env::temp_dir().join("test_no_rust_tests");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create a Rust file without tests
    std::fs::write(temp_dir.join("src/lib.rs"), "fn hello() {}").unwrap();

    assert!(!has_rust_tests(&temp_dir));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_011_has_rust_tests_with_tests_dir() {
    let temp_dir = std::env::temp_dir().join("test_with_tests_dir");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("tests")).unwrap();

    assert!(has_rust_tests(&temp_dir));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_012_wasm_support_no_cargo_toml() {
    let support = has_wasm_support(Path::new("/nonexistent/path"));
    assert!(!support.has_wasm_feature);
    assert!(!support.has_wasm_bindgen);
    assert!(!support.has_web_sys);
}

#[test]
fn test_fals_aud_012_wasm_support_with_wasm_bindgen() {
    let temp_dir = std::env::temp_dir().join("test_wasm_support");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"
[package]
name = "test"
version = "0.1.0"

[features]
wasm = ["wasm-bindgen"]

[dependencies]
wasm-bindgen = "0.2"
"#,
    )
    .unwrap();

    let support = has_wasm_support(&temp_dir);
    assert!(support.has_wasm_feature);
    assert!(support.has_wasm_bindgen);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_013_serde_config_no_cargo() {
    let support = has_serde_config(Path::new("/nonexistent/path"));
    assert!(!support.has_serde);
    assert!(!support.has_config_struct);
}

#[test]
fn test_fals_aud_014_dependency_audit_with_forbidden() {
    let temp_dir = std::env::temp_dir().join("test_dep_audit");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

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

    let result = audit_cargo_dependencies(&temp_dir, &["pyo3", "napi"]);
    assert!(result.has_violations());
    assert!(result.forbidden.contains(&"pyo3".to_string()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_015_audit_scripting_with_excluded() {
    let temp_dir = std::env::temp_dir().join("test_scripting_excluded");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("node_modules")).unwrap();
    std::fs::create_dir_all(temp_dir.join("venv")).unwrap();

    // Files in excluded dirs should be ignored
    std::fs::write(temp_dir.join("node_modules/test.js"), "").unwrap();
    std::fs::write(temp_dir.join("venv/helper.py"), "").unwrap();

    let result = audit_scripting_files(&temp_dir);
    assert!(!result.has_violations());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_016_wasm_support_struct_fields() {
    let support = WasmSupport {
        has_wasm_feature: true,
        has_wasm_bindgen: true,
        has_web_sys: true,
        has_wasm_pack: false,
        has_wasm_module: true,
        has_wasm_target: true,
    };

    assert!(support.has_wasm_feature);
    assert!(support.has_wasm_module);
    assert!(!support.has_wasm_pack);
    assert!(support.has_wasm_target);
}

#[test]
fn test_fals_aud_017_serde_config_support_struct_fields() {
    let support = SerdeConfigSupport {
        has_serde: true,
        has_serde_yaml: false,
        has_serde_json: true,
        has_toml: false,
        has_validator: false,
        has_deserialize_structs: true,
        has_config_struct: true,
    };

    assert!(support.has_serde);
    assert!(!support.has_serde_yaml);
    assert!(support.has_config_struct);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_fals_aud_018_wasm_support_level_full() {
    let support = WasmSupport {
        has_wasm_feature: true,
        has_wasm_bindgen: true,
        has_web_sys: true,
        has_wasm_pack: true,
        has_wasm_module: true,
        has_wasm_target: true,
    };
    assert_eq!(support.level(), "Full");
    assert!(support.is_supported());
}

#[test]
fn test_fals_aud_019_wasm_support_level_partial() {
    let support = WasmSupport {
        has_wasm_feature: true,
        has_wasm_bindgen: false,
        has_web_sys: false,
        has_wasm_pack: false,
        has_wasm_module: false,
        has_wasm_target: false,
    };
    assert_eq!(support.level(), "Partial");
    assert!(support.is_supported());
}

#[test]
fn test_fals_aud_020_wasm_support_level_basic() {
    let support = WasmSupport {
        has_wasm_feature: false,
        has_wasm_bindgen: false,
        has_web_sys: false,
        has_wasm_pack: false,
        has_wasm_module: true,
        has_wasm_target: false,
    };
    assert_eq!(support.level(), "Basic");
    assert!(support.is_supported());
}

#[test]
fn test_fals_aud_021_wasm_support_level_none() {
    let support = WasmSupport {
        has_wasm_feature: false,
        has_wasm_bindgen: false,
        has_web_sys: false,
        has_wasm_pack: false,
        has_wasm_module: false,
        has_wasm_target: false,
    };
    assert_eq!(support.level(), "None");
    assert!(!support.is_supported());
}

#[test]
fn test_fals_aud_022_serde_config_has_typed_config() {
    let with_typed = SerdeConfigSupport {
        has_serde: true,
        has_serde_yaml: false,
        has_serde_json: false,
        has_toml: false,
        has_validator: false,
        has_deserialize_structs: false,
        has_config_struct: true,
    };
    assert!(with_typed.has_typed_config());

    let without_serde = SerdeConfigSupport {
        has_serde: false,
        has_serde_yaml: false,
        has_serde_json: false,
        has_toml: false,
        has_validator: false,
        has_deserialize_structs: false,
        has_config_struct: true,
    };
    assert!(!without_serde.has_typed_config());

    let without_config = SerdeConfigSupport {
        has_serde: true,
        has_serde_yaml: false,
        has_serde_json: false,
        has_toml: false,
        has_validator: false,
        has_deserialize_structs: false,
        has_config_struct: false,
    };
    assert!(!without_config.has_typed_config());
}

#[test]
fn test_fals_aud_023_serde_config_has_validation() {
    let with_validator = SerdeConfigSupport {
        has_serde: false,
        has_serde_yaml: false,
        has_serde_json: false,
        has_toml: false,
        has_validator: true,
        has_deserialize_structs: false,
        has_config_struct: false,
    };
    assert!(with_validator.has_validation());

    let with_serde = SerdeConfigSupport {
        has_serde: true,
        has_serde_yaml: false,
        has_serde_json: false,
        has_toml: false,
        has_validator: false,
        has_deserialize_structs: false,
        has_config_struct: false,
    };
    assert!(with_serde.has_validation());

    let without_both = SerdeConfigSupport {
        has_serde: false,
        has_serde_yaml: false,
        has_serde_json: false,
        has_toml: false,
        has_validator: false,
        has_deserialize_structs: false,
        has_config_struct: false,
    };
    assert!(!without_both.has_validation());
}

#[test]
fn test_fals_aud_024_wasm_support_with_cargo_config() {
    let temp_dir = std::env::temp_dir().join("test_wasm_cargo_config");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join(".cargo")).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    std::fs::write(
        temp_dir.join(".cargo/config.toml"),
        "[build]\ntarget = \"wasm32-unknown-unknown\"\n",
    )
    .unwrap();

    let support = has_wasm_support(&temp_dir);
    assert!(support.has_wasm_target);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_025_wasm_support_with_wasm_module() {
    let temp_dir = std::env::temp_dir().join("test_wasm_module");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    std::fs::write(temp_dir.join("src/wasm.rs"), "// WASM module\n").unwrap();

    let support = has_wasm_support(&temp_dir);
    assert!(support.has_wasm_module);
    assert_eq!(support.level(), "Basic");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_026_serde_config_with_yml_variant() {
    let temp_dir = std::env::temp_dir().join("test_serde_yml");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"[package]
name = "test"
version = "0.1.0"

[dependencies]
serde = "1.0"
serde_yml = "0.1"
garde = "0.1"
"#,
    )
    .unwrap();

    let support = has_serde_config(&temp_dir);
    assert!(support.has_serde);
    assert!(support.has_serde_yaml);
    assert!(support.has_validator);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_027_has_rust_tests_with_cfg_test() {
    let temp_dir = std::env::temp_dir().join("test_cfg_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
fn hello() {}

#[cfg(test)]
mod tests {
#[test]
fn test_hello() {}
}
"#,
    )
    .unwrap();

    assert!(has_rust_tests(&temp_dir));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_028_dependency_audit_alternative_formats() {
    let temp_dir = std::env::temp_dir().join("test_dep_formats");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Test with different dependency formats
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"[package]
name = "test"
version = "0.1.0"

[dependencies]
pyo3 = "0.20"
napi= { version = "2.0" }
mlua = { path = "../mlua" }
"#,
    )
    .unwrap();

    let result = audit_cargo_dependencies(&temp_dir, &["pyo3", "napi", "mlua"]);
    assert!(result.forbidden.contains(&"pyo3".to_string()));
    assert!(result.forbidden.contains(&"napi".to_string()));
    assert!(result.forbidden.contains(&"mlua".to_string()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_029_scripting_audit_with_more_exclusions() {
    let temp_dir = std::env::temp_dir().join("test_scripting_more");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("__pycache__")).unwrap();
    std::fs::create_dir_all(temp_dir.join("target")).unwrap();
    std::fs::create_dir_all(temp_dir.join("dist")).unwrap();
    std::fs::create_dir_all(temp_dir.join("examples")).unwrap();
    std::fs::create_dir_all(temp_dir.join("migrations")).unwrap();
    std::fs::create_dir_all(temp_dir.join("book")).unwrap();
    std::fs::create_dir_all(temp_dir.join("docs")).unwrap();
    std::fs::create_dir_all(temp_dir.join("fixtures")).unwrap();
    std::fs::create_dir_all(temp_dir.join("testdata")).unwrap();
    std::fs::create_dir_all(temp_dir.join(".venv")).unwrap();

    // All these should be excluded
    std::fs::write(temp_dir.join("__pycache__/module.pyc"), "").unwrap();
    std::fs::write(temp_dir.join("target/script.py"), "").unwrap();
    std::fs::write(temp_dir.join("dist/bundle.js"), "").unwrap();
    std::fs::write(temp_dir.join("examples/demo.py"), "").unwrap();
    std::fs::write(temp_dir.join("migrations/001.py"), "").unwrap();
    std::fs::write(temp_dir.join("book/highlight.js"), "").unwrap();
    std::fs::write(temp_dir.join("docs/api.js"), "").unwrap();
    std::fs::write(temp_dir.join("fixtures/sample.py"), "").unwrap();
    std::fs::write(temp_dir.join("testdata/input.py"), "").unwrap();
    std::fs::write(temp_dir.join(".venv/activate.py"), "").unwrap();

    let result = audit_scripting_files(&temp_dir);
    assert!(!result.has_violations());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_030_file_audit_result_debug() {
    let result = FileAuditResult {
        matches: vec![PathBuf::from("test.py")],
        scanned: 10,
        patterns: vec!["*.py".to_string()],
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("FileAuditResult"));
    assert!(debug.contains("test.py"));
}

#[test]
fn test_fals_aud_031_dependency_audit_result_debug() {
    let result = DependencyAuditResult {
        forbidden: vec!["pyo3".to_string()],
        checked: vec!["pyo3".to_string()],
        cargo_tree_output: Some("output".to_string()),
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("DependencyAuditResult"));
    assert!(debug.contains("pyo3"));
}

#[test]
fn test_fals_aud_032_wasm_support_debug() {
    let support = WasmSupport::default();
    let debug = format!("{:?}", support);
    assert!(debug.contains("WasmSupport"));
}

#[test]
fn test_fals_aud_033_serde_config_support_debug() {
    let support = SerdeConfigSupport::default();
    let debug = format!("{:?}", support);
    assert!(debug.contains("SerdeConfigSupport"));
}

#[test]
fn test_fals_aud_034_wasm_support_partial_bindgen() {
    // Test partial level with wasm-bindgen but no wasm module
    let support = WasmSupport {
        has_wasm_feature: false,
        has_wasm_bindgen: true,
        has_web_sys: false,
        has_wasm_pack: false,
        has_wasm_module: false,
        has_wasm_target: false,
    };
    assert_eq!(support.level(), "Partial");
}

#[test]
fn test_fals_aud_035_serde_config_with_deserialize() {
    let temp_dir = std::env::temp_dir().join("test_serde_deserialize");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1.0\"\n",
    )
    .unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Settings {
name: String,
}
"#,
    )
    .unwrap();

    let support = has_serde_config(&temp_dir);
    assert!(support.has_serde);
    assert!(support.has_deserialize_structs);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_036_serde_config_with_config_struct() {
    let temp_dir = std::env::temp_dir().join("test_serde_config_struct");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1.0\"\n",
    )
    .unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
use serde::Deserialize;

#[derive(Deserialize)]
pub struct AppConfig {
name: String,
}
"#,
    )
    .unwrap();

    let support = has_serde_config(&temp_dir);
    assert!(support.has_serde);
    assert!(support.has_config_struct);
    assert!(support.has_typed_config());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_037_test_frameworks_node_modules_excluded() {
    let temp_dir = std::env::temp_dir().join("test_frameworks_excluded");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("node_modules")).unwrap();
    std::fs::create_dir_all(temp_dir.join("venv")).unwrap();

    std::fs::write(temp_dir.join("node_modules/jest.config.js"), "").unwrap();
    std::fs::write(temp_dir.join("venv/conftest.py"), "").unwrap();

    let result = audit_test_frameworks(&temp_dir);
    assert!(!result.has_violations());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_038_clone_file_audit_result() {
    let result = FileAuditResult {
        matches: vec![PathBuf::from("a.py"), PathBuf::from("b.py")],
        scanned: 100,
        patterns: vec!["*.py".to_string(), "*.js".to_string()],
    };
    let cloned = result.clone();
    assert_eq!(cloned.matches.len(), 2);
    assert_eq!(cloned.scanned, 100);
    assert_eq!(cloned.patterns.len(), 2);
}

#[test]
fn test_fals_aud_039_clone_dependency_audit_result() {
    let result = DependencyAuditResult {
        forbidden: vec!["pyo3".to_string()],
        checked: vec!["pyo3".to_string(), "napi".to_string()],
        cargo_tree_output: Some("tree".to_string()),
    };
    let cloned = result.clone();
    assert_eq!(cloned.forbidden.len(), 1);
    assert_eq!(cloned.checked.len(), 2);
    assert!(cloned.cargo_tree_output.is_some());
}

#[test]
fn test_fals_aud_040_clone_wasm_support() {
    let support = WasmSupport {
        has_wasm_feature: true,
        has_wasm_bindgen: true,
        has_web_sys: false,
        has_wasm_pack: false,
        has_wasm_module: true,
        has_wasm_target: false,
    };
    let cloned = support.clone();
    assert_eq!(cloned.has_wasm_feature, support.has_wasm_feature);
    assert_eq!(cloned.has_wasm_module, support.has_wasm_module);
}

#[test]
fn test_fals_aud_041_clone_serde_config_support() {
    let support = SerdeConfigSupport {
        has_serde: true,
        has_serde_yaml: true,
        has_serde_json: false,
        has_toml: true,
        has_validator: false,
        has_deserialize_structs: true,
        has_config_struct: true,
    };
    let cloned = support.clone();
    assert_eq!(cloned.has_serde, support.has_serde);
    assert_eq!(cloned.has_toml, support.has_toml);
}

// =========================================================================
// Coverage: audit_cargo_dependencies - cargo tree & nonexistent Cargo.toml
// =========================================================================

#[test]
fn test_fals_aud_043_audit_cargo_deps_no_cargo_toml() {
    let temp_dir = std::env::temp_dir().join("test_audit_deps_no_cargo");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // No Cargo.toml at all => empty forbidden, cargo tree should also fail
    let result = audit_cargo_dependencies(&temp_dir, &["pyo3", "napi"]);
    assert!(!result.has_violations());
    assert!(result.checked.is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_044_audit_cargo_deps_quoted_format() {
    let temp_dir = std::env::temp_dir().join("test_audit_deps_quoted");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Use the "dep" format (quoted dependency name)
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"[package]
name = "test"
version = "0.1.0"

[dependencies]
"pyo3" = "0.20"
"#,
    )
    .unwrap();

    let result = audit_cargo_dependencies(&temp_dir, &["pyo3"]);
    assert!(result.has_violations());
    assert!(result.forbidden.contains(&"pyo3".to_string()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// Coverage: scan_deserialize_structs - glob error early return
// =========================================================================

#[test]
fn test_fals_aud_045_serde_config_no_src_dir() {
    // A path with Cargo.toml but no src/ dir => scan_deserialize_structs glob fails
    let temp_dir = std::env::temp_dir().join("test_serde_no_src");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1.0\"\n",
    )
    .unwrap();
    // Intentionally no src/ directory

    let support = has_serde_config(&temp_dir);
    assert!(support.has_serde);
    // No src dir to scan, so no deserialize structs found
    assert!(!support.has_deserialize_structs);
    assert!(!support.has_config_struct);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// Coverage: scan_deserialize_structs - file read error (unreadable file)
// =========================================================================

#[test]
fn test_fals_aud_046_serde_scan_deserialize_only_no_config() {
    // Has Deserialize but not a config struct
    let temp_dir = std::env::temp_dir().join("test_serde_deser_only");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1.0\"\n",
    )
    .unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
use serde::Deserialize;

#[derive(Deserialize)]
pub struct UserData {
    name: String,
}
"#,
    )
    .unwrap();

    let support = has_serde_config(&temp_dir);
    assert!(support.has_deserialize_structs);
    // "UserData" does not contain "config" (case-insensitive)
    assert!(!support.has_config_struct);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// =========================================================================
// Coverage: test_frameworks with actual test file matches
// =========================================================================

#[test]
fn test_fals_aud_047_test_frameworks_with_matches() {
    let temp_dir = std::env::temp_dir().join("test_fw_matches");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create actual test framework files that should be found
    std::fs::write(temp_dir.join("app.test.js"), "test('hello', () => {})").unwrap();
    std::fs::write(temp_dir.join("app.spec.ts"), "describe('app', () => {})").unwrap();
    std::fs::write(temp_dir.join("test_module.py"), "def test_hello(): pass").unwrap();
    std::fs::write(temp_dir.join("conftest.py"), "import pytest").unwrap();

    let result = audit_test_frameworks(&temp_dir);
    assert!(result.has_violations());
    assert!(result.violation_count() >= 3);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_042_wasm_support_with_web_feature() {
    let temp_dir = std::env::temp_dir().join("test_wasm_web_feature");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"[package]
name = "test"
version = "0.1.0"

[features]
web = []

[dependencies]
web-sys = "0.3"
wasm-pack = "0.1"
"#,
    )
    .unwrap();

    let support = has_wasm_support(&temp_dir);
    assert!(support.has_wasm_feature);
    assert!(support.has_web_sys);
    assert!(support.has_wasm_pack);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fals_aud_048_serde_config_with_yaml_ng_variant() {
    let temp_dir = std::env::temp_dir().join("test_serde_yaml_ng");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"[package]
name = "test"
version = "0.1.0"

[dependencies]
serde = "1.0"
serde_yaml_ng = "0.10"
"#,
    )
    .unwrap();

    let support = has_serde_config(&temp_dir);
    assert!(support.has_serde);
    assert!(support.has_serde_yaml, "serde_yaml_ng should be detected as has_serde_yaml");

    let _ = std::fs::remove_dir_all(&temp_dir);
}
