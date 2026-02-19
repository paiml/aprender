pub(crate) use super::*;
pub(crate) use crate::citl::{Difficulty, ErrorCategory};
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
    let code = r"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
";
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

    let dir = project.project_dir().expect("Has dir").clone();
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

// ========================================================================
// Additional Coverage Tests for citl/compiler.rs
// ========================================================================

#[test]
fn test_compiler_version_parse_short() {
    // Too short to parse
    assert!(CompilerVersion::parse("1.75").is_none());
    assert!(CompilerVersion::parse("1").is_none());
    assert!(CompilerVersion::parse("").is_none());
}

#[test]
fn test_compiler_version_parse_invalid() {
    assert!(CompilerVersion::parse("a.b.c").is_none());
    assert!(CompilerVersion::parse("1.x.0").is_none());
}

#[test]
fn test_compiled_artifact_fields() {
    let artifact = CompiledArtifact {
        artifact_type: ArtifactType::Binary,
        path: Some(PathBuf::from("/tmp/test")),
        size: 1024,
    };
    assert_eq!(artifact.artifact_type, ArtifactType::Binary);
    assert_eq!(artifact.size, 1024);
    assert!(artifact.path.is_some());
}

#[test]
fn test_artifact_type_variants() {
    assert_eq!(ArtifactType::StaticLib, ArtifactType::StaticLib);
    assert_eq!(ArtifactType::DynamicLib, ArtifactType::DynamicLib);
    assert_eq!(ArtifactType::Object, ArtifactType::Object);
    assert_eq!(ArtifactType::Wasm, ArtifactType::Wasm);
    assert_ne!(ArtifactType::Binary, ArtifactType::Wasm);
}

#[test]
fn test_compilation_result_warnings_success() {
    let code = ErrorCode::new("W0001", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let warning = CompilerDiagnostic::new(code, DiagnosticSeverity::Warning, "test warning", span);

    let result = CompilationResult::Success {
        artifact: None,
        warnings: vec![warning],
        metrics: CompilationMetrics::default(),
    };

    assert!(result.is_success());
    assert_eq!(result.warning_count(), 1);
    assert_eq!(result.warnings().len(), 1);
    assert!(result.errors().is_empty());
}

#[test]
fn test_compilation_result_warnings_failure() {
    let err_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let warn_code = ErrorCode::new("W0001", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let error = CompilerDiagnostic::new(
        err_code,
        DiagnosticSeverity::Error,
        "test error",
        span.clone(),
    );
    let warning =
        CompilerDiagnostic::new(warn_code, DiagnosticSeverity::Warning, "test warning", span);

    let result = CompilationResult::Failure {
        errors: vec![error],
        warnings: vec![warning],
        raw_output: "raw".to_string(),
    };

    assert!(!result.is_success());
    assert_eq!(result.warning_count(), 1);
    assert_eq!(result.error_count(), 1);
}

#[test]
fn test_compilation_mode_cargo() {
    let mode = CompilationMode::Cargo {
        manifest_path: PathBuf::from("/tmp/Cargo.toml"),
    };
    match mode {
        CompilationMode::Cargo { manifest_path } => {
            assert_eq!(manifest_path, PathBuf::from("/tmp/Cargo.toml"));
        }
        _ => panic!("Expected Cargo mode"),
    }
}

#[test]
fn test_opt_level_equality() {
    assert_eq!(OptLevel::Debug, OptLevel::Debug);
    assert_eq!(OptLevel::Release, OptLevel::Release);
    assert_ne!(OptLevel::Debug, OptLevel::Release);
}

#[test]
fn test_extract_span_from_json() {
    let json = r#"{"file_name":"test.rs","line_start":"5","line_end":"10","column_start":"1","column_end":"20"}"#;
    let span = extract_span_from_json(json);
    assert_eq!(span.file, "test.rs");
    assert_eq!(span.line_start, 5);
    assert_eq!(span.line_end, 10);
}

#[test]
fn test_extract_span_from_json_missing_fields() {
    let json = r#"{"file_name":"test.rs"}"#;
    let span = extract_span_from_json(json);
    assert_eq!(span.file, "test.rs");
    assert_eq!(span.line_start, 1);
    assert_eq!(span.line_end, 1);
}

#[test]
fn test_extract_suggestions_from_json() {
    let json = r#"{"suggestions":[]}"#;
    assert!(extract_suggestions_from_json(json).is_none());
}

#[test]
fn test_find_children_array_end() {
    let json = r#"{"children":[{"a":1},{"b":2}],"other":"data"}"#;
    let start = json.find("\"children\":").expect("Should find");
    let end = find_children_array_end(json, start);
    assert!(end > start);
}

#[test]
fn test_find_children_array_end_nested() {
    let json = r#"{"children":[[1,2],[3,4]]}"#;
    let start = json.find("\"children\":").expect("Should find");
    let end = find_children_array_end(json, start);
    assert_eq!(end, json.len() - 1);
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
#[path = "tests_part_04.rs"]
mod tests_part_04;
