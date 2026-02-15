
#[test]
fn test_extract_top_level_json_value_before_children() {
    let json = r#"{"level":"error","children":[{"level":"note"}]}"#;
    let children_start = json.find("\"children\":").expect("Should find");
    let result = extract_top_level_json_value(json, "\"level\":\"", children_start);
    assert_eq!(result, Some("error".to_string()));
}

#[test]
fn test_extract_value_from_segment() {
    let segment = r#""key":"value", other"#;
    assert_eq!(
        extract_value_from_segment(segment, "\"key\":\""),
        Some("value".to_string())
    );
    assert!(extract_value_from_segment(segment, "\"nonexistent\":\"").is_none());
}

#[test]
fn test_extract_json_string_with_children() {
    // Test that we get top-level level/message, not from children
    let json =
        r#"{"level":"error","children":[{"level":"note","message":"inner"}],"message":"outer"}"#;
    assert_eq!(
        extract_json_string(json, "level"),
        Some("error".to_string())
    );
}

#[test]
fn test_rust_compiler_default() {
    let compiler = RustCompiler::default();
    assert_eq!(compiler.name(), "rustc");
}

#[test]
fn test_rust_compiler_parse_diagnostic() {
    let compiler = RustCompiler::new();
    let json = r#"{"level":"error","message":"test message","code":{"code":"E0308"}}"#;
    let diag = compiler.parse_diagnostic(json);
    assert!(diag.is_some());
    let d = diag.expect("Should parse");
    assert_eq!(d.severity, DiagnosticSeverity::Error);
    assert_eq!(d.message, "test message");
}

#[test]
fn test_rust_compiler_parse_diagnostic_invalid() {
    let compiler = RustCompiler::new();
    let diag = compiler.parse_diagnostic("not json");
    assert!(diag.is_none());
}

#[test]
fn test_rust_compiler_parse_json_diagnostics() {
    let compiler = RustCompiler::new();
    let output = r#"{"level":"error","message":"type mismatch","code":{"code":"E0308"}}
{"level":"warning","message":"unused variable","code":{"code":"unused"}}"#;

    let (errors, warnings) = compiler.parse_json_diagnostics(output);
    assert_eq!(errors.len(), 1);
    assert_eq!(warnings.len(), 1);
}

#[test]
fn test_rust_compiler_parse_json_diagnostics_skip_aborting() {
    let compiler = RustCompiler::new();
    let output =
        r#"{"level":"error","message":"aborting due to 1 previous error","code":{"code":"E0000"}}"#;

    let (errors, warnings) = compiler.parse_json_diagnostics(output);
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

#[test]
fn test_compile_options_with_values() {
    let mut options = CompileOptions::default();
    options.opt_level = OptLevel::Release;
    options.extra_flags.push("-v".to_string());
    options
        .env
        .insert("RUST_LOG".to_string(), "debug".to_string());
    options.working_dir = Some(PathBuf::from("/tmp"));

    assert_eq!(options.opt_level, OptLevel::Release);
    assert_eq!(options.extra_flags.len(), 1);
    assert!(options.working_dir.is_some());
}

#[test]
fn test_compilation_metrics_with_values() {
    let metrics = CompilationMetrics {
        duration: Duration::from_secs(5),
        memory_bytes: Some(1024 * 1024),
        units: 3,
    };

    assert_eq!(metrics.duration, Duration::from_secs(5));
    assert_eq!(metrics.memory_bytes, Some(1024 * 1024));
    assert_eq!(metrics.units, 3);
}

#[test]
fn test_lookup_error_code_known() {
    let code = lookup_error_code("E0308");
    assert_eq!(code.code, "E0308");
}

#[test]
fn test_lookup_error_code_unknown() {
    let code = lookup_error_code("E9999");
    assert_eq!(code.code, "E9999");
}

#[test]
fn test_cargo_project_generate_cargo_toml() {
    let project = CargoProject::new("test_proj")
        .edition(RustEdition::E2021)
        .dependency("serde", "1.0");

    let toml = project.generate_cargo_toml();
    assert!(toml.contains("name = \"test_proj\""));
    assert!(toml.contains("edition = \"2021\""));
    assert!(toml.contains("serde = \"1.0\""));
}

#[test]
fn test_compiler_version_with_commit() {
    let version = CompilerVersion {
        major: 1,
        minor: 80,
        patch: 0,
        full: "rustc 1.80.0 (abc123)".to_string(),
        commit: Some("abc123".to_string()),
    };

    assert_eq!(version.commit, Some("abc123".to_string()));
    assert_eq!(format!("{version}"), "1.80.0");
}

// ==================== Coverage: CompilerVersion PartialEq ====================

#[test]
fn test_compiler_version_equality() {
    let v1 = CompilerVersion {
        major: 1,
        minor: 75,
        patch: 0,
        full: "1.75.0".to_string(),
        commit: None,
    };
    let v2 = CompilerVersion {
        major: 1,
        minor: 75,
        patch: 0,
        full: "1.75.0".to_string(),
        commit: None,
    };
    let v3 = CompilerVersion {
        major: 1,
        minor: 80,
        patch: 0,
        full: "1.80.0".to_string(),
        commit: None,
    };
    assert_eq!(v1, v2);
    assert_ne!(v1, v3);
}

// ==================== Coverage: CompilerVersion parse edge cases ====================

#[test]
fn test_compiler_version_parse_with_pre_release() {
    let version = CompilerVersion::parse("1.82.0-beta.1").expect("Should parse");
    assert_eq!(version.major, 1);
    assert_eq!(version.minor, 82);
    assert_eq!(version.patch, 0);
}

// ==================== Coverage: RustEdition Debug ====================

#[test]
fn test_rust_edition_debug() {
    let edition = RustEdition::E2024;
    let debug_str = format!("{:?}", edition);
    assert!(debug_str.contains("E2024"));
}

// ==================== Coverage: CompilationMode Debug ====================

#[test]
fn test_compilation_mode_debug() {
    let standalone = CompilationMode::Standalone;
    let debug_str = format!("{:?}", standalone);
    assert!(debug_str.contains("Standalone"));

    let cargo = CompilationMode::Cargo {
        manifest_path: PathBuf::from("/tmp/Cargo.toml"),
    };
    let debug_str = format!("{:?}", cargo);
    assert!(debug_str.contains("Cargo"));

    let check = CompilationMode::CargoCheck {
        manifest_path: PathBuf::from("/tmp/Cargo.toml"),
    };
    let debug_str = format!("{:?}", check);
    assert!(debug_str.contains("CargoCheck"));
}

// ==================== Coverage: CompileOptions Debug ====================

#[test]
fn test_compile_options_debug() {
    let options = CompileOptions::default();
    let debug_str = format!("{:?}", options);
    assert!(debug_str.contains("CompileOptions"));
    assert!(debug_str.contains("Debug"));
}

// ==================== Coverage: CompiledArtifact Debug/Clone ====================

#[test]
fn test_compiled_artifact_debug_clone() {
    let artifact = CompiledArtifact {
        artifact_type: ArtifactType::Wasm,
        path: None,
        size: 0,
    };
    let debug_str = format!("{:?}", artifact);
    assert!(debug_str.contains("Wasm"));

    let cloned = artifact.clone();
    assert_eq!(cloned.artifact_type, ArtifactType::Wasm);
    assert!(cloned.path.is_none());
    assert_eq!(cloned.size, 0);
}

// ==================== Coverage: CompilationMetrics Debug/Clone ====================

#[test]
fn test_compilation_metrics_debug_clone() {
    let metrics = CompilationMetrics {
        duration: Duration::from_millis(500),
        memory_bytes: Some(4096),
        units: 2,
    };
    let debug_str = format!("{:?}", metrics);
    assert!(debug_str.contains("CompilationMetrics"));

    let cloned = metrics.clone();
    assert_eq!(cloned.duration, Duration::from_millis(500));
    assert_eq!(cloned.memory_bytes, Some(4096));
    assert_eq!(cloned.units, 2);
}

// ==================== Coverage: CompilationResult Clone ====================

#[test]
fn test_compilation_result_clone_success() {
    let result = CompilationResult::Success {
        artifact: Some(CompiledArtifact {
            artifact_type: ArtifactType::StaticLib,
            path: Some(PathBuf::from("/tmp/lib.a")),
            size: 2048,
        }),
        warnings: vec![],
        metrics: CompilationMetrics::default(),
    };
    let cloned = result.clone();
    assert!(cloned.is_success());
    assert_eq!(cloned.error_count(), 0);
}

#[test]
fn test_compilation_result_clone_failure() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let error = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "test", span);

    let result = CompilationResult::Failure {
        errors: vec![error],
        warnings: vec![],
        raw_output: "raw output".to_string(),
    };
    let cloned = result.clone();
    assert!(!cloned.is_success());
    assert_eq!(cloned.error_count(), 1);
}

// ==================== Coverage: parse_single_json_diagnostic severity variants ====================

#[test]
fn test_parse_diagnostic_warning() {
    let compiler = RustCompiler::new();
    let json = r#"{"level":"warning","message":"unused variable","code":{"code":"W0001"}}"#;
    let diag = compiler.parse_diagnostic(json);
    assert!(diag.is_some());
    let d = diag.expect("Should parse");
    assert_eq!(d.severity, DiagnosticSeverity::Warning);
}

#[test]
fn test_parse_diagnostic_note() {
    let compiler = RustCompiler::new();
    let json = r#"{"level":"note","message":"consider using","code":{"code":"N0001"}}"#;
    let diag = compiler.parse_diagnostic(json);
    assert!(diag.is_some());
    let d = diag.expect("Should parse");
    assert_eq!(d.severity, DiagnosticSeverity::Note);
}

#[test]
fn test_parse_diagnostic_help() {
    let compiler = RustCompiler::new();
    let json = r#"{"level":"help","message":"try adding a reference","code":{"code":"H0001"}}"#;
    let diag = compiler.parse_diagnostic(json);
    assert!(diag.is_some());
    let d = diag.expect("Should parse");
    assert_eq!(d.severity, DiagnosticSeverity::Help);
}

#[test]
fn test_parse_diagnostic_unknown_level() {
    let compiler = RustCompiler::new();
    let json = r#"{"level":"ice","message":"internal error","code":{"code":"ICE"}}"#;
    let diag = compiler.parse_diagnostic(json);
    assert!(diag.is_none());
}

// ==================== Coverage: parse_single_json_diagnostic with expected/found ====================

#[test]
fn test_parse_diagnostic_with_expected_found() {
    let compiler = RustCompiler::new();
    let json = r#"{"level":"error","message":"mismatched types","code":{"code":"E0308"},"expected":"i32","found":"String"}"#;
    let diag = compiler.parse_diagnostic(json);
    assert!(diag.is_some());
    let d = diag.expect("Should parse");
    assert!(d.expected.is_some());
    assert!(d.found.is_some());
}

// ==================== Coverage: parse_single_json_diagnostic missing code ====================

#[test]
fn test_parse_diagnostic_no_code() {
    let compiler = RustCompiler::new();
    let json = r#"{"level":"error","message":"some error"}"#;
    let diag = compiler.parse_diagnostic(json);
    assert!(diag.is_some());
    let d = diag.expect("Should parse");
    assert_eq!(d.code.code, "unknown");
}

// ==================== Coverage: extract_json_string without children ====================

#[test]
fn test_extract_json_string_level_no_children() {
    let json = r#"{"level":"warning","message":"test warning"}"#;
    assert_eq!(
        extract_json_string(json, "level"),
        Some("warning".to_string())
    );
    assert_eq!(
        extract_json_string(json, "message"),
        Some("test warning".to_string())
    );
}

// ==================== Coverage: extract_json_string for non-level/message keys ====================

#[test]
fn test_extract_json_string_non_special_key() {
    let json = r#"{"level":"error","file_name":"test.rs","children":[{"level":"note"}]}"#;
    assert_eq!(
        extract_json_string(json, "file_name"),
        Some("test.rs".to_string())
    );
}

// ==================== Coverage: extract_top_level_json_value after children ====================

#[test]
fn test_extract_top_level_json_value_after_children() {
    // In cargo format, the top-level "message" can come after "children"
    let json = r#"{"children":[{"message":"inner"}],"message":"outer"}"#;
    let children_start = json.find("\"children\":").expect("Should find");
    let result = extract_top_level_json_value(json, "\"message\":\"", children_start);
    assert_eq!(result, Some("outer".to_string()));
}

// ==================== Coverage: extract_nested_json_string missing outer key ====================

#[test]
fn test_extract_nested_json_string_missing_key() {
    let json = r#"{"other":"value"}"#;
    assert!(extract_nested_json_string(json, "code", "code").is_none());
}

// ==================== Coverage: find_children_array_end deeply nested ====================

#[test]
fn test_find_children_array_end_empty_array() {
    let json = r#"{"children":[]}"#;
    let start = json.find("\"children\":").expect("Should find");
    let end = find_children_array_end(json, start);
    assert!(end > start);
    assert!(end <= json.len());
}

// ==================== Coverage: CargoProject name() accessor ====================

#[test]
fn test_cargo_project_name_accessor() {
    let project = CargoProject::new("my_project");
    assert_eq!(project.name(), "my_project");
}

// ==================== Coverage: CargoProject edition builder ====================

#[test]
fn test_cargo_project_edition_builder() {
    let project = CargoProject::new("test").edition(RustEdition::E2018);
    let toml = project.generate_cargo_toml();
    assert!(toml.contains("edition = \"2018\""));
}

// ==================== Coverage: CargoProject project_dir before write ====================

#[test]
fn test_cargo_project_dir_before_write() {
    let project = CargoProject::new("unwritten");
    assert!(project.project_dir().is_none());
    assert!(project.manifest_path().is_none());
}

// ==================== Coverage: CargoProject generate_cargo_toml with no deps ====================

#[test]
fn test_cargo_project_toml_no_deps() {
    let project = CargoProject::new("nodeps").edition(RustEdition::E2021);
    let toml = project.generate_cargo_toml();
    assert!(toml.contains("name = \"nodeps\""));
    assert!(toml.contains("edition = \"2021\""));
    assert!(toml.contains("[dependencies]"));
}
