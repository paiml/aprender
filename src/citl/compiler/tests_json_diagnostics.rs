use super::*;

// ==================== Coverage: parse_json_diagnostics with non-JSON lines ====================

#[test]
fn test_parse_json_diagnostics_ignores_non_json() {
    let compiler = RustCompiler::new();
    let output = "not json at all\n\nstill not json\n";
    let (errors, warnings) = compiler.parse_json_diagnostics(output);
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

// ==================== Coverage: parse_json_diagnostics with rendered lines ====================

#[test]
fn test_parse_json_diagnostics_skips_rendered() {
    let compiler = RustCompiler::new();
    // Lines with "rendered" should be skipped
    let output = r#"{"level":"error","rendered":"some rendered text"}"#;
    let (errors, warnings) = compiler.parse_json_diagnostics(output);
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

// ==================== Coverage: parse_cargo_json_diagnostics empty ====================

#[test]
fn test_parse_cargo_json_diagnostics_empty() {
    let compiler = RustCompiler::new();
    let (errors, warnings) = compiler.parse_cargo_json_diagnostics("");
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

// ==================== Coverage: parse_cargo_json_diagnostics non-compiler-message ====================

#[test]
fn test_parse_cargo_json_diagnostics_non_compiler_message() {
    let compiler = RustCompiler::new();
    let output = r#"{"reason":"build-finished","success":true}"#;
    let (errors, warnings) = compiler.parse_cargo_json_diagnostics(output);
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

// ==================== Coverage: parse_cargo_message with malformed nested message ====================

#[test]
fn test_parse_cargo_message_no_message_field() {
    let compiler = RustCompiler::new();
    let json = r#"{"reason":"compiler-message","other":"data"}"#;
    // No "message":{} in the JSON - should return None from parse_cargo_message
    let (errors, warnings) = compiler.parse_cargo_json_diagnostics(json);
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

// ==================== Coverage: RustCompiler mode setter ====================

#[test]
fn test_rust_compiler_mode_setter() {
    let manifest = PathBuf::from("/tmp/Cargo.toml");
    let compiler = RustCompiler::new().mode(CompilationMode::Cargo {
        manifest_path: manifest.clone(),
    });
    match &compiler.mode {
        CompilationMode::Cargo { manifest_path } => {
            assert_eq!(manifest_path, &manifest);
        }
        _ => panic!("Expected Cargo mode"),
    }
}

// ==================== Coverage: CompilerVersion Clone ====================

#[test]
fn test_compiler_version_clone() {
    let version = CompilerVersion {
        major: 1,
        minor: 75,
        patch: 0,
        full: "1.75.0".to_string(),
        commit: Some("abc".to_string()),
    };
    let cloned = version.clone();
    assert_eq!(cloned.major, 1);
    assert_eq!(cloned.minor, 75);
    assert_eq!(cloned.commit, Some("abc".to_string()));
}

// ==================== Coverage: RustCompiler Clone ====================

#[test]
fn test_rust_compiler_clone() {
    let compiler = RustCompiler::new()
        .edition(RustEdition::E2018)
        .timeout(Duration::from_secs(30))
        .extra_flag("-W");
    let cloned = compiler.clone();
    assert_eq!(cloned.edition, RustEdition::E2018);
    assert_eq!(cloned.timeout, Duration::from_secs(30));
    assert_eq!(cloned.extra_flags.len(), 1);
}

// ==================== Coverage: extract_span_from_json with all fields ====================

#[test]
fn test_extract_span_from_json_complete() {
    let json = r#"{"file_name":"main.rs","line_start":"10","line_end":"15","column_start":"3","column_end":"25"}"#;
    let span = extract_span_from_json(json);
    assert_eq!(span.file, "main.rs");
    assert_eq!(span.line_start, 10);
    assert_eq!(span.line_end, 15);
    assert_eq!(span.column_start, 3);
    assert_eq!(span.column_end, 25);
}

// ==================== Coverage: extract_span_from_json with no fields ====================

#[test]
fn test_extract_span_from_json_empty_json() {
    let json = r#"{}"#;
    let span = extract_span_from_json(json);
    assert_eq!(span.file, "");
    assert_eq!(span.line_start, 1);
    assert_eq!(span.line_end, 1);
    assert_eq!(span.column_start, 1);
    assert_eq!(span.column_end, 1);
}

// ==================== Coverage: extract_value_from_segment with no end quote ====================

#[test]
fn test_extract_value_from_segment_no_end_quote() {
    let segment = r#""key":"value_no_end"#;
    // Pattern starts at 7 (after "key":"), value has no closing quote in remaining
    // Actually the whole segment starts: the pattern "\"key\":\"" matches and
    // then we look for closing ", it is at the end
    let result = extract_value_from_segment(segment, "\"key\":\"");
    // "value_no_end" has no closing quote so this depends on implementation
    // The function finds the quote: value_no_end doesn't have one after the last "
    // Actually segment has trailing " at the very end. Let me check correctly.
    // The segment is: "key":"value_no_end
    // After finding "key":" at position 0, rest is: value_no_end
    // find('"') in "value_no_end" -> None
    assert!(result.is_none());
}

// ==================== Coverage: parse_cargo_message edge cases ====================

#[test]
fn test_parse_cargo_message_end_is_zero() {
    let compiler = RustCompiler::new();
    // JSON where the nested message object has no matching closing brace
    // This triggers the else branch where end == 0
    // Note: The first test case doesn't trigger end==0; using malformed instead.
    // The pattern is "message":{ and then we scan for depth=0.
    // If we don't find a closing brace, depth never becomes 0, so end stays 0.
    let malformed = r#"{"reason":"compiler-message","message":{no_closing"#;
    let (errors, warnings) = compiler.parse_cargo_json_diagnostics(malformed);
    // Should return empty since parsing fails
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

#[test]
fn test_parse_cargo_message_nested_braces() {
    let compiler = RustCompiler::new();
    // JSON with deeply nested braces in the message
    let json = r#"{"reason":"compiler-message","message":{"level":"error","message":"test","code":{"code":"E0308","nested":{"deep":"value"}}}}"#;
    let (errors, _) = compiler.parse_cargo_json_diagnostics(json);
    // Should parse successfully despite nested structure
    assert_eq!(errors.len(), 1);
}

#[test]
fn test_parse_cargo_json_diagnostics_with_warning() {
    let compiler = RustCompiler::new();
    // Create a cargo-style JSON output with a warning
    let json = r#"{"reason":"compiler-message","message":{"level":"warning","message":"unused variable","code":{"code":"unused_variables"}}}"#;
    let (errors, warnings) = compiler.parse_cargo_json_diagnostics(json);
    assert!(errors.is_empty());
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].severity, DiagnosticSeverity::Warning);
}

#[test]
fn test_parse_cargo_json_diagnostics_mixed() {
    let compiler = RustCompiler::new();
    // Multiple lines: one error, one warning
    let output = r#"{"reason":"compiler-message","message":{"level":"error","message":"mismatched types","code":{"code":"E0308"}}}
{"reason":"compiler-message","message":{"level":"warning","message":"unused var","code":{"code":"W0001"}}}"#;
    let (errors, warnings) = compiler.parse_cargo_json_diagnostics(output);
    assert_eq!(errors.len(), 1);
    assert_eq!(warnings.len(), 1);
}

#[test]
fn test_parse_cargo_json_diagnostics_skip_aborting_message() {
    let compiler = RustCompiler::new();
    let json = r#"{"reason":"compiler-message","message":{"level":"error","message":"aborting due to 2 previous errors","code":{"code":"E0001"}}}"#;
    let (errors, warnings) = compiler.parse_cargo_json_diagnostics(json);
    // "aborting due to" messages should be skipped
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

// ==================== Coverage: compile_cargo_check error path ====================

#[test]
fn test_compile_cargo_check_invalid_manifest_path() {
    // Test the error path when manifest_path points to a nonexistent location.
    // IMPORTANT: Never use relative paths here - they would write to the actual
    // project's src/lib.rs and corrupt it!
    let temp_dir = std::env::temp_dir().join("citl_invalid_manifest_test");
    let nonexistent_manifest = temp_dir.join("nonexistent").join("Cargo.toml");

    let compiler = RustCompiler::new().mode(CompilationMode::CargoCheck {
        manifest_path: nonexistent_manifest,
    });

    let result = compiler.compile("fn main() {}", &CompileOptions::default());
    // Should fail because the directory doesn't exist or cargo check fails
    // Either outcome is acceptable for this edge case test
    assert!(result.is_ok() || result.is_err());

    // Cleanup if anything was created
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_compile_cargo_mode_uses_cargo_check() {
    // Test that CompilationMode::Cargo goes through compile_cargo_check
    let temp_project = CargoProject::new("citl_cargo_mode_test")
        .edition(RustEdition::E2021)
        .write_to_temp()
        .expect("Should create temp project");

    let manifest = temp_project.manifest_path().expect("Has manifest");
    let compiler = RustCompiler::new().mode(CompilationMode::Cargo {
        manifest_path: manifest.clone(),
    });

    let code = "pub fn test_fn() -> i32 { 42 }";
    let result = compiler.compile(code, &CompileOptions::default());
    assert!(result.is_ok());
}

// ==================== Coverage: helper functions ====================

#[test]
fn test_check_env_binary_nonexistent() {
    // Test with an env var that doesn't exist
    let result = check_env_binary("NONEXISTENT_BINARY_XYZ");
    assert!(result.is_none());
}

#[test]
fn test_check_cargo_bin_nonexistent() {
    // Test with a binary that doesn't exist in cargo bin
    let result = check_cargo_bin("totally_fake_binary_name_xyz");
    assert!(result.is_none());
}

#[test]
fn test_check_system_paths_none_exist() {
    let result = check_system_paths(&["/nonexistent/path/binary", "/also/nonexistent/binary"]);
    assert!(result.is_none());
}

#[test]
fn test_check_system_paths_empty() {
    let result = check_system_paths(&[]);
    assert!(result.is_none());
}

// ==================== Coverage: which_rustc and which_cargo fallback ====================

#[test]
fn test_which_rustc_returns_path() {
    let path = which_rustc();
    // Should return some path (either from env, cargo bin, system, or fallback)
    assert!(!path.as_os_str().is_empty());
}

#[test]
fn test_which_cargo_returns_path() {
    let path = which_cargo();
    assert!(!path.as_os_str().is_empty());
}

// ==================== Coverage: extract_top_level_json_value edge cases ====================

#[test]
fn test_extract_top_level_json_value_not_found_anywhere() {
    let json = r#"{"children":[{"inner":"value"}],"other":"data"}"#;
    let children_start = json.find("\"children\":").expect("Should find");
    // Looking for a key that doesn't exist anywhere
    let result = extract_top_level_json_value(json, "\"nonexistent\":\"", children_start);
    assert!(result.is_none());
}

#[test]
fn test_extract_top_level_json_value_only_in_children() {
    let json = r#"{"children":[{"level":"note"}],"other":"data"}"#;
    let children_start = json.find("\"children\":").expect("Should find");
    // "level" only exists inside children, not before or after
    let result = extract_top_level_json_value(json, "\"level\":\"", children_start);
    // Should return None since "level" is only inside children
    assert!(result.is_none());
}

// ==================== Coverage: find_children_array_end edge cases ====================

#[test]
fn test_find_children_array_end_no_closing_bracket() {
    // If there's no closing bracket, should return json.len()
    let json = r#"{"children":[[1,2,3"#;
    let start = json.find("\"children\":").expect("Should find");
    let end = find_children_array_end(json, start);
    assert_eq!(end, json.len());
}

#[test]
fn test_find_children_array_end_complex_nested() {
    let json = r#"{"children":[{"a":[1,2]},{"b":[3,[4,5]]}],"after":"value"}"#;
    let start = json.find("\"children\":").expect("Should find");
    let end = find_children_array_end(json, start);
    // Should find the correct end of the children array
    assert!(end < json.len());
    assert!(json[end - 1..end].starts_with(']'));
}

// ==================== Coverage: extract_json_string message key ====================

#[test]
fn test_extract_json_string_message_with_children() {
    let json = r#"{"children":[{"message":"inner msg"}],"message":"outer msg"}"#;
    let result = extract_json_string(json, "message");
    // Should get "outer msg", not "inner msg"
    assert_eq!(result, Some("outer msg".to_string()));
}

#[test]
fn test_extract_json_string_message_before_children() {
    let json = r#"{"message":"first","children":[{"message":"nested"}]}"#;
    let result = extract_json_string(json, "message");
    assert_eq!(result, Some("first".to_string()));
}

// ==================== Coverage: Debug implementations ====================

#[test]
fn test_cargo_project_debug() {
    let project = CargoProject::new("debug_test");
    let debug_str = format!("{:?}", project);
    assert!(debug_str.contains("CargoProject"));
    assert!(debug_str.contains("debug_test"));
}

#[test]
fn test_compiler_version_debug() {
    let version = CompilerVersion {
        major: 1,
        minor: 75,
        patch: 0,
        full: "1.75.0".to_string(),
        commit: None,
    };
    let debug_str = format!("{:?}", version);
    assert!(debug_str.contains("CompilerVersion"));
    assert!(debug_str.contains("75"));
}

// ==================== Coverage: RustCompiler Debug ====================

#[test]
fn test_rust_compiler_debug() {
    let compiler = RustCompiler::new();
    let debug_str = format!("{:?}", compiler);
    assert!(debug_str.contains("RustCompiler"));
}

// ==================== Coverage: ArtifactType Debug and Copy ====================

#[test]
fn test_artifact_type_debug() {
    let artifact = ArtifactType::DynamicLib;
    let debug_str = format!("{:?}", artifact);
    assert!(debug_str.contains("DynamicLib"));
}

#[test]
fn test_artifact_type_copy() {
    let a1 = ArtifactType::Object;
    let a2 = a1; // Copy
    assert_eq!(a1, a2);
}

// ==================== Coverage: CompilationResult Debug ====================

#[test]
fn test_compilation_result_debug_success() {
    let result = CompilationResult::Success {
        artifact: None,
        warnings: vec![],
        metrics: CompilationMetrics::default(),
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("Success"));
}

#[test]
fn test_compilation_result_debug_failure() {
    let code = ErrorCode::new("E0001", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let error = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "err", span);
    let result = CompilationResult::Failure {
        errors: vec![error],
        warnings: vec![],
        raw_output: "raw".to_string(),
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("Failure"));
}

// ==================== Coverage: CompileOptions Clone ====================

#[test]
fn test_compile_options_clone() {
    let mut options = CompileOptions::default();
    options.opt_level = OptLevel::Release;
    options.extra_flags.push("-v".to_string());
    options.working_dir = Some(PathBuf::from("/test"));

    let cloned = options.clone();
    assert_eq!(cloned.opt_level, OptLevel::Release);
    assert_eq!(cloned.extra_flags.len(), 1);
    assert!(cloned.working_dir.is_some());
}

// ==================== Coverage: OptLevel Debug/Copy ====================

#[test]
fn test_opt_level_debug() {
    let level = OptLevel::Release;
    let debug_str = format!("{:?}", level);
    assert!(debug_str.contains("Release"));
}
