use super::*;

#[test]
fn test_integration_pattern_library_workflow() {
    // Test full pattern library workflow: add, search, record outcome
    let mut lib = PatternLibrary::new();
    let encoder = ErrorEncoder::new();

    // Add a pattern
    let code = e0308();
    let diag = test_diagnostic(
        code.clone(),
        "mismatched types: expected `String`, found `i32`",
    );

    let source_code = "pub fn foo() -> String { 42 }";
    let embedding = encoder.encode(&diag, source_code);
    let template = FixTemplate::new("$expr.to_string()", "Convert to String").with_code("E0308");

    lib.add_pattern(embedding.clone(), template);
    assert_eq!(lib.len(), 1);

    // Search for similar pattern
    let results = lib.search(&embedding, 5);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].similarity > 0.99,
        "Same embedding should have high similarity"
    );

    // Record outcome
    lib.record_outcome(0, true);
    lib.record_outcome(0, true);
    lib.record_outcome(0, false);

    let pattern = lib.get(0).expect("Pattern should exist");
    assert_eq!(pattern.success_count, 3); // 1 initial + 2 recorded
    assert_eq!(pattern.failure_count, 1);
}

#[test]
fn test_integration_citl_full_pipeline() {
    // Test full CITL pipeline: compile -> diagnose -> suggest
    let mut citl = test_citl_with_max_iter(5);

    // Code with type error (integer where String expected)
    let code = "pub fn foo() -> String { 42 }";

    // Compile and check for errors
    let compiler = citl.compiler();
    let result = compiler.compile(code, &CompileOptions::default());
    let compilation = result.expect("Should return result");

    assert!(!compilation.is_success());
    assert!(compilation.error_count() > 0);

    // Get the first error and try suggesting a fix
    let errors = compilation.errors();
    let first_error = &errors[0];

    // Suggest fix (requires diagnostic and source)
    let _suggestion = citl.suggest_fix(first_error, code);
    // May or may not have a suggestion depending on pattern library state
    // The important thing is that the pipeline doesn't panic

    // Try fix_all
    let fix_result = citl.fix_all(code);
    // Won't fully fix since we don't have the exact pattern, but should not panic
    assert!(fix_result.iterations <= 5);
}

#[test]
fn test_integration_cargo_mode_compiles_with_deps() {
    // Test Cargo mode can compile code with dependencies
    let project = CargoProject::new("citl_integration_test")
        .edition(RustEdition::E2021)
        .write_to_temp()
        .expect("Should create project");

    let manifest = project.manifest_path().expect("Has manifest");
    let compiler = RustCompiler::new().mode(CompilationMode::CargoCheck {
        manifest_path: manifest.clone(),
    });

    // Valid code
    let code = r"
pub fn multiply(a: f64, b: f64) -> f64 {
    a * b
}

pub fn is_positive(n: i32) -> bool {
    n > 0
}
";

    let result = compiler.compile(code, &CompileOptions::default());
    assert!(result.is_ok());

    let compilation = result.expect("Should return result");
    assert!(
        compilation.is_success(),
        "Valid code in cargo mode should compile"
    );
}

#[test]
fn test_integration_cargo_mode_detects_type_errors() {
    // Test Cargo mode detects type errors
    let project = CargoProject::new("citl_integration_error")
        .edition(RustEdition::E2021)
        .write_to_temp()
        .expect("Should create project");

    let manifest = project.manifest_path().expect("Has manifest");
    let compiler = RustCompiler::new().mode(CompilationMode::CargoCheck {
        manifest_path: manifest.clone(),
    });

    // Code with type error
    let code = "pub fn bad() -> String { 42 }";

    let result = compiler.compile(code, &CompileOptions::default());
    assert!(result.is_ok());

    let compilation = result.expect("Should return result");
    assert!(!compilation.is_success());
    assert!(compilation.error_count() > 0);
}

#[test]
fn test_integration_similar_errors_produce_similar_embeddings() {
    // Test that similar errors produce similar embeddings
    let encoder = ErrorEncoder::new();
    let source_code = "let x: String = 42;";

    let diag1 = test_diagnostic(
        e0308(),
        "mismatched types: expected `String`, found `i32`",
    );

    let diag2 = test_diagnostic(
        e0308(),
        "mismatched types: expected `String`, found `u32`",
    );

    let emb1 = encoder.encode(&diag1, source_code);
    let emb2 = encoder.encode(&diag2, source_code);

    let similarity = emb1.cosine_similarity(&emb2);
    assert!(
        similarity > 0.8,
        "Similar errors should have high similarity, got {similarity}"
    );
}

#[test]
fn test_integration_different_errors_produce_different_embeddings() {
    // Test that different error types produce different embeddings
    let encoder = ErrorEncoder::new();
    let source1 = "let x: String = 42;";
    let source2 = "let x = String::new(); let y = x; let z = x;";

    let diag1 = test_diagnostic(e0308(), "mismatched types");

    let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
    let diag2 = test_diagnostic(code2, "use of moved value");

    let emb1 = encoder.encode(&diag1, source1);
    let emb2 = encoder.encode(&diag2, source2);

    let similarity = emb1.cosine_similarity(&emb2);
    assert!(
        similarity < 0.9,
        "Different errors should have lower similarity, got {similarity}"
    );
}

// ==================== Additional Coverage Tests ====================

#[test]
fn test_citl_builder_default() {
    let builder = CITLBuilder::default();
    // Default builder has no compiler set
    let result = builder.build();
    assert!(result.is_err());
}

#[test]
fn test_citl_builder_pattern_library_path() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .pattern_library("/nonexistent/path/patterns.db")
        .build()
        .expect("Should build with nonexistent path (creates new library)");
    // Pattern library should be empty (new) since path doesn't exist
    assert!(citl.pattern_library.is_empty());
}

#[test]
fn test_citl_add_pattern_self_training_disabled() {
    let mut citl = test_citl_mut();

    // Disable self-training
    citl.config.enable_self_training = false;

    // Try to add a pattern
    let embedding = ErrorEmbedding::new(vec![0.0; 256], e0308(), 12345);
    let fix = FixTemplate::new("$expr.to_string()", "Convert to String");

    citl.add_pattern(embedding, fix, true);

    // Pattern should NOT be added when self-training is disabled
    assert!(citl.pattern_library.is_empty());
}

#[test]
fn test_citl_add_pattern_unsuccessful_fix() {
    let mut citl = test_citl_mut();

    // Self-training enabled but fix was unsuccessful
    let embedding = ErrorEmbedding::new(vec![0.0; 256], e0308(), 12345);
    let fix = FixTemplate::new("$expr.to_string()", "Convert to String");

    citl.add_pattern(embedding, fix, false); // success = false

    // Pattern should NOT be added for unsuccessful fix
    assert!(citl.pattern_library.is_empty());
}

#[test]
fn test_apply_fix_start_offset_out_of_bounds() {
    let citl = test_citl();

    let source = "let x = 42;";
    let fix =
        SuggestedFix::new("replacement".to_string(), 0.9, "Test".to_string()).with_span(100, 105); // start >= source.len()

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, source); // Should return original source unchanged
}

#[test]
fn test_apply_fix_end_offset_out_of_bounds() {
    let citl = test_citl();

    let source = "let x = 42;";
    let fix =
        SuggestedFix::new("replacement".to_string(), 0.9, "Test".to_string()).with_span(0, 100); // end > source.len()

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, source); // Should return original source unchanged
}

#[test]
fn test_suggested_fix_with_error_code() {
    let fix = SuggestedFix::new("fixed".to_string(), 0.8, "Description".to_string())
        .with_error_code("E0308");
    assert_eq!(fix.error_code, "E0308");
}

#[test]
fn test_suggested_fix_with_span_and_error_code() {
    let fix = SuggestedFix::new("fixed".to_string(), 0.8, "Description".to_string())
        .with_span(10, 20)
        .with_error_code("E0382");
    assert_eq!(fix.start_offset, 10);
    assert_eq!(fix.end_offset, 20);
    assert_eq!(fix.error_code, "E0382");
}

#[test]
fn test_error_code_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let code2 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let code3 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);

    set.insert(code1.clone());
    set.insert(code2.clone());
    set.insert(code3.clone());

    // code1 and code2 are equal, so only 2 entries
    assert_eq!(set.len(), 2);
}

#[test]
fn test_error_category_clone_and_debug() {
    let cat = ErrorCategory::TypeMismatch;
    let cloned = cat.clone();
    assert_eq!(cat, cloned);
    let debug_str = format!("{:?}", cat);
    assert!(debug_str.contains("TypeMismatch"));
}

#[test]
fn test_difficulty_clone_and_debug() {
    let diff = Difficulty::Hard;
    let cloned = diff.clone();
    assert_eq!(diff, cloned);
    let debug_str = format!("{:?}", diff);
    assert!(debug_str.contains("Hard"));
}

#[test]
fn test_language_clone_debug_hash() {
    use std::collections::HashSet;

    let lang = Language::Python;
    let cloned = lang.clone();
    assert_eq!(lang, cloned);

    let debug_str = format!("{:?}", lang);
    assert!(debug_str.contains("Python"));

    let mut set = HashSet::new();
    set.insert(Language::Python);
    set.insert(Language::C);
    set.insert(Language::Rust);
    assert_eq!(set.len(), 3);
}

#[test]
fn test_citl_config_clone_and_debug() {
    let config = CITLConfig::default();
    let cloned = config.clone();
    assert_eq!(config.max_iterations, cloned.max_iterations);
    assert!((config.confidence_threshold - cloned.confidence_threshold).abs() < f32::EPSILON);
    assert_eq!(config.enable_self_training, cloned.enable_self_training);

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("max_iterations"));
}

#[test]
fn test_suggested_fix_debug_clone() {
    let fix = SuggestedFix::new("code".to_string(), 0.9, "desc".to_string());
    let cloned = fix.clone();
    assert_eq!(fix.replacement, cloned.replacement);
    assert!((fix.confidence - cloned.confidence).abs() < f32::EPSILON);

    let debug_str = format!("{:?}", fix);
    assert!(debug_str.contains("replacement"));
}

#[test]
fn test_fix_result_debug_clone() {
    let result = FixResult::success("code".to_string(), 1);
    let cloned = result.clone();
    assert_eq!(result.success, cloned.success);
    assert_eq!(result.iterations, cloned.iterations);

    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("success"));
}

#[test]
fn test_instantiate_template_with_expected_type() {
    let citl = test_citl();

    let template = FixTemplate::new("$expr as $type", "Type cast");
    let mut diag = test_diagnostic(e0308(), "mismatched types");
    diag.expected = Some(TypeInfo::new("String"));

    let result = citl.instantiate_template(&template, &diag, "let x = 42;");
    assert!(result.contains("String"));
}

#[test]
fn test_instantiate_template_with_found_type() {
    let citl = test_citl();

    let template = FixTemplate::new("convert $found to target", "Type conversion");
    let mut diag = test_diagnostic(e0308(), "mismatched types");
    diag.found = Some(TypeInfo::new("i32"));

    let result = citl.instantiate_template(&template, &diag, "let x = 42;");
    assert!(result.contains("i32"));
}
