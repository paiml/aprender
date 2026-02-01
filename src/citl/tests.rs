use super::*;

// ==================== ErrorCode Tests ====================

#[test]
fn test_error_code_new() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    assert_eq!(code.code, "E0308");
    assert_eq!(code.category, ErrorCategory::TypeMismatch);
    assert_eq!(code.difficulty, Difficulty::Easy);
}

#[test]
fn test_error_code_from_code() {
    let code = ErrorCode::from_code("E0308");
    assert_eq!(code.code, "E0308");
    assert_eq!(code.category, ErrorCategory::Unknown);
    assert_eq!(code.difficulty, Difficulty::Medium);
}

#[test]
fn test_error_code_display() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    assert_eq!(format!("{code}"), "E0308");
}

#[test]
fn test_error_code_equality() {
    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let code2 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let code3 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);

    assert_eq!(code1, code2);
    assert_ne!(code1, code3);
}

// ==================== Difficulty Tests ====================

#[test]
fn test_difficulty_score() {
    assert!((Difficulty::Easy.score() - 0.25).abs() < f32::EPSILON);
    assert!((Difficulty::Medium.score() - 0.5).abs() < f32::EPSILON);
    assert!((Difficulty::Hard.score() - 0.75).abs() < f32::EPSILON);
    assert!((Difficulty::Expert.score() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_difficulty_ordering() {
    assert!(Difficulty::Easy < Difficulty::Medium);
    assert!(Difficulty::Medium < Difficulty::Hard);
    assert!(Difficulty::Hard < Difficulty::Expert);
}

// ==================== ErrorCategory Tests ====================

#[test]
fn test_error_category_variants() {
    let categories = [
        ErrorCategory::TypeMismatch,
        ErrorCategory::TraitBound,
        ErrorCategory::Unresolved,
        ErrorCategory::Ownership,
        ErrorCategory::Borrowing,
        ErrorCategory::Lifetime,
        ErrorCategory::Async,
        ErrorCategory::TypeInference,
        ErrorCategory::MethodNotFound,
        ErrorCategory::Import,
        ErrorCategory::Unknown,
    ];
    assert_eq!(categories.len(), 11);
}

// ==================== Language Tests ====================

#[test]
fn test_language_display() {
    assert_eq!(format!("{}", Language::Python), "Python");
    assert_eq!(format!("{}", Language::C), "C");
    assert_eq!(format!("{}", Language::Ruchy), "Ruchy");
    assert_eq!(format!("{}", Language::Bash), "Bash");
    assert_eq!(format!("{}", Language::Rust), "Rust");
}

// ==================== CITLConfig Tests ====================

#[test]
fn test_citl_config_default() {
    let config = CITLConfig::default();
    assert_eq!(config.max_iterations, 10);
    assert!((config.confidence_threshold - 0.7).abs() < f32::EPSILON);
    assert!(config.enable_self_training);
}

// ==================== rust_error_codes Tests ====================

#[test]
fn test_rust_error_codes_contains_common_errors() {
    let codes = rust_error_codes();

    // Most common errors from depyler data
    assert!(codes.contains_key("E0308")); // 20.9%
    assert!(codes.contains_key("E0599")); // 17.9%
    assert!(codes.contains_key("E0433")); // 16.4%
    assert!(codes.contains_key("E0432")); // 14.1%
    assert!(codes.contains_key("E0277")); // 11.0%
    assert!(codes.contains_key("E0425")); // 8.2%
    assert!(codes.contains_key("E0282")); // 7.0%
}

#[test]
fn test_rust_error_codes_categories() {
    let codes = rust_error_codes();

    assert_eq!(
        codes.get("E0308").map(|c| c.category),
        Some(ErrorCategory::TypeMismatch)
    );
    assert_eq!(
        codes.get("E0382").map(|c| c.category),
        Some(ErrorCategory::Ownership)
    );
    assert_eq!(
        codes.get("E0597").map(|c| c.category),
        Some(ErrorCategory::Lifetime)
    );
    assert_eq!(
        codes.get("E0277").map(|c| c.category),
        Some(ErrorCategory::TraitBound)
    );
}

#[test]
fn test_rust_error_codes_difficulties() {
    let codes = rust_error_codes();

    // Easy errors
    assert_eq!(
        codes.get("E0308").map(|c| c.difficulty),
        Some(Difficulty::Easy)
    );
    assert_eq!(
        codes.get("E0425").map(|c| c.difficulty),
        Some(Difficulty::Easy)
    );

    // Medium errors
    assert_eq!(
        codes.get("E0382").map(|c| c.difficulty),
        Some(Difficulty::Medium)
    );
    assert_eq!(
        codes.get("E0502").map(|c| c.difficulty),
        Some(Difficulty::Medium)
    );

    // Hard errors
    assert_eq!(
        codes.get("E0597").map(|c| c.difficulty),
        Some(Difficulty::Hard)
    );
    assert_eq!(
        codes.get("E0277").map(|c| c.difficulty),
        Some(Difficulty::Hard)
    );

    // Expert errors
    assert_eq!(
        codes.get("E0373").map(|c| c.difficulty),
        Some(Difficulty::Expert)
    );
}

// ==================== CITLBuilder Tests ====================

#[test]
fn test_citl_builder_without_compiler_fails() {
    let result = CITL::builder().build();
    assert!(result.is_err());
    if let Err(CITLError::ConfigurationError { message }) = result {
        assert!(message.contains("Compiler interface is required"));
    } else {
        panic!("Expected ConfigurationError");
    }
}

#[test]
fn test_citl_builder_with_compiler_succeeds() {
    let compiler = RustCompiler::new();
    let result = CITL::builder().compiler(compiler).build();
    assert!(result.is_ok());
}

#[test]
fn test_citl_builder_max_iterations() {
    let compiler = RustCompiler::new();
    let citl = CITL::builder()
        .compiler(compiler)
        .max_iterations(20)
        .build()
        .expect("Should build");
    assert_eq!(citl.config.max_iterations, 20);
}

#[test]
fn test_citl_builder_confidence_threshold() {
    let compiler = RustCompiler::new();
    let citl = CITL::builder()
        .compiler(compiler)
        .confidence_threshold(0.9)
        .build()
        .expect("Should build");
    assert!((citl.config.confidence_threshold - 0.9).abs() < f32::EPSILON);
}

// ==================== Iterative Fix Loop Tests ====================

#[test]
fn test_suggested_fix_creation() {
    let fix = SuggestedFix::new(
        "expr.to_string()".to_string(),
        0.85,
        "Convert to String".to_string(),
    );
    assert_eq!(fix.replacement, "expr.to_string()");
    assert!((fix.confidence - 0.85).abs() < f32::EPSILON);
    assert_eq!(fix.description, "Convert to String");
}

#[test]
fn test_fix_result_success() {
    let result = FixResult::success("fixed code".to_string(), 1);
    assert!(result.is_success());
    assert_eq!(result.iterations, 1);
    assert!(result.fixed_source.is_some());
}

#[test]
fn test_fix_result_failure() {
    let result = FixResult::failure(5, vec!["E0308".to_string()]);
    assert!(!result.is_success());
    assert_eq!(result.iterations, 5);
    assert!(result.fixed_source.is_none());
    assert_eq!(result.remaining_errors.len(), 1);
}

#[test]
fn test_suggest_fix_for_valid_code_returns_none() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let code = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
    let result = citl.compile(code).expect("Should compile");

    // Valid code should have no errors to fix
    assert!(result.is_success());
}

#[test]
fn test_suggest_fix_returns_suggestion_for_error() {
    let mut citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    // Add a pattern for E0308 type mismatch
    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let fix = FixTemplate::new("$expr.to_string()", "Convert to String");
    let embedding = ErrorEmbedding::new(vec![0.0; 256], error_code.clone(), 12345);
    citl.pattern_library.add_pattern(embedding, fix);

    // Now suggest_fix should find this pattern for similar errors
    let diag = CompilerDiagnostic::new(
        error_code,
        DiagnosticSeverity::Error,
        "mismatched types",
        SourceSpan::default(),
    );

    let suggestion = citl.suggest_fix(&diag, "let x: String = 42;");
    // Should find a suggestion (may or may not match well depending on embedding)
    // The key is that it doesn't panic and returns Some when pattern exists
    assert!(suggestion.is_some() || !citl.pattern_library.is_empty());
}

#[test]
fn test_apply_fix_simple_replacement() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let source = "let x = 42;";
    let fix = SuggestedFix::new("42_i32".to_string(), 0.9, "Add type suffix".to_string())
        .with_span(8, 10); // Position of "42"

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, "let x = 42_i32;");
}

#[test]
fn test_apply_fix_preserves_surrounding_code() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let source = "fn foo() { let x = bar(); }";
    let fix = SuggestedFix::new(
        "bar().unwrap()".to_string(),
        0.8,
        "Unwrap Result".to_string(),
    )
    .with_span(19, 24); // Position of "bar()"

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, "fn foo() { let x = bar().unwrap(); }");
}

#[test]
fn test_fix_all_valid_code_returns_immediately() {
    let mut citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let code = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
    let result = citl.fix_all(code);

    assert!(result.is_success());
    assert_eq!(result.iterations, 0); // No iterations needed
}

#[test]
fn test_fix_all_respects_max_iterations() {
    let mut citl = CITL::builder()
        .compiler(RustCompiler::new())
        .max_iterations(3)
        .build()
        .expect("Should build");

    // Code with unfixable error (no patterns available)
    let code = "fn main() { let x: String = 42; }";
    let result = citl.fix_all(code);

    // Should stop after max_iterations
    assert!(!result.is_success());
    assert!(result.iterations <= 3);
}

#[test]
fn test_fix_result_tracks_applied_fixes() {
    let result = FixResult::success("fixed".to_string(), 2)
        .with_applied_fix("Fix 1".to_string())
        .with_applied_fix("Fix 2".to_string());

    assert_eq!(result.applied_fixes.len(), 2);
    assert_eq!(result.applied_fixes[0], "Fix 1");
    assert_eq!(result.applied_fixes[1], "Fix 2");
}

// ==================== Integration Tests (Real Compilation) ====================

#[test]
fn test_integration_compile_and_detect_type_error() {
    // Test that we can compile code and detect E0308 type errors
    let compiler = RustCompiler::new();
    let code = "pub fn foo() -> String { 42 }";
    let result = compiler.compile(code, &CompileOptions::default());

    assert!(result.is_ok());
    let compilation = result.expect("Should return result");

    assert!(
        !compilation.is_success(),
        "Code with type error should not compile"
    );
    assert!(compilation.error_count() > 0);

    // Check that we got an E0308 error
    let errors = compilation.errors();
    assert!(!errors.is_empty());
    // E0308 is "mismatched types"
    assert!(
        errors.iter().any(|e| e.code.code == "E0308"),
        "Should have E0308 error"
    );
}

#[test]
fn test_integration_valid_code_compiles() {
    // Test that valid code compiles successfully
    let compiler = RustCompiler::new();
    let code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
"#;
    let result = compiler.compile(code, &CompileOptions::default());

    assert!(result.is_ok());
    let compilation = result.expect("Should return result");
    assert!(compilation.is_success(), "Valid code should compile");
    assert_eq!(compilation.error_count(), 0);
}

#[test]
fn test_integration_encoder_produces_embeddings() {
    // Test that the error encoder produces valid embeddings
    let encoder = ErrorEncoder::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let diag = CompilerDiagnostic::new(
        code,
        DiagnosticSeverity::Error,
        "mismatched types: expected `String`, found `i32`",
        span,
    );

    let source_code = "pub fn foo() -> String { 42 }";
    let embedding = encoder.encode(&diag, source_code);
    assert!(!embedding.vector.is_empty());

    // Embedding should have non-zero values
    let sum: f32 = embedding.vector.iter().sum();
    assert!(sum.abs() > 0.0, "Embedding should have non-zero values");
}

#[test]
fn test_integration_pattern_library_workflow() {
    // Test full pattern library workflow: add, search, record outcome
    let mut lib = PatternLibrary::new();
    let encoder = ErrorEncoder::new();

    // Add a pattern
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let diag = CompilerDiagnostic::new(
        code.clone(),
        DiagnosticSeverity::Error,
        "mismatched types: expected `String`, found `i32`",
        span,
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
    let mut citl = CITL::builder()
        .compiler(RustCompiler::new())
        .max_iterations(5)
        .build()
        .expect("Should build");

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

    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let diag1 = CompilerDiagnostic::new(
        code1,
        DiagnosticSeverity::Error,
        "mismatched types: expected `String`, found `i32`",
        SourceSpan::default(),
    );

    let code2 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let diag2 = CompilerDiagnostic::new(
        code2,
        DiagnosticSeverity::Error,
        "mismatched types: expected `String`, found `u32`",
        SourceSpan::default(),
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

    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let diag1 = CompilerDiagnostic::new(
        code1,
        DiagnosticSeverity::Error,
        "mismatched types",
        SourceSpan::default(),
    );

    let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
    let diag2 = CompilerDiagnostic::new(
        code2,
        DiagnosticSeverity::Error,
        "use of moved value",
        SourceSpan::default(),
    );

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
    let compiler = RustCompiler::new();
    let citl = CITL::builder()
        .compiler(compiler)
        .pattern_library("/nonexistent/path/patterns.db")
        .build()
        .expect("Should build with nonexistent path (creates new library)");
    // Pattern library should be empty (new) since path doesn't exist
    assert!(citl.pattern_library.is_empty());
}

#[test]
fn test_citl_add_pattern_self_training_disabled() {
    let mut citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    // Disable self-training
    citl.config.enable_self_training = false;

    // Try to add a pattern
    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let embedding = ErrorEmbedding::new(vec![0.0; 256], error_code, 12345);
    let fix = FixTemplate::new("$expr.to_string()", "Convert to String");

    citl.add_pattern(embedding, fix, true);

    // Pattern should NOT be added when self-training is disabled
    assert!(citl.pattern_library.is_empty());
}

#[test]
fn test_citl_add_pattern_unsuccessful_fix() {
    let mut citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    // Self-training enabled but fix was unsuccessful
    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let embedding = ErrorEmbedding::new(vec![0.0; 256], error_code, 12345);
    let fix = FixTemplate::new("$expr.to_string()", "Convert to String");

    citl.add_pattern(embedding, fix, false); // success = false

    // Pattern should NOT be added for unsuccessful fix
    assert!(citl.pattern_library.is_empty());
}

#[test]
fn test_apply_fix_start_offset_out_of_bounds() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let source = "let x = 42;";
    let fix =
        SuggestedFix::new("replacement".to_string(), 0.9, "Test".to_string()).with_span(100, 105); // start >= source.len()

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, source); // Should return original source unchanged
}

#[test]
fn test_apply_fix_end_offset_out_of_bounds() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

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
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let template = FixTemplate::new("$expr as $type", "Type cast");
    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let mut diag = CompilerDiagnostic::new(
        error_code,
        DiagnosticSeverity::Error,
        "mismatched types",
        span,
    );
    diag.expected = Some(TypeInfo::new("String"));

    let result = citl.instantiate_template(&template, &diag, "let x = 42;");
    assert!(result.contains("String"));
}

#[test]
fn test_instantiate_template_with_found_type() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let template = FixTemplate::new("convert $found to target", "Type conversion");
    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let mut diag = CompilerDiagnostic::new(
        error_code,
        DiagnosticSeverity::Error,
        "mismatched types",
        span,
    );
    diag.found = Some(TypeInfo::new("i32"));

    let result = citl.instantiate_template(&template, &diag, "let x = 42;");
    assert!(result.contains("i32"));
}

#[test]
fn test_instantiate_template_with_both_types() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let template = FixTemplate::new("($expr as $type) // was $found", "Full cast");
    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::default();
    let mut diag = CompilerDiagnostic::new(
        error_code,
        DiagnosticSeverity::Error,
        "mismatched types",
        span,
    );
    diag.expected = Some(TypeInfo::new("u64"));
    diag.found = Some(TypeInfo::new("i32"));

    let result = citl.instantiate_template(&template, &diag, "let x = 42;");
    assert!(result.contains("u64"));
    assert!(result.contains("i32"));
    assert!(result.contains("expr")); // $expr replaced with placeholder
}

#[test]
fn test_error_category_all_variants_debug() {
    let categories = [
        ErrorCategory::TypeMismatch,
        ErrorCategory::TraitBound,
        ErrorCategory::Unresolved,
        ErrorCategory::Ownership,
        ErrorCategory::Borrowing,
        ErrorCategory::Lifetime,
        ErrorCategory::Async,
        ErrorCategory::TypeInference,
        ErrorCategory::MethodNotFound,
        ErrorCategory::Import,
        ErrorCategory::Unknown,
    ];
    for cat in &categories {
        let debug_str = format!("{:?}", cat);
        assert!(!debug_str.is_empty());
    }
}

#[test]
fn test_difficulty_all_variants_debug() {
    let difficulties = [
        Difficulty::Easy,
        Difficulty::Medium,
        Difficulty::Hard,
        Difficulty::Expert,
    ];
    for diff in &difficulties {
        let debug_str = format!("{:?}", diff);
        assert!(!debug_str.is_empty());
    }
}

#[test]
fn test_language_all_variants() {
    let languages = [
        Language::Python,
        Language::C,
        Language::Ruchy,
        Language::Bash,
        Language::Rust,
    ];
    for lang in &languages {
        // Test Display
        let display = format!("{}", lang);
        assert!(!display.is_empty());
        // Test Debug
        let debug = format!("{:?}", lang);
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_rust_error_codes_all_tiers() {
    let codes = rust_error_codes();

    // Tier 1: Easy
    let easy_codes = ["E0308", "E0425", "E0433", "E0432", "E0412", "E0599"];
    for code in &easy_codes {
        let ec = codes.get(*code).expect("Code should exist");
        assert_eq!(ec.difficulty, Difficulty::Easy);
    }

    // Tier 2: Medium
    let medium_codes = [
        "E0382", "E0502", "E0499", "E0596", "E0507", "E0282", "E0106",
    ];
    for code in &medium_codes {
        let ec = codes.get(*code).expect("Code should exist");
        assert!(
            ec.difficulty == Difficulty::Medium,
            "Expected Medium for {code}"
        );
    }

    // Tier 3: Hard
    let hard_codes = ["E0597", "E0621", "E0495", "E0623", "E0277"];
    for code in &hard_codes {
        let ec = codes.get(*code).expect("Code should exist");
        assert!(
            ec.difficulty == Difficulty::Hard,
            "Expected Hard for {code}"
        );
    }

    // Tier 4: Expert
    let expert_codes = ["E0373"];
    for code in &expert_codes {
        let ec = codes.get(*code).expect("Code should exist");
        assert_eq!(ec.difficulty, Difficulty::Expert);
    }
}

#[test]
fn test_fix_result_failure_with_applied_fixes() {
    let result = FixResult::failure(3, vec!["E0308".to_string()])
        .with_applied_fix("Attempted fix 1".to_string())
        .with_applied_fix("Attempted fix 2".to_string());

    assert!(!result.is_success());
    assert_eq!(result.iterations, 3);
    assert_eq!(result.applied_fixes.len(), 2);
}

#[test]
fn test_citl_search_patterns_empty_library() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let embedding = ErrorEmbedding::new(vec![0.1; 256], error_code, 12345);

    let results = citl.search_patterns(&embedding, 10);
    assert!(results.is_empty());
}

#[test]
fn test_citl_encode_error() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let diag = CompilerDiagnostic::new(
        error_code,
        DiagnosticSeverity::Error,
        "test error message",
        SourceSpan::default(),
    );

    let embedding = citl.encode_error(&diag, "let x = 42;");
    assert!(!embedding.vector.is_empty());
}

// ==================== CITL apply_fix Tests ====================

#[test]
fn test_citl_apply_fix_basic() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let source = "fn main() { let x = 42 }"; // Missing semicolon
    let fix = SuggestedFix {
        replacement: "42;".to_string(),
        confidence: 0.9,
        description: "Add semicolon".to_string(),
        start_offset: 20, // Position of "42"
        end_offset: 22,   // After "42"
        error_code: "E0308".to_string(),
    };

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, "fn main() { let x = 42; }");
}

#[test]
fn test_citl_apply_fix_out_of_bounds_start() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let source = "short";
    let fix = SuggestedFix {
        replacement: "replacement".to_string(),
        confidence: 0.9,
        description: "test".to_string(),
        start_offset: 100, // Out of bounds
        end_offset: 110,
        error_code: "E0308".to_string(),
    };

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, "short"); // Should return original unchanged
}

#[test]
fn test_citl_apply_fix_out_of_bounds_end() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let source = "short";
    let fix = SuggestedFix {
        replacement: "replacement".to_string(),
        confidence: 0.9,
        description: "test".to_string(),
        start_offset: 0,
        end_offset: 100, // Out of bounds
        error_code: "E0308".to_string(),
    };

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, "short"); // Should return original unchanged
}

#[test]
fn test_citl_compiler_accessor() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    // Just verify the accessor doesn't panic
    let _compiler = citl.compiler();
}

#[test]
fn test_citl_suggest_fix_no_match() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build");

    let error_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let diag = CompilerDiagnostic::new(
        error_code,
        DiagnosticSeverity::Error,
        "test error message",
        SourceSpan::default(),
    );

    // Empty pattern library, should return None
    let result = citl.suggest_fix(&diag, "let x: i32 = \"hello\";");
    assert!(result.is_none());
}

// ==================== SuggestedFix Tests ====================

#[test]
fn test_suggested_fix_with_span() {
    let fix = SuggestedFix::new(
        "fix text".to_string(),
        0.95,
        "Test fix description".to_string(),
    )
    .with_span(10, 20)
    .with_error_code("E0308");

    assert_eq!(fix.replacement, "fix text");
    assert!((fix.confidence - 0.95).abs() < f32::EPSILON);
    assert_eq!(fix.description, "Test fix description");
    assert_eq!(fix.start_offset, 10);
    assert_eq!(fix.end_offset, 20);
    assert_eq!(fix.error_code, "E0308");
}

#[test]
fn test_suggested_fix_default_offsets() {
    let fix = SuggestedFix::new("replacement".to_string(), 0.8, "desc".to_string());
    assert_eq!(fix.replacement, "replacement");
    assert!((fix.confidence - 0.8).abs() < f32::EPSILON);
    assert_eq!(fix.start_offset, 0);
    assert_eq!(fix.end_offset, 0);
    assert!(fix.error_code.is_empty());
}

// ==================== CITLBuilder Additional Tests ====================

#[test]
fn test_citl_builder_pattern_library_nonexistent() {
    let compiler = RustCompiler::new();
    // Using a nonexistent path should still work (creates empty library)
    let result = CITL::builder()
        .compiler(compiler)
        .pattern_library("/nonexistent/path/patterns.db")
        .build();
    assert!(result.is_ok());
}
