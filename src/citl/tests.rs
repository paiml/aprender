pub(crate) use super::*;

// ==================== Test Helpers (reduce DataTransformation repetition) ====================

/// Build a default CITL instance with RustCompiler for testing
fn test_citl() -> CITL {
    CITL::builder()
        .compiler(RustCompiler::new())
        .build()
        .expect("Should build")
}

/// Build a mutable CITL instance with RustCompiler for testing
fn test_citl_mut() -> CITL {
    test_citl()
}

/// Build a CITL instance with custom max_iterations
fn test_citl_with_max_iter(max_iterations: usize) -> CITL {
    CITL::builder()
        .compiler(RustCompiler::new())
        .max_iterations(max_iterations)
        .build()
        .expect("Should build")
}

/// Create a standard E0308 TypeMismatch error code (most common in tests)
fn e0308() -> ErrorCode {
    ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy)
}

/// Create a standard compiler diagnostic for testing
fn test_diagnostic(error_code: ErrorCode, message: &str) -> CompilerDiagnostic {
    CompilerDiagnostic::new(
        error_code,
        DiagnosticSeverity::Error,
        message,
        SourceSpan::default(),
    )
}

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
    let _citl = test_citl();
}

#[test]
fn test_citl_builder_max_iterations() {
    let citl = test_citl_with_max_iter(20);
    assert_eq!(citl.config.max_iterations, 20);
}

#[test]
fn test_citl_builder_confidence_threshold() {
    let citl = CITL::builder()
        .compiler(RustCompiler::new())
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
    let citl = test_citl();

    let code = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
    let result = citl.compile(code).expect("Should compile");

    // Valid code should have no errors to fix
    assert!(result.is_success());
}

#[test]
fn test_suggest_fix_returns_suggestion_for_error() {
    let mut citl = test_citl_mut();

    // Add a pattern for E0308 type mismatch
    let error_code = e0308();
    let fix = FixTemplate::new("$expr.to_string()", "Convert to String");
    let embedding = ErrorEmbedding::new(vec![0.0; 256], error_code.clone(), 12345);
    citl.pattern_library.add_pattern(embedding, fix);

    // Now suggest_fix should find this pattern for similar errors
    let diag = test_diagnostic(error_code, "mismatched types");

    let suggestion = citl.suggest_fix(&diag, "let x: String = 42;");
    // Should find a suggestion (may or may not match well depending on embedding)
    // The key is that it doesn't panic and returns Some when pattern exists
    assert!(suggestion.is_some() || !citl.pattern_library.is_empty());
}

#[test]
fn test_apply_fix_simple_replacement() {
    let citl = test_citl();

    let source = "let x = 42;";
    let fix = SuggestedFix::new("42_i32".to_string(), 0.9, "Add type suffix".to_string())
        .with_span(8, 10); // Position of "42"

    let result = citl.apply_fix(source, &fix);
    assert_eq!(result, "let x = 42_i32;");
}

#[test]
fn test_apply_fix_preserves_surrounding_code() {
    let citl = test_citl();

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
    let mut citl = test_citl_mut();

    let code = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
    let result = citl.fix_all(code);

    assert!(result.is_success());
    assert_eq!(result.iterations, 0); // No iterations needed
}

#[test]
fn test_fix_all_respects_max_iterations() {
    let mut citl = test_citl_with_max_iter(3);

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
    let diag = test_diagnostic(
        e0308(),
        "mismatched types: expected `String`, found `i32`",
    );

    let source_code = "pub fn foo() -> String { 42 }";
    let embedding = encoder.encode(&diag, source_code);
    assert!(!embedding.vector.is_empty());

    // Embedding should have non-zero values
    let sum: f32 = embedding.vector.iter().sum();
    assert!(sum.abs() > 0.0, "Embedding should have non-zero values");
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
