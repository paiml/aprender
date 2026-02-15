
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
