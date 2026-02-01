use super::*;

// ==================== PatternLibrary Tests ====================

#[test]
fn test_pattern_library_new() {
    let lib = PatternLibrary::new();
    assert!(lib.is_empty());
    assert_eq!(lib.len(), 0);
}

#[test]
fn test_pattern_library_add_pattern() {
    let mut lib = PatternLibrary::new();

    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let embedding = ErrorEmbedding::new(vec![1.0; 256], code, 12345);
    let template = FixTemplate::new("$expr.to_string()", "Convert to String");

    lib.add_pattern(embedding, template);

    assert_eq!(lib.len(), 1);
    assert!(!lib.is_empty());
}

#[test]
fn test_pattern_library_search() {
    let mut lib = PatternLibrary::new();

    // Add a pattern
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let mut vec1 = vec![0.0; 256];
    vec1[0] = 1.0;
    let embedding1 = ErrorEmbedding::new(vec1, code.clone(), 12345);
    let template1 = FixTemplate::new("$expr.to_string()", "Convert to String");
    lib.add_pattern(embedding1, template1);

    // Search with similar embedding
    let mut query_vec = vec![0.0; 256];
    query_vec[0] = 1.0;
    let query = ErrorEmbedding::new(query_vec, code, 0);

    let results = lib.search(&query, 5);
    assert_eq!(results.len(), 1);
    assert!(results[0].similarity > 0.9);
}

#[test]
fn test_pattern_library_search_empty() {
    let lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let query = ErrorEmbedding::new(vec![1.0; 256], code, 0);

    let results = lib.search(&query, 5);
    assert!(results.is_empty());
}

#[test]
fn test_pattern_library_get_by_code() {
    let mut lib = PatternLibrary::new();

    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);

    lib.add_pattern(
        ErrorEmbedding::new(vec![1.0; 256], code1.clone(), 0),
        FixTemplate::new("fix1", "Fix 1"),
    );
    lib.add_pattern(
        ErrorEmbedding::new(vec![1.0; 256], code1, 0),
        FixTemplate::new("fix2", "Fix 2"),
    );
    lib.add_pattern(
        ErrorEmbedding::new(vec![1.0; 256], code2, 0),
        FixTemplate::new("fix3", "Fix 3"),
    );

    let e0308_patterns = lib.get_by_code("E0308");
    assert_eq!(e0308_patterns.len(), 2);

    let e0382_patterns = lib.get_by_code("E0382");
    assert_eq!(e0382_patterns.len(), 1);
}

#[test]
fn test_pattern_library_record_outcome() {
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    lib.add_pattern(
        ErrorEmbedding::new(vec![1.0; 256], code, 0),
        FixTemplate::new("fix", "Fix"),
    );

    lib.record_outcome(0, true);
    lib.record_outcome(0, true);
    lib.record_outcome(0, false);

    let pattern = lib.get(0).expect("Pattern should exist");
    // Initial add_pattern sets success_count to 1, then we add 2 more successes
    assert_eq!(pattern.success_count, 3);
    assert_eq!(pattern.failure_count, 1);
}

// ==================== ErrorFixPattern Tests ====================

#[test]
fn test_error_fix_pattern_success_rate() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 0,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 7,
        failure_count: 3,
    };

    assert!((pattern.success_rate() - 0.7).abs() < 0.001);
    assert_eq!(pattern.total_applications(), 10);
}

#[test]
fn test_error_fix_pattern_zero_applications() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 0,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 0,
        failure_count: 0,
    };

    assert!((pattern.success_rate() - 0.0).abs() < 0.001);
}

// ==================== FixTemplate Tests ====================

#[test]
fn test_fix_template_new() {
    let template = FixTemplate::new("$expr.to_string()", "Convert to String");
    assert_eq!(template.pattern, "$expr.to_string()");
    assert_eq!(template.description, "Convert to String");
    assert!((template.confidence - 0.5).abs() < 0.001);
}

#[test]
fn test_fix_template_apply() {
    let template = FixTemplate::new("$expr.to_string()", "Convert to String");
    let mut bindings = HashMap::new();
    bindings.insert("expr".to_string(), "my_value".to_string());

    let result = template.apply(&bindings);
    assert_eq!(result, "my_value.to_string()");
}

#[test]
fn test_fix_template_applies_to() {
    let template = FixTemplate::new("fix", "Fix")
        .with_code("E0308")
        .with_code("E0382");

    assert!(template.applies_to("E0308"));
    assert!(template.applies_to("E0382"));
    assert!(!template.applies_to("E0597"));
}

#[test]
fn test_fix_template_applies_to_any() {
    let template = FixTemplate::new("fix", "Fix");
    // Empty applicable_codes means applies to any
    assert!(template.applies_to("E0308"));
    assert!(template.applies_to("E0999"));
}

// ==================== Placeholder Tests ====================

#[test]
fn test_placeholder_expression() {
    let ph = Placeholder::expression("expr");
    assert_eq!(ph.name, "expr");
    assert_eq!(ph.constraint, PlaceholderConstraint::Expression);
}

#[test]
fn test_placeholder_type_name() {
    let ph = Placeholder::type_name("T");
    assert_eq!(ph.name, "T");
    assert_eq!(ph.constraint, PlaceholderConstraint::Type);
}

#[test]
fn test_placeholder_identifier() {
    let ph = Placeholder::identifier("var");
    assert_eq!(ph.name, "var");
    assert_eq!(ph.constraint, PlaceholderConstraint::Identifier);
}

// ==================== PatternMatch Tests ====================

#[test]
fn test_pattern_match_combined_score() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 0,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 8,
        failure_count: 2,
    };

    let pm = PatternMatch {
        pattern,
        similarity: 0.8,
        success_rate: 0.8,
    };

    // Combined score should be 0.8 * 0.8 = 0.64
    assert!((pm.combined_score() - 0.64).abs() < 0.001);
}

// ==================== PatternStats Tests ====================

#[test]
fn test_pattern_stats_new() {
    let stats = PatternStats::new();
    // Default success rate for unseen patterns is 0.5
    assert!((stats.success_rate(0) - 0.5).abs() < 0.001);
}

#[test]
fn test_pattern_stats_record() {
    let mut stats = PatternStats::new();
    stats.record(0, true);
    stats.record(0, true);
    stats.record(0, false);

    // 2 successes / 3 total = 0.667
    assert!((stats.success_rate(0) - 0.667).abs() < 0.01);
}

// ==================== Templates Tests ====================

#[test]
fn test_templates_to_string() {
    let template = templates::to_string_conversion();
    assert!(template.pattern.contains("to_string"));
    assert!(template.applies_to("E0308"));
}

#[test]
fn test_templates_clone() {
    let template = templates::clone_value();
    assert!(template.pattern.contains("clone"));
    assert!(template.applies_to("E0382"));
}

// ==================== E0382 Templates (Use of Moved Value) ====================

#[test]
fn test_template_borrow_instead_of_move() {
    let template = templates::borrow_instead_of_move();
    assert!(template.pattern.contains('&'));
    assert!(template.applies_to("E0382"));
    assert!(template.confidence > 0.7);
}

#[test]
fn test_template_rc_wrap() {
    let template = templates::rc_wrap();
    assert!(template.pattern.contains("Rc::new"));
    assert!(template.applies_to("E0382"));
}

#[test]
fn test_template_arc_wrap() {
    let template = templates::arc_wrap();
    assert!(template.pattern.contains("Arc::new"));
    assert!(template.applies_to("E0382"));
}

// ==================== E0277 Templates (Trait Bound Not Satisfied) ====================

#[test]
fn test_template_derive_debug() {
    let template = templates::derive_debug();
    assert!(template.pattern.contains("Debug"));
    assert!(template.applies_to("E0277"));
}

#[test]
fn test_template_derive_clone_trait() {
    let template = templates::derive_clone_trait();
    assert!(template.pattern.contains("Clone"));
    assert!(template.applies_to("E0277"));
}

#[test]
fn test_template_impl_display() {
    let template = templates::impl_display();
    assert!(template.pattern.contains("Display"));
    assert!(template.applies_to("E0277"));
}

#[test]
fn test_template_impl_from() {
    let template = templates::impl_from();
    assert!(template.pattern.contains("From"));
    assert!(template.applies_to("E0277"));
}

// ==================== E0515 Templates (Cannot Return Reference) ====================

#[test]
fn test_template_return_owned() {
    let template = templates::return_owned();
    assert!(template.applies_to("E0515"));
    assert!(template.description.contains("owned"));
}

#[test]
fn test_template_return_cloned() {
    let template = templates::return_cloned();
    assert!(template.pattern.contains("clone"));
    assert!(template.applies_to("E0515"));
}

#[test]
fn test_template_use_cow() {
    let template = templates::use_cow();
    assert!(template.pattern.contains("Cow"));
    assert!(template.applies_to("E0515"));
}

#[test]
fn test_all_templates() {
    let all = templates::all_templates();
    // 10 original + 9 new = 19 templates
    assert!(all.len() >= 19);
}

// ==================== Cosine Similarity Tests ====================

#[test]
fn test_cosine_similarity_same() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert!(cosine_similarity(&a, &b).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_different_lengths() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
}

// ==================== Pattern Persistence Tests ====================

#[test]
fn test_pattern_library_save_load_empty() {
    let lib = PatternLibrary::new();
    let path = "/tmp/citl_test_empty.pat";

    // Save
    lib.save(path).expect("Save should succeed");

    // Load
    let loaded = PatternLibrary::load(path).expect("Load should succeed");
    assert!(loaded.is_empty());

    // Cleanup
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_pattern_library_save_load_roundtrip() {
    let mut lib = PatternLibrary::new();

    // Add patterns
    let code1 = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let embedding1 = ErrorEmbedding::new(vec![1.0, 2.0, 3.0, 4.0], code1.clone(), 12345);
    let template1 = FixTemplate::new("$expr.to_string()", "Convert to String")
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0308")
        .with_confidence(0.9);
    lib.add_pattern(embedding1, template1);

    let code2 = ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium);
    let embedding2 = ErrorEmbedding::new(vec![5.0, 6.0, 7.0, 8.0], code2.clone(), 67890);
    let template2 = FixTemplate::new("$expr.clone()", "Clone value")
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0382")
        .with_confidence(0.8);
    lib.add_pattern(embedding2, template2);

    // Record some outcomes to update stats
    lib.record_outcome(0, true);
    lib.record_outcome(0, false);

    let path = "/tmp/citl_test_roundtrip.pat";

    // Save
    lib.save(path).expect("Save should succeed");

    // Load
    let loaded = PatternLibrary::load(path).expect("Load should succeed");

    // Verify patterns
    assert_eq!(loaded.len(), 2);

    let pattern0 = loaded.get(0).expect("Pattern 0 should exist");
    assert_eq!(pattern0.error_code.code, "E0308");
    assert_eq!(pattern0.context_hash, 12345);
    assert_eq!(pattern0.fix_template.pattern, "$expr.to_string()");

    let pattern1 = loaded.get(1).expect("Pattern 1 should exist");
    assert_eq!(pattern1.error_code.code, "E0382");
    assert_eq!(pattern1.context_hash, 67890);

    // Cleanup
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_pattern_library_load_nonexistent() {
    let result = PatternLibrary::load("/nonexistent/path/to/file.pat");
    assert!(result.is_err());
}

#[test]
fn test_pattern_library_save_load_preserves_embeddings() {
    let mut lib = PatternLibrary::new();

    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let embedding_vec = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let embedding = ErrorEmbedding::new(embedding_vec.clone(), code, 99999);
    let template = FixTemplate::new("fix", "Fix");
    lib.add_pattern(embedding, template);

    let path = "/tmp/citl_test_embeddings.pat";

    lib.save(path).expect("Save should succeed");
    let loaded = PatternLibrary::load(path).expect("Load should succeed");

    // Search with same embedding should have high similarity
    let query_code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let query = ErrorEmbedding::new(embedding_vec, query_code, 0);
    let results = loaded.search(&query, 1);

    assert_eq!(results.len(), 1);
    assert!(
        results[0].similarity > 0.99,
        "Embedding should match closely"
    );

    // Cleanup
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_pattern_library_load_corrupted_file() {
    let path = "/tmp/citl_test_corrupted.pat";

    // Write garbage data
    std::fs::write(path, b"not a valid pattern file").expect("Write should succeed");

    let result = PatternLibrary::load(path);
    assert!(result.is_err());

    // Cleanup
    let _ = std::fs::remove_file(path);
}

#[test]
fn test_pattern_library_file_has_magic_header() {
    let lib = PatternLibrary::new();
    let path = "/tmp/citl_test_magic.pat";

    lib.save(path).expect("Save should succeed");

    // Read raw bytes and check magic header
    let bytes = std::fs::read(path).expect("Read should succeed");
    assert!(bytes.len() >= 4);
    assert_eq!(&bytes[0..4], b"CITL", "File should have CITL magic header");

    // Cleanup
    let _ = std::fs::remove_file(path);
}
