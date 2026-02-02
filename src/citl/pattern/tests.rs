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

// ==================== Coverage: PatternLibrary Default Impl ====================

#[test]
fn test_pattern_library_default() {
    let lib = PatternLibrary::default();
    assert!(lib.is_empty());
    assert_eq!(lib.len(), 0);
}

// ==================== Coverage: PatternStats Default Impl ====================

#[test]
fn test_pattern_stats_default() {
    let stats = PatternStats::default();
    // Default success rate for unseen patterns is 0.5
    assert!((stats.success_rate(0) - 0.5).abs() < 0.001);
    assert!((stats.success_rate(99) - 0.5).abs() < 0.001);
}

// ==================== Coverage: PatternStats record failure path ====================

#[test]
fn test_pattern_stats_record_failures() {
    let mut stats = PatternStats::new();
    stats.record(0, false);
    stats.record(0, false);
    stats.record(0, false);

    // 0 successes / 3 total = 0.0
    assert!((stats.success_rate(0) - 0.0).abs() < 0.001);
}

// ==================== Coverage: record_outcome out-of-bounds ====================

#[test]
fn test_pattern_library_record_outcome_out_of_bounds() {
    let mut lib = PatternLibrary::new();
    // Recording outcome for nonexistent pattern should be a no-op
    lib.record_outcome(999, true);
    lib.record_outcome(999, false);
    assert!(lib.is_empty());
}

// ==================== Coverage: get() returns None for invalid index ====================

#[test]
fn test_pattern_library_get_none() {
    let lib = PatternLibrary::new();
    assert!(lib.get(0).is_none());
    assert!(lib.get(100).is_none());
}

// ==================== Coverage: get_by_code on empty library ====================

#[test]
fn test_pattern_library_get_by_code_empty() {
    let lib = PatternLibrary::new();
    let patterns = lib.get_by_code("E0308");
    assert!(patterns.is_empty());
}

// ==================== Coverage: cosine_similarity empty vectors ====================

#[test]
fn test_cosine_similarity_empty() {
    let a: Vec<f32> = Vec::new();
    let b: Vec<f32> = Vec::new();
    assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
}

// ==================== Coverage: cosine_similarity zero-norm vectors ====================

#[test]
fn test_cosine_similarity_zero_norm() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);

    let c = vec![1.0, 0.0, 0.0];
    let d = vec![0.0, 0.0, 0.0];
    assert!((cosine_similarity(&c, &d) - 0.0).abs() < 0.001);
}

// ==================== Coverage: parse_error_category all variants ====================

#[test]
fn test_parse_error_category_all_variants() {
    assert_eq!(parse_error_category(0), ErrorCategory::TypeMismatch);
    assert_eq!(parse_error_category(1), ErrorCategory::TraitBound);
    assert_eq!(parse_error_category(2), ErrorCategory::Unresolved);
    assert_eq!(parse_error_category(3), ErrorCategory::Ownership);
    assert_eq!(parse_error_category(4), ErrorCategory::Borrowing);
    assert_eq!(parse_error_category(5), ErrorCategory::Lifetime);
    assert_eq!(parse_error_category(6), ErrorCategory::Async);
    assert_eq!(parse_error_category(7), ErrorCategory::TypeInference);
    assert_eq!(parse_error_category(8), ErrorCategory::MethodNotFound);
    assert_eq!(parse_error_category(9), ErrorCategory::Import);
    // Unknown bytes default to TypeMismatch
    assert_eq!(parse_error_category(10), ErrorCategory::TypeMismatch);
    assert_eq!(parse_error_category(255), ErrorCategory::TypeMismatch);
}

// ==================== Coverage: parse_difficulty all variants ====================

#[test]
fn test_parse_difficulty_all_variants() {
    assert_eq!(parse_difficulty(0), Difficulty::Easy);
    assert_eq!(parse_difficulty(1), Difficulty::Medium); // default
    assert_eq!(parse_difficulty(2), Difficulty::Hard);
    assert_eq!(parse_difficulty(3), Difficulty::Expert);
    // Unknown defaults to Medium
    assert_eq!(parse_difficulty(4), Difficulty::Medium);
    assert_eq!(parse_difficulty(255), Difficulty::Medium);
}

// ==================== Coverage: parse_placeholder_constraint all variants ====================

#[test]
fn test_parse_placeholder_constraint_all_variants() {
    assert_eq!(
        parse_placeholder_constraint(0),
        PlaceholderConstraint::Expression
    );
    assert_eq!(parse_placeholder_constraint(1), PlaceholderConstraint::Type);
    assert_eq!(
        parse_placeholder_constraint(2),
        PlaceholderConstraint::Identifier
    );
    assert_eq!(
        parse_placeholder_constraint(3),
        PlaceholderConstraint::Literal
    );
    // Unknown defaults to Any
    assert_eq!(parse_placeholder_constraint(4), PlaceholderConstraint::Any);
    assert_eq!(
        parse_placeholder_constraint(255),
        PlaceholderConstraint::Any
    );
}

// ==================== Coverage: FixTemplate with_placeholder and with_confidence ====================

#[test]
fn test_fix_template_builder_chain() {
    let template = FixTemplate::new("$a + $b", "Add two things")
        .with_placeholder(Placeholder::expression("a"))
        .with_placeholder(Placeholder::identifier("b"))
        .with_code("E0308")
        .with_code("E0382")
        .with_confidence(0.95);

    assert_eq!(template.placeholders.len(), 2);
    assert_eq!(template.applicable_codes.len(), 2);
    assert!((template.confidence - 0.95).abs() < 0.001);
}

// ==================== Coverage: FixTemplate apply with no matching placeholders ====================

#[test]
fn test_fix_template_apply_no_matching_bindings() {
    let template = FixTemplate::new("constant_text", "No placeholders");
    let bindings = HashMap::new();
    let result = template.apply(&bindings);
    assert_eq!(result, "constant_text");
}

// ==================== Coverage: FixTemplate apply with multiple placeholders ====================

#[test]
fn test_fix_template_apply_multiple_placeholders() {
    let template = FixTemplate::new("$a.into::<$b>()", "Convert types");
    let mut bindings = HashMap::new();
    bindings.insert("a".to_string(), "my_val".to_string());
    bindings.insert("b".to_string(), "String".to_string());
    let result = template.apply(&bindings);
    assert_eq!(result, "my_val.into::<String>()");
}

// ==================== Coverage: Placeholder::new direct ====================

#[test]
fn test_placeholder_new_direct() {
    let ph = Placeholder::new(
        "custom",
        "A custom placeholder",
        PlaceholderConstraint::Literal,
    );
    assert_eq!(ph.name, "custom");
    assert_eq!(ph.description, "A custom placeholder");
    assert_eq!(ph.constraint, PlaceholderConstraint::Literal);
}

// ==================== Coverage: Placeholder::new with Any constraint ====================

#[test]
fn test_placeholder_any_constraint() {
    let ph = Placeholder::new("anything", "Anything goes", PlaceholderConstraint::Any);
    assert_eq!(ph.constraint, PlaceholderConstraint::Any);
}

// ==================== Coverage: search with k >= n (no partial sort) ====================

#[test]
fn test_pattern_library_search_k_greater_than_n() {
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

    // Add 2 patterns
    let mut vec1 = vec![0.0; 8];
    vec1[0] = 1.0;
    lib.add_pattern(
        ErrorEmbedding::new(vec1, code.clone(), 0),
        FixTemplate::new("fix1", "Fix 1"),
    );

    let mut vec2 = vec![0.0; 8];
    vec2[1] = 1.0;
    lib.add_pattern(
        ErrorEmbedding::new(vec2, code.clone(), 0),
        FixTemplate::new("fix2", "Fix 2"),
    );

    // Search with k=10 but only 2 patterns exist
    let mut query_vec = vec![0.0; 8];
    query_vec[0] = 1.0;
    let query = ErrorEmbedding::new(query_vec, code, 0);
    let results = lib.search(&query, 10);
    assert_eq!(results.len(), 2);
}

// ==================== Coverage: search with k == n (edge case) ====================

#[test]
fn test_pattern_library_search_k_equals_n() {
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

    let mut vec1 = vec![0.0; 4];
    vec1[0] = 1.0;
    lib.add_pattern(
        ErrorEmbedding::new(vec1, code.clone(), 0),
        FixTemplate::new("fix1", "Fix 1"),
    );

    let mut query_vec = vec![0.0; 4];
    query_vec[0] = 1.0;
    let query = ErrorEmbedding::new(query_vec, code, 0);

    // k == n (both 1)
    let results = lib.search(&query, 1);
    assert_eq!(results.len(), 1);
}

// ==================== Coverage: load bad version number ====================

#[test]
fn test_pattern_library_load_bad_version() {
    let path = "/tmp/citl_test_bad_version.pat";

    // Write valid magic header but invalid version
    let mut data = Vec::new();
    data.extend_from_slice(b"CITL");
    data.push(99); // Bad version
    std::fs::write(path, &data).expect("Write should succeed");

    let result = PatternLibrary::load(path);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("version"));

    let _ = std::fs::remove_file(path);
}

// ==================== Coverage: templates as_str and add_mut_reference ====================

#[test]
fn test_template_as_str_conversion() {
    let template = templates::as_str_conversion();
    assert!(template.pattern.contains("as_str"));
    assert!(template.applies_to("E0308"));
    assert!(template.confidence > 0.5);
}

#[test]
fn test_template_add_reference() {
    let template = templates::add_reference();
    assert!(template.pattern.contains('&'));
    assert!(template.applies_to("E0308"));
}

#[test]
fn test_template_add_mut_reference() {
    let template = templates::add_mut_reference();
    assert!(template.pattern.contains("&mut"));
    assert!(template.applies_to("E0308"));
}

#[test]
fn test_template_dereference() {
    let template = templates::dereference();
    assert!(template.pattern.contains('*'));
    assert!(template.applies_to("E0308"));
}

#[test]
fn test_template_into_conversion() {
    let template = templates::into_conversion();
    assert!(template.pattern.contains("into"));
    assert!(template.applies_to("E0308"));
}

#[test]
fn test_template_vec_new() {
    let template = templates::vec_new();
    assert!(template.pattern.contains("Vec::new"));
    assert!(template.applies_to("E0308"));
}

#[test]
fn test_template_string_new() {
    let template = templates::string_new();
    assert!(template.pattern.contains("String::new"));
    assert!(template.applies_to("E0308"));
}

#[test]
fn test_template_unwrap_or_default() {
    let template = templates::unwrap_or_default();
    assert!(template.pattern.contains("unwrap_or_default"));
    assert!(template.applies_to("E0308"));
}

// ==================== Coverage: ErrorFixPattern Clone trait ====================

#[test]
fn test_error_fix_pattern_clone() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 42,
        fix_template: FixTemplate::new("fix", "A fix"),
        success_count: 5,
        failure_count: 2,
    };
    let cloned = pattern.clone();
    assert_eq!(cloned.error_code.code, "E0308");
    assert_eq!(cloned.context_hash, 42);
    assert_eq!(cloned.success_count, 5);
    assert_eq!(cloned.failure_count, 2);
}

// ==================== Coverage: PatternMatch Debug ====================

#[test]
fn test_pattern_match_debug() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 0,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 1,
        failure_count: 0,
    };
    let pm = PatternMatch {
        pattern,
        similarity: 0.9,
        success_rate: 1.0,
    };
    let debug_str = format!("{:?}", pm);
    assert!(debug_str.contains("PatternMatch"));
    assert!(debug_str.contains("0.9"));
}

// ==================== Coverage: save/load with multiple placeholder types ====================

#[test]
fn test_pattern_save_load_all_placeholder_types() {
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0277", ErrorCategory::TraitBound, Difficulty::Hard);
    let template = FixTemplate::new("impl $trait for $type", "Implement trait")
        .with_placeholder(Placeholder::expression("expr"))
        .with_placeholder(Placeholder::type_name("type"))
        .with_placeholder(Placeholder::identifier("trait"))
        .with_placeholder(Placeholder::new(
            "lit",
            "Literal val",
            PlaceholderConstraint::Literal,
        ))
        .with_placeholder(Placeholder::new(
            "any",
            "Anything",
            PlaceholderConstraint::Any,
        ))
        .with_code("E0277")
        .with_code("E0308")
        .with_confidence(0.75);

    lib.add_pattern(ErrorEmbedding::new(vec![1.0, 2.0], code, 777), template);

    let path = "/tmp/citl_test_all_ph_types.pat";
    lib.save(path).expect("Save should succeed");
    let loaded = PatternLibrary::load(path).expect("Load should succeed");

    assert_eq!(loaded.len(), 1);
    let p = loaded.get(0).expect("Pattern should exist");
    assert_eq!(p.fix_template.placeholders.len(), 5);
    assert_eq!(p.fix_template.applicable_codes.len(), 2);
    assert!((p.fix_template.confidence - 0.75).abs() < 0.001);

    let _ = std::fs::remove_file(path);
}

// ==================== Coverage: save/load with all ErrorCategory + Difficulty combos ====================

#[test]
fn test_pattern_save_load_category_difficulty_combos() {
    let mut lib = PatternLibrary::new();

    // Use different categories and difficulties
    let combos = [
        ("E0001", ErrorCategory::TraitBound, Difficulty::Easy),
        ("E0002", ErrorCategory::Unresolved, Difficulty::Medium),
        ("E0003", ErrorCategory::Borrowing, Difficulty::Hard),
        ("E0004", ErrorCategory::Lifetime, Difficulty::Expert),
        ("E0005", ErrorCategory::Async, Difficulty::Easy),
        ("E0006", ErrorCategory::TypeInference, Difficulty::Medium),
        ("E0007", ErrorCategory::MethodNotFound, Difficulty::Hard),
        ("E0008", ErrorCategory::Import, Difficulty::Expert),
    ];

    for (code_str, cat, diff) in &combos {
        let code = ErrorCode::new(code_str, *cat, *diff);
        lib.add_pattern(
            ErrorEmbedding::new(vec![1.0], code, 0),
            FixTemplate::new("fix", "Fix"),
        );
    }

    let path = "/tmp/citl_test_cat_diff.pat";
    lib.save(path).expect("Save should succeed");
    let loaded = PatternLibrary::load(path).expect("Load should succeed");

    assert_eq!(loaded.len(), combos.len());

    // Verify each category/difficulty was preserved
    for (i, (code_str, cat, diff)) in combos.iter().enumerate() {
        let p = loaded.get(i).expect("Pattern should exist");
        assert_eq!(p.error_code.code, *code_str);
        assert_eq!(p.error_code.category, *cat);
        assert_eq!(p.error_code.difficulty, *diff);
    }

    let _ = std::fs::remove_file(path);
}

// ==================== Additional Coverage Tests ====================

#[test]
fn test_search_partial_sort_path() {
    // Add enough patterns to trigger partial sort (k < n)
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

    // Add 10 patterns with different embeddings
    for i in 0..10 {
        let mut vec = vec![0.0; 8];
        vec[i % 8] = 1.0;
        lib.add_pattern(
            ErrorEmbedding::new(vec, code.clone(), i as u64),
            FixTemplate::new(&format!("fix{i}"), &format!("Fix {i}")),
        );
    }

    // Search with k=3 (partial sort path: k < n)
    let mut query_vec = vec![0.0; 8];
    query_vec[0] = 1.0;
    let query = ErrorEmbedding::new(query_vec, code, 0);
    let results = lib.search(&query, 3);
    assert_eq!(results.len(), 3);
}

#[test]
fn test_search_full_sort_path() {
    // k >= n should not use partial sort
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);

    // Add 3 patterns
    for i in 0..3 {
        let vec = vec![1.0; 4];
        lib.add_pattern(
            ErrorEmbedding::new(vec, code.clone(), i as u64),
            FixTemplate::new(&format!("fix{i}"), &format!("Fix {i}")),
        );
    }

    // Search with k=10 (k >= n path)
    let query_vec = vec![1.0; 4];
    let query = ErrorEmbedding::new(query_vec, code, 0);
    let results = lib.search(&query, 10);
    assert_eq!(results.len(), 3);
}

#[test]
fn test_pattern_match_clone() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 0,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 1,
        failure_count: 0,
    };
    let pm = PatternMatch {
        pattern,
        similarity: 0.95,
        success_rate: 0.8,
    };
    let cloned = pm.clone();
    assert!((cloned.similarity - 0.95).abs() < 0.001);
    assert!((cloned.success_rate - 0.8).abs() < 0.001);
}

#[test]
fn test_fix_template_clone() {
    let template = FixTemplate::new("$a.into()", "Into conversion")
        .with_placeholder(Placeholder::expression("a"))
        .with_code("E0308")
        .with_confidence(0.85);
    let cloned = template.clone();
    assert_eq!(cloned.pattern, template.pattern);
    assert_eq!(cloned.placeholders.len(), template.placeholders.len());
    assert_eq!(
        cloned.applicable_codes.len(),
        template.applicable_codes.len()
    );
}

#[test]
fn test_placeholder_clone() {
    let ph = Placeholder::new("test", "Test desc", PlaceholderConstraint::Literal);
    let cloned = ph.clone();
    assert_eq!(cloned.name, ph.name);
    assert_eq!(cloned.description, ph.description);
    assert_eq!(cloned.constraint, ph.constraint);
}

#[test]
fn test_error_fix_pattern_debug() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 12345,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 5,
        failure_count: 2,
    };
    let debug = format!("{pattern:?}");
    assert!(debug.contains("ErrorFixPattern"));
    assert!(debug.contains("12345"));
}

#[test]
fn test_fix_template_debug() {
    let template = FixTemplate::new("pattern", "desc");
    let debug = format!("{template:?}");
    assert!(debug.contains("FixTemplate"));
}

#[test]
fn test_placeholder_debug() {
    let ph = Placeholder::expression("x");
    let debug = format!("{ph:?}");
    assert!(debug.contains("Placeholder"));
}

#[test]
fn test_pattern_library_debug() {
    let lib = PatternLibrary::new();
    let debug = format!("{lib:?}");
    assert!(debug.contains("PatternLibrary"));
}

#[test]
fn test_pattern_stats_clone() {
    let mut stats = PatternStats::new();
    stats.record(0, true);
    stats.record(0, false);
    let cloned = stats.clone();
    assert!((cloned.success_rate(0) - stats.success_rate(0)).abs() < 0.001);
}

#[test]
fn test_pattern_stats_multiple_patterns() {
    let mut stats = PatternStats::new();
    stats.record(0, true);
    stats.record(0, true);
    stats.record(1, false);
    stats.record(1, false);
    stats.record(2, true);

    assert!((stats.success_rate(0) - 1.0).abs() < 0.001);
    assert!((stats.success_rate(1) - 0.0).abs() < 0.001);
    assert!((stats.success_rate(2) - 1.0).abs() < 0.001);
}

#[test]
fn test_placeholder_constraint_copy() {
    let constraint = PlaceholderConstraint::Type;
    let copied = constraint;
    assert_eq!(copied, PlaceholderConstraint::Type);
}

#[test]
fn test_fix_template_apply_partial_bindings() {
    let template = FixTemplate::new("$a + $b + $c", "Add three");
    let mut bindings = HashMap::new();
    bindings.insert("a".to_string(), "x".to_string());
    bindings.insert("c".to_string(), "z".to_string());
    // $b is not bound
    let result = template.apply(&bindings);
    assert_eq!(result, "x + $b + z");
}

#[test]
fn test_save_load_empty_strings() {
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let template = FixTemplate::new("", "");
    lib.add_pattern(ErrorEmbedding::new(vec![1.0], code, 0), template);

    let path = "/tmp/citl_test_empty_str.pat";
    lib.save(path).expect("Save should succeed");
    let loaded = PatternLibrary::load(path).expect("Load should succeed");

    let p = loaded.get(0).expect("Pattern should exist");
    assert!(p.error_code.code.is_empty());
    assert!(p.fix_template.pattern.is_empty());

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_save_load_unicode_strings() {
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let template = FixTemplate::new(
        "$expr.to_string() // 変換",
        "Unicode description: 日本語テスト",
    );
    lib.add_pattern(ErrorEmbedding::new(vec![1.0, 2.0], code, 0), template);

    let path = "/tmp/citl_test_unicode.pat";
    lib.save(path).expect("Save should succeed");
    let loaded = PatternLibrary::load(path).expect("Load should succeed");

    let p = loaded.get(0).expect("Pattern should exist");
    assert!(p.fix_template.pattern.contains("変換"));
    assert!(p.fix_template.description.contains("日本語"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_cosine_similarity_negative_values() {
    let a = vec![-1.0, -2.0, -3.0];
    let b = vec![-1.0, -2.0, -3.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_mixed_values() {
    let a = vec![1.0, -1.0, 0.0];
    let b = vec![-1.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    // Opposite vectors should have similarity -1
    assert!((sim + 1.0).abs() < 0.001);
}

#[test]
fn test_pattern_library_search_all_similar() {
    // All patterns have the same embedding
    let mut lib = PatternLibrary::new();
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let vec = vec![1.0, 1.0, 1.0, 1.0];

    for i in 0..5 {
        lib.add_pattern(
            ErrorEmbedding::new(vec.clone(), code.clone(), i as u64),
            FixTemplate::new(&format!("fix{i}"), &format!("Fix {i}")),
        );
    }

    let query = ErrorEmbedding::new(vec.clone(), code, 0);
    let results = lib.search(&query, 3);
    assert_eq!(results.len(), 3);
    // All should have similarity 1.0
    for result in &results {
        assert!((result.similarity - 1.0).abs() < 0.001);
    }
}

#[test]
fn test_error_fix_pattern_all_failures() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 0,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 0,
        failure_count: 100,
    };
    assert!((pattern.success_rate() - 0.0).abs() < 0.001);
    assert_eq!(pattern.total_applications(), 100);
}

#[test]
fn test_error_fix_pattern_all_successes() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let pattern = ErrorFixPattern {
        error_code: code,
        context_hash: 0,
        fix_template: FixTemplate::new("fix", "Fix"),
        success_count: 50,
        failure_count: 0,
    };
    assert!((pattern.success_rate() - 1.0).abs() < 0.001);
}

#[test]
fn test_templates_all_have_pattern() {
    let all = templates::all_templates();
    for template in &all {
        assert!(!template.pattern.is_empty(), "Template has empty pattern");
    }
}

#[test]
fn test_templates_all_have_description() {
    let all = templates::all_templates();
    for template in &all {
        assert!(
            !template.description.is_empty(),
            "Template has empty description"
        );
    }
}
