
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
