
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
