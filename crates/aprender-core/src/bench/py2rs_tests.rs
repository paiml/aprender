pub(crate) use super::*;

#[test]
fn test_level_all() {
    let levels = Py2RsLevel::all();
    assert_eq!(levels.len(), 10);
    assert_eq!(levels[0], Py2RsLevel::Hello);
    assert_eq!(levels[9], Py2RsLevel::Metaprogramming);
}

#[test]
fn test_level_numbers() {
    assert_eq!(Py2RsLevel::Hello.number(), 1);
    assert_eq!(Py2RsLevel::Metaprogramming.number(), 10);
}

#[test]
fn test_level_weights() {
    // Sum should be 68.5
    let total: f32 = Py2RsLevel::all().iter().map(|l| l.weight()).sum();
    assert!((total - 68.5).abs() < 0.01);
}

#[test]
fn test_level_difficulty() {
    assert_eq!(Py2RsLevel::Hello.difficulty(), Difficulty::Trivial);
    assert_eq!(Py2RsLevel::ControlFlow.difficulty(), Difficulty::Medium);
    assert_eq!(Py2RsLevel::Metaprogramming.difficulty(), Difficulty::Expert);
}

#[test]
fn test_py2rs_score_creation() {
    let mut score = Py2RsScore::new("test-model");
    assert_eq!(score.max_level, 0);
    assert!(score.single_shot_levels.is_empty());

    score.add_level(LevelResult::passed(
        Py2RsLevel::Hello,
        1,
        Duration::from_millis(50),
    ));
    score.add_level(LevelResult::passed(
        Py2RsLevel::Variables,
        2,
        Duration::from_millis(100),
    ));
    score.finalize();

    assert_eq!(score.max_level, 2);
    assert_eq!(score.single_shot_levels.len(), 1); // Only Hello was turn 1
}

#[test]
fn test_py2rs_score_composite() {
    let mut score = Py2RsScore::new("perfect");

    // Pass all levels on turn 1
    for level in Py2RsLevel::all() {
        score.add_level(LevelResult::passed(level, 1, Duration::from_millis(10)));
    }
    score.finalize();

    assert!((score.composite - 100.0).abs() < 0.01);
}

#[test]
fn test_py2rs_score_partial() {
    let mut score = Py2RsScore::new("partial");

    // Pass only level 1 on turn 1 (weight 1.0)
    score.add_level(LevelResult::passed(
        Py2RsLevel::Hello,
        1,
        Duration::from_millis(10),
    ));
    score.finalize();

    // 1.0 / 68.5 * 100 = 1.46%
    assert!(score.composite > 1.0 && score.composite < 2.0);
}

#[test]
fn test_level_result_creation() {
    let passed = LevelResult::passed(Py2RsLevel::Functions, 2, Duration::from_millis(100));
    assert!(passed.passed);
    assert_eq!(passed.turn, 2);
    assert_eq!(passed.level, 3);

    let failed = LevelResult::failed(
        Py2RsLevel::Concurrency,
        5,
        "async not supported",
        Duration::from_secs(1),
    );
    assert!(!failed.passed);
    assert!(failed.error.is_some());
}

#[test]
fn test_visual_summary() {
    let mut score = Py2RsScore::new("test");
    score.add_level(LevelResult::passed(Py2RsLevel::Hello, 1, Duration::ZERO));
    score.add_level(LevelResult::passed(
        Py2RsLevel::Variables,
        2,
        Duration::ZERO,
    ));
    score.add_level(LevelResult::failed(
        Py2RsLevel::Functions,
        5,
        "error",
        Duration::ZERO,
    ));

    let visual = score.visual_summary();
    assert!(visual.contains('●')); // Turn 1 pass
    assert!(visual.contains('◐')); // Turn 2+ pass
    assert!(visual.contains('○')); // Failed
}

#[test]
fn test_generate_canonical_examples() {
    let examples = generate_canonical_examples();
    assert_eq!(examples.len(), 10);
    assert!(examples[0].id.contains("L1"));
    assert!(examples[9].id.contains("L10"));
}

#[test]
fn test_run_benchmark() {
    let score = run_benchmark("test-6b", 5);
    assert!(score.max_level > 0);
    assert!(score.composite > 0.0);
}

#[test]
fn test_compare_models() {
    let models = vec![
        ("model-2b", 4_000_000_000_u64),
        ("model-6b", 12_000_000_000_u64),
        ("model-16b", 32_000_000_000_u64),
    ];

    let comparison = compare_models(&models, 5);

    assert_eq!(comparison.results.len(), 3);
    assert!(!comparison.pareto_frontier.is_empty());
    assert!(!comparison.recommendations.is_empty());
}

#[test]
fn test_python_examples() {
    for level in Py2RsLevel::all() {
        let python = level.python_example();
        assert!(!python.is_empty());
        // Each example should have some Python-specific syntax
        assert!(
            python.contains("def ")
                || python.contains("print")
                || python.contains("class ")
                || python.contains("import ")
                || python.contains("@")
                || python.contains("=")
        );
    }
}

#[test]
fn test_format_comparison_table() {
    let models = vec![("small", 1000_u64), ("large", 10000_u64)];
    let comparison = compare_models(&models, 3);

    let scores: Vec<_> = models.iter().map(|(id, _)| run_benchmark(id, 3)).collect();

    let table = format_comparison_table(&comparison, &scores);

    assert!(table.contains("py2rs-canonical"));
    assert!(table.contains("small"));
    assert!(table.contains("large"));
    assert!(table.contains("Legend"));
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_level_names() {
    assert_eq!(Py2RsLevel::Hello.name(), "Hello");
    assert_eq!(Py2RsLevel::Variables.name(), "Variables");
    assert_eq!(Py2RsLevel::Functions.name(), "Functions");
    assert_eq!(Py2RsLevel::Collections.name(), "Collections");
    assert_eq!(Py2RsLevel::ControlFlow.name(), "ControlFlow");
    assert_eq!(Py2RsLevel::ErrorHandling.name(), "ErrorHandling");
    assert_eq!(Py2RsLevel::OopTraits.name(), "OOP→Traits");
    assert_eq!(Py2RsLevel::Concurrency.name(), "Concurrency");
    assert_eq!(Py2RsLevel::FfiUnsafe.name(), "FFI/Unsafe");
    assert_eq!(Py2RsLevel::Metaprogramming.name(), "Metaprogramming");
}

#[test]
fn test_level_difficulty_all() {
    assert_eq!(Py2RsLevel::Variables.difficulty(), Difficulty::Trivial);
    assert_eq!(Py2RsLevel::Functions.difficulty(), Difficulty::Easy);
    assert_eq!(Py2RsLevel::Collections.difficulty(), Difficulty::Easy);
    assert_eq!(Py2RsLevel::ControlFlow.difficulty(), Difficulty::Medium);
    assert_eq!(Py2RsLevel::ErrorHandling.difficulty(), Difficulty::Medium);
    assert_eq!(Py2RsLevel::OopTraits.difficulty(), Difficulty::Hard);
    assert_eq!(Py2RsLevel::Concurrency.difficulty(), Difficulty::Hard);
    assert_eq!(Py2RsLevel::FfiUnsafe.difficulty(), Difficulty::Expert);
}

#[test]
fn test_level_weight_all() {
    assert!((Py2RsLevel::Variables.weight() - 1.5).abs() < 0.01);
    assert!((Py2RsLevel::Functions.weight() - 2.0).abs() < 0.01);
    assert!((Py2RsLevel::Collections.weight() - 3.0).abs() < 0.01);
    assert!((Py2RsLevel::ControlFlow.weight() - 4.0).abs() < 0.01);
    assert!((Py2RsLevel::ErrorHandling.weight() - 5.0).abs() < 0.01);
    assert!((Py2RsLevel::OopTraits.weight() - 7.0).abs() < 0.01);
    assert!((Py2RsLevel::Concurrency.weight() - 10.0).abs() < 0.01);
    assert!((Py2RsLevel::FfiUnsafe.weight() - 15.0).abs() < 0.01);
    assert!((Py2RsLevel::Metaprogramming.weight() - 20.0).abs() < 0.01);
}

#[test]
fn test_level_symbol_missing() {
    let score = Py2RsScore::new("empty");
    // No level results added
    assert_eq!(score.level_symbol(1), '○');
    assert_eq!(score.level_symbol(10), '○');
}

#[test]
fn test_score_add_failed_level() {
    let mut score = Py2RsScore::new("test");
    score.add_level(LevelResult::failed(
        Py2RsLevel::Metaprogramming,
        5,
        "too hard",
        Duration::from_secs(1),
    ));

    assert_eq!(score.max_level, 0); // Failed doesn't update max
    assert!(score.single_shot_levels.is_empty());
}

#[test]
fn test_score_zero_composite() {
    let mut score = Py2RsScore::new("fail");
    // Add only failed levels (no turn 1 successes)
    for level in Py2RsLevel::all() {
        score.add_level(LevelResult::failed(level, 5, "error", Duration::ZERO));
    }
    score.finalize();

    assert!((score.composite - 0.0).abs() < 0.01);
}

#[test]
fn test_mock_model_result_variations() {
    // Test different model sizes
    let (passed_2b, _) = mock_model_result(Py2RsLevel::Hello, "model-2b");
    let (passed_large, _) = mock_model_result(Py2RsLevel::Hello, "model-large");
    let (passed_unknown, _) = mock_model_result(Py2RsLevel::Hello, "random");

    assert!(passed_2b);
    assert!(passed_large);
    assert!(passed_unknown);
}

#[test]
fn test_mock_model_high_level() {
    // Level 8 (Concurrency) is hard for small models but achievable for large
    let (passed_small, _) = mock_model_result(Py2RsLevel::Concurrency, "model-2b");
    let (passed_large, _) = mock_model_result(Py2RsLevel::Concurrency, "model-16b");

    assert!(!passed_small); // 2b can't do level 8 (capability 5)
    assert!(passed_large); // 16b can do level 8 (capability 9)
}

#[test]
fn test_level_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Py2RsLevel::Hello);
    set.insert(Py2RsLevel::Variables);
    assert!(set.contains(&Py2RsLevel::Hello));
    assert!(!set.contains(&Py2RsLevel::Functions));
}

#[test]
fn test_level_clone_copy() {
    let level = Py2RsLevel::Concurrency;
    let copied = level;
    let cloned = level.clone();
    assert_eq!(level, copied);
    assert_eq!(level, cloned);
}

#[test]
fn test_py2rs_score_debug() {
    let score = Py2RsScore::new("debug-test");
    let debug_str = format!("{:?}", score);
    assert!(debug_str.contains("Py2RsScore"));
    assert!(debug_str.contains("debug-test"));
}

#[test]
fn test_level_result_debug() {
    let result = LevelResult::passed(Py2RsLevel::Hello, 1, Duration::from_millis(100));
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("LevelResult"));
}

#[test]
fn test_py2rs_score_clone() {
    let mut score = Py2RsScore::new("clone-test");
    score.add_level(LevelResult::passed(Py2RsLevel::Hello, 1, Duration::ZERO));
    score.finalize();

    let cloned = score.clone();
    assert_eq!(cloned.model_id, score.model_id);
    assert_eq!(cloned.max_level, score.max_level);
}

#[test]
fn test_level_result_clone() {
    let result = LevelResult::passed(Py2RsLevel::Functions, 2, Duration::from_millis(50));
    let cloned = result.clone();
    assert_eq!(cloned.level, result.level);
    assert_eq!(cloned.passed, result.passed);
}
