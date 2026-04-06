// ============================================================================
// Cache Integration Tests
// ============================================================================

#[test]
fn test_cache_with_eda_generator() {
    let mut cache = SyntheticCache::<String>::new(100_000);
    let generator = EdaGenerator::new(EdaConfig::default());

    let seeds = vec!["git status".to_string(), "cargo build".to_string()];
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.0)
        .with_seed(42);

    let result1 = cache
        .get_or_generate(&seeds, &config, &generator)
        .expect("First generation should succeed");

    assert_eq!(cache.stats().misses, 1, "First call should be a miss");
    assert_eq!(cache.stats().generations, 1, "Should have one generation");

    let result2 = cache
        .get_or_generate(&seeds, &config, &generator)
        .expect("Cached retrieval should succeed");

    assert_eq!(result1, result2, "Cached result should match original");
    assert_eq!(cache.stats().hits, 1, "Second call should be a hit");
    assert_eq!(
        cache.stats().generations,
        1,
        "Should still have one generation"
    );

    let hit_rate = cache.stats().hit_rate();
    assert!(
        (hit_rate - 0.5).abs() < f32::EPSILON,
        "Hit rate should be 50%"
    );
}

#[test]
fn test_cache_lru_eviction_integration() {
    let mut cache = SyntheticCache::<String>::new(500);
    let generator = EdaGenerator::new(EdaConfig::default());

    for i in 0..5 {
        let seeds = vec![format!("command_{}", i)];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_seed(i as u64);
        cache
            .get_or_generate(&seeds, &config, &generator)
            .expect("Generation should succeed");
    }

    assert!(
        cache.stats().evictions > 0,
        "Should have evicted entries: {}",
        cache.stats().evictions
    );
}

// ============================================================================
// End-to-End Pipeline Tests
// ============================================================================

#[test]
fn test_full_synthetic_pipeline() {
    let mut cache = SyntheticCache::<String>::new(50_000);

    let generator = EdaGenerator::new(EdaConfig::default());

    let seeds = vec![
        "git status".to_string(),
        "cargo test".to_string(),
        "docker ps".to_string(),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(2.0)
        .with_quality_threshold(0.3)
        .with_seed(42);

    let result = cache
        .get_or_generate(&seeds, &config, &generator)
        .expect("Pipeline should succeed");

    assert!(!result.is_empty(), "Should generate samples");
    assert_eq!(cache.stats().generations, 1, "Should cache generation");

    let mut total_quality = 0.0;
    for (generated, seed) in result.iter().zip(seeds.iter().cycle()) {
        total_quality += generator.quality_score(generated, seed);
    }
    let avg_quality = total_quality / result.len() as f32;
    assert!(avg_quality > 0.0, "Should have positive average quality");
}

#[test]
fn test_multiple_generators_pipeline() {
    let eda_gen = EdaGenerator::new(EdaConfig::default());

    let git_template =
        Template::new("git {action}").with_slot("action", &["status", "log", "diff", "branch"]);
    let cargo_template =
        Template::new("cargo {action}").with_slot("action", &["build", "test", "run", "check"]);

    let template_gen = TemplateGenerator::new()
        .with_template(git_template)
        .with_template(cargo_template);

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(2.0)
        .with_quality_threshold(0.0)
        .with_seed(42);

    let placeholder_seeds = vec![
        "seed1".to_string(),
        "seed2".to_string(),
        "seed3".to_string(),
    ];
    let template_results = template_gen
        .generate(&placeholder_seeds, &config)
        .expect("Template generation should succeed");

    assert!(
        !template_results.is_empty(),
        "Template should generate results"
    );

    let eda_results = eda_gen
        .generate(&template_results, &config)
        .expect("EDA generation should succeed");

    assert!(!eda_results.is_empty(), "EDA should generate results");

    let diversity = eda_gen.diversity_score(&eda_results);
    assert!(
        diversity >= 0.0,
        "Final results should have non-negative diversity"
    );
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[test]
fn test_generators_never_panic_on_empty_input() {
    let eda_gen = EdaGenerator::new(EdaConfig::default());
    let template_gen = TemplateGenerator::new();

    let empty: Vec<String> = vec![];
    let config = SyntheticConfig::default();

    let eda_result = eda_gen.generate(&empty, &config);
    assert!(eda_result.is_ok(), "EDA should handle empty input");

    let template_result = template_gen.generate(&empty, &config);
    assert!(
        template_result.is_ok(),
        "Template should handle empty input"
    );
}

#[test]
fn test_quality_scores_bounded() {
    let eda_gen = EdaGenerator::new(EdaConfig::default());

    let samples = vec!["test".to_string(), "command".to_string()];
    let config = SyntheticConfig::default().with_augmentation_ratio(1.0);

    if let Ok(results) = eda_gen.generate(&samples, &config) {
        for (gen, seed) in results.iter().zip(samples.iter().cycle()) {
            let q = eda_gen.quality_score(gen, seed);
            assert!((0.0..=1.0).contains(&q), "Quality must be in [0, 1]: {q}");
        }
    }
}

#[test]
fn test_diversity_scores_bounded() {
    let eda_gen = EdaGenerator::new(EdaConfig::default());

    let samples = vec![
        "git status".to_string(),
        "cargo build".to_string(),
        "docker run".to_string(),
    ];
    let config = SyntheticConfig::default().with_augmentation_ratio(2.0);

    if let Ok(results) = eda_gen.generate(&samples, &config) {
        let diversity = eda_gen.diversity_score(&results);
        assert!(
            (0.0..=1.0).contains(&diversity),
            "Diversity must be in [0, 1]: {diversity}"
        );
    }
}

#[test]
fn test_config_builder_methods() {
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.5)
        .with_quality_threshold(0.5)
        .with_diversity_weight(0.3)
        .with_max_attempts(10)
        .with_seed(12345);

    assert!((config.augmentation_ratio - 1.5).abs() < f32::EPSILON);
    assert!((config.quality_threshold - 0.5).abs() < f32::EPSILON);
    assert!((config.diversity_weight - 0.3).abs() < f32::EPSILON);
    assert_eq!(config.max_attempts, 10);
    assert_eq!(config.seed, 12345);
}
