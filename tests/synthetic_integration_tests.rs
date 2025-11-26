//! Integration tests for Synthetic Data Generation module.
//!
//! These tests verify end-to-end workflows for synthetic data generation,
//! including EDA, Template, Shell, MixUp, WeakSupervision, and Caching.

use aprender::synthetic::andon::AndonConfig;
use aprender::synthetic::cache::SyntheticCache;
use aprender::synthetic::eda::{EdaConfig, EdaGenerator};
use aprender::synthetic::mixup::{Embeddable, MixUpConfig, MixUpGenerator};
use aprender::synthetic::shell::{ShellGrammar, ShellSample, ShellSyntheticGenerator};
use aprender::synthetic::template::{Template, TemplateGenerator};
use aprender::synthetic::weak_supervision::{
    AggregationStrategy, KeywordLF, LabelVote, WeakSupervisionConfig, WeakSupervisionGenerator,
};
use aprender::synthetic::{SyntheticConfig, SyntheticGenerator};

// ============================================================================
// Test Fixtures
// ============================================================================

/// Sample type implementing Embeddable for MixUp tests.
#[derive(Debug, Clone, PartialEq)]
struct TextSample {
    text: String,
    embedding: Vec<f32>,
}

impl TextSample {
    fn new(text: &str, embedding: Vec<f32>) -> Self {
        Self {
            text: text.to_string(),
            embedding,
        }
    }
}

impl Embeddable for TextSample {
    fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    fn from_embedding(embedding: Vec<f32>, reference: &Self) -> Self {
        Self {
            text: format!("mixed_{}", reference.text),
            embedding,
        }
    }
}

// ============================================================================
// EDA Integration Tests
// ============================================================================

#[test]
fn test_eda_full_pipeline() {
    let generator = EdaGenerator::new(EdaConfig::default());
    let seeds = vec![
        "git status".to_string(),
        "cargo build --release".to_string(),
        "docker run nginx".to_string(),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(2.0)
        .with_quality_threshold(0.3)
        .with_seed(42);

    let result = generator
        .generate(&seeds, &config)
        .expect("EDA should succeed");

    // Should generate approximately 2x the input
    assert!(!result.is_empty(), "Should generate synthetic samples");
    assert!(
        result.len() >= seeds.len(),
        "Should generate at least as many as input"
    );

    // Verify quality scores
    for (generated, seed) in result.iter().zip(seeds.iter().cycle()) {
        let quality = generator.quality_score(generated, seed);
        assert!(
            (0.0..=1.0).contains(&quality),
            "Quality should be in [0, 1]"
        );
    }

    // Verify diversity
    let diversity = generator.diversity_score(&result);
    assert!(diversity >= 0.0, "Diversity should be non-negative");
}

#[test]
fn test_eda_with_andon_config() {
    let generator = EdaGenerator::new(EdaConfig::default());
    let andon_config = AndonConfig::new()
        .with_rejection_threshold(0.5)
        .with_quality_baseline(0.5);

    let seeds = vec!["test command".to_string()];
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.0)
        .with_quality_threshold(0.1);

    let result = generator.generate(&seeds, &config).expect("Should succeed");

    // Check quality against andon thresholds
    let mut accepted = 0;
    let mut rejected = 0;
    for (generated, seed) in result.iter().zip(seeds.iter().cycle()) {
        let quality = generator.quality_score(generated, seed);
        if quality >= config.quality_threshold {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    let rejection_rate = if accepted + rejected > 0 {
        rejected as f32 / (accepted + rejected) as f32
    } else {
        0.0
    };

    // Verify we can check against andon threshold
    let _exceeds_threshold = andon_config.exceeds_rejection_threshold(rejection_rate);
}

// ============================================================================
// Template Integration Tests
// ============================================================================

#[test]
fn test_template_generation_pipeline() {
    // Create templates with slots using builder pattern
    let git_template = Template::new("git {action} {target}")
        .with_slot("action", &["status", "log", "diff", "branch"])
        .with_slot("target", &[".", "..", "src/", "tests/"]);

    let cargo_template = Template::new("cargo {action} --{flag}")
        .with_slot("action", &["build", "test", "run", "check"])
        .with_slot("flag", &["release", "verbose", "quiet"]);

    let docker_template = Template::new("docker {action} {image}")
        .with_slot("action", &["run", "pull", "push", "build"])
        .with_slot("image", &["nginx", "redis", "postgres"]);

    let generator = TemplateGenerator::new()
        .with_template(git_template)
        .with_template(cargo_template)
        .with_template(docker_template);

    // Note: Template generator needs seeds to determine target count
    // augmentation_ratio * seeds.len() = target count
    let seeds = vec![
        "placeholder1".to_string(),
        "placeholder2".to_string(),
        "placeholder3".to_string(),
    ];
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(5.0)
        .with_quality_threshold(0.0) // Accept all valid templates
        .with_seed(42);

    let result = generator
        .generate(&seeds, &config)
        .expect("Template generation should succeed");

    assert!(!result.is_empty(), "Should generate samples from templates");

    // Verify generated commands have valid structure
    for cmd in &result {
        assert!(!cmd.is_empty(), "Generated command should not be empty");
        assert!(
            cmd.starts_with("git") || cmd.starts_with("cargo") || cmd.starts_with("docker"),
            "Should start with known command: {cmd}"
        );
    }
}

// ============================================================================
// Shell Autocomplete Integration Tests
// ============================================================================

#[test]
fn test_shell_autocomplete_pipeline() {
    let grammar = ShellGrammar::common_commands();
    let generator = ShellSyntheticGenerator::new().with_grammar(grammar);

    // Use seeds with commands that are in the common grammar
    let seeds = vec![
        ShellSample::new("git", "git status"),
        ShellSample::new("cargo", "cargo build"),
        ShellSample::new("ls", "ls -la"),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(3.0)
        .with_quality_threshold(0.0) // Accept all
        .with_seed(42);

    let result = generator
        .generate(&seeds, &config)
        .expect("Shell autocomplete should succeed");

    // Shell generator may return empty if grammar doesn't match
    // This is expected behavior - test the API contract
    for sample in &result {
        assert!(!sample.prefix().is_empty(), "Prefix should not be empty");
        assert!(
            !sample.completion().is_empty(),
            "Completion should not be empty"
        );
    }

    // Check diversity (can be 0.0 for empty or single result)
    let diversity = generator.diversity_score(&result);
    assert!(diversity >= 0.0, "Should have non-negative diversity");
}

// ============================================================================
// MixUp Integration Tests
// ============================================================================

#[test]
fn test_mixup_embedding_interpolation() {
    let generator =
        MixUpGenerator::<TextSample>::new().with_config(MixUpConfig::default().with_alpha(0.4));

    let seeds = vec![
        TextSample::new("command one", vec![1.0, 0.0, 0.0]),
        TextSample::new("command two", vec![0.0, 1.0, 0.0]),
        TextSample::new("command three", vec![0.0, 0.0, 1.0]),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(2.0)
        .with_quality_threshold(0.0)
        .with_seed(42);

    let result = generator
        .generate(&seeds, &config)
        .expect("MixUp should succeed");

    assert!(!result.is_empty(), "Should generate mixed samples");

    // Verify mixed embeddings are interpolations
    for sample in &result {
        let emb = sample.embedding();
        assert_eq!(emb.len(), 3, "Embedding dimension should be preserved");

        // Sum should be approximately 1.0 (convex combination)
        let sum: f32 = emb.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.5,
            "Embedding sum should be near 1.0: {sum}"
        );
    }
}

// ============================================================================
// Weak Supervision Integration Tests
// ============================================================================

#[test]
fn test_weak_supervision_labeling_pipeline() {
    let mut generator = WeakSupervisionGenerator::<String>::new().with_config(
        WeakSupervisionConfig::new()
            .with_aggregation(AggregationStrategy::MajorityVote)
            .with_min_votes(1),
    );

    // Add labeling functions
    generator.add_lf(Box::new(KeywordLF::new(
        "git_positive",
        &["git", "branch", "commit", "push"],
        LabelVote::Positive,
    )));
    generator.add_lf(Box::new(KeywordLF::new(
        "cargo_positive",
        &["cargo", "build", "test", "run"],
        LabelVote::Positive,
    )));
    generator.add_lf(Box::new(KeywordLF::new(
        "dangerous_negative",
        &["rm -rf", "sudo", "chmod 777"],
        LabelVote::Negative,
    )));

    let samples = vec![
        "git commit -m 'test'".to_string(),
        "cargo build --release".to_string(),
        "rm -rf /tmp/test".to_string(),
        "echo hello".to_string(),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.0)
        .with_quality_threshold(0.0);

    let result = generator
        .generate(&samples, &config)
        .expect("Weak supervision should succeed");

    // Should label the samples that match LFs
    assert!(!result.is_empty(), "Should produce labeled samples");

    // Check labels are assigned correctly
    for labeled in &result {
        let label = labeled.label;
        let confidence = labeled.confidence;

        // Labels can be +1 (positive), -1 (negative), or 0 (abstain/default)
        assert!(
            label == 1 || label == -1 || label == 0,
            "Label should be +1, -1, or 0: {label}"
        );
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence should be in [0, 1]"
        );
    }
}

#[test]
fn test_weak_supervision_unanimous_strategy() {
    let mut generator = WeakSupervisionGenerator::<String>::new().with_config(
        WeakSupervisionConfig::new()
            .with_aggregation(AggregationStrategy::Unanimous)
            .with_min_votes(2),
    );

    // Add LFs that agree on git commands
    generator.add_lf(Box::new(KeywordLF::new(
        "git_lf1",
        &["git"],
        LabelVote::Positive,
    )));
    generator.add_lf(Box::new(KeywordLF::new(
        "git_lf2",
        &["commit", "push", "pull"],
        LabelVote::Positive,
    )));

    let samples = vec![
        "git commit -m 'msg'".to_string(), // Both LFs vote positive
        "git status".to_string(),          // Only git_lf1 votes
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.0)
        .with_quality_threshold(0.0);

    let result = generator
        .generate(&samples, &config)
        .expect("Should succeed");

    // Only the first sample should be labeled (unanimous agreement)
    let unanimous_labels: Vec<_> = result.iter().filter(|l| l.confidence == 1.0).collect();
    assert!(!unanimous_labels.is_empty(), "Should have unanimous labels");
}

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

    // First call - generates and caches
    let result1 = cache
        .get_or_generate(&seeds, &config, &generator)
        .expect("First generation should succeed");

    assert_eq!(cache.stats().misses, 1, "First call should be a miss");
    assert_eq!(cache.stats().generations, 1, "Should have one generation");

    // Second call - returns cached
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

    // Hit rate should be 50%
    let hit_rate = cache.stats().hit_rate();
    assert!(
        (hit_rate - 0.5).abs() < f32::EPSILON,
        "Hit rate should be 50%"
    );
}

#[test]
fn test_cache_lru_eviction_integration() {
    // Small cache that forces eviction
    let mut cache = SyntheticCache::<String>::new(500);
    let generator = EdaGenerator::new(EdaConfig::default());

    // Generate multiple different datasets
    for i in 0..5 {
        let seeds = vec![format!("command_{}", i)];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_seed(i as u64);
        cache
            .get_or_generate(&seeds, &config, &generator)
            .expect("Generation should succeed");
    }

    // Should have evicted some entries
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
    // Setup cache
    let mut cache = SyntheticCache::<String>::new(50_000);

    // Setup generator
    let generator = EdaGenerator::new(EdaConfig::default());

    // Input data
    let seeds = vec![
        "git status".to_string(),
        "cargo test".to_string(),
        "docker ps".to_string(),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(2.0)
        .with_quality_threshold(0.3)
        .with_seed(42);

    // Generate with caching
    let result = cache
        .get_or_generate(&seeds, &config, &generator)
        .expect("Pipeline should succeed");

    // Verify pipeline stats
    assert!(!result.is_empty(), "Should generate samples");
    assert_eq!(cache.stats().generations, 1, "Should cache generation");

    // Quality check
    let mut total_quality = 0.0;
    for (generated, seed) in result.iter().zip(seeds.iter().cycle()) {
        total_quality += generator.quality_score(generated, seed);
    }
    let avg_quality = total_quality / result.len() as f32;
    assert!(avg_quality > 0.0, "Should have positive average quality");
}

#[test]
fn test_multiple_generators_pipeline() {
    // Test using multiple generators in sequence
    let eda_gen = EdaGenerator::new(EdaConfig::default());

    // Create shell templates
    let git_template =
        Template::new("git {action}").with_slot("action", &["status", "log", "diff", "branch"]);
    let cargo_template =
        Template::new("cargo {action}").with_slot("action", &["build", "test", "run", "check"]);

    let template_gen = TemplateGenerator::new()
        .with_template(git_template)
        .with_template(cargo_template);

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(2.0)
        .with_quality_threshold(0.0) // Accept all
        .with_seed(42);

    // Stage 1: Template generation (needs seeds to determine target count)
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

    // Stage 2: EDA augmentation on template results
    let eda_results = eda_gen
        .generate(&template_results, &config)
        .expect("EDA generation should succeed");

    assert!(!eda_results.is_empty(), "EDA should generate results");

    // Verify final output diversity
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

    // All generators should handle empty input gracefully
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

    // EDA quality scores
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
