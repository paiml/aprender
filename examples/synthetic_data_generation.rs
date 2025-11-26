//! Example: Synthetic Data Generation for ML Training
//!
//! This example demonstrates how to use aprender's synthetic data generation
//! module to augment training data for machine learning models.
//!
//! Run with: cargo run --example synthetic_data_generation

use aprender::synthetic::cache::SyntheticCache;
use aprender::synthetic::eda::{EdaConfig, EdaGenerator};
use aprender::synthetic::template::{Template, TemplateGenerator};
use aprender::synthetic::weak_supervision::{
    AggregationStrategy, KeywordLF, LabelVote, WeakSupervisionConfig, WeakSupervisionGenerator,
};
use aprender::synthetic::{SyntheticConfig, SyntheticGenerator};

fn main() {
    println!("=== Synthetic Data Generation Demo ===\n");

    // Demo 1: EDA (Easy Data Augmentation) for text
    demo_eda_augmentation();

    // Demo 2: Template-based generation
    demo_template_generation();

    // Demo 3: Weak supervision labeling
    demo_weak_supervision();

    // Demo 4: Caching for efficient generation
    demo_caching();

    println!("\n=== Demo Complete ===");
}

fn demo_eda_augmentation() {
    println!("--- 1. EDA Text Augmentation ---\n");

    let generator = EdaGenerator::new(EdaConfig::default());

    let seeds = vec![
        "git commit -m 'fix bug'".to_string(),
        "cargo build --release".to_string(),
        "docker run nginx".to_string(),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(2.0) // Generate 2x original
        .with_quality_threshold(0.3)
        .with_seed(42);

    println!("Original commands ({}):", seeds.len());
    for cmd in &seeds {
        println!("  {cmd}");
    }

    let augmented = generator
        .generate(&seeds, &config)
        .expect("EDA generation failed");

    println!("\nAugmented commands ({}):", augmented.len());
    for (i, cmd) in augmented.iter().take(6).enumerate() {
        let quality = generator.quality_score(cmd, &seeds[i % seeds.len()]);
        println!("  {cmd} (quality: {quality:.2})");
    }

    let diversity = generator.diversity_score(&augmented);
    println!("\nDiversity score: {diversity:.3}");
}

fn demo_template_generation() {
    println!("\n--- 2. Template-Based Generation ---\n");

    // Define templates with slot patterns
    let git_template = Template::new("git {action} {args}")
        .with_slot("action", &["commit", "push", "pull", "checkout", "merge"])
        .with_slot("args", &["-m 'update'", "--all", "main", "-b feature"]);

    let cargo_template = Template::new("cargo {cmd} {flags}")
        .with_slot("cmd", &["build", "test", "run", "check", "clippy"])
        .with_slot("flags", &["--release", "--all-features", "--verbose", ""]);

    let generator = TemplateGenerator::new()
        .with_template(git_template)
        .with_template(cargo_template);

    // Provide seeds to determine output count
    let seeds = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(3.0)
        .with_seed(123);

    let generated = generator
        .generate(&seeds, &config)
        .expect("Template generation failed");

    println!("Generated from templates ({}):", generated.len());
    for cmd in generated.iter().take(8) {
        println!("  {cmd}");
    }

    println!(
        "\nTotal possible combinations: {}",
        generator.total_combinations()
    );
}

fn demo_weak_supervision() {
    println!("\n--- 3. Weak Supervision Labeling ---\n");

    let mut generator = WeakSupervisionGenerator::<String>::new().with_config(
        WeakSupervisionConfig::new()
            .with_aggregation(AggregationStrategy::MajorityVote)
            .with_min_votes(1)
            .with_min_confidence(0.5),
    );

    // Add labeling functions (domain heuristics)
    generator.add_lf(Box::new(KeywordLF::new(
        "version_control",
        &["git", "svn", "hg", "commit", "push", "pull"],
        LabelVote::Positive,
    )));

    generator.add_lf(Box::new(KeywordLF::new(
        "build_tool",
        &["cargo", "make", "npm", "build", "compile"],
        LabelVote::Positive,
    )));

    generator.add_lf(Box::new(KeywordLF::new(
        "dangerous",
        &["rm -rf", "sudo rm", "format", "dd if="],
        LabelVote::Negative,
    )));

    // Unlabeled samples to classify
    let samples = vec![
        "git push origin main".to_string(),
        "cargo test --all".to_string(),
        "rm -rf /tmp/cache".to_string(),
        "echo hello world".to_string(),
        "npm install lodash".to_string(),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.0)
        .with_quality_threshold(0.0);

    let labeled = generator
        .generate(&samples, &config)
        .expect("Weak supervision failed");

    println!("Labeled samples:");
    for sample in &labeled {
        let label_str = match sample.label {
            1 => "SAFE",
            -1 => "UNSAFE",
            _ => "UNKNOWN",
        };
        println!(
            "  [{}] (conf: {:.2}) {}",
            label_str, sample.confidence, sample.sample
        );
    }

    println!("\nLabeling functions: {}", generator.num_lfs());
}

fn demo_caching() {
    println!("\n--- 4. Cached Generation ---\n");

    let mut cache = SyntheticCache::<String>::new(100_000); // 100KB cache
    let generator = EdaGenerator::new(EdaConfig::default());

    let seeds = vec!["ls -la".to_string(), "pwd".to_string()];
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.5)
        .with_seed(42);

    // First call - cache miss
    let _result1 = cache
        .get_or_generate(&seeds, &config, &generator)
        .expect("First generation failed");

    println!("After first call:");
    println!("  Cache misses: {}", cache.stats().misses);
    println!("  Cache hits: {}", cache.stats().hits);
    println!("  Generations: {}", cache.stats().generations);

    // Second call - cache hit
    let _result2 = cache
        .get_or_generate(&seeds, &config, &generator)
        .expect("Second call failed");

    println!("\nAfter second call:");
    println!("  Cache misses: {}", cache.stats().misses);
    println!("  Cache hits: {}", cache.stats().hits);
    println!("  Generations: {}", cache.stats().generations);
    println!("  Hit rate: {:.1}%", cache.stats().hit_rate() * 100.0);
    println!("  Cache size: {} bytes", cache.size());
    println!("  Entries: {}", cache.len());
}
