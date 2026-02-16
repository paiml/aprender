pub(crate) use super::*;

// ============================================================================
// EXTREME TDD: Template Tests
// ============================================================================

#[test]
fn test_template_new() {
    let t = Template::new("git {cmd}");
    assert_eq!(t.pattern(), "git {cmd}");
    assert!(t.slots.is_empty());
    assert!((t.weight() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_template_with_slot() {
    let t = Template::new("git {cmd}").with_slot("cmd", &["status", "log"]);

    assert_eq!(t.slot_names().len(), 1);
    let values = t.slot_values("cmd").expect("should have values");
    assert_eq!(values.len(), 2);
    assert!(values.contains(&"status".to_string()));
}

#[test]
fn test_template_with_weight() {
    let t = Template::new("test").with_weight(2.5);
    assert!((t.weight() - 2.5).abs() < f32::EPSILON);
}

#[test]
fn test_template_weight_non_negative() {
    let t = Template::new("test").with_weight(-1.0);
    assert!((t.weight() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_template_combination_count() {
    let t = Template::new("git {cmd} {target}")
        .with_slot("cmd", &["a", "b", "c"])
        .with_slot("target", &["x", "y"]);

    assert_eq!(t.combination_count(), 6); // 3 * 2
}

#[test]
fn test_template_combination_count_no_slots() {
    let t = Template::new("git status");
    assert_eq!(t.combination_count(), 1);
}

#[test]
fn test_template_fill() {
    let t = Template::new("git {cmd} {target}")
        .with_slot("cmd", &["push"])
        .with_slot("target", &["origin"]);

    let mut values = HashMap::new();
    values.insert("cmd".to_string(), "push".to_string());
    values.insert("target".to_string(), "origin".to_string());

    assert_eq!(t.fill(&values), "git push origin");
}

#[test]
fn test_template_fill_indexed() {
    let t = Template::new("{a} {b}")
        .with_slot("a", &["x", "y"])
        .with_slot("b", &["1", "2"]);

    // With sorted keys: a, b
    // Index 0: a[0]=x, b[0]=1 -> "x 1"
    // Index 1: a[1]=y, b[0]=1 -> "y 1"
    // Index 2: a[0]=x, b[1]=2 -> "x 2"
    // Index 3: a[1]=y, b[1]=2 -> "y 2"

    let s0 = t.fill_indexed(0);
    let s1 = t.fill_indexed(1);
    let s2 = t.fill_indexed(2);
    let s3 = t.fill_indexed(3);

    // All should be valid combinations
    let valid = ["x 1", "x 2", "y 1", "y 2"];
    assert!(valid.contains(&s0.as_str()));
    assert!(valid.contains(&s1.as_str()));
    assert!(valid.contains(&s2.as_str()));
    assert!(valid.contains(&s3.as_str()));
}

// ============================================================================
// EXTREME TDD: TemplateConfig Tests
// ============================================================================

#[test]
fn test_template_config_default() {
    let config = TemplateConfig::default();
    assert!(config.use_weights);
    assert!(config.unique_outputs);
    assert_eq!(config.max_unique_attempts, 100);
}

#[test]
fn test_template_config_builder() {
    let config = TemplateConfig::new()
        .with_use_weights(false)
        .with_unique_outputs(false)
        .with_max_unique_attempts(50);

    assert!(!config.use_weights);
    assert!(!config.unique_outputs);
    assert_eq!(config.max_unique_attempts, 50);
}

#[test]
fn test_template_config_min_attempts() {
    let config = TemplateConfig::new().with_max_unique_attempts(0);
    assert_eq!(config.max_unique_attempts, 1);
}

// ============================================================================
// EXTREME TDD: TemplateGenerator Tests
// ============================================================================

#[test]
fn test_template_generator_new() {
    let gen = TemplateGenerator::new();
    assert!(gen.templates().is_empty());
}

#[test]
fn test_template_generator_add_template() {
    let mut gen = TemplateGenerator::new();
    gen.add_template(Template::new("test"));
    assert_eq!(gen.templates().len(), 1);
}

#[test]
fn test_template_generator_with_template() {
    let gen = TemplateGenerator::new()
        .with_template(Template::new("a"))
        .with_template(Template::new("b"));

    assert_eq!(gen.templates().len(), 2);
}

#[test]
fn test_template_generator_total_combinations() {
    let gen = TemplateGenerator::new()
        .with_template(Template::new("{a}").with_slot("a", &["1", "2"]))
        .with_template(Template::new("{b}").with_slot("b", &["x", "y", "z"]));

    assert_eq!(gen.total_combinations(), 5); // 2 + 3
}

#[test]
fn test_template_generator_generate_samples() {
    let gen = TemplateGenerator::new()
        .with_template(Template::new("git {cmd}").with_slot("cmd", &["status", "log", "diff"]));

    let samples = gen.generate_samples(5, 42);
    assert_eq!(samples.len(), 5);

    for sample in &samples {
        assert!(sample.starts_with("git "));
    }
}

#[test]
fn test_template_generator_generate_samples_empty() {
    let gen = TemplateGenerator::new();
    let samples = gen.generate_samples(5, 42);
    assert!(samples.is_empty());
}

#[test]
fn test_template_generator_deterministic() {
    let gen = TemplateGenerator::new()
        .with_template(Template::new("cmd {arg}").with_slot("arg", &["a", "b", "c", "d", "e"]));

    let samples1 = gen.generate_samples(10, 42);
    let samples2 = gen.generate_samples(10, 42);

    assert_eq!(samples1, samples2);
}

#[test]
fn test_template_generator_different_seeds() {
    let gen = TemplateGenerator::new().with_template(
        Template::new("{a} {b}")
            .with_slot("a", &["x", "y", "z"])
            .with_slot("b", &["1", "2", "3"]),
    );

    let samples1 = gen.generate_samples(10, 42);
    let samples2 = gen.generate_samples(10, 123);

    // Should produce different sequences
    assert_ne!(samples1, samples2);
}

#[test]
fn test_template_generator_unique_outputs() {
    let config = TemplateConfig::new().with_unique_outputs(true);
    let gen = TemplateGenerator::with_config(config)
        .with_template(Template::new("{a}").with_slot("a", &["x", "y", "z", "w", "v"]));

    let samples = gen.generate_samples(5, 42);

    // All should be unique
    let unique: std::collections::HashSet<_> = samples.iter().collect();
    assert_eq!(unique.len(), samples.len());
}

#[test]
fn test_template_generator_shell_commands() {
    let gen = TemplateGenerator::shell_commands();
    assert!(!gen.templates().is_empty());

    let samples = gen.generate_samples(20, 42);
    assert_eq!(samples.len(), 20);

    // Should have variety of commands
    let has_git = samples.iter().any(|s| s.starts_with("git"));
    let has_cargo = samples.iter().any(|s| s.starts_with("cargo"));

    assert!(has_git || has_cargo);
}

// ============================================================================
// EXTREME TDD: SyntheticGenerator Trait Tests
// ============================================================================

#[test]
fn test_template_synthetic_generator_trait() {
    let gen = TemplateGenerator::shell_commands();
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.0)
        .with_quality_threshold(0.0);

    let seeds = vec!["git status".to_string()];
    let result = gen.generate(&seeds, &config);

    assert!(result.is_ok());
    assert!(!result.expect("should succeed").is_empty());
}

#[test]
fn test_template_quality_score() {
    let gen = TemplateGenerator::new();

    let score = gen.quality_score(&"git status".to_string(), &String::new());
    assert!(score > 0.5);

    let score = gen.quality_score(&String::new(), &String::new());
    assert!((score - 0.0).abs() < f32::EPSILON);

    let score = gen.quality_score(&"ab".to_string(), &String::new());
    assert!(score < 0.5);
}

#[test]
fn test_template_diversity_score() {
    let gen = TemplateGenerator::new();

    // All unique
    let batch = vec![
        "git status".to_string(),
        "cargo build".to_string(),
        "npm install".to_string(),
    ];
    let diversity = gen.diversity_score(&batch);
    assert!((diversity - 1.0).abs() < f32::EPSILON);

    // Some duplicates
    let batch = vec![
        "git status".to_string(),
        "git status".to_string(),
        "cargo build".to_string(),
    ];
    let diversity = gen.diversity_score(&batch);
    assert!(diversity < 1.0);
    assert!(diversity > 0.5);
}
