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

    assert!(!result.is_empty(), "Should generate synthetic samples");
    assert!(
        result.len() >= seeds.len(),
        "Should generate at least as many as input"
    );

    for (generated, seed) in result.iter().zip(seeds.iter().cycle()) {
        let quality = generator.quality_score(generated, seed);
        assert!(
            (0.0..=1.0).contains(&quality),
            "Quality should be in [0, 1]"
        );
    }

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

    let _exceeds_threshold = andon_config.exceeds_rejection_threshold(rejection_rate);
}

// ============================================================================
// Template Integration Tests
// ============================================================================

#[test]
fn test_template_generation_pipeline() {
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

    let seeds = vec![
        "placeholder1".to_string(),
        "placeholder2".to_string(),
        "placeholder3".to_string(),
    ];
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(5.0)
        .with_quality_threshold(0.0)
        .with_seed(42);

    let result = generator
        .generate(&seeds, &config)
        .expect("Template generation should succeed");

    assert!(!result.is_empty(), "Should generate samples from templates");

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

    let seeds = vec![
        ShellSample::new("git", "git status"),
        ShellSample::new("cargo", "cargo build"),
        ShellSample::new("ls", "ls -la"),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(3.0)
        .with_quality_threshold(0.0)
        .with_seed(42);

    let result = generator
        .generate(&seeds, &config)
        .expect("Shell autocomplete should succeed");

    for sample in &result {
        assert!(!sample.prefix().is_empty(), "Prefix should not be empty");
        assert!(
            !sample.completion().is_empty(),
            "Completion should not be empty"
        );
    }

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

    for sample in &result {
        let emb = sample.embedding();
        assert_eq!(emb.len(), 3, "Embedding dimension should be preserved");

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

    assert!(!result.is_empty(), "Should produce labeled samples");

    for labeled in &result {
        let label = labeled.label;
        let confidence = labeled.confidence;

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
        "git commit -m 'msg'".to_string(),
        "git status".to_string(),
    ];

    let config = SyntheticConfig::default()
        .with_augmentation_ratio(1.0)
        .with_quality_threshold(0.0);

    let result = generator
        .generate(&samples, &config)
        .expect("Should succeed");

    let unanimous_labels: Vec<_> = result.iter().filter(|l| l.confidence == 1.0).collect();
    assert!(!unanimous_labels.is_empty(), "Should have unanimous labels");
}
