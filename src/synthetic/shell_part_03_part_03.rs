
    #[test]
    fn test_generator_respects_quality_threshold() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![ShellSample::new("git st", "git status")];

        // High threshold - fewer results
        let high_config = SyntheticConfig::default()
            .with_augmentation_ratio(5.0)
            .with_quality_threshold(0.95);
        let high_result = gen
            .generate(&seeds, &high_config)
            .expect("generation failed");

        // Low threshold - more results
        let low_config = SyntheticConfig::default()
            .with_augmentation_ratio(5.0)
            .with_quality_threshold(0.1);
        let low_result = gen
            .generate(&seeds, &low_config)
            .expect("generation failed");

        assert!(low_result.len() >= high_result.len());
    }

    #[test]
    fn test_generator_respects_target_count() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![
            ShellSample::new("git st", "git status"),
            ShellSample::new("cargo b", "cargo build"),
            ShellSample::new("npm i", "npm install"),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(0.5) // Target: 1.5 -> 1 sample
            .with_quality_threshold(0.1);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Should stop at target (may be fewer if not enough generated)
        assert!(result.len() <= 2);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_pipeline() {
        let gen = ShellSyntheticGenerator::new();

        let seeds = vec![
            ShellSample::new("git", "git status")
                .with_history(vec!["cd repo".to_string()])
                .with_cwd("/home/user/repo"),
            ShellSample::new("cargo", "cargo build --release").with_cwd("/home/user/project"),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0)
            .with_quality_threshold(0.3);

        let synthetic = gen.generate(&seeds, &config).expect("generation failed");

        // Verify generated samples are valid
        for sample in &synthetic {
            // Should have non-empty completion
            assert!(!sample.completion().is_empty());

            // Should be grammatically valid (generator filters invalid)
            assert!(gen.grammar.is_valid_command(sample.completion()));

            // Quality score should be in valid range
            let quality = gen.quality_score(sample, &seeds[0]);
            assert!((0.0..=1.0).contains(&quality));
        }

        // Verify diversity
        let diversity = gen.diversity_score(&synthetic);
        assert!((0.0..=1.0).contains(&diversity));
    }

    #[test]
    fn test_deterministic_generation() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![ShellSample::new("git st", "git status")];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0)
            .with_seed(12345);

        let result1 = gen.generate(&seeds, &config).expect("generation failed");
        let result2 = gen.generate(&seeds, &config).expect("generation failed");

        // Same seed should produce same results
        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_eq!(r1.completion(), r2.completion());
        }
    }
