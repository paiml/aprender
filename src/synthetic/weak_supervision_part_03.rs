
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // LabelVote Tests
    // ========================================================================

    #[test]
    fn test_label_vote_is_abstain() {
        assert!(!LabelVote::Positive.is_abstain());
        assert!(!LabelVote::Negative.is_abstain());
        assert!(!LabelVote::Class(5).is_abstain());
        assert!(LabelVote::Abstain.is_abstain());
    }

    #[test]
    fn test_label_vote_to_label() {
        assert_eq!(LabelVote::Positive.to_label(), Some(1));
        assert_eq!(LabelVote::Negative.to_label(), Some(0));
        assert_eq!(LabelVote::Class(42).to_label(), Some(42));
        assert_eq!(LabelVote::Abstain.to_label(), None);
    }

    // ========================================================================
    // WeakSupervisionConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = WeakSupervisionConfig::default();
        assert_eq!(config.aggregation, AggregationStrategy::MajorityVote);
        assert!((config.min_confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.min_votes, 1);
        assert!(!config.include_abstained);
    }

    #[test]
    fn test_config_with_aggregation() {
        let config = WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Unanimous);
        assert_eq!(config.aggregation, AggregationStrategy::Unanimous);
    }

    #[test]
    fn test_config_with_min_confidence() {
        let config = WeakSupervisionConfig::new().with_min_confidence(0.8);
        assert!((config.min_confidence - 0.8).abs() < f32::EPSILON);

        // Should clamp
        let config = WeakSupervisionConfig::new().with_min_confidence(1.5);
        assert!((config.min_confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_with_min_votes() {
        let config = WeakSupervisionConfig::new().with_min_votes(3);
        assert_eq!(config.min_votes, 3);

        // Should be at least 1
        let config = WeakSupervisionConfig::new().with_min_votes(0);
        assert_eq!(config.min_votes, 1);
    }

    #[test]
    fn test_config_with_include_abstained() {
        let config = WeakSupervisionConfig::new().with_include_abstained(true, 99);
        assert!(config.include_abstained);
        assert_eq!(config.default_label, 99);
    }

    // ========================================================================
    // LabeledSample Tests
    // ========================================================================

    #[test]
    fn test_labeled_sample_new() {
        let sample = LabeledSample::new("test".to_string(), 1, 0.9);
        assert_eq!(sample.sample, "test");
        assert_eq!(sample.label, 1);
        assert!((sample.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_labeled_sample_with_votes() {
        let sample = LabeledSample::new("test".to_string(), 1, 0.9).with_votes(
            2,
            vec![
                ("lf1".to_string(), LabelVote::Positive),
                ("lf2".to_string(), LabelVote::Positive),
            ],
        );
        assert_eq!(sample.num_votes, 2);
        assert_eq!(sample.votes.len(), 2);
    }

    // ========================================================================
    // Built-in LF Tests
    // ========================================================================

    #[test]
    fn test_keyword_lf() {
        let lf = KeywordLF::new("git_cmd", &["git", "commit"], LabelVote::Positive);

        assert_eq!(lf.name(), "git_cmd");
        assert_eq!(lf.apply(&"git status".to_string()), LabelVote::Positive);
        assert_eq!(lf.apply(&"commit message".to_string()), LabelVote::Positive);
        assert_eq!(lf.apply(&"cargo build".to_string()), LabelVote::Abstain);
    }

    #[test]
    fn test_keyword_lf_case_insensitive() {
        let lf = KeywordLF::new("test", &["git"], LabelVote::Positive);
        assert_eq!(lf.apply(&"GIT status".to_string()), LabelVote::Positive);
    }

    #[test]
    fn test_keyword_lf_with_weight() {
        let lf = KeywordLF::new("test", &["x"], LabelVote::Positive).with_weight(2.5);
        assert!((lf.weight() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_length_lf() {
        let lf = LengthLF::new("short", 0, 5, LabelVote::Negative);

        assert_eq!(lf.name(), "short");
        assert_eq!(lf.apply(&"hi".to_string()), LabelVote::Negative);
        assert_eq!(lf.apply(&"hello".to_string()), LabelVote::Negative);
        assert_eq!(lf.apply(&"hello world".to_string()), LabelVote::Abstain);
    }

    #[test]
    fn test_pattern_lf() {
        let lf = PatternLF::new("has_equals", "=", LabelVote::Positive);

        assert_eq!(lf.name(), "has_equals");
        assert_eq!(lf.apply(&"x=1".to_string()), LabelVote::Positive);
        assert_eq!(lf.apply(&"no equals".to_string()), LabelVote::Abstain);
    }

    // ========================================================================
    // WeakSupervisionGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_new() {
        let gen = WeakSupervisionGenerator::<String>::new();
        assert_eq!(gen.num_lfs(), 0);
    }

    #[test]
    fn test_generator_add_lf() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new(
            "test",
            &["x"],
            LabelVote::Positive,
        )));
        assert_eq!(gen.num_lfs(), 1);
    }

    #[test]
    fn test_generator_no_lfs() {
        let gen = WeakSupervisionGenerator::<String>::new();
        let samples = vec!["test".to_string()];
        let result = gen
            .generate(&samples, &SyntheticConfig::default())
            .expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_generator_majority_vote() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["git"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf2",
            &["status"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf3",
            &["cargo"],
            LabelVote::Negative,
        )));

        let samples = vec!["git status".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 1); // Positive (2 votes vs 0)
    }

    #[test]
    fn test_generator_weighted_vote() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new()
                .with_aggregation(AggregationStrategy::WeightedVote)
                .with_min_confidence(0.0),
        );

        gen.add_lf(Box::new(
            KeywordLF::new("lf1", &["test"], LabelVote::Positive).with_weight(1.0),
        ));
        gen.add_lf(Box::new(
            KeywordLF::new("lf2", &["test"], LabelVote::Negative).with_weight(3.0),
        ));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 0); // Negative wins (weight 3 vs 1)
    }

    #[test]
    fn test_generator_unanimous() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Unanimous),
        );

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf2",
            &["test"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 1);
        assert!((result[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_unanimous_disagree() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Unanimous),
        );

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "lf2",
            &["test"],
            LabelVote::Negative,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.is_empty()); // No unanimous agreement
    }

    #[test]
    fn test_generator_any_vote() {
        let mut gen = WeakSupervisionGenerator::<String>::new()
            .with_config(WeakSupervisionConfig::new().with_aggregation(AggregationStrategy::Any));

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_generator_min_votes() {
        let mut gen = WeakSupervisionGenerator::<String>::new()
            .with_config(WeakSupervisionConfig::new().with_min_votes(2));

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["test"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.is_empty()); // Only 1 vote, need 2
    }

    #[test]
    fn test_generator_all_abstain() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["nomatch"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.is_empty()); // All abstained
    }

    #[test]
    fn test_generator_include_abstained() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new()
                .with_include_abstained(true, 99)
                .with_min_confidence(0.0),
        );

        gen.add_lf(Box::new(KeywordLF::new(
            "lf1",
            &["nomatch"],
            LabelVote::Positive,
        )));

        let samples = vec!["test".to_string()];
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, 99); // Default label
    }

    #[test]
    fn test_quality_score() {
        let gen = WeakSupervisionGenerator::<String>::new();

        let high_conf = LabeledSample::new("test".to_string(), 1, 0.9);
        let low_conf = LabeledSample::new("test".to_string(), 1, 0.3);

        assert!(gen.quality_score(&high_conf, &String::new()) > 0.5);
        assert!(gen.quality_score(&low_conf, &String::new()) < 0.5);
    }

    #[test]
    fn test_diversity_score() {
        let gen = WeakSupervisionGenerator::<String>::new();

        // Empty batch
        let score = gen.diversity_score(&[]);
        assert!((score - 0.0).abs() < f32::EPSILON);

        // Single label
        let single_label = vec![
            LabeledSample::new("a".to_string(), 1, 1.0),
            LabeledSample::new("b".to_string(), 1, 1.0),
        ];
        let score = gen.diversity_score(&single_label);
        assert!((score - 0.0).abs() < f32::EPSILON); // No diversity

        // Two labels, equal distribution
        let diverse = vec![
            LabeledSample::new("a".to_string(), 0, 1.0),
            LabeledSample::new("b".to_string(), 1, 1.0),
        ];
        let score = gen.diversity_score(&diverse);
        assert!((score - 1.0).abs() < f32::EPSILON); // Max diversity
    }

    #[test]
    fn test_generator_respects_target() {
        let mut gen = WeakSupervisionGenerator::<String>::new();
        gen.add_lf(Box::new(KeywordLF::new("lf1", &["a"], LabelVote::Positive)));

        let samples = vec![
            "a1".to_string(),
            "a2".to_string(),
            "a3".to_string(),
            "a4".to_string(),
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(0.5) // Target: 2
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");
        assert!(result.len() <= 2);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_weak_supervision_pipeline() {
        let mut gen = WeakSupervisionGenerator::<String>::new().with_config(
            WeakSupervisionConfig::new()
                .with_aggregation(AggregationStrategy::MajorityVote)
                .with_min_confidence(0.5),
        );

        // Add multiple LFs
        gen.add_lf(Box::new(KeywordLF::new(
            "git_positive",
            &["git"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(KeywordLF::new(
            "cargo_positive",
            &["cargo"],
            LabelVote::Positive,
        )));
        gen.add_lf(Box::new(LengthLF::new(
            "short_negative",
            0,
            5,
            LabelVote::Negative,
        )));
        gen.add_lf(Box::new(PatternLF::new(
            "equals_positive",
            "=",
            LabelVote::Positive,
        )));

        let samples = vec![
            "git status".to_string(),        // git_positive votes
            "cargo build".to_string(),       // cargo_positive votes
            "hi".to_string(),                // short_negative votes
            "git log --oneline".to_string(), // git_positive votes
            "x=1".to_string(),               // equals_positive + short_negative
        ];

        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.0);

        let result = gen.generate(&samples, &config).expect("generation failed");

        // Verify labels make sense
        for labeled in &result {
            assert!(labeled.num_votes > 0);
            assert!(!labeled.votes.is_empty());
        }

        // Check diversity
        let diversity = gen.diversity_score(&result);
        assert!((0.0..=1.0).contains(&diversity));
    }
}
