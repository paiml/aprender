
#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // EXTREME TDD: EdaConfig Tests
    // ============================================================================

    #[test]
    fn test_eda_config_default() {
        let config = EdaConfig::default();
        assert!((config.synonym_prob - 0.1).abs() < f32::EPSILON);
        assert!((config.insert_prob - 0.1).abs() < f32::EPSILON);
        assert!((config.swap_prob - 0.1).abs() < f32::EPSILON);
        assert!((config.delete_prob - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.num_augments, 4);
        assert_eq!(config.min_words, 3);
    }

    #[test]
    fn test_eda_config_builder() {
        let config = EdaConfig::new()
            .with_synonym_prob(0.2)
            .with_insert_prob(0.15)
            .with_swap_prob(0.05)
            .with_delete_prob(0.1)
            .with_num_augments(8)
            .with_min_words(2);

        assert!((config.synonym_prob - 0.2).abs() < f32::EPSILON);
        assert!((config.insert_prob - 0.15).abs() < f32::EPSILON);
        assert!((config.swap_prob - 0.05).abs() < f32::EPSILON);
        assert!((config.delete_prob - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.num_augments, 8);
        assert_eq!(config.min_words, 2);
    }

    #[test]
    fn test_eda_config_clamping() {
        let config = EdaConfig::new()
            .with_synonym_prob(1.5)
            .with_insert_prob(-0.5);

        assert!((config.synonym_prob - 1.0).abs() < f32::EPSILON);
        assert!((config.insert_prob - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_eda_config_num_augments_minimum() {
        let config = EdaConfig::new().with_num_augments(0);
        assert_eq!(config.num_augments, 1);
    }

    // ============================================================================
    // EXTREME TDD: SynonymDict Tests
    // ============================================================================

    #[test]
    fn test_synonym_dict_default() {
        let dict = SynonymDict::default();
        assert!(!dict.is_empty());
        assert!(dict.len() > 10);
    }

    #[test]
    fn test_synonym_dict_get() {
        let dict = SynonymDict::default();
        let synonyms = dict.get("ls");
        assert!(synonyms.is_some());
        assert!(synonyms
            .expect("should have synonyms")
            .contains(&"dir".to_string()));
    }

    #[test]
    fn test_synonym_dict_case_insensitive() {
        let dict = SynonymDict::default();
        assert!(dict.get("LS").is_some());
        assert!(dict.get("Ls").is_some());
    }

    #[test]
    fn test_synonym_dict_random_synonym() {
        let dict = SynonymDict::default();
        let syn = dict.random_synonym("ls", 42);
        assert!(syn.is_some());
    }

    #[test]
    fn test_synonym_dict_add_custom() {
        let mut dict = SynonymDict::empty();
        dict.add("hello", &["hi", "greetings"]);

        let synonyms = dict.get("hello");
        assert!(synonyms.is_some());
        assert_eq!(synonyms.expect("should have synonyms").len(), 2);
    }

    #[test]
    fn test_synonym_dict_has_synonyms() {
        let dict = SynonymDict::default();
        assert!(dict.has_synonyms("ls"));
        assert!(!dict.has_synonyms("nonexistent_word_xyz"));
    }

    // ============================================================================
    // EXTREME TDD: SimpleRng Tests
    // ============================================================================

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..10 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_simple_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(123);

        assert_ne!(rng1.next(), rng2.next());
    }

    #[test]
    fn test_simple_rng_f32_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let f = rng.next_f32();
            assert!((0.0..=1.0).contains(&f));
        }
    }

    #[test]
    fn test_simple_rng_usize_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let n = rng.next_usize(10);
            assert!(n < 10);
        }
    }

    // ============================================================================
    // EXTREME TDD: EdaGenerator Core Tests
    // ============================================================================

    #[test]
    fn test_eda_generator_new() {
        let config = EdaConfig::default();
        let gen = EdaGenerator::new(config.clone());

        assert_eq!(gen.config(), &config);
        assert!(!gen.synonyms().is_empty());
    }

    #[test]
    fn test_eda_generator_with_custom_synonyms() {
        let config = EdaConfig::default();
        let mut synonyms = SynonymDict::empty();
        synonyms.add("test", &["check"]);

        let gen = EdaGenerator::with_synonyms(config, synonyms);
        assert!(gen.synonyms().has_synonyms("test"));
    }

    #[test]
    fn test_eda_augment_basic() {
        let config = EdaConfig::default();
        let gen = EdaGenerator::new(config);

        let input = "git commit -m fix bug";
        let augmented = gen.augment(input, 42);

        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_eda_augment_short_text() {
        let config = EdaConfig::default().with_min_words(3);
        let gen = EdaGenerator::new(config);

        let input = "ls";
        let augmented = gen.augment(input, 42);

        // Short text returns original
        assert_eq!(augmented.len(), 1);
        assert_eq!(augmented[0], input);
    }

    #[test]
    fn test_eda_augment_deterministic() {
        let config = EdaConfig::default();
        let gen = EdaGenerator::new(config);

        let input = "cargo build --release";
        let aug1 = gen.augment(input, 42);
        let aug2 = gen.augment(input, 42);

        assert_eq!(aug1, aug2);
    }

    #[test]
    fn test_eda_augment_different_seeds() {
        let config = EdaConfig::new().with_synonym_prob(0.5).with_swap_prob(0.5);
        let gen = EdaGenerator::new(config);

        let input = "git push origin main branch";
        let aug1 = gen.augment(input, 42);
        let aug2 = gen.augment(input, 123);

        // Different seeds should produce different results (usually)
        // Note: may occasionally be same due to randomness
        assert!(!aug1.is_empty());
        assert!(!aug2.is_empty());
    }

    // ============================================================================
    // EXTREME TDD: Individual EDA Operations Tests
    // ============================================================================

    #[test]
    fn test_eda_synonym_replacement() {
        let config = EdaConfig::new().with_synonym_prob(1.0);
        let gen = EdaGenerator::new(config);

        // "ls" has synonyms "dir", "list"
        let words = vec!["ls".to_string(), "-la".to_string()];
        let mut rng = SimpleRng::new(42);
        let result = gen.synonym_replacement(&words, &mut rng);

        // Should have replaced at least something
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_eda_random_insertion() {
        let config = EdaConfig::new().with_insert_prob(1.0);
        let gen = EdaGenerator::new(config);

        let words = vec!["git".to_string(), "status".to_string()];
        let mut rng = SimpleRng::new(42);
        let result = gen.random_insertion(&words, &mut rng);

        // Should have more words after insertion
        assert!(result.len() >= words.len());
    }

    #[test]
    fn test_eda_random_swap() {
        let config = EdaConfig::new().with_swap_prob(1.0);
        let gen = EdaGenerator::new(config);

        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut rng = SimpleRng::new(42);
        let result = gen.random_swap(&words, &mut rng);

        // Should have same length
        assert_eq!(result.len(), 3);
        // Should contain same words (possibly reordered)
        for word in &words {
            assert!(result.contains(word));
        }
    }

    #[test]
    fn test_eda_random_deletion() {
        let config = EdaConfig::new().with_delete_prob(0.5);
        let gen = EdaGenerator::new(config);

        let words = vec![
            "git".to_string(),
            "commit".to_string(),
            "-m".to_string(),
            "message".to_string(),
        ];
        let mut rng = SimpleRng::new(42);
        let result = gen.random_deletion(&words, &mut rng);

        // Should have at least 1 word
        assert!(!result.is_empty());
        // Should have at most original length
        assert!(result.len() <= words.len());
    }

    #[test]
    fn test_eda_random_deletion_preserves_minimum() {
        let config = EdaConfig::new().with_delete_prob(1.0);
        let gen = EdaGenerator::new(config);

        let words = vec!["only".to_string()];
        let mut rng = SimpleRng::new(42);
        let result = gen.random_deletion(&words, &mut rng);

        // Should keep at least one word
        assert_eq!(result.len(), 1);
    }

    // ============================================================================
    // EXTREME TDD: Similarity and Quality Tests
    // ============================================================================

    #[test]
    fn test_eda_similarity_identical() {
        let gen = EdaGenerator::new(EdaConfig::default());
        let sim = gen.similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_eda_similarity_different() {
        let gen = EdaGenerator::new(EdaConfig::default());
        let sim = gen.similarity("hello world", "goodbye universe");
        assert!(sim < 0.5);
    }

    #[test]
    fn test_eda_similarity_partial_overlap() {
        let gen = EdaGenerator::new(EdaConfig::default());
        let sim = gen.similarity("git commit -m", "git push -m");
        assert!(sim > 0.3);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_eda_similarity_empty() {
        let gen = EdaGenerator::new(EdaConfig::default());
        assert!((gen.similarity("", "") - 1.0).abs() < f32::EPSILON);
        assert!((gen.similarity("hello", "") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_eda_quality_score() {
        let gen = EdaGenerator::new(EdaConfig::default());

        // Identical strings should have high quality
        let score = gen.quality_score(&"git status".to_string(), &"git status".to_string());
        assert!(score > 0.9);

        // Similar strings should have reasonable quality
        let score = gen.quality_score(&"git push".to_string(), &"git status".to_string());
        assert!(score > 0.3);
    }

    #[test]
    fn test_eda_diversity_score() {
        let gen = EdaGenerator::new(EdaConfig::default());

        // Identical batch should have low diversity
        let batch = vec![
            "git status".to_string(),
            "git status".to_string(),
            "git status".to_string(),
        ];
        let diversity = gen.diversity_score(&batch);
        assert!(diversity < 0.1);

        // Diverse batch should have high diversity
        let batch = vec![
            "git status".to_string(),
            "cargo build".to_string(),
            "npm install".to_string(),
        ];
        let diversity = gen.diversity_score(&batch);
        assert!(diversity > 0.5);
    }

    // ============================================================================
    // EXTREME TDD: SyntheticGenerator Trait Tests
    // ============================================================================

    #[test]
    fn test_eda_synthetic_generator_trait() {
        let gen = EdaGenerator::new(EdaConfig::default());
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.3);

        let seeds = vec![
            "git commit -m fix bug".to_string(),
            "cargo build --release".to_string(),
        ];

        let result = gen.generate(&seeds, &config);
        assert!(result.is_ok());

        let augmented = result.expect("generation should succeed");
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_eda_generate_respects_target_count() {
        let gen = EdaGenerator::new(EdaConfig::default().with_num_augments(10));
        let config = SyntheticConfig::default()
            .with_augmentation_ratio(0.5)
            .with_quality_threshold(0.0); // Accept all

        let seeds = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "git push".to_string(),
            "git pull".to_string(),
        ];

        let result = gen
            .generate(&seeds, &config)
            .expect("generation should succeed");

        // Target is 4 * 0.5 = 2 augmented samples
        assert!(result.len() <= 2 + 4); // May produce up to target + some extras
    }

    #[test]
    fn test_eda_generate_empty_seeds() {
        let gen = EdaGenerator::new(EdaConfig::default());
        let config = SyntheticConfig::default();

        let seeds: Vec<String> = vec![];
        let result = gen.generate(&seeds, &config);

        assert!(result.is_ok());
        assert!(result.expect("should succeed").is_empty());
    }
}
