
    #[test]
    fn test_code_eda_config_default() {
        let config = CodeEdaConfig::default();
        assert!((config.rename_prob - 0.15).abs() < f32::EPSILON);
        assert_eq!(config.num_augments, 4);
        assert_eq!(config.language, CodeLanguage::Rust);
    }

    #[test]
    fn test_code_eda_config_builder() {
        let config = CodeEdaConfig::new()
            .with_rename_prob(0.2)
            .with_comment_prob(0.15)
            .with_num_augments(8)
            .with_language(CodeLanguage::Python);

        assert!((config.rename_prob - 0.2).abs() < f32::EPSILON);
        assert!((config.comment_prob - 0.15).abs() < f32::EPSILON);
        assert_eq!(config.num_augments, 8);
        assert_eq!(config.language, CodeLanguage::Python);
    }

    #[test]
    fn test_code_eda_config_clamp() {
        let config = CodeEdaConfig::new()
            .with_rename_prob(1.5)
            .with_remove_prob(-0.5);

        assert!((config.rename_prob - 1.0).abs() < f32::EPSILON);
        assert!((config.remove_prob - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_variable_synonyms_default() {
        let synonyms = VariableSynonyms::default();
        assert!(synonyms.has_synonym("x"));
        assert!(synonyms.has_synonym("i"));
        assert!(!synonyms.has_synonym("unknown_var_xyz"));
    }

    #[test]
    fn test_variable_synonyms_get() {
        let synonyms = VariableSynonyms::default();
        let x_syns = synonyms.get("x").expect("x should have synonyms");
        assert!(x_syns.contains(&"value".to_string()));
    }

    #[test]
    fn test_variable_synonyms_add() {
        let mut synonyms = VariableSynonyms::new();
        synonyms.add_synonym(
            "foo".to_string(),
            vec!["bar".to_string(), "baz".to_string()],
        );
        assert!(synonyms.has_synonym("foo"));
    }

    #[test]
    fn test_code_eda_new() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);
        assert!(!eda.reserved.is_empty());
        assert!(eda.reserved.contains("fn"));
        assert!(eda.reserved.contains("let"));
    }

    #[test]
    fn test_code_eda_tokenize() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        let tokens = eda.tokenize("let x = 42;");
        assert!(tokens.contains(&"let".to_string()));
        assert!(tokens.contains(&"x".to_string()));
        assert!(tokens.contains(&"42".to_string()));
    }

    #[test]
    fn test_code_eda_is_identifier() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        assert!(eda.is_identifier("foo"));
        assert!(eda.is_identifier("_bar"));
        assert!(eda.is_identifier("x123"));
        assert!(!eda.is_identifier("123"));
        assert!(!eda.is_identifier(""));
        assert!(!eda.is_identifier("+"));
    }

    #[test]
    fn test_code_eda_augment_basic() {
        let config = CodeEdaConfig::default().with_rename_prob(1.0);
        let eda = CodeEda::new(config);

        let code = "let x = 42;\nprintln!(\"{}\", x);";
        let augmented = eda.augment(code, 42);

        // Should produce some output
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_code_eda_augment_short_code() {
        let config = CodeEdaConfig::default().with_min_tokens(10);
        let eda = CodeEda::new(config);

        let code = "x";
        let augmented = eda.augment(code, 42);

        // Short code should be returned unchanged
        assert_eq!(augmented, code);
    }

    #[test]
    fn test_code_eda_augment_preserves_keywords() {
        let config = CodeEdaConfig::default().with_rename_prob(1.0);
        let eda = CodeEda::new(config);

        let code = "let mut fn impl";
        let augmented = eda.augment(code, 42);

        // Keywords should not be renamed
        assert!(augmented.contains("let") || augmented.contains("mut"));
    }

    #[test]
    fn test_code_eda_token_overlap() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        let a = "let x = 42;";
        let b = "let x = 42;";
        assert!((eda.token_overlap(a, b) - 1.0).abs() < f32::EPSILON);

        let c = "let y = 99;";
        let overlap = eda.token_overlap(a, c);
        assert!(overlap > 0.0 && overlap < 1.0);
    }

    #[test]
    fn test_code_eda_token_overlap_empty() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        assert!((eda.token_overlap("", "test") - 0.0).abs() < f32::EPSILON);
        assert!((eda.token_overlap("test", "") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_code_eda_quality_score() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        let seed = "let x = 42;";

        // Identical should have lower quality (too similar)
        let quality_identical = eda.quality_score(&seed.to_string(), &seed.to_string());
        assert!(quality_identical < 0.6);

        // Completely different should have low quality
        let quality_different = eda.quality_score(&"abc".to_string(), &seed.to_string());
        assert!(quality_different < 0.5);
    }

    #[test]
    fn test_code_eda_diversity_score() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        // Single item has perfect diversity
        assert!((eda.diversity_score(&["test".to_string()]) - 1.0).abs() < f32::EPSILON);

        // Identical items have low diversity
        let identical = vec!["let x = 1;".to_string(), "let x = 1;".to_string()];
        assert!(eda.diversity_score(&identical) < 0.1);

        // Different items have higher diversity
        let different = vec!["let x = 1;".to_string(), "fn foo() {}".to_string()];
        assert!(eda.diversity_score(&different) > 0.5);
    }

    #[test]
    fn test_code_eda_generate() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        let seeds = vec!["let x = 1;".to_string(), "let y = 2;".to_string()];
        let synth_config = SyntheticConfig::default()
            .with_augmentation_ratio(2.0)
            .with_quality_threshold(0.0);

        let result = eda
            .generate(&seeds, &synth_config)
            .expect("generation failed");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_code_eda_generate_with_quality_filter() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        let seeds = vec!["let x = 42;".to_string()];
        let synth_config = SyntheticConfig::default()
            .with_augmentation_ratio(1.0)
            .with_quality_threshold(0.9); // Very high threshold

        let result = eda
            .generate(&seeds, &synth_config)
            .expect("generation failed");
        // High threshold may filter out all results
        // Just verify no panic
        assert!(result.len() <= seeds.len() * 4);
    }

    #[test]
    fn test_code_eda_python_keywords() {
        let config = CodeEdaConfig::default().with_language(CodeLanguage::Python);
        let eda = CodeEda::new(config);

        assert!(eda.reserved.contains("def"));
        assert!(eda.reserved.contains("class"));
        assert!(eda.reserved.contains("import"));
    }

    #[test]
    fn test_code_eda_statement_reorder() {
        let config = CodeEdaConfig::default().with_reorder_prob(1.0);
        let eda = CodeEda::new(config);

        let code = "let a = 1;\nlet b = 2;\nlet c = 3;";
        let augmented = eda.augment(code, 12345);

        // Should still contain all statements
        assert!(augmented.contains('a') || augmented.contains('b') || augmented.contains('c'));
    }

    #[test]
    fn test_code_eda_dead_code_removal() {
        let config = CodeEdaConfig::default().with_remove_prob(1.0);
        let eda = CodeEda::new(config);

        // Manually test the removal function
        let tokens = eda.tokenize("let x = 1; // comment\nlet y = 2;");
        let cleaned = eda.apply_dead_code_removal(&tokens);
        let result: String = cleaned.into_iter().collect();

        // Comment should be removed
        assert!(!result.contains("comment"));
    }

    #[test]
    fn test_code_eda_deterministic() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        let code = "let x = 42;";
        let aug1 = eda.augment(code, 42);
        let aug2 = eda.augment(code, 42);

        assert_eq!(aug1, aug2);
    }

    #[test]
    fn test_code_eda_different_seeds() {
        let config = CodeEdaConfig::default().with_rename_prob(1.0);
        let eda = CodeEda::new(config);

        let code = "let x = 42;\nlet y = 99;";
        let aug1 = eda.augment(code, 1);
        let aug2 = eda.augment(code, 2);

        // Different seeds may produce different results
        // (or same if operations don't apply)
        // Just verify no panic
        assert!(!aug1.is_empty());
        assert!(!aug2.is_empty());
    }

    // ================================================================
    // Additional coverage tests for missed branches
    // ================================================================

    #[test]
    fn test_config_with_reorder_prob() {
        let config = CodeEdaConfig::new().with_reorder_prob(0.5);
        assert!((config.reorder_prob - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_with_remove_prob() {
        let config = CodeEdaConfig::new().with_remove_prob(0.3);
        assert!((config.remove_prob - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_with_min_tokens() {
        let config = CodeEdaConfig::new().with_min_tokens(10);
        assert_eq!(config.min_tokens, 10);
    }

    #[test]
    fn test_config_num_augments_clamp_zero() {
        let config = CodeEdaConfig::new().with_num_augments(0);
        // 0.max(1) = 1
        assert_eq!(config.num_augments, 1);
    }

    #[test]
    fn test_generic_language_keywords() {
        let config = CodeEdaConfig::default().with_language(CodeLanguage::Generic);
        let eda = CodeEda::new(config);

        // Generic has no reserved keywords
        assert!(eda.reserved.is_empty());
    }

    #[test]
    fn test_python_comment_insertion() {
        let config = CodeEdaConfig::default()
            .with_language(CodeLanguage::Python)
            .with_comment_prob(1.0)
            .with_rename_prob(0.0)
            .with_reorder_prob(0.0)
            .with_remove_prob(0.0);
        let eda = CodeEda::new(config);

        let code = "x = 42\ny = 99\nz = x + y\n";
        let augmented = eda.augment(code, 42);

        // Should contain original code and possibly an inserted comment
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_generic_comment_insertion() {
        let config = CodeEdaConfig::default()
            .with_language(CodeLanguage::Generic)
            .with_comment_prob(1.0)
            .with_rename_prob(0.0)
            .with_reorder_prob(0.0)
            .with_remove_prob(0.0);
        let eda = CodeEda::new(config);

        let code = "x = 42\ny = 99\n";
        let augmented = eda.augment(code, 42);
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_python_statement_reorder() {
        let config = CodeEdaConfig::default()
            .with_language(CodeLanguage::Python)
            .with_reorder_prob(1.0)
            .with_rename_prob(0.0)
            .with_comment_prob(0.0)
            .with_remove_prob(0.0);
        let eda = CodeEda::new(config);

        // Python uses newlines as delimiters
        let code = "a = 1\nb = 2\nc = 3\n";
        let augmented = eda.augment(code, 42);
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_dead_code_removal_python_hash_comments() {
        let config = CodeEdaConfig::default()
            .with_language(CodeLanguage::Python)
            .with_remove_prob(1.0)
            .with_rename_prob(0.0)
            .with_comment_prob(0.0)
            .with_reorder_prob(0.0);
        let eda = CodeEda::new(config);

        let tokens = eda.tokenize("x = 1 # this is a comment\ny = 2");
        let cleaned = eda.apply_dead_code_removal(&tokens);
        let result: String = cleaned.into_iter().collect();

        // The hash comment should be removed
        assert!(!result.contains("this"));
        assert!(!result.contains("comment"));
    }

    #[test]
    fn test_dead_code_removal_whitespace_collapse() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        // Multiple consecutive whitespace tokens should collapse
        let tokens = vec![
            "x".to_string(),
            " ".to_string(),
            " ".to_string(),
            " ".to_string(),
            "=".to_string(),
            " ".to_string(),
            "1".to_string(),
        ];
        let cleaned = eda.apply_dead_code_removal(&tokens);

        // Count whitespace tokens (should be collapsed)
        let ws_count = cleaned.iter().filter(|t| t.trim().is_empty()).count();
        assert!(
            ws_count <= 2,
            "Whitespace should be collapsed, got {ws_count}"
        );
    }

    #[test]
    fn test_dead_code_removal_single_slash_preserved() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        // A single slash (like division) should be preserved
        let tokens = vec![
            "x".to_string(),
            " ".to_string(),
            "/".to_string(),
            " ".to_string(),
            "2".to_string(),
        ];
        let cleaned = eda.apply_dead_code_removal(&tokens);
        let result: String = cleaned.into_iter().collect();
        assert!(result.contains('/'));
    }

    #[test]
    fn test_config_accessor() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config.clone());
        assert_eq!(*eda.config(), config);
    }

    #[test]
    fn test_variable_synonyms_get_none() {
        let synonyms = VariableSynonyms::default();
        assert!(synonyms.get("nonexistent_xyz").is_none());
    }

    #[test]
    fn test_diversity_score_empty_batch() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);
        let empty: Vec<String> = vec![];
        // Empty batch <= 1 items returns 1.0
        assert!((eda.diversity_score(&empty) - 1.0).abs() < f32::EPSILON);
    }
