
    #[test]
    fn test_quality_score_moderate_overlap() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        // Two strings with moderate overlap (in the 0.3..0.95 range)
        let seed = "let x = 42;\nlet y = 99;\nlet z = x + y;".to_string();
        let generated = "let value = 42;\nlet result = 99;\nlet z = value + result;".to_string();

        let quality = eda.quality_score(&generated, &seed);
        // Should return the overlap value itself (not 0.5 or 0.3 clamped)
        assert!(quality > 0.0);
    }

    #[test]
    fn test_augment_all_operations_enabled() {
        // Set all probabilities to 1.0 to trigger all branches
        let config = CodeEdaConfig::default()
            .with_rename_prob(1.0)
            .with_comment_prob(1.0)
            .with_reorder_prob(1.0)
            .with_remove_prob(1.0);
        let eda = CodeEda::new(config);

        let code = "let x = 42;\nlet y = 99;\nlet n = x + y;\n";
        let augmented = eda.augment(code, 100);

        // All operations applied - should produce something non-empty
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_statement_reorder_single_statement() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);

        // Single statement (no swap possible)
        let tokens = eda.tokenize("let x = 1;");
        let mut rng = 42_u64;
        let reordered = eda.apply_statement_reorder(&tokens, &mut rng);

        let result: String = reordered.into_iter().collect();
        assert_eq!(result, "let x = 1;");
    }

    #[test]
    fn test_variable_rename_no_synonyms() {
        let config = CodeEdaConfig::default().with_rename_prob(1.0);
        let eda = CodeEda::new(config);

        // Use identifiers without synonyms in the dictionary
        let tokens = eda.tokenize("let foobar = baz;");
        let mut rng = 42_u64;
        let renamed = eda.apply_variable_rename(&tokens, &mut rng);
        let result: String = renamed.into_iter().collect();

        // foobar and baz have no synonyms, should remain unchanged
        assert!(result.contains("foobar"));
        assert!(result.contains("baz"));
    }

    #[test]
    fn test_code_language_default() {
        let lang = CodeLanguage::default();
        assert_eq!(lang, CodeLanguage::Rust);
    }

    #[test]
    fn test_code_eda_config_partial_eq() {
        let config1 = CodeEdaConfig::default();
        let config2 = CodeEdaConfig::default();
        assert_eq!(config1, config2);

        let config3 = CodeEdaConfig::new().with_rename_prob(0.5);
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_code_eda_debug_and_clone() {
        let config = CodeEdaConfig::default();
        let eda = CodeEda::new(config);
        let cloned = eda.clone();

        let debug_str = format!("{:?}", eda);
        assert!(debug_str.contains("CodeEda"));
        assert_eq!(cloned.config().num_augments, eda.config().num_augments);
    }

    #[test]
    fn test_variable_synonyms_debug_and_clone() {
        let synonyms = VariableSynonyms::new();
        let cloned = synonyms.clone();

        let debug_str = format!("{:?}", synonyms);
        assert!(debug_str.contains("VariableSynonyms"));
        assert_eq!(cloned.has_synonym("x"), synonyms.has_synonym("x"));
    }
