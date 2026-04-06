
    // ========================================================================
    // ShellSample Tests
    // ========================================================================

    #[test]
    fn test_shell_sample_new() {
        let sample = ShellSample::new("git st", "git status");
        assert_eq!(sample.prefix(), "git st");
        assert_eq!(sample.completion(), "git status");
        assert!(sample.history().is_empty());
        assert!(sample.cwd().is_empty());
    }

    #[test]
    fn test_shell_sample_with_history() {
        let sample =
            ShellSample::new("cargo b", "cargo build").with_history(vec!["cd project".to_string()]);
        assert_eq!(sample.history().len(), 1);
        assert_eq!(sample.history()[0], "cd project");
    }

    #[test]
    fn test_shell_sample_with_cwd() {
        let sample = ShellSample::new("ls", "ls -la").with_cwd("/home/user");
        assert_eq!(sample.cwd(), "/home/user");
    }

    #[test]
    fn test_shell_sample_command_name() {
        let sample = ShellSample::new("git", "git status --short");
        assert_eq!(sample.command_name(), Some("git"));

        let empty = ShellSample::new("", "");
        assert_eq!(empty.command_name(), None);
    }

    #[test]
    fn test_shell_sample_arguments() {
        let sample = ShellSample::new("cargo", "cargo build --release --target wasm32");
        let args = sample.arguments();
        assert_eq!(args, vec!["build", "--release", "--target", "wasm32"]);
    }

    #[test]
    fn test_shell_sample_is_valid_completion() {
        let valid = ShellSample::new("git st", "git status");
        assert!(valid.is_valid_completion());

        let invalid = ShellSample::new("cargo", "git status");
        assert!(!invalid.is_valid_completion());
    }

    #[test]
    fn test_shell_sample_clone() {
        let sample = ShellSample::new("ls", "ls -la")
            .with_history(vec!["pwd".to_string()])
            .with_cwd("/tmp");
        let cloned = sample.clone();
        assert_eq!(sample, cloned);
    }

    #[test]
    fn test_shell_sample_debug() {
        let sample = ShellSample::new("test", "test command");
        let debug = format!("{sample:?}");
        assert!(debug.contains("ShellSample"));
        assert!(debug.contains("prefix"));
    }

    // ========================================================================
    // ShellGrammar Tests
    // ========================================================================

    #[test]
    fn test_grammar_new_empty() {
        let grammar = ShellGrammar::new();
        assert!(grammar.commands().is_empty());
    }

    #[test]
    fn test_grammar_add_command() {
        let mut grammar = ShellGrammar::new();
        grammar.add_command("mycommand");
        assert!(grammar.commands().contains("mycommand"));
    }

    #[test]
    fn test_grammar_add_subcommands() {
        let mut grammar = ShellGrammar::new();
        grammar.add_command("git");
        grammar.add_subcommands("git", &["status", "commit"]);

        let subs = grammar.get_subcommands("git").expect("should have subs");
        assert!(subs.contains("status"));
        assert!(subs.contains("commit"));
    }

    #[test]
    fn test_grammar_common_commands() {
        let grammar = ShellGrammar::common_commands();

        // Check git
        assert!(grammar.commands().contains("git"));
        let git_subs = grammar
            .get_subcommands("git")
            .expect("git should have subs");
        assert!(git_subs.contains("status"));
        assert!(git_subs.contains("commit"));

        // Check cargo
        assert!(grammar.commands().contains("cargo"));
        let cargo_subs = grammar
            .get_subcommands("cargo")
            .expect("cargo should have subs");
        assert!(cargo_subs.contains("build"));
        assert!(cargo_subs.contains("test"));

        // Check common commands
        assert!(grammar.commands().contains("ls"));
        assert!(grammar.commands().contains("cd"));
    }

    #[test]
    fn test_grammar_is_valid_command_known() {
        let grammar = ShellGrammar::common_commands();

        assert!(grammar.is_valid_command("git status"));
        assert!(grammar.is_valid_command("cargo build"));
        assert!(grammar.is_valid_command("ls -la"));
        assert!(grammar.is_valid_command("docker run"));
    }

    #[test]
    fn test_grammar_is_valid_command_with_options() {
        let grammar = ShellGrammar::common_commands();

        assert!(grammar.is_valid_command("git -h"));
        assert!(grammar.is_valid_command("cargo --version"));
        assert!(grammar.is_valid_command("git status --short"));
    }

    #[test]
    fn test_grammar_is_valid_command_empty() {
        let grammar = ShellGrammar::common_commands();
        assert!(!grammar.is_valid_command(""));
        assert!(!grammar.is_valid_command("   "));
    }

    #[test]
    fn test_grammar_is_valid_command_unknown() {
        let grammar = ShellGrammar::common_commands();
        assert!(!grammar.is_valid_command("unknowncommand"));
        assert!(!grammar.is_valid_command("notacommand --flag"));
    }

    #[test]
    fn test_grammar_is_valid_command_bad_subcommand() {
        let grammar = ShellGrammar::common_commands();
        // "notasub" is not a known git subcommand
        assert!(!grammar.is_valid_command("git notasub"));
    }

    #[test]
    fn test_grammar_is_valid_option() {
        let grammar = ShellGrammar::common_commands();

        assert!(grammar.is_valid_option("-h"));
        assert!(grammar.is_valid_option("--help"));
        assert!(grammar.is_valid_option("--version"));
        assert!(grammar.is_valid_option("--some-flag")); // Unknown but valid format
        assert!(!grammar.is_valid_option("notanoption"));
    }

    #[test]
    fn test_grammar_default() {
        let grammar = ShellGrammar::default();
        assert!(grammar.commands().contains("git"));
    }

    #[test]
    fn test_grammar_clone() {
        let grammar = ShellGrammar::common_commands();
        let cloned = grammar.clone();
        assert_eq!(grammar.commands().len(), cloned.commands().len());
    }

    // ========================================================================
    // ShellGeneratorConfig Tests
    // ========================================================================

    #[test]
    fn test_generator_config_default() {
        let config = ShellGeneratorConfig::default();
        assert!(config.enable_template);
        assert!(config.enable_permutation);
        assert!(config.enable_context_variation);
        assert_eq!(config.max_permute_args, 3);
    }

    // ========================================================================
    // ShellSyntheticGenerator Tests
    // ========================================================================

    #[test]
    fn test_generator_new() {
        let gen = ShellSyntheticGenerator::new();
        assert!(!gen.substitutions.is_empty());
    }

    #[test]
    fn test_generator_with_grammar() {
        let mut grammar = ShellGrammar::new();
        grammar.add_command("custom");

        let gen = ShellSyntheticGenerator::new().with_grammar(grammar);
        assert!(gen.grammar.commands().contains("custom"));
    }

    #[test]
    fn test_generator_with_config() {
        let config = ShellGeneratorConfig {
            enable_template: false,
            ..Default::default()
        };
        let gen = ShellSyntheticGenerator::new().with_config(config);
        assert!(!gen.config.enable_template);
    }

    #[test]
    fn test_generator_add_substitution() {
        let mut gen = ShellSyntheticGenerator::new();
        gen.add_substitution("myarg", &["variant1", "variant2"]);

        assert!(gen.substitutions.contains_key("myarg"));
        assert_eq!(gen.substitutions["myarg"].len(), 2);
    }

    #[test]
    fn test_generator_semantic_similarity() {
        // Identical
        let sim = ShellSyntheticGenerator::semantic_similarity("git status", "git status");
        assert!((sim - 1.0).abs() < f32::EPSILON);

        // Partial overlap
        let sim = ShellSyntheticGenerator::semantic_similarity("git status", "git commit");
        assert!(sim > 0.0 && sim < 1.0);

        // No overlap
        let sim = ShellSyntheticGenerator::semantic_similarity("cargo build", "npm install");
        assert!((sim - 0.0).abs() < f32::EPSILON);

        // Empty
        let sim = ShellSyntheticGenerator::semantic_similarity("", "");
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generator_context_coherence() {
        // Base sample
        let base = ShellSample::new("git", "git status");
        let score = ShellSyntheticGenerator::context_coherence(&base);
        assert!(score >= 0.5);

        // With history
        let with_hist =
            ShellSample::new("git", "git status").with_history(vec!["cd repo".to_string()]);
        let score_hist = ShellSyntheticGenerator::context_coherence(&with_hist);
        assert!(score_hist > score);

        // With cwd
        let with_cwd = ShellSample::new("git", "git status").with_cwd("/home/user");
        let score_cwd = ShellSyntheticGenerator::context_coherence(&with_cwd);
        assert!(score_cwd > score);

        // Valid completion bonus
        let valid = ShellSample::new("git st", "git status");
        let invalid = ShellSample::new("cargo", "git status");
        assert!(
            ShellSyntheticGenerator::context_coherence(&valid)
                > ShellSyntheticGenerator::context_coherence(&invalid)
        );
    }

    #[test]
    fn test_generator_generate_basic() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![
            ShellSample::new("git st", "git status"),
            ShellSample::new("cargo b", "cargo build"),
        ];
        let config = SyntheticConfig::default().with_augmentation_ratio(2.0);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Should generate some samples (exact count depends on strategies)
        assert!(!result.is_empty());
    }

    #[test]
    fn test_generator_generate_empty_seeds() {
        let gen = ShellSyntheticGenerator::new();
        let seeds: Vec<ShellSample> = vec![];
        let config = SyntheticConfig::default();

        let result = gen.generate(&seeds, &config).expect("generation failed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_generator_generate_deduplicates() {
        let gen = ShellSyntheticGenerator::new();
        let seeds = vec![
            ShellSample::new("git st", "git status"),
            ShellSample::new("git st", "git status"), // Duplicate
        ];
        let config = SyntheticConfig::default().with_augmentation_ratio(1.0);

        let result = gen.generate(&seeds, &config).expect("generation failed");

        // Check no duplicate completions
        let completions: HashSet<_> = result.iter().map(ShellSample::completion).collect();
        assert_eq!(completions.len(), result.len());
    }

    #[test]
    fn test_generator_quality_score() {
        let gen = ShellSyntheticGenerator::new();

        let seed = ShellSample::new("git st", "git status");
        let similar = ShellSample::new("git st", "git status --short");
        let different = ShellSample::new("cargo", "cargo build");

        let score_similar = gen.quality_score(&similar, &seed);
        let score_different = gen.quality_score(&different, &seed);

        // Similar should have higher quality
        assert!(score_similar > score_different);
    }

    #[test]
    fn test_generator_quality_score_invalid_grammar() {
        let gen = ShellSyntheticGenerator::new();

        let seed = ShellSample::new("git", "git status");
        let invalid = ShellSample::new("unk", "unknowncommand");

        let score = gen.quality_score(&invalid, &seed);

        // Grammar component should be 0
        assert!(score < 0.5);
    }

    #[test]
    fn test_generator_diversity_score() {
        let gen = ShellSyntheticGenerator::new();

        // Empty batch
        assert!((gen.diversity_score(&[]) - 0.0).abs() < f32::EPSILON);

        // Single sample
        let single = vec![ShellSample::new("git", "git status")];
        assert!((gen.diversity_score(&single) - 1.0).abs() < f32::EPSILON);

        // Diverse batch
        let diverse = vec![
            ShellSample::new("git", "git status"),
            ShellSample::new("cargo", "cargo build"),
            ShellSample::new("npm", "npm install"),
        ];
        let div_score = gen.diversity_score(&diverse);
        assert!((div_score - 1.0).abs() < f32::EPSILON);

        // Homogeneous batch (same command)
        let homogeneous = vec![
            ShellSample::new("git", "git status"),
            ShellSample::new("git", "git status"),
            ShellSample::new("git", "git status"),
        ];
        let homo_score = gen.diversity_score(&homogeneous);
        assert!(homo_score < 1.0);
    }

    #[test]
    fn test_generator_default() {
        let gen = ShellSyntheticGenerator::default();
        assert!(gen.grammar.commands().contains("git"));
    }

    #[test]
    fn test_generator_template_substitution() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("cargo build", "cargo build --release");

        let results = gen.generate_from_template(&seed, 0);

        // Should generate some variants with --debug or without the flag
        // (depends on substitution rules and rng)
        // We mainly test it doesn't panic
        assert!(results.len() <= 10); // Reasonable upper bound
    }

    #[test]
    fn test_generator_permute_arguments() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("git checkout", "git checkout main develop");

        let results = gen.permute_arguments(&seed, 0);

        // Should have swapped version
        for r in &results {
            assert_ne!(r.completion(), seed.completion());
        }
    }

    #[test]
    fn test_generator_permute_single_arg() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("git", "git status");

        let results = gen.permute_arguments(&seed, 0);

        // Single arg can't be permuted
        assert!(results.is_empty());
    }

    #[test]
    fn test_generator_vary_context() {
        let gen = ShellSyntheticGenerator::new();
        let seed = ShellSample::new("ls", "ls -la").with_cwd("/original");

        let results = gen.vary_context(&seed, 0);

        // Should have at least history variation
        assert!(!results.is_empty());

        // Check that context was varied
        let has_different_cwd = results.iter().any(|r| r.cwd() != seed.cwd());
        let has_different_history = results.iter().any(|r| r.history() != seed.history());
        assert!(has_different_cwd || has_different_history);
    }
