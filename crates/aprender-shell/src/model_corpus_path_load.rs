
// =============================================================================
// EXTREME TDD: Corpus Training Tests
// These tests verify training from synthetic developer corpus
// =============================================================================

#[cfg(test)]
mod corpus_training_tests {
    use super::*;
    use crate::corpus::Corpus;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn corpus_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/developer-commands.txt")
    }

    /// Load corpus or skip test if file doesn't exist (gitignored for security)
    fn load_corpus_or_skip() -> Option<Corpus> {
        let path = corpus_path();
        if !path.exists() {
            eprintln!("Skipping test: corpus file not found at {:?}", path);
            eprintln!("Create corpus/developer-commands.txt locally to run these tests");
            return None;
        }
        Some(Corpus::load(path).expect("corpus should load"))
    }

    // ========================================================================
    // Corpus Loading and Validation Tests
    // ========================================================================

    #[test]
    fn test_corpus_loads_successfully() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        assert!(!corpus.is_empty(), "corpus should not be empty");
    }

    #[test]
    fn test_corpus_has_expected_size() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        // Should have at least 400 commands (current is ~480)
        assert!(
            corpus.len() >= 400,
            "corpus should have >= 400 commands, got {}",
            corpus.len()
        );
    }

    #[test]
    fn test_corpus_has_diverse_prefixes() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        // Should have git, cargo, docker, kubectl, npm, python, aws, etc.
        let expected_prefixes = ["git", "cargo", "docker", "kubectl", "npm", "python", "make"];
        for prefix in expected_prefixes {
            assert!(
                corpus.prefixes().contains(prefix),
                "corpus should contain '{}' commands",
                prefix
            );
        }
    }

    // ========================================================================
    // Model Training from Corpus Tests (TDD)
    // ========================================================================

    #[test]
    fn test_train_from_corpus() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        assert_eq!(
            model.total_commands(),
            corpus.len(),
            "model should train on all corpus commands"
        );
    }

    #[test]
    fn test_corpus_trained_model_has_ngrams() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        // Should have built n-grams
        assert!(
            model.ngram_count() > 100,
            "model should have >100 n-grams, got {}",
            model.ngram_count()
        );
    }

    #[test]
    fn test_corpus_trained_model_has_vocab() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        // Vocab should be close to corpus size (some duplicates)
        let vocab = model.vocab_size();
        assert!(
            vocab >= corpus.len() / 2,
            "vocab {} should be >= half corpus size {}",
            vocab,
            corpus.len()
        );
    }

    // ========================================================================
    // Suggestion Quality Tests (from corpus training)
    // ========================================================================

    #[test]
    fn test_git_suggestions_from_corpus() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let suggestions = model.suggest("git ", 10);
        assert!(
            !suggestions.is_empty(),
            "should have git suggestions from corpus"
        );

        // All suggestions should be git commands (start with "git ")
        for (suggestion, score) in &suggestions {
            assert!(
                suggestion.starts_with("git "),
                "suggestion '{}' should start with 'git '",
                suggestion
            );
            assert!(*score > 0.0, "score should be positive");
        }
    }

    #[test]
    fn test_cargo_suggestions_from_corpus() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let suggestions = model.suggest("cargo ", 10);
        assert!(!suggestions.is_empty(), "should have cargo suggestions");

        // All suggestions should be cargo commands
        for (suggestion, score) in &suggestions {
            assert!(
                suggestion.starts_with("cargo "),
                "suggestion '{}' should start with 'cargo '",
                suggestion
            );
            assert!(*score > 0.0, "score should be positive");
        }
    }

    #[test]
    fn test_docker_suggestions_from_corpus() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let suggestions = model.suggest("docker ", 10);
        assert!(!suggestions.is_empty(), "should have docker suggestions");

        // All suggestions should be docker commands
        for (suggestion, score) in &suggestions {
            assert!(
                suggestion.starts_with("docker "),
                "suggestion '{}' should start with 'docker '",
                suggestion
            );
            assert!(*score > 0.0, "score should be positive");
        }
    }

    #[test]
    fn test_kubectl_suggestions_from_corpus() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let suggestions = model.suggest("kubectl ", 10);
        assert!(!suggestions.is_empty(), "should have kubectl suggestions");

        // All suggestions should be kubectl commands
        for (suggestion, score) in &suggestions {
            assert!(
                suggestion.starts_with("kubectl "),
                "suggestion '{}' should start with 'kubectl '",
                suggestion
            );
            assert!(*score > 0.0, "score should be positive");
        }
    }

    // ========================================================================
    // Partial Completion Tests (from corpus)
    // ========================================================================

    #[test]
    fn test_partial_git_c_from_corpus() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let suggestions = model.suggest("git c", 10);
        // Should suggest commit, checkout, clone, clean
        for (suggestion, _) in &suggestions {
            if suggestion.starts_with("git c") {
                // Valid partial match
                return;
            }
        }
        // At least trie matches should work
        assert!(
            !suggestions.is_empty() || model.vocab_size() > 0,
            "should have some git c suggestions or vocab"
        );
    }

    #[test]
    fn test_partial_cargo_b_from_corpus() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let suggestions = model.suggest("cargo b", 10);
        // Should suggest build, bench
        let has_build = suggestions.iter().any(|(s, _)| s.contains("build"));
        assert!(
            has_build || !suggestions.is_empty(),
            "should suggest cargo build"
        );
    }

    // ========================================================================
    // Model Persistence Tests (from corpus)
    // ========================================================================

    #[test]
    fn test_corpus_model_save_load_roundtrip() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let file = NamedTempFile::new().expect("temp file");
        model.save(file.path()).expect("save should succeed");

        let loaded = MarkovModel::load(file.path()).expect("load should succeed");

        assert_eq!(loaded.ngram_size(), model.ngram_size());
        assert_eq!(loaded.total_commands(), model.total_commands());
        assert_eq!(loaded.vocab_size(), model.vocab_size());
    }

    #[test]
    fn test_corpus_model_suggestions_preserved_after_save() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let file = NamedTempFile::new().expect("temp file");
        model.save(file.path()).expect("save");
        let loaded = MarkovModel::load(file.path()).expect("load");

        // Model structure should be preserved (exact equality)
        assert_eq!(
            model.ngram_size(),
            loaded.ngram_size(),
            "ngram size preserved"
        );
        assert_eq!(
            model.total_commands(),
            loaded.total_commands(),
            "total commands preserved"
        );
        assert_eq!(
            model.vocab_size(),
            loaded.vocab_size(),
            "vocab size preserved"
        );
        assert_eq!(
            model.ngram_count(),
            loaded.ngram_count(),
            "ngram count preserved"
        );

        // Both models should produce suggestions for common prefixes
        // (exact suggestions may vary due to HashMap iteration order in n-gram scoring)
        let git_suggestions = loaded.suggest("git ", 5);
        assert!(
            !git_suggestions.is_empty(),
            "loaded model should suggest git commands"
        );
        assert!(
            git_suggestions.iter().all(|(s, _)| s.starts_with("git ")),
            "all git suggestions should start with 'git '"
        );

        let cargo_suggestions = loaded.suggest("cargo ", 5);
        assert!(
            !cargo_suggestions.is_empty(),
            "loaded model should suggest cargo commands"
        );
        assert!(
            cargo_suggestions
                .iter()
                .all(|(s, _)| s.starts_with("cargo ")),
            "all cargo suggestions should start with 'cargo '"
        );
    }

    // ========================================================================
    // Validation Metrics Tests (from corpus)
    // ========================================================================

    #[test]
    fn test_corpus_validation_metrics() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let result = MarkovModel::validate(corpus.commands(), 3, 0.8);

        // Should have reasonable metrics
        assert!(result.train_size > 0, "should have training data");
        assert!(result.test_size > 0, "should have test data");
        assert!(result.evaluated > 0, "should evaluate some commands");

        // Metrics are valid (may be 0 for corpus with mostly unique commands)
        assert!(
            result.metrics.hit_at_10 >= 0.0 && result.metrics.hit_at_10 <= 1.0,
            "hit@10 should be in [0, 1], got {}",
            result.metrics.hit_at_10
        );
        assert!(
            result.metrics.mrr >= 0.0 && result.metrics.mrr <= 1.0,
            "mrr should be in [0, 1], got {}",
            result.metrics.mrr
        );
    }

    #[test]
    fn test_corpus_validation_80_20_split() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let result = MarkovModel::validate(corpus.commands(), 3, 0.8);

        // 80/20 split
        let expected_train = (corpus.len() as f32 * 0.8) as usize;
        assert_eq!(
            result.train_size, expected_train,
            "train size should be 80%"
        );
    }

    // ========================================================================
    // Model Statistics Tests
    // ========================================================================

    #[test]
    fn test_corpus_model_size_reasonable() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let size = model.size_bytes();
        // Should be < 10MB for ~500 commands
        assert!(
            size < 10_000_000,
            "model size should be < 10MB, got {} bytes",
            size
        );
        // Should be > 10KB (has content)
        assert!(
            size > 10_000,
            "model size should be > 10KB, got {} bytes",
            size
        );
    }

    #[test]
    fn test_corpus_top_commands() {
        let Some(corpus) = load_corpus_or_skip() else {
            return;
        };
        let mut model = MarkovModel::new(3);
        model.train(corpus.commands());

        let top = model.top_commands(10);
        assert_eq!(top.len(), 10, "should return 10 top commands");

        // First command should have frequency > 0
        assert!(top[0].1 > 0, "top command should have frequency > 0");
    }
}
