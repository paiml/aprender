
/// Validation metrics for shell model using aprender's ranking metrics.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Number of commands in training set
    pub train_size: usize,
    /// Number of commands in test set
    pub test_size: usize,
    /// Number of commands evaluated (with >= 2 tokens)
    pub evaluated: usize,
    /// Ranking metrics from aprender (Hit@K, MRR)
    pub metrics: RankingMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_suggest() {
        let commands = vec![
            "git status".to_string(),
            "git commit -m test".to_string(),
            "git push".to_string(),
            "git status".to_string(),
            "git log".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let suggestions = model.suggest("git ", 3);
        assert!(!suggestions.is_empty());

        // "status" should be suggested (appears twice)
        let has_status = suggestions.iter().any(|(s, _)| s.contains("status"));
        assert!(has_status);
    }

    #[test]
    fn test_ngram_counts() {
        let commands = vec!["ls -la".to_string(), "ls -la /tmp".to_string()];

        let mut model = MarkovModel::new(2);
        model.train(&commands);

        assert!(model.ngram_count() > 0);
        assert_eq!(model.vocab_size(), 2);
    }

    // ==================== EXTREME TDD: Partial Token Tests ====================

    #[test]
    fn test_partial_token_completion() {
        // CRITICAL: "git c" should complete to "git commit", "git checkout"
        // NOT return corrupted full commands like "git commit-m"
        let commands = vec![
            "git commit -m test".to_string(),
            "git checkout main".to_string(),
            "git clone url".to_string(),
            "git status".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let suggestions = model.suggest("git c", 5);
        assert!(
            !suggestions.is_empty(),
            "Should have suggestions for 'git c'"
        );

        // All suggestions should start with "git c"
        for (suggestion, _) in &suggestions {
            assert!(
                suggestion.starts_with("git c"),
                "Suggestion '{}' should start with 'git c'",
                suggestion
            );
        }

        // Should suggest commit, checkout, clone
        let suggestion_text: String = suggestions.iter().map(|(s, _)| s.as_str()).collect();
        assert!(
            suggestion_text.contains("commit")
                || suggestion_text.contains("checkout")
                || suggestion_text.contains("clone"),
            "Should suggest commit/checkout/clone, got: {:?}",
            suggestions
        );
    }

    #[test]
    fn test_is_corrupted_command() {
        // Test the corruption detection helper
        assert!(
            MarkovModel::is_corrupted_command("git commit-m test"),
            "Should detect 'commit-m' as corrupted"
        );
        assert!(
            MarkovModel::is_corrupted_command("git add-A"),
            "Should detect 'add-A' as corrupted"
        );
        assert!(
            !MarkovModel::is_corrupted_command("git commit -m test"),
            "Should NOT detect valid 'commit -m' as corrupted"
        );
        assert!(
            !MarkovModel::is_corrupted_command("git checkout feature-branch"),
            "Should NOT detect 'feature-branch' as corrupted"
        );
    }

    #[test]
    fn test_partial_token_filters_corrupted() {
        // Even if corrupted commands exist, partial completion should not return them
        let commands = vec![
            "git commit -m test".to_string(),
            "git commit-m broken".to_string(), // corrupted - no space
            "git checkout main".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let suggestions = model.suggest("git co", 5);

        // Should NOT include "git commit-m" - that's corrupted
        for (suggestion, _) in &suggestions {
            assert!(
                !suggestion.contains("commit-m"),
                "Should not suggest corrupted 'commit-m', got: {}",
                suggestion
            );
        }
    }

    #[test]
    fn test_partial_token_single_char() {
        // "git s" should suggest "git status", "git stash"
        let commands = vec![
            "git status".to_string(),
            "git status".to_string(),
            "git stash".to_string(),
            "git show".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let suggestions = model.suggest("git s", 5);
        assert!(!suggestions.is_empty());

        // All should start with "git s"
        for (suggestion, _) in &suggestions {
            assert!(
                suggestion.starts_with("git s"),
                "Expected 'git s*', got: {}",
                suggestion
            );
        }

        // status should rank highest (appears twice)
        assert!(
            suggestions[0].0.contains("status"),
            "Most frequent 'status' should be first, got: {}",
            suggestions[0].0
        );
    }

    #[test]
    fn test_trailing_space_vs_no_space() {
        // "git " (with space) = predict next token
        // "git" (no space) = complete current token
        let commands = vec![
            "git status".to_string(),
            "grep pattern".to_string(),
            "git commit".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        // With trailing space: predict next token
        let with_space = model.suggest("git ", 5);
        assert!(with_space
            .iter()
            .any(|(s, _)| s == "git status" || s == "git commit"));

        // Without trailing space: complete "git" to commands starting with "git"
        let without_space = model.suggest("git", 5);
        // Should suggest git commands, not grep
        assert!(without_space.iter().all(|(s, _)| s.starts_with("git")));
    }

    // ==================== Issue #92: Malformed Suggestions ====================

    #[test]
    fn test_is_corrupted_double_spaces() {
        // Double spaces indicate corruption
        assert!(
            MarkovModel::is_corrupted_command("cargo-lambda  help"),
            "Should detect double spaces as corrupted"
        );
        assert!(
            MarkovModel::is_corrupted_command("git  status"),
            "Should detect double spaces as corrupted"
        );
        assert!(
            !MarkovModel::is_corrupted_command("git status"),
            "Single space is valid"
        );
    }

    #[test]
    fn test_is_corrupted_trailing_backslash() {
        // Trailing backslashes indicate incomplete multiline
        assert!(
            MarkovModel::is_corrupted_command("git rm -r --cached vendor/\\"),
            "Should detect trailing backslash"
        );
        assert!(
            MarkovModel::is_corrupted_command("cargo lambda deploy \\\\"),
            "Should detect trailing escape"
        );
        assert!(
            !MarkovModel::is_corrupted_command("git rm -r --cached vendor/"),
            "Path without backslash is valid"
        );
    }

    #[test]
    fn test_is_corrupted_typos() {
        // Common typos where space merged with next word
        assert!(
            MarkovModel::is_corrupted_command("gitr push"),
            "Should detect 'gitr' as typo"
        );
        assert!(
            MarkovModel::is_corrupted_command("giti pull"),
            "Should detect 'giti' as typo"
        );
        assert!(
            MarkovModel::is_corrupted_command("cargoo build"),
            "Should detect 'cargoo' as typo"
        );
        assert!(
            !MarkovModel::is_corrupted_command("git push"),
            "Valid command should pass"
        );
        assert!(
            !MarkovModel::is_corrupted_command("cargo build"),
            "Valid command should pass"
        );
    }
}

// ============================================================================
// Property-Based Tests for Model Format (QA Report Fix Verification)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use tempfile::NamedTempFile;

    // Strategy for generating valid shell commands
    fn arb_command() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("git status".to_string()),
            Just("git commit -m 'test'".to_string()),
            Just("git push origin main".to_string()),
            Just("cargo build --release".to_string()),
            Just("cargo test".to_string()),
            Just("docker run -it ubuntu".to_string()),
            Just("kubectl get pods".to_string()),
            Just("npm install".to_string()),
            Just("ls -la".to_string()),
            Just("cd ..".to_string()),
            // Generate random commands
            "[a-z]{3,10}( -[a-z])?( [a-z]{2,8})?".prop_map(|s| s),
        ]
    }

    // Strategy for generating command lists
    fn arb_commands(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
        proptest::collection::vec(arb_command(), min..max)
    }

    proptest! {
        /// Property: Model save/load roundtrip preserves data
        #[test]
        fn prop_roundtrip_preserves_data(commands in arb_commands(5, 50)) {
            let mut model = MarkovModel::new(3);
            model.train(&commands);

            let file = NamedTempFile::new().expect("temp file");
            model.save(file.path()).expect("save");

            let loaded = MarkovModel::load(file.path()).expect("load");

            prop_assert_eq!(loaded.n, model.n, "n-gram size mismatch");
            prop_assert_eq!(loaded.total_commands, model.total_commands, "command count mismatch");
            prop_assert_eq!(loaded.command_freq.len(), model.command_freq.len(), "vocab mismatch");
        }

        /// Property: Model uses NgramLm type (0x0010), not Custom (0x00FF)
        #[test]
        fn prop_model_type_is_ngram_lm(commands in arb_commands(3, 20)) {
            let mut model = MarkovModel::new(3);
            model.train(&commands);

            let file = NamedTempFile::new().expect("temp file");
            model.save(file.path()).expect("save");

            let bytes = fs::read(file.path()).expect("read");

            // Model type is at bytes 6-7
            let model_type = u16::from_le_bytes([bytes[6], bytes[7]]);
            prop_assert_eq!(model_type, 0x0010, "Model type should be NgramLm (0x0010)");
        }

        /// Property: Model file has valid APRN magic
        #[test]
        fn prop_magic_is_aprn(commands in arb_commands(3, 20)) {
            let mut model = MarkovModel::new(3);
            model.train(&commands);

            let file = NamedTempFile::new().expect("temp file");
            model.save(file.path()).expect("save");

            let bytes = fs::read(file.path()).expect("read");
            prop_assert_eq!(&bytes[0..4], b"APRN", "Magic should be APRN");
        }

        /// Property: Command frequencies preserved after roundtrip
        #[test]
        fn prop_command_freq_preserved_after_roundtrip(commands in arb_commands(10, 50)) {
            let mut model = MarkovModel::new(3);
            model.train(&commands);

            // Get command frequencies before save
            let before_freq = model.command_freq.clone();

            let file = NamedTempFile::new().expect("temp file");
            model.save(file.path()).expect("save");
            let loaded = MarkovModel::load(file.path()).expect("load");

            // Compare command frequencies (exact match expected)
            prop_assert_eq!(loaded.command_freq, before_freq, "command_freq should match after roundtrip");
        }

        /// Property: N-gram size is preserved
        #[test]
        fn prop_ngram_size_preserved(n in 2usize..=5) {
            let commands: Vec<String> = vec![
                "git status".to_string(),
                "git commit".to_string(),
                "cargo build".to_string(),
            ];

            let mut model = MarkovModel::new(n);
            model.train(&commands);

            let file = NamedTempFile::new().expect("temp file");
            model.save(file.path()).expect("save");
            let loaded = MarkovModel::load(file.path()).expect("load");

            prop_assert_eq!(loaded.n, n, "n-gram size should be preserved");
        }

        /// Property: Empty model can be saved and loaded
        #[test]
        fn prop_empty_model_roundtrip(n in 2usize..=5) {
            let model = MarkovModel::new(n);

            let file = NamedTempFile::new().expect("temp file");
            model.save(file.path()).expect("save");
            let loaded = MarkovModel::load(file.path()).expect("load");

            prop_assert_eq!(loaded.n, n);
            prop_assert_eq!(loaded.total_commands, 0);
            prop_assert!(loaded.command_freq.is_empty());
        }

        /// Property: File size is reasonable (not a zip bomb)
        #[test]
        fn prop_file_size_reasonable(commands in arb_commands(10, 100)) {
            let mut model = MarkovModel::new(3);
            model.train(&commands);

            let file = NamedTempFile::new().expect("temp file");
            model.save(file.path()).expect("save");

            let metadata = fs::metadata(file.path()).expect("metadata");
            let size = metadata.len();

            // File should be < 1MB for 100 commands
            prop_assert!(size < 1_000_000, "File too large: {} bytes", size);
            // File should be > 100 bytes (has actual content)
            prop_assert!(size > 100, "File too small: {} bytes", size);
        }
    }
}
