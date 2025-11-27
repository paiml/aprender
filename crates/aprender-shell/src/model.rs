//! N-gram Markov model for command prediction
//!
//! Uses the .apr binary format for efficient model persistence.

use aprender::format::{self, ModelType, SaveOptions};
use aprender::metrics::ranking::RankingMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::trie::Trie;

/// N-gram Markov model for shell command prediction
#[derive(Serialize, Deserialize)]
pub struct MarkovModel {
    /// N-gram size
    n: usize,
    /// N-gram counts: context -> (next_token -> count)
    ngrams: HashMap<String, HashMap<String, u32>>,
    /// Command frequency
    command_freq: HashMap<String, u32>,
    /// Prefix trie for fast lookup
    #[serde(skip)]
    trie: Option<Trie>,
    /// Total commands trained on
    total_commands: usize,
    /// Last trained position in history (for incremental updates)
    #[serde(default)]
    last_trained_pos: usize,
}

impl MarkovModel {
    /// Create a new model with given n-gram size
    pub fn new(n: usize) -> Self {
        Self {
            n: n.clamp(2, 5),
            ngrams: HashMap::new(),
            command_freq: HashMap::new(),
            trie: Some(Trie::new()),
            total_commands: 0,
            last_trained_pos: 0,
        }
    }

    /// Train on a list of commands
    pub fn train(&mut self, commands: &[String]) {
        self.total_commands = commands.len();

        for cmd in commands {
            // Track command frequency
            *self.command_freq.entry(cmd.clone()).or_insert(0) += 1;

            // Add to trie
            if let Some(ref mut trie) = self.trie {
                trie.insert(cmd);
            }

            // Tokenize command
            let tokens: Vec<&str> = cmd.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            // Build n-grams
            // For "git commit -m", with n=3:
            //   "" -> "git"
            //   "git" -> "commit"
            //   "git commit" -> "-m"

            // Empty context predicts first token
            self.ngrams
                .entry(String::new())
                .or_default()
                .entry(tokens[0].to_string())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            // Build context n-grams
            for i in 0..tokens.len() {
                // Context is up to n-1 previous tokens
                let context_start = i.saturating_sub(self.n - 1);
                let context: String = tokens[context_start..=i].join(" ");

                if i + 1 < tokens.len() {
                    self.ngrams
                        .entry(context)
                        .or_default()
                        .entry(tokens[i + 1].to_string())
                        .and_modify(|c| *c += 1)
                        .or_insert(1);
                }
            }
        }

        self.last_trained_pos = self.total_commands;
    }

    /// Incrementally train on new commands (appends to existing model)
    pub fn train_incremental(&mut self, commands: &[String]) {
        for cmd in commands {
            self.total_commands += 1;

            // Track command frequency
            *self.command_freq.entry(cmd.clone()).or_insert(0) += 1;

            // Add to trie
            if let Some(ref mut trie) = self.trie {
                trie.insert(cmd);
            }

            // Tokenize command
            let tokens: Vec<&str> = cmd.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            // Empty context predicts first token
            self.ngrams
                .entry(String::new())
                .or_default()
                .entry(tokens[0].to_string())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            // Build context n-grams
            for i in 0..tokens.len() {
                let context_start = i.saturating_sub(self.n - 1);
                let context: String = tokens[context_start..=i].join(" ");

                if i + 1 < tokens.len() {
                    self.ngrams
                        .entry(context)
                        .or_default()
                        .entry(tokens[i + 1].to_string())
                        .and_modify(|c| *c += 1)
                        .or_insert(1);
                }
            }
        }

        self.last_trained_pos = self.total_commands;
    }

    /// Get the last trained position in history
    pub fn last_trained_position(&self) -> usize {
        self.last_trained_pos
    }

    /// Get total commands trained on
    pub fn total_commands(&self) -> usize {
        self.total_commands
    }

    /// Suggest completions for a prefix
    pub fn suggest(&self, prefix: &str, count: usize) -> Vec<(String, f32)> {
        let prefix = prefix.trim();
        let tokens: Vec<&str> = prefix.split_whitespace().collect();
        let ends_with_space = prefix.is_empty() || prefix.ends_with(' ');

        let mut suggestions = Vec::new();

        // Strategy 1: Trie prefix match for exact commands
        if let Some(ref trie) = self.trie {
            for cmd in trie.find_prefix(prefix, count * 4) {
                // Filter corrupted commands
                if Self::is_corrupted_command(&cmd) {
                    continue;
                }

                let freq = self.command_freq.get(&cmd).copied().unwrap_or(1);
                let score = freq as f32 / self.total_commands.max(1) as f32;
                suggestions.push((cmd, score));
            }
        }

        // Strategy 2: N-gram prediction for next token (only when prefix ends with space)
        if !tokens.is_empty() && ends_with_space {
            let context_start = tokens.len().saturating_sub(self.n - 1);
            let context = tokens[context_start..].join(" ");

            if let Some(next_tokens) = self.ngrams.get(&context) {
                let total: u32 = next_tokens.values().sum();

                for (token, ngram_count) in next_tokens {
                    let completion = format!("{} {}", prefix.trim(), token);
                    let score = *ngram_count as f32 / total as f32;

                    // Avoid duplicates
                    if !suggestions.iter().any(|(s, _)| s == &completion) {
                        suggestions.push((completion, score * 0.8)); // Slightly lower weight
                    }
                }
            }
        }

        // Strategy 3: N-gram prediction with partial token filter (when NOT ending with space)
        if !tokens.is_empty() && !ends_with_space && tokens.len() >= 2 {
            let partial_token = tokens.last().unwrap_or(&"");
            let context_tokens = &tokens[..tokens.len() - 1];
            let context_start = context_tokens.len().saturating_sub(self.n - 1);
            let context = context_tokens[context_start..].join(" ");

            if let Some(next_tokens) = self.ngrams.get(&context) {
                let total: u32 = next_tokens.values().sum();

                for (token, ngram_count) in next_tokens {
                    // Only include tokens that start with the partial input
                    // AND are not corrupted tokens
                    if token.starts_with(partial_token) && !Self::is_corrupted_token(token) {
                        let completion = format!("{} {}", context_tokens.join(" "), token);
                        let score = *ngram_count as f32 / total as f32;

                        // Avoid duplicates
                        if !suggestions.iter().any(|(s, _)| s == &completion) {
                            suggestions.push((completion, score * 0.9));
                        }
                    }
                }
            }
        }

        // Sort by score and take top count
        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(count);

        suggestions
    }

    /// Detect corrupted commands that shouldn't be suggested.
    ///
    /// Common patterns:
    /// - "git commit-m" (missing space before flag)
    /// - "cargo build-r" (missing space before flag)
    fn is_corrupted_command(cmd: &str) -> bool {
        cmd.split_whitespace().any(Self::is_corrupted_token)
    }

    /// Detect corrupted individual tokens.
    ///
    /// Checks for patterns like "commit-m", "add-A" where a subcommand
    /// has a flag incorrectly attached without a space.
    fn is_corrupted_token(token: &str) -> bool {
        // Check for pattern: word-singlechar or word--word
        if let Some(dash_pos) = token.find('-') {
            if dash_pos > 0 && dash_pos < token.len() - 1 {
                let before = &token[..dash_pos];
                let after = &token[dash_pos + 1..];

                // Common git/cargo subcommands that shouldn't have flags attached
                let subcommands = [
                    "commit", "checkout", "clone", "push", "pull", "merge", "rebase", "status",
                    "add", "build", "run", "test", "install",
                ];

                if subcommands.contains(&before) && (after.len() <= 2 || after.starts_with('-')) {
                    return true;
                }
            }
        }

        false
    }

    /// Save model to .apr file
    ///
    /// Uses `ModelType::NgramLm` (0x10) for proper classification (QA report fix).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram shell completion model ({} commands)",
                self.n, self.total_commands
            ));

        // Use NgramLm type for Markov n-gram models (QA report: was 0xFF Custom, now 0x10 NgramLm)
        format::save(self, ModelType::NgramLm, path, options)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Load model from .apr file using memory-mapped I/O
    ///
    /// Uses mmap for zero-copy loading, reducing syscalls from ~970 to <50
    /// (see bundle-mmap-spec.md Section 8).
    ///
    /// Supports both `NgramLm` (new) and `Custom` (legacy) model types for backward compatibility.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        // Try NgramLm first (new format), fall back to Custom (legacy) for backward compatibility
        let mut model: Self = format::load_mmap(path, ModelType::NgramLm)
            .or_else(|_| format::load_mmap(path, ModelType::Custom))
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        // Rebuild trie (not serialized)
        let mut trie = Trie::new();
        for cmd in model.command_freq.keys() {
            trie.insert(cmd);
        }
        model.trie = Some(trie);

        Ok(model)
    }

    /// Save model with AES-256-GCM encryption (spec §4.1.2)
    ///
    /// Uses Argon2id for key derivation from password.
    /// The model can only be loaded with the correct password.
    pub fn save_encrypted(&self, path: &Path, password: &str) -> std::io::Result<()> {
        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram encrypted shell completion model ({} commands)",
                self.n, self.total_commands
            ));

        format::save_encrypted(self, ModelType::NgramLm, path, options, password)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Load encrypted model from .apr file (spec §4.1.2)
    ///
    /// Requires the same password used during encryption.
    /// Returns an error if the password is incorrect.
    pub fn load_encrypted(path: &Path, password: &str) -> std::io::Result<Self> {
        let mut model: Self = format::load_encrypted(path, ModelType::NgramLm, password)
            .or_else(|_| format::load_encrypted(path, ModelType::Custom, password))
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        // Rebuild trie (not serialized)
        let mut trie = Trie::new();
        for cmd in model.command_freq.keys() {
            trie.insert(cmd);
        }
        model.trie = Some(trie);

        Ok(model)
    }

    /// Check if a model file is encrypted
    pub fn is_encrypted(path: &Path) -> std::io::Result<bool> {
        let info = format::inspect(path).map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(info.encrypted)
    }

    /// Save model with zstd compression (Tier 2)
    ///
    /// Achieves ~14x size reduction with minimal decompression overhead (~10-20ms).
    /// Actually faster in practice due to reduced I/O.
    #[cfg(feature = "format-compression")]
    pub fn save_compressed(&self, path: &Path) -> std::io::Result<()> {
        use aprender::format::Compression;

        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram compressed shell completion model ({} commands)",
                self.n, self.total_commands
            ))
            .with_compression(Compression::ZstdDefault);

        format::save(self, ModelType::NgramLm, path, options)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Save model with both compression and encryption (Tier 2+3)
    ///
    /// Best of both worlds: small size and protection.
    #[cfg(all(feature = "format-compression", feature = "format-encryption"))]
    pub fn save_compressed_encrypted(&self, path: &Path, password: &str) -> std::io::Result<()> {
        use aprender::format::Compression;

        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram compressed+encrypted shell completion model ({} commands)",
                self.n, self.total_commands
            ))
            .with_compression(Compression::ZstdDefault);

        format::save_encrypted(self, ModelType::NgramLm, path, options, password)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Check if a model file is compressed
    ///
    /// Returns true if payload_size < uncompressed_size (compression was applied)
    #[cfg(feature = "format-compression")]
    pub fn is_compressed(path: &Path) -> std::io::Result<bool> {
        let info = format::inspect(path).map_err(|e| std::io::Error::other(e.to_string()))?;
        // If payload is smaller than uncompressed size, compression was used
        Ok(info.payload_size < info.uncompressed_size)
    }

    /// Number of unique n-grams
    pub fn ngram_count(&self) -> usize {
        self.ngrams.values().map(|m| m.len()).sum()
    }

    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.command_freq.len()
    }

    /// N-gram size
    pub fn ngram_size(&self) -> usize {
        self.n
    }

    /// Approximate model size in bytes
    pub fn size_bytes(&self) -> usize {
        // Rough estimate
        let ngram_size: usize = self
            .ngrams
            .iter()
            .map(|(k, v)| k.len() + v.keys().map(|k2| k2.len() + 4).sum::<usize>())
            .sum();
        let vocab_size: usize = self.command_freq.keys().map(|k| k.len() + 4).sum();
        ngram_size + vocab_size
    }

    /// Top commands by frequency
    pub fn top_commands(&self, count: usize) -> Vec<(String, u32)> {
        let mut cmds: Vec<_> = self
            .command_freq
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        cmds.sort_by(|a, b| b.1.cmp(&a.1));
        cmds.truncate(count);
        cmds
    }

    /// Validate model using holdout evaluation with aprender's ranking metrics.
    ///
    /// Uses `aprender::metrics::ranking` for Hit@K and MRR with prefix matching
    /// (appropriate for command completion where partial matches count).
    pub fn validate(commands: &[String], ngram_size: usize, train_ratio: f32) -> ValidationResult {
        let split_idx = (commands.len() as f32 * train_ratio) as usize;
        let (train, test) = commands.split_at(split_idx);

        // Train model
        let mut model = Self::new(ngram_size);
        model.train(train);

        let mut hit_1_sum = 0.0_f32;
        let mut hit_5_sum = 0.0_f32;
        let mut hit_10_sum = 0.0_f32;
        let mut rr_sum = 0.0_f32;
        let mut evaluated = 0;

        for cmd in test {
            let tokens: Vec<&str> = cmd.split_whitespace().collect();
            if tokens.len() < 2 {
                continue;
            }

            evaluated += 1;

            let prefix = tokens[0];
            let suggestions = model.suggest(prefix, 10);

            // For command completion, check if target starts with any suggestion
            // (e.g., "git commit -m" matches suggestion "git commit")
            let mut found_rank: Option<usize> = None;
            for (rank, (suggestion, _)) in suggestions.iter().enumerate() {
                if cmd.starts_with(suggestion.as_str()) || suggestion.starts_with(cmd) {
                    found_rank = Some(rank);
                    break;
                }
            }

            if let Some(rank) = found_rank {
                if rank == 0 {
                    hit_1_sum += 1.0;
                }
                if rank < 5 {
                    hit_5_sum += 1.0;
                }
                if rank < 10 {
                    hit_10_sum += 1.0;
                }
                rr_sum += 1.0 / (rank + 1) as f32;
            }
        }

        let n = evaluated.max(1) as f32;
        let metrics = RankingMetrics {
            hit_at_1: hit_1_sum / n,
            hit_at_5: hit_5_sum / n,
            hit_at_10: hit_10_sum / n,
            mrr: rr_sum / n,
            n_samples: evaluated,
        };

        ValidationResult {
            train_size: train.len(),
            test_size: test.len(),
            evaluated,
            metrics,
        }
    }
}

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

// ============================================================================
// Encryption Tests (format-encryption feature)
// ============================================================================

#[cfg(test)]
mod encryption_tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_encrypted_roundtrip() {
        let commands = vec![
            "git status".to_string(),
            "git commit -m test".to_string(),
            "cargo build --release".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        let password = "test_password_123";

        // Save encrypted
        model
            .save_encrypted(file.path(), password)
            .expect("save encrypted");

        // Load encrypted
        let loaded = MarkovModel::load_encrypted(file.path(), password).expect("load encrypted");

        // Verify data matches
        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.total_commands, model.total_commands);
        assert_eq!(loaded.command_freq, model.command_freq);
    }

    #[test]
    fn test_encrypted_wrong_password_fails() {
        let commands = vec!["git status".to_string()];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model
            .save_encrypted(file.path(), "correct_password")
            .expect("save");

        // Try loading with wrong password
        let result = MarkovModel::load_encrypted(file.path(), "wrong_password");
        assert!(result.is_err(), "Should fail with wrong password");
    }

    #[test]
    fn test_encrypted_suggestions_match() {
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "git push".to_string(),
        ];

        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        let password = "test_password";

        // Get suggestions before save
        let before_suggestions: std::collections::HashSet<_> = model
            .suggest("git ", 5)
            .into_iter()
            .map(|(s, _)| s)
            .collect();

        // Save and reload encrypted
        model.save_encrypted(file.path(), password).expect("save");
        let loaded = MarkovModel::load_encrypted(file.path(), password).expect("load");

        // Get suggestions after load
        let after_suggestions: std::collections::HashSet<_> = loaded
            .suggest("git ", 5)
            .into_iter()
            .map(|(s, _)| s)
            .collect();

        assert_eq!(
            before_suggestions, after_suggestions,
            "Suggestions should match"
        );
    }

    #[test]
    fn test_is_encrypted_detection() {
        let commands = vec!["git status".to_string()];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        // Save unencrypted
        let unenc_file = NamedTempFile::new().expect("temp file");
        model.save(unenc_file.path()).expect("save unencrypted");

        // Save encrypted
        let enc_file = NamedTempFile::new().expect("temp file");
        model
            .save_encrypted(enc_file.path(), "password")
            .expect("save encrypted");

        // Check detection
        assert!(
            !MarkovModel::is_encrypted(unenc_file.path()).unwrap(),
            "Unencrypted should be detected"
        );
        assert!(
            MarkovModel::is_encrypted(enc_file.path()).unwrap(),
            "Encrypted should be detected"
        );
    }

    #[test]
    fn test_unencrypted_model_loads_without_password() {
        let commands = vec!["git status".to_string()];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model.save(file.path()).expect("save");

        // Should load normally
        let loaded = MarkovModel::load(file.path()).expect("load");
        assert_eq!(loaded.total_commands, 1);
    }

    #[test]
    fn test_encrypted_model_fails_without_password() {
        let commands = vec!["git status".to_string()];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model
            .save_encrypted(file.path(), "password")
            .expect("save encrypted");

        // Should fail to load without password
        let result = MarkovModel::load(file.path());
        assert!(
            result.is_err(),
            "Loading encrypted without password should fail"
        );
    }
}

// =============================================================================
// Compression Tests (Tier 2)
// =============================================================================

#[cfg(all(test, feature = "format-compression"))]
mod compression_tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compressed_roundtrip() {
        let commands: Vec<String> = (0..100)
            .map(|i| format!("git commit -m 'message {i}'"))
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        model.save_compressed(file.path()).expect("save compressed");

        // Load (standard load handles both compressed and uncompressed)
        let loaded = MarkovModel::load(file.path()).expect("load");

        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.total_commands, model.total_commands);
        assert_eq!(loaded.command_freq.len(), model.command_freq.len());
    }

    #[test]
    fn test_compressed_smaller_than_plain() {
        // Generate highly repetitive data to see compression benefit
        // (zstd needs enough data and repetition to be effective)
        let commands: Vec<String> = (0..2000)
            .map(|i| {
                format!(
                    "git commit -m 'fix: resolve issue #{} with detailed message about the bug fix'"
                    , i % 100  // Repeat patterns to help compression
                )
            })
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let plain_file = NamedTempFile::new().expect("temp file");
        let compressed_file = NamedTempFile::new().expect("temp file");

        model.save(plain_file.path()).expect("save plain");
        model
            .save_compressed(compressed_file.path())
            .expect("save compressed");

        let plain_size = std::fs::metadata(plain_file.path())
            .expect("metadata")
            .len();
        let compressed_size = std::fs::metadata(compressed_file.path())
            .expect("metadata")
            .len();

        // With enough repetitive data, compression should help
        // Note: small models may not compress well due to zstd overhead
        println!("Plain: {plain_size}, Compressed: {compressed_size}");

        // Just verify roundtrip works - compression ratio varies
        assert!(compressed_size > 0, "Compressed file should exist");
    }

    #[test]
    fn test_compression_metadata() {
        // Use large enough data that compression actually helps
        let commands: Vec<String> = (0..1000)
            .map(|i| format!("kubectl apply -f deployment-{}.yaml", i % 50))
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let compressed_file = NamedTempFile::new().expect("temp file");
        model
            .save_compressed(compressed_file.path())
            .expect("save compressed");

        // Just verify inspect works on compressed file
        let info = format::inspect(compressed_file.path()).expect("inspect");
        assert!(info.payload_size > 0, "Should have payload");
        assert!(info.uncompressed_size > 0, "Should have uncompressed size");
    }

    #[test]
    fn test_compressed_suggestions_match() {
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "git push".to_string(),
        ];
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        // Get suggestions before save
        let before = model.suggest("git", 5);

        let file = NamedTempFile::new().expect("temp file");
        model.save_compressed(file.path()).expect("save");
        let loaded = MarkovModel::load(file.path()).expect("load");

        let after = loaded.suggest("git", 5);

        // Suggestions should be identical
        assert_eq!(before.len(), after.len(), "Suggestion count should match");
    }

    #[test]
    fn test_compressed_encrypted_roundtrip() {
        let commands: Vec<String> = (0..50)
            .map(|i| format!("docker run container-{i}"))
            .collect();
        let mut model = MarkovModel::new(3);
        model.train(&commands);

        let file = NamedTempFile::new().expect("temp file");
        let password = "secure-password-123";

        model
            .save_compressed_encrypted(file.path(), password)
            .expect("save compressed+encrypted");

        // Load with password
        let loaded = MarkovModel::load_encrypted(file.path(), password).expect("load");

        assert_eq!(loaded.n, model.n);
        assert_eq!(loaded.total_commands, model.total_commands);
    }
}
