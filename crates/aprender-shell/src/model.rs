//! N-gram Markov model for command prediction
//!
//! Uses the .apr binary format for efficient model persistence.

use aprender::format::{self, ModelType, SaveOptions};
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

        let mut suggestions = Vec::new();

        // Strategy 1: Trie prefix match for exact commands
        if let Some(ref trie) = self.trie {
            for cmd in trie.find_prefix(prefix, count * 2) {
                let freq = self.command_freq.get(&cmd).copied().unwrap_or(1);
                let score = freq as f32 / self.total_commands.max(1) as f32;
                suggestions.push((cmd, score));
            }
        }

        // Strategy 2: N-gram prediction for next token
        if !tokens.is_empty() {
            let context_start = tokens.len().saturating_sub(self.n - 1);
            let context = tokens[context_start..].join(" ");

            if let Some(next_tokens) = self.ngrams.get(&context) {
                let total: u32 = next_tokens.values().sum();

                for (token, count) in next_tokens {
                    let completion = format!("{} {}", prefix, token);
                    let score = *count as f32 / total as f32;

                    // Avoid duplicates
                    if !suggestions.iter().any(|(s, _)| s == &completion) {
                        suggestions.push((completion, score * 0.8)); // Slightly lower weight
                    }
                }
            }
        }

        // Sort by score and take top count
        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(count);

        suggestions
    }

    /// Save model to .apr file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(&format!(
                "{}-gram shell completion model ({} commands)",
                self.n, self.total_commands
            ));

        format::save(self, ModelType::Custom, path, options)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    /// Load model from .apr file
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut model: Self = format::load(path, ModelType::Custom)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Rebuild trie (not serialized)
        let mut trie = Trie::new();
        for cmd in model.command_freq.keys() {
            trie.insert(cmd);
        }
        model.trie = Some(trie);

        Ok(model)
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
            .map(|(k, v)| k.len() + v.iter().map(|(k2, _)| k2.len() + 4).sum::<usize>())
            .sum();
        let vocab_size: usize = self.command_freq.iter().map(|(k, _)| k.len() + 4).sum();
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
}
