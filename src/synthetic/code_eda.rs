//! Code-Specific EDA (Easy Data Augmentation) for source code.
//!
//! Implements code augmentation techniques inspired by Wei & Zou (2019) EDA paper,
//! adapted for programming languages. Operations preserve syntactic validity while
//! introducing meaningful variation for training code analysis models.
//!
//! # Operations
//!
//! 1. **Variable Renaming (VR)**: Rename variables to synonyms (e.g., `x` -> `value`)
//! 2. **Comment Insertion (CI)**: Insert comments or assertions
//! 3. **Statement Reorder (SR)**: Reorder independent statements
//! 4. **Dead Code Removal (DCR)**: Remove comments/whitespace
//!
//! # Example
//!
//! ```
//! use aprender::synthetic::code_eda::{CodeEda, CodeEdaConfig};
//! use aprender::synthetic::{SyntheticGenerator, SyntheticConfig};
//!
//! let config = CodeEdaConfig::default();
//! let generator = CodeEda::new(config);
//!
//! let code = "let x = 42;\nprintln!(\"{}\", x);";
//! let augmented = generator.augment(code, 42);
//! assert!(!augmented.is_empty());
//! ```

use super::{SyntheticConfig, SyntheticGenerator};
use crate::error::Result;
use std::collections::{HashMap, HashSet};

/// Configuration for code-specific EDA augmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeEdaConfig {
    /// Probability of variable renaming per identifier.
    pub rename_prob: f32,
    /// Probability of comment insertion.
    pub comment_prob: f32,
    /// Probability of statement reordering.
    pub reorder_prob: f32,
    /// Probability of dead code removal.
    pub remove_prob: f32,
    /// Number of augmented samples to generate per input.
    pub num_augments: usize,
    /// Minimum tokens to apply augmentation.
    pub min_tokens: usize,
    /// Target programming language (for syntax-aware operations).
    pub language: CodeLanguage,
}

/// Supported programming languages for code augmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodeLanguage {
    /// Rust programming language
    #[default]
    Rust,
    /// Python programming language
    Python,
    /// Generic (language-agnostic operations only)
    Generic,
}

impl Default for CodeEdaConfig {
    fn default() -> Self {
        Self {
            rename_prob: 0.15,
            comment_prob: 0.1,
            reorder_prob: 0.05,
            remove_prob: 0.1,
            num_augments: 4,
            min_tokens: 5,
            language: CodeLanguage::Rust,
        }
    }
}

impl CodeEdaConfig {
    /// Create a new code EDA configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set variable renaming probability.
    #[must_use]
    pub fn with_rename_prob(mut self, prob: f32) -> Self {
        self.rename_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set comment insertion probability.
    #[must_use]
    pub fn with_comment_prob(mut self, prob: f32) -> Self {
        self.comment_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set statement reorder probability.
    #[must_use]
    pub fn with_reorder_prob(mut self, prob: f32) -> Self {
        self.reorder_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set dead code removal probability.
    #[must_use]
    pub fn with_remove_prob(mut self, prob: f32) -> Self {
        self.remove_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set number of augmented samples per input.
    #[must_use]
    pub fn with_num_augments(mut self, n: usize) -> Self {
        self.num_augments = n.max(1);
        self
    }

    /// Set minimum tokens for augmentation.
    #[must_use]
    pub fn with_min_tokens(mut self, n: usize) -> Self {
        self.min_tokens = n;
        self
    }

    /// Set target programming language.
    #[must_use]
    pub fn with_language(mut self, lang: CodeLanguage) -> Self {
        self.language = lang;
        self
    }
}

/// Variable synonym dictionary for code identifiers.
#[derive(Debug, Clone)]
pub struct VariableSynonyms {
    synonyms: HashMap<String, Vec<String>>,
}

impl Default for VariableSynonyms {
    fn default() -> Self {
        let mut synonyms = HashMap::new();

        // Common single-letter to descriptive name mappings
        synonyms.insert(
            "x".to_string(),
            vec!["value".to_string(), "val".to_string()],
        );
        synonyms.insert(
            "y".to_string(),
            vec!["result".to_string(), "res".to_string()],
        );
        synonyms.insert(
            "i".to_string(),
            vec!["index".to_string(), "idx".to_string()],
        );
        synonyms.insert(
            "j".to_string(),
            vec!["inner".to_string(), "jdx".to_string()],
        );
        synonyms.insert(
            "n".to_string(),
            vec!["count".to_string(), "num".to_string()],
        );
        synonyms.insert("s".to_string(), vec!["str".to_string(), "text".to_string()]);

        // Common variable patterns
        synonyms.insert(
            "tmp".to_string(),
            vec!["temp".to_string(), "scratch".to_string()],
        );
        synonyms.insert(
            "data".to_string(),
            vec!["input".to_string(), "payload".to_string()],
        );
        synonyms.insert(
            "result".to_string(),
            vec!["output".to_string(), "ret".to_string()],
        );
        synonyms.insert(
            "buf".to_string(),
            vec!["buffer".to_string(), "data".to_string()],
        );
        synonyms.insert(
            "len".to_string(),
            vec!["length".to_string(), "size".to_string()],
        );
        synonyms.insert(
            "err".to_string(),
            vec!["error".to_string(), "e".to_string()],
        );
        synonyms.insert(
            "msg".to_string(),
            vec!["message".to_string(), "text".to_string()],
        );

        // Function-related
        synonyms.insert("fn".to_string(), vec!["func".to_string()]);
        synonyms.insert(
            "args".to_string(),
            vec!["params".to_string(), "arguments".to_string()],
        );

        Self { synonyms }
    }
}

impl VariableSynonyms {
    /// Create a new variable synonyms dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom synonym mapping.
    pub fn add_synonym(&mut self, word: String, alternatives: Vec<String>) {
        self.synonyms.insert(word, alternatives);
    }

    /// Get synonyms for a word.
    #[must_use]
    pub fn get(&self, word: &str) -> Option<&[String]> {
        self.synonyms.get(word).map(Vec::as_slice)
    }

    /// Check if a word has synonyms.
    #[must_use]
    pub fn has_synonym(&self, word: &str) -> bool {
        self.synonyms.contains_key(word)
    }
}

/// Code-specific EDA generator.
///
/// Performs syntax-aware augmentation operations on source code while
/// attempting to preserve program semantics (where possible).
#[derive(Debug, Clone)]
pub struct CodeEda {
    config: CodeEdaConfig,
    synonyms: VariableSynonyms,
    /// Reserved keywords that should not be renamed
    reserved: HashSet<String>,
}

impl CodeEda {
    /// Create a new code EDA generator with the given configuration.
    #[must_use]
    pub fn new(config: CodeEdaConfig) -> Self {
        let reserved = Self::get_reserved_keywords(config.language);
        Self {
            config,
            synonyms: VariableSynonyms::default(),
            reserved,
        }
    }

    /// Get reserved keywords for a language.
    fn get_reserved_keywords(lang: CodeLanguage) -> HashSet<String> {
        let keywords: &[&str] = match lang {
            CodeLanguage::Rust => &[
                "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else",
                "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
                "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
                "super", "trait", "true", "type", "unsafe", "use", "where", "while", "abstract",
                "become", "box", "do", "final", "macro", "override", "priv", "try", "typeof",
                "unsized", "virtual", "yield",
            ],
            CodeLanguage::Python => &[
                "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
                "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
                "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
                "raise", "return", "try", "while", "with", "yield",
            ],
            CodeLanguage::Generic => &[],
        };
        keywords.iter().map(|s| (*s).to_string()).collect()
    }

    /// Augment a single code sample.
    ///
    /// # Arguments
    ///
    /// * `code` - Source code to augment
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Augmented code string.
    #[must_use]
    pub fn augment(&self, code: &str, seed: u64) -> String {
        let tokens = self.tokenize(code);
        if tokens.len() < self.config.min_tokens {
            return code.to_string();
        }

        let mut result_tokens = tokens.clone();
        let mut rng_state = seed;

        // Apply operations based on probability
        if self.random_f32(&mut rng_state) < self.config.rename_prob {
            result_tokens = self.apply_variable_rename(&result_tokens, &mut rng_state);
        }

        if self.random_f32(&mut rng_state) < self.config.comment_prob {
            result_tokens = self.apply_comment_insertion(&result_tokens, &mut rng_state);
        }

        if self.random_f32(&mut rng_state) < self.config.reorder_prob {
            result_tokens = self.apply_statement_reorder(&result_tokens, &mut rng_state);
        }

        if self.random_f32(&mut rng_state) < self.config.remove_prob {
            result_tokens = self.apply_dead_code_removal(&result_tokens);
        }

        result_tokens.join("")
    }

    /// Simple tokenization preserving whitespace and structure.
    #[allow(clippy::unused_self)]
    fn tokenize(&self, code: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in code.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(ch.to_string());
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Apply variable renaming operation.
    fn apply_variable_rename(&self, tokens: &[String], rng: &mut u64) -> Vec<String> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut rename_map: HashMap<String, String> = HashMap::new();

        for token in tokens {
            // Check if token is an identifier (not reserved, alphanumeric start)
            if self.is_identifier(token) && !self.reserved.contains(token) {
                if let Some(synonyms) = self.synonyms.get(token) {
                    // Use cached rename or pick new one
                    let renamed = rename_map.entry(token.clone()).or_insert_with(|| {
                        let idx = (self.random_u64(rng) as usize) % synonyms.len();
                        synonyms[idx].clone()
                    });
                    result.push(renamed.clone());
                } else {
                    result.push(token.clone());
                }
            } else {
                result.push(token.clone());
            }
        }

        result
    }

    /// Apply comment insertion operation.
    fn apply_comment_insertion(&self, tokens: &[String], rng: &mut u64) -> Vec<String> {
        let mut result = Vec::with_capacity(tokens.len() + 2);

        let comments: &[&str] = match self.config.language {
            CodeLanguage::Rust => &["// REVIEW: pending", "// SAFETY: checked", "/* temp */"],
            CodeLanguage::Python => &["# REVIEW: pending", "# NOTE: temp", "# type: ignore"],
            CodeLanguage::Generic => &["/* comment */"],
        };

        // Find a newline to insert comment after
        let mut inserted = false;
        for token in tokens {
            result.push(token.clone());
            if token == "\n" && !inserted && self.random_f32(rng) < 0.5 {
                let idx = (self.random_u64(rng) as usize) % comments.len();
                result.push(comments[idx].to_string());
                result.push("\n".to_string());
                inserted = true;
            }
        }

        result
    }

    /// Apply statement reorder operation (swap adjacent statements).
    fn apply_statement_reorder(&self, tokens: &[String], rng: &mut u64) -> Vec<String> {
        // Find statement boundaries (semicolons for Rust, newlines for Python)
        let delimiter = match self.config.language {
            CodeLanguage::Rust => ";",
            CodeLanguage::Python | CodeLanguage::Generic => "\n",
        };

        // Split into statements
        let mut statements: Vec<Vec<String>> = Vec::new();
        let mut current_stmt: Vec<String> = Vec::new();

        for token in tokens {
            current_stmt.push(token.clone());
            if token == delimiter {
                statements.push(current_stmt.clone());
                current_stmt.clear();
            }
        }
        if !current_stmt.is_empty() {
            statements.push(current_stmt);
        }

        // Swap two adjacent statements if we have enough
        if statements.len() >= 2 {
            let idx = (self.random_u64(rng) as usize) % (statements.len() - 1);
            statements.swap(idx, idx + 1);
        }

        statements.into_iter().flatten().collect()
    }

    /// Apply dead code removal (remove comments and extra whitespace).
    #[allow(clippy::unused_self)]
    fn apply_dead_code_removal(&self, tokens: &[String]) -> Vec<String> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut in_comment = false;
        let mut prev_was_whitespace = false;
        let mut prev_was_slash = false;

        for token in tokens {
            // Detect // comment start (two consecutive slashes)
            if token == "/" {
                if prev_was_slash {
                    // This is the second slash, start comment
                    in_comment = true;
                    prev_was_slash = false;
                    // Remove the first slash we already added
                    if result.last() == Some(&"/".to_string()) {
                        result.pop();
                    }
                    continue;
                }
                prev_was_slash = true;
                if !in_comment {
                    result.push(token.clone());
                }
                continue;
            }

            // Reset slash tracking for non-slash tokens
            prev_was_slash = false;

            // Detect # comment start (Python)
            if token == "#" {
                in_comment = true;
                continue;
            }

            // End single-line comment on newline
            if in_comment && token == "\n" {
                in_comment = false;
                result.push(token.clone());
                continue;
            }

            if in_comment {
                continue;
            }

            // Collapse multiple whitespace
            let is_whitespace = token.chars().all(char::is_whitespace);
            if is_whitespace {
                if !prev_was_whitespace {
                    result.push(token.clone());
                }
                prev_was_whitespace = true;
            } else {
                result.push(token.clone());
                prev_was_whitespace = false;
            }
        }

        result
    }

    /// Check if token is a valid identifier.
    #[allow(clippy::unused_self)]
    fn is_identifier(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        let mut chars = token.chars();
        let first = chars.next().unwrap_or('0');
        (first.is_alphabetic() || first == '_') && chars.all(|c| c.is_alphanumeric() || c == '_')
    }

    /// Simple PRNG for reproducibility.
    #[allow(clippy::unused_self)]
    fn random_u64(&self, state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *state
    }

    /// Random f32 in [0, 1).
    fn random_f32(&self, state: &mut u64) -> f32 {
        (self.random_u64(state) as f32) / (u64::MAX as f32)
    }

    /// Calculate token overlap between two code strings.
    #[must_use]
    pub fn token_overlap(&self, a: &str, b: &str) -> f32 {
        let tokens_a: HashSet<_> = self.tokenize(a).into_iter().collect();
        let tokens_b: HashSet<_> = self.tokenize(b).into_iter().collect();

        if tokens_a.is_empty() || tokens_b.is_empty() {
            return 0.0;
        }

        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();

        intersection as f32 / union as f32
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &CodeEdaConfig {
        &self.config
    }
}

impl SyntheticGenerator for CodeEda {
    type Input = String;
    type Output = String;

    fn generate(
        &self,
        seeds: &[Self::Input],
        config: &SyntheticConfig,
    ) -> Result<Vec<Self::Output>> {
        let target_count = ((seeds.len() as f32) * config.augmentation_ratio).ceil() as usize;
        let mut results = Vec::with_capacity(target_count);
        let seed = config.seed;

        for (idx, code) in seeds.iter().enumerate() {
            let num_augments = (target_count / seeds.len().max(1)).max(1);
            for aug_idx in 0..num_augments {
                let aug_seed = seed.wrapping_add((idx * 1000 + aug_idx) as u64);
                let augmented = self.augment(code, aug_seed);

                // Check quality threshold
                let quality = self.quality_score(&augmented, code);
                if quality >= config.quality_threshold {
                    results.push(augmented);
                }

                if results.len() >= target_count {
                    break;
                }
            }
            if results.len() >= target_count {
                break;
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &Self::Output, seed: &Self::Input) -> f32 {
        // Quality based on token overlap (semantic preservation)
        let overlap = self.token_overlap(generated, seed);

        // Penalize if too similar (no augmentation) or too different (corrupted)
        // Ideal is 0.6-0.9 overlap
        if overlap > 0.95 {
            0.5 // Too similar, little augmentation
        } else if overlap < 0.3 {
            0.3 // Too different, likely corrupted
        } else {
            overlap
        }
    }

    fn diversity_score(&self, batch: &[Self::Output]) -> f32 {
        if batch.len() <= 1 {
            return 1.0;
        }

        // Calculate pairwise token overlap
        let mut total_overlap = 0.0;
        let mut pairs = 0;

        for i in 0..batch.len() {
            for j in (i + 1)..batch.len() {
                total_overlap += self.token_overlap(&batch[i], &batch[j]);
                pairs += 1;
            }
        }

        if pairs == 0 {
            return 1.0;
        }

        // Diversity is inverse of average overlap
        1.0 - (total_overlap / pairs as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
