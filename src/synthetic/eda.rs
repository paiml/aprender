//! Easy Data Augmentation (EDA) for text data.
//!
//! Implements Wei & Zou (2019) "EDA: Easy Data Augmentation Techniques
//! for Boosting Performance on Text Classification Tasks" (EMNLP).
//!
//! # Operations
//!
//! 1. **Synonym Replacement (SR)**: Replace words with synonyms
//! 2. **Random Insertion (RI)**: Insert random synonyms
//! 3. **Random Swap (RS)**: Swap word positions
//! 4. **Random Deletion (RD)**: Randomly remove words
//!
//! # Example
//!
//! ```
//! use aprender::synthetic::eda::{EdaGenerator, EdaConfig};
//!
//! let config = EdaConfig::default();
//! let generator = EdaGenerator::new(config);
//!
//! let input = "git commit -m fix bug";
//! let augmented = generator.augment(input, 42);
//! assert!(!augmented.is_empty());
//! ```

use super::{SyntheticConfig, SyntheticGenerator};
use crate::error::Result;
use std::collections::HashMap;

/// Configuration for EDA text augmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct EdaConfig {
    /// Probability of applying synonym replacement per word.
    pub synonym_prob: f32,
    /// Probability of random insertion.
    pub insert_prob: f32,
    /// Probability of random swap.
    pub swap_prob: f32,
    /// Probability of random deletion.
    pub delete_prob: f32,
    /// Number of augmented samples to generate per input.
    pub num_augments: usize,
    /// Minimum word count to apply augmentation.
    pub min_words: usize,
}

impl Default for EdaConfig {
    fn default() -> Self {
        Self {
            synonym_prob: 0.1,
            insert_prob: 0.1,
            swap_prob: 0.1,
            delete_prob: 0.1,
            num_augments: 4,
            min_words: 3,
        }
    }
}

impl EdaConfig {
    /// Create a new EDA configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set synonym replacement probability.
    #[must_use]
    pub fn with_synonym_prob(mut self, prob: f32) -> Self {
        self.synonym_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set random insertion probability.
    #[must_use]
    pub fn with_insert_prob(mut self, prob: f32) -> Self {
        self.insert_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set random swap probability.
    #[must_use]
    pub fn with_swap_prob(mut self, prob: f32) -> Self {
        self.swap_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set random deletion probability.
    #[must_use]
    pub fn with_delete_prob(mut self, prob: f32) -> Self {
        self.delete_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Set number of augmented samples per input.
    #[must_use]
    pub fn with_num_augments(mut self, n: usize) -> Self {
        self.num_augments = n.max(1);
        self
    }

    /// Set minimum words for augmentation.
    #[must_use]
    pub fn with_min_words(mut self, n: usize) -> Self {
        self.min_words = n;
        self
    }
}

/// Simple synonym dictionary for shell commands and common words.
#[derive(Debug, Clone)]
pub struct SynonymDict {
    synonyms: HashMap<String, Vec<String>>,
}

impl Default for SynonymDict {
    fn default() -> Self {
        let mut synonyms = HashMap::new();

        // Shell command synonyms (common alternatives)
        synonyms.insert(
            "ls".to_string(),
            vec!["dir".to_string(), "list".to_string()],
        );
        synonyms.insert(
            "rm".to_string(),
            vec!["delete".to_string(), "remove".to_string()],
        );
        synonyms.insert("cp".to_string(), vec!["copy".to_string()]);
        synonyms.insert(
            "mv".to_string(),
            vec!["move".to_string(), "rename".to_string()],
        );
        synonyms.insert(
            "cat".to_string(),
            vec!["type".to_string(), "show".to_string()],
        );
        synonyms.insert(
            "grep".to_string(),
            vec!["find".to_string(), "search".to_string()],
        );

        // Git subcommand alternatives (conceptually similar)
        synonyms.insert("checkout".to_string(), vec!["switch".to_string()]);
        synonyms.insert("fetch".to_string(), vec!["pull".to_string()]);

        // Common flags synonyms
        synonyms.insert("-a".to_string(), vec!["--all".to_string()]);
        synonyms.insert("-v".to_string(), vec!["--verbose".to_string()]);
        synonyms.insert(
            "-r".to_string(),
            vec!["--recursive".to_string(), "-R".to_string()],
        );
        synonyms.insert("-f".to_string(), vec!["--force".to_string()]);
        synonyms.insert("-n".to_string(), vec!["--dry-run".to_string()]);

        // Common words
        synonyms.insert(
            "file".to_string(),
            vec!["document".to_string(), "path".to_string()],
        );
        synonyms.insert(
            "directory".to_string(),
            vec!["folder".to_string(), "dir".to_string()],
        );
        synonyms.insert(
            "test".to_string(),
            vec!["check".to_string(), "verify".to_string()],
        );
        synonyms.insert(
            "build".to_string(),
            vec!["compile".to_string(), "make".to_string()],
        );
        synonyms.insert(
            "run".to_string(),
            vec!["execute".to_string(), "start".to_string()],
        );
        synonyms.insert(
            "fix".to_string(),
            vec!["repair".to_string(), "patch".to_string()],
        );
        synonyms.insert(
            "update".to_string(),
            vec!["upgrade".to_string(), "refresh".to_string()],
        );
        synonyms.insert(
            "add".to_string(),
            vec!["append".to_string(), "include".to_string()],
        );

        Self { synonyms }
    }
}

impl SynonymDict {
    /// Create a new synonym dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty synonym dictionary.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            synonyms: HashMap::new(),
        }
    }

    /// Add a synonym mapping.
    pub fn add(&mut self, word: &str, synonyms: &[&str]) {
        self.synonyms.insert(
            word.to_lowercase(),
            synonyms.iter().map(|s| (*s).to_string()).collect(),
        );
    }

    /// Get synonyms for a word.
    #[must_use]
    pub fn get(&self, word: &str) -> Option<&Vec<String>> {
        self.synonyms.get(&word.to_lowercase())
    }

    /// Get a random synonym for a word using a seed.
    #[must_use]
    pub fn random_synonym(&self, word: &str, seed: u64) -> Option<&str> {
        self.synonyms.get(&word.to_lowercase()).map(|syns| {
            let idx = (seed as usize) % syns.len();
            syns[idx].as_str()
        })
    }

    /// Check if word has synonyms.
    #[must_use]
    pub fn has_synonyms(&self, word: &str) -> bool {
        self.synonyms.contains_key(&word.to_lowercase())
    }

    /// Get all words in the dictionary.
    #[must_use]
    pub fn words(&self) -> Vec<&String> {
        self.synonyms.keys().collect()
    }

    /// Get count of words with synonyms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.synonyms.len()
    }

    /// Check if dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.synonyms.is_empty()
    }
}

/// Simple Linear Congruential Generator for deterministic randomness.
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next() as usize) % max
    }
}

/// EDA (Easy Data Augmentation) generator for text data.
///
/// Implements the four core EDA operations from Wei & Zou (2019).
#[derive(Debug, Clone)]
pub struct EdaGenerator {
    config: EdaConfig,
    synonyms: SynonymDict,
}

impl EdaGenerator {
    /// Create a new EDA generator with configuration.
    #[must_use]
    pub fn new(config: EdaConfig) -> Self {
        Self {
            config,
            synonyms: SynonymDict::default(),
        }
    }

    /// Create a new EDA generator with custom synonym dictionary.
    #[must_use]
    pub fn with_synonyms(config: EdaConfig, synonyms: SynonymDict) -> Self {
        Self { config, synonyms }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &EdaConfig {
        &self.config
    }

    /// Get the synonym dictionary.
    #[must_use]
    pub fn synonyms(&self) -> &SynonymDict {
        &self.synonyms
    }

    /// Augment a single text input.
    ///
    /// Returns multiple augmented versions based on `config.num_augments`.
    #[must_use]
    pub fn augment(&self, text: &str, seed: u64) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.len() < self.config.min_words {
            return vec![text.to_string()];
        }

        let mut results = Vec::with_capacity(self.config.num_augments);
        let mut rng = SimpleRng::new(seed);

        for _ in 0..self.config.num_augments {
            let mut augmented = words.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();

            // Apply operations with configured probabilities
            if rng.next_f32() < self.config.synonym_prob {
                augmented = self.synonym_replacement(&augmented, &mut rng);
            }
            if rng.next_f32() < self.config.insert_prob {
                augmented = self.random_insertion(&augmented, &mut rng);
            }
            if rng.next_f32() < self.config.swap_prob {
                augmented = self.random_swap(&augmented, &mut rng);
            }
            if rng.next_f32() < self.config.delete_prob {
                augmented = self.random_deletion(&augmented, &mut rng);
            }

            let result = augmented.join(" ");
            if !result.is_empty() && result != text {
                results.push(result);
            }
        }

        // Ensure we return at least the original if no augmentations applied
        if results.is_empty() {
            results.push(text.to_string());
        }

        results
    }

    /// Synonym replacement: Replace n words with synonyms.
    fn synonym_replacement(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        let mut result = words.to_vec();
        let n = (words.len() as f32 * self.config.synonym_prob).ceil() as usize;

        for _ in 0..n {
            if result.is_empty() {
                break;
            }
            let idx = rng.next_usize(result.len());
            if let Some(syn) = self.synonyms.random_synonym(&result[idx], rng.next()) {
                result[idx] = syn.to_string();
            }
        }

        result
    }

    /// Random insertion: Insert n random synonyms.
    fn random_insertion(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        let mut result = words.to_vec();
        let n = (words.len() as f32 * self.config.insert_prob).ceil() as usize;

        let dict_words = self.synonyms.words();
        if dict_words.is_empty() {
            return result;
        }

        for _ in 0..n {
            // Pick a random word from dictionary
            let word_idx = rng.next_usize(dict_words.len());
            let word = dict_words[word_idx];

            // Get a random synonym
            if let Some(syn) = self.synonyms.random_synonym(word, rng.next()) {
                // Insert at random position
                let pos = rng.next_usize(result.len() + 1);
                result.insert(pos, syn.to_string());
            }
        }

        result
    }

    /// Random swap: Swap n pairs of words.
    fn random_swap(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        if words.len() < 2 {
            return words.to_vec();
        }

        let mut result = words.to_vec();
        let n = (words.len() as f32 * self.config.swap_prob).ceil() as usize;

        for _ in 0..n.max(1) {
            let i = rng.next_usize(result.len());
            let j = rng.next_usize(result.len());
            if i != j {
                result.swap(i, j);
            }
        }

        result
    }

    /// Random deletion: Delete words with probability p.
    fn random_deletion(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        if words.len() <= 1 {
            return words.to_vec();
        }

        let result: Vec<String> = words
            .iter()
            .filter(|_| rng.next_f32() > self.config.delete_prob)
            .cloned()
            .collect();

        // Ensure at least one word remains
        if result.is_empty() {
            let idx = rng.next_usize(words.len());
            return vec![words[idx].clone()];
        }

        result
    }

    /// Calculate similarity between original and augmented text.
    #[must_use]
    pub fn similarity(&self, original: &str, augmented: &str) -> f32 {
        let orig_words: std::collections::HashSet<_> = original.split_whitespace().collect();
        let aug_words: std::collections::HashSet<_> = augmented.split_whitespace().collect();

        if orig_words.is_empty() && aug_words.is_empty() {
            return 1.0;
        }
        if orig_words.is_empty() || aug_words.is_empty() {
            return 0.0;
        }

        let intersection = orig_words.intersection(&aug_words).count();
        let union = orig_words.union(&aug_words).count();

        intersection as f32 / union as f32
    }
}

impl SyntheticGenerator for EdaGenerator {
    type Input = String;
    type Output = String;

    fn generate(&self, seeds: &[String], config: &SyntheticConfig) -> Result<Vec<String>> {
        let target = config.target_count(seeds.len());
        let mut results = Vec::with_capacity(target);

        for (seed_idx, seed_text) in seeds.iter().enumerate() {
            let augmented = self.augment(seed_text, seed_idx as u64);
            for aug in augmented {
                if self.quality_score(&aug, seed_text) >= config.quality_threshold {
                    results.push(aug);
                }
                if results.len() >= target {
                    return Ok(results);
                }
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &String, seed: &String) -> f32 {
        // Quality = similarity (not too different) + length preservation
        let similarity = self.similarity(seed, generated);
        let len_ratio = generated.len() as f32 / seed.len().max(1) as f32;
        let len_score = if (0.5..=2.0).contains(&len_ratio) {
            1.0
        } else {
            0.5
        };

        0.7 * similarity + 0.3 * len_score
    }

    fn diversity_score(&self, batch: &[String]) -> f32 {
        if batch.len() < 2 {
            return 1.0;
        }

        // Compute pairwise Jaccard distances
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..batch.len() {
            for j in (i + 1)..batch.len() {
                let sim = self.similarity(&batch[i], &batch[j]);
                total_dist += 1.0 - sim;
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            total_dist / count as f32
        }
    }
}

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
