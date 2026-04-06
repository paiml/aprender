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

include!("eda_generator_impl.rs");
include!("eda_tests.rs");
