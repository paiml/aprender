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

include!("code_eda_impl.rs");
include!("code_eda_part_03.rs");
