//! Tokenization algorithms for text preprocessing.
//!
//! This module provides various tokenization strategies:
//! - Whitespace tokenization (splits on Unicode whitespace)
//! - Word tokenization (alphanumeric + punctuation handling)
//! - Character tokenization (splits into individual characters)
//! - Subword tokenization for LLMs:
//!   - BPE (Byte Pair Encoding) - GPT, `LLaMA`, Mistral
//!   - `WordPiece` - BERT, `DistilBERT`
//!   - Unigram/SentencePiece - T5, ALBERT, `XLNet`
//!
//! All tokenizers implement the `Tokenizer` trait and follow zero-unwrap safety.
//!
//! # Subword Tokenization
//!
//! Subword tokenizers learn a vocabulary from a corpus and split text into
//! subword units. This handles out-of-vocabulary words by decomposing them
//! into known subwords.
//!
//! ```ignore
//! use aprender::text::tokenize::BpeTokenizer;
//!
//! // Train BPE on corpus
//! let corpus = vec!["hello world", "hello there"];
//! let tokenizer = BpeTokenizer::train(&corpus, 100)?;
//!
//! // Encode text to token IDs
//! let ids = tokenizer.encode("hello")?;
//!
//! // Decode back to text
//! let text = tokenizer.decode(&ids)?;
//! ```
//!
//! # References
//!
//! - Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units
//! - Wu et al. (2016): Google's Neural Machine Translation System
//! - Kudo & Richardson (2018): `SentencePiece`: A simple and language independent subword tokenizer

use crate::text::Tokenizer;
use crate::AprenderError;
use std::collections::HashMap;

/// Whitespace tokenizer that splits text on Unicode whitespace characters.
///
/// This is the simplest tokenizer, splitting on any Unicode whitespace
/// (spaces, tabs, newlines, etc.). It preserves punctuation attached to words.
///
/// # Examples
///
/// ```
/// use aprender::text::{Tokenizer, tokenize::WhitespaceTokenizer};
///
/// let tokenizer = WhitespaceTokenizer::new();
///
/// // Basic tokenization
/// let tokens = tokenizer.tokenize("Hello, world!").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["Hello,", "world!"]);
///
/// // Handles multiple spaces
/// let tokens = tokenizer.tokenize("foo   bar").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["foo", "bar"]);
///
/// // Handles newlines and tabs
/// let tokens = tokenizer.tokenize("line1\nline2\ttab").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["line1", "line2", "tab"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::WhitespaceTokenizer;
    ///
    /// let tokenizer = WhitespaceTokenizer::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        // Split on whitespace and filter out empty strings
        let tokens: Vec<String> = text.split_whitespace().map(ToString::to_string).collect();

        Ok(tokens)
    }
}

/// Word tokenizer that splits on whitespace and separates punctuation.
///
/// This tokenizer is more sophisticated than whitespace splitting:
/// - Splits on Unicode whitespace
/// - Separates punctuation from words
/// - Preserves contractions (e.g., "don't" stays together)
///
/// # Examples
///
/// ```
/// use aprender::text::{Tokenizer, tokenize::WordTokenizer};
///
/// let tokenizer = WordTokenizer::new();
///
/// // Separates punctuation
/// let tokens = tokenizer.tokenize("Hello, world!").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["Hello", ",", "world", "!"]);
///
/// // Preserves contractions
/// let tokens = tokenizer.tokenize("I don't know.").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["I", "don't", "know", "."]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct WordTokenizer;

impl WordTokenizer {
    /// Create a new word tokenizer.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::WordTokenizer;
    ///
    /// let tokenizer = WordTokenizer::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Check if a character should be treated as a separator.
    ///
    /// Separators include most punctuation except apostrophes (for contractions).
    fn is_separator(c: char) -> bool {
        c.is_ascii_punctuation() && c != '\''
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                // End current token on whitespace
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            } else if Self::is_separator(ch) {
                // Push current word, then push punctuation as separate token
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(ch.to_string());
            } else {
                // Accumulate alphanumeric and apostrophes
                current.push(ch);
            }
        }

        // Don't forget the last token
        if !current.is_empty() {
            tokens.push(current);
        }

        Ok(tokens)
    }
}

/// Character tokenizer that splits text into individual characters.
///
/// This tokenizer is useful for character-level NLP models.
/// It preserves all characters including whitespace and punctuation.
///
/// # Examples
///
/// ```
/// use aprender::text::{Tokenizer, tokenize::CharTokenizer};
///
/// let tokenizer = CharTokenizer::new();
///
/// let tokens = tokenizer.tokenize("Hi!").expect("tokenize should succeed");
/// assert_eq!(tokens, vec!["H", "i", "!"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CharTokenizer;

impl CharTokenizer {
    /// Create a new character tokenizer.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::CharTokenizer;
    ///
    /// let tokenizer = CharTokenizer::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for CharTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        let tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        Ok(tokens)
    }
}

/// Sentence tokenizer that splits text into sentences.
///
/// Uses punctuation-based rules to detect sentence boundaries,
/// handling common abbreviations and edge cases.
///
/// # Examples
///
/// ```
/// use aprender::text::tokenize::SentenceTokenizer;
///
/// let tokenizer = SentenceTokenizer::new();
///
/// let sentences = tokenizer.split("Hello world. How are you? I'm fine!");
/// assert_eq!(sentences, vec!["Hello world.", "How are you?", "I'm fine!"]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SentenceTokenizer {
    /// Common abbreviations that don't end sentences
    abbreviations: Vec<&'static str>,
}

impl SentenceTokenizer {
    /// Create a new sentence tokenizer with default abbreviations.
    #[must_use]
    pub fn new() -> Self {
        Self {
            abbreviations: vec![
                "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd", "corp",
                "st", "ave", "blvd", "rd", "dept", "gov", "gen", "col", "lt", "sgt", "rev", "hon",
                "pres", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov",
                "dec", "i.e", "e.g", "cf", "al", "vol", "no", "fig", "pp", "ph.d", "m.d", "b.a",
                "m.a", "d.d.s",
            ],
        }
    }

    /// Split text into sentences.
    #[must_use]
    pub fn split(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut sentences = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();

        let mut i = 0;
        while i < len {
            let c = chars[i];
            current.push(c);

            // Check for sentence-ending punctuation
            if c == '.' || c == '?' || c == '!' {
                // Look ahead to see if this is really a sentence end
                let is_end = if i + 1 < len {
                    let next = chars[i + 1];
                    // End if followed by space + uppercase, or end of text
                    if next.is_whitespace() {
                        // Check if followed by uppercase
                        let mut j = i + 2;
                        while j < len && chars[j].is_whitespace() {
                            j += 1;
                        }
                        j >= len || chars[j].is_uppercase()
                    } else {
                        false
                    }
                } else {
                    true // End of text
                };

                // Check for abbreviation (for periods only)
                let is_abbrev = if c == '.' {
                    self.is_abbreviation(&current)
                } else {
                    false
                };

                if is_end && !is_abbrev {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    current.clear();
                }
            }
            i += 1;
        }

        // Add remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    fn is_abbreviation(&self, text: &str) -> bool {
        // Extract the last word before the period
        let text = text.trim_end_matches('.');
        let last_word = text.split_whitespace().last().unwrap_or("");
        let lower = last_word.to_lowercase();
        self.abbreviations.contains(&lower.as_str())
    }
}

// ============================================================================
// SUBWORD TOKENIZERS
// ============================================================================

/// Special tokens used by subword tokenizers.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Unknown token for OOV words
    pub unk: String,
    /// Beginning of sequence token
    pub bos: Option<String>,
    /// End of sequence token
    pub eos: Option<String>,
    /// Padding token
    pub pad: Option<String>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            unk: "<unk>".to_string(),
            bos: Some("<s>".to_string()),
            eos: Some("</s>".to_string()),
            pad: Some("<pad>".to_string()),
        }
    }
}

/// Byte Pair Encoding (BPE) tokenizer.
///
/// BPE iteratively merges the most frequent pair of adjacent tokens,
/// building a subword vocabulary that handles rare words by decomposition.
///
/// Used by GPT, GPT-2, `RoBERTa`, `LLaMA`, Mistral, and many other LLMs.
///
/// # Algorithm
///
/// 1. Initialize vocabulary with all characters (+ special tokens)
/// 2. Count frequency of all adjacent token pairs
/// 3. Merge the most frequent pair into a new token
/// 4. Repeat until vocabulary size is reached
///
/// # Examples
///
/// ```
/// use aprender::text::tokenize::BpeTokenizer;
///
/// // Train on a small corpus
/// let corpus = vec!["low", "lower", "newest", "widest"];
/// let tokenizer = BpeTokenizer::train(&corpus, 50).expect("training should succeed");
///
/// // Encode text to token IDs
/// let ids = tokenizer.encode("low").expect("encode should succeed");
/// assert!(!ids.is_empty());
///
/// // Decode back to text
/// let text = tokenizer.decode(&ids).expect("decode should succeed");
/// assert_eq!(text, "low");
/// ```
///
/// # References
///
/// - Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units
/// - Gage (1994): A New Algorithm for Data Compression
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping (inverse of vocab)
    inverse_vocab: HashMap<u32, String>,
    /// Ordered list of merge rules (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Special tokens configuration
    special_tokens: SpecialTokens,
    /// End-of-word marker (used during encoding)
    end_of_word: String,
}

impl BpeTokenizer {
    /// Train a BPE tokenizer on the given corpus.
    ///
    /// # Arguments
    ///
    /// * `corpus` - Slice of text documents to train on
    /// * `vocab_size` - Target vocabulary size (including special tokens)
    ///
    /// # Returns
    ///
    /// * `Ok(BpeTokenizer)` - Trained tokenizer
    /// * `Err(AprenderError)` - If `vocab_size` is too small
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let corpus = vec!["hello world", "hello there", "world wide web"];
    /// let tokenizer = BpeTokenizer::train(&corpus, 100).expect("training should succeed");
    ///
    /// assert!(tokenizer.vocab_size() >= 26); // At least all letters
    /// ```
    pub fn train(corpus: &[&str], vocab_size: usize) -> Result<Self, AprenderError> {
        Self::train_with_special_tokens(corpus, vocab_size, SpecialTokens::default())
    }

    /// Train BPE with custom special tokens.
    ///
    /// # Arguments
    ///
    /// * `corpus` - Slice of text documents to train on
    /// * `vocab_size` - Target vocabulary size (including special tokens)
    /// * `special_tokens` - Custom special tokens configuration
    pub fn train_with_special_tokens(
        corpus: &[&str],
        vocab_size: usize,
        special_tokens: SpecialTokens,
    ) -> Result<Self, AprenderError> {
        if vocab_size < 10 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "vocab_size".to_string(),
                value: vocab_size.to_string(),
                constraint: "must be at least 10".to_string(),
            });
        }

        let end_of_word = "</w>".to_string();

        // Initialize vocab with special tokens
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut next_id: u32 = 0;

        // Add special tokens
        vocab.insert(special_tokens.unk.clone(), next_id);
        next_id += 1;

        if let Some(ref bos) = special_tokens.bos {
            vocab.insert(bos.clone(), next_id);
            next_id += 1;
        }
        if let Some(ref eos) = special_tokens.eos {
            vocab.insert(eos.clone(), next_id);
            next_id += 1;
        }
        if let Some(ref pad) = special_tokens.pad {
            vocab.insert(pad.clone(), next_id);
            next_id += 1;
        }

        // Count character frequencies and build initial word splits
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for doc in corpus {
            for word in doc.split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Convert words to character sequences with end-of-word marker
        // word_splits: word -> (frequency, character sequence)
        let mut word_splits: HashMap<String, (usize, Vec<String>)> = HashMap::new();
        for (word, freq) in &word_freqs {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if !chars.is_empty() {
                // Add end-of-word marker to last character
                if let Some(last) = chars.last_mut() {
                    last.push_str(&end_of_word);
                }
            }
            word_splits.insert(word.clone(), (*freq, chars));
        }

        // Add all characters to vocab
        for (_, splits) in word_splits.values() {
            for token in splits {
                if !vocab.contains_key(token) {
                    vocab.insert(token.clone(), next_id);
                    next_id += 1;
                }
            }
        }

        // Iteratively merge most frequent pairs
        let mut merges: Vec<(String, String)> = Vec::new();

        while vocab.len() < vocab_size {
            // Count pair frequencies
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
            for (freq, splits) in word_splits.values() {
                if splits.len() < 2 {
                    continue;
                }
                for window in splits.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .map(|(pair, _)| pair.clone());

            let Some((left, right)) = best_pair else {
                break; // No more pairs to merge
            };

            // Create merged token
            let merged = format!("{left}{right}");

            // Add to vocab
            if !vocab.contains_key(&merged) {
                vocab.insert(merged.clone(), next_id);
                next_id += 1;
            }

            // Record merge rule
            merges.push((left.clone(), right.clone()));

            // Apply merge to all word splits
            for (_, splits) in word_splits.values_mut() {
                let mut i = 0;
                while i < splits.len().saturating_sub(1) {
                    if splits[i] == left && splits[i + 1] == right {
                        merged.clone_into(&mut splits[i]);
                        splits.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Build inverse vocab
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Ok(Self {
            vocab,
            inverse_vocab,
            merges,
            special_tokens,
            end_of_word,
        })
    }

    /// Create a BPE tokenizer from pre-built vocabulary and merges.
    ///
    /// # Arguments
    ///
    /// * `vocab` - Token to ID mapping
    /// * `merges` - Ordered list of merge rules
    #[must_use]
    pub fn from_vocab(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            inverse_vocab,
            merges,
            special_tokens: SpecialTokens::default(),
            end_of_word: "</w>".to_string(),
        }
    }

    /// Load a BPE tokenizer from `HuggingFace` vocab.json and merges.txt files.
    ///
    /// This is the standard format used by GPT-2, Whisper, and many other models.
    ///
    /// # Arguments
    ///
    /// * `vocab_json` - JSON content of vocab.json (token -> id mapping)
    /// * `merges_txt` - Content of merges.txt (one merge per line)
    ///
    /// # Returns
    ///
    /// * `Ok(BpeTokenizer)` - Loaded tokenizer
    /// * `Err(AprenderError)` - If parsing fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let vocab_json = r#"{"hello": 0, "world": 1, "<|endoftext|>": 2}"#;
    /// let merges_txt = "h e\nhe l\nhel lo";
    ///
    /// let tokenizer = BpeTokenizer::from_huggingface(vocab_json, merges_txt)
    ///     .expect("loading should succeed");
    /// assert_eq!(tokenizer.vocab_size(), 3);
    /// ```
    ///
    /// # Format Details
    ///
    /// The vocab.json file is a JSON object mapping tokens to IDs:
    /// ```json
    /// {"hello": 0, "world": 1, "<|endoftext|>": 50256}
    /// ```
    ///
    /// The merges.txt file contains one merge rule per line:
    /// ```text
    /// #version: 0.2
    /// h e
    /// he l
    /// hel lo
    /// ```
    ///
    /// # References
    ///
    /// - Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units
    /// - `HuggingFace` Tokenizers: <https://huggingface.co/docs/tokenizers>
    pub fn from_huggingface(vocab_json: &str, merges_txt: &str) -> Result<Self, AprenderError> {
        // Parse vocab.json - simple JSON parsing without external dependency
        let vocab = Self::parse_vocab_json(vocab_json)?;

        // Parse merges.txt
        let merges = Self::parse_merges_txt(merges_txt);

        // Detect end-of-word marker from vocab (GPT-2 uses "Ġ", others use "</w>")
        let end_of_word = if vocab.keys().any(|k| k.contains("Ġ")) {
            "Ġ".to_string()
        } else {
            "</w>".to_string()
        };

        // Detect special tokens from vocab
        let unk = vocab
            .keys()
            .find(|k| k.contains("unk") || k.contains("UNK"))
            .cloned()
            .unwrap_or_else(|| "<unk>".to_string());

        let eos = vocab
            .keys()
            .find(|k| k.contains("endoftext") || k.contains("</s>") || k.contains("eos"))
            .cloned();

        let bos = vocab
            .keys()
            .find(|k| k.contains("startoftext") || k.contains("<s>") || k.contains("bos"))
            .cloned();

        let pad = vocab
            .keys()
            .find(|k| k.contains("pad") || k.contains("PAD"))
            .cloned();

        let special_tokens = SpecialTokens { unk, bos, eos, pad };

        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Ok(Self {
            vocab,
            inverse_vocab,
            merges,
            special_tokens,
            end_of_word,
        })
    }

    /// Parse vocab.json content into a `HashMap`.
    ///
    /// Simple JSON parsing without external dependencies. Handles basic JSON format:
    /// `{"token1": 0, "token2": 1, ...}`
    fn parse_vocab_json(json: &str) -> Result<HashMap<String, u32>, AprenderError> {
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(AprenderError::Serialization(
                "Invalid vocab.json: must be a JSON object".to_string(),
            ));
        }

        let mut vocab = HashMap::new();
        let content = &json[1..json.len() - 1]; // Remove { and }

        if content.trim().is_empty() {
            return Ok(vocab);
        }

        // Parse key-value pairs
        let chars = content.chars();
        let mut in_string = false;
        let mut escape_next = false;
        let mut current_key = String::new();
        let mut current_value = String::new();
        let mut parsing_key = true;

        for c in chars {
            if escape_next {
                if parsing_key {
                    current_key.push(c);
                } else {
                    current_value.push(c);
                }
                escape_next = false;
                continue;
            }

            match c {
                '\\' => {
                    escape_next = true;
                }
                '"' => {
                    in_string = !in_string;
                }
                ':' if !in_string => {
                    parsing_key = false;
                }
                ',' if !in_string => {
                    // End of pair
                    let key = current_key.trim().to_string();
                    let value: u32 = current_value.trim().parse().map_err(|_| {
                        AprenderError::Serialization(format!(
                            "Invalid token ID for '{key}': '{current_value}'"
                        ))
                    })?;

                    if !key.is_empty() {
                        vocab.insert(key, value);
                    }

                    current_key.clear();
                    current_value.clear();
                    parsing_key = true;
                }
                _ if in_string => {
                    if parsing_key {
                        current_key.push(c);
                    } else {
                        current_value.push(c);
                    }
                }
                _ if !in_string && !c.is_whitespace() => {
                    if !parsing_key {
                        current_value.push(c);
                    }
                }
                _ => {}
            }
        }

        // Handle last pair
        let key = current_key.trim().to_string();
        if !key.is_empty() && !current_value.is_empty() {
            let value: u32 = current_value.trim().parse().map_err(|_| {
                AprenderError::Serialization(format!(
                    "Invalid token ID for '{key}': '{current_value}'"
                ))
            })?;
            vocab.insert(key, value);
        }

        Ok(vocab)
    }

    /// Parse merges.txt content into a list of merge rules.
    ///
    /// The format is one merge per line: "token1 token2"
    /// Lines starting with # are treated as comments.
    fn parse_merges_txt(content: &str) -> Vec<(String, String)> {
        let mut merges = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Split on first space
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() != 2 {
                continue; // Skip malformed lines
            }

            let left = parts[0].to_string();
            let right = parts[1].to_string();

            if !left.is_empty() && !right.is_empty() {
                merges.push((left, right));
            }
        }

        merges
    }

    /// Check if a token is a special token (UNK, BOS, EOS, PAD).
    #[must_use]
    pub fn is_special_token(&self, token: &str) -> bool {
        token == self.special_tokens.unk
            || self.special_tokens.bos.as_deref() == Some(token)
            || self.special_tokens.eos.as_deref() == Some(token)
            || self.special_tokens.pad.as_deref() == Some(token)
    }

    /// Get the EOS token if configured.
    #[must_use]
    pub fn eos_token(&self) -> Option<&str> {
        self.special_tokens.eos.as_deref()
    }

    /// Get the BOS token if configured.
    #[must_use]
    pub fn bos_token(&self) -> Option<&str> {
        self.special_tokens.bos.as_deref()
    }

    /// Get the UNK token.
    #[must_use]
    pub fn unk_token(&self) -> &str {
        &self.special_tokens.unk
    }

    /// Get the end-of-word marker used by this tokenizer.
    ///
    /// Returns `</w>` for standard BPE or "Ġ" for GPT-2 style tokenizers.
    #[must_use]
    pub fn end_of_word_marker(&self) -> &str {
        &self.end_of_word
    }

    /// Encode text into token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u32>)` - Token IDs
    /// * `Err(AprenderError)` - If encoding fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let corpus = vec!["hello", "world"];
    /// let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");
    ///
    /// let ids = tokenizer.encode("hello").expect("encode");
    /// assert!(!ids.is_empty());
    /// ```
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, AprenderError> {
        let mut token_ids = Vec::new();

        for word in text.split_whitespace() {
            // Convert word to character sequence with end-of-word marker
            let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if !tokens.is_empty() {
                if let Some(last) = tokens.last_mut() {
                    last.push_str(&self.end_of_word);
                }
            }

            // Apply merges in order
            for (left, right) in &self.merges {
                let merged = format!("{left}{right}");
                let mut i = 0;
                while i < tokens.len().saturating_sub(1) {
                    if &tokens[i] == left && &tokens[i + 1] == right {
                        merged.clone_into(&mut tokens[i]);
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            // Convert tokens to IDs
            let unk_id = self
                .vocab
                .get(&self.special_tokens.unk)
                .copied()
                .unwrap_or(0);

            for token in tokens {
                let id = self.vocab.get(&token).copied().unwrap_or(unk_id);
                token_ids.push(id);
            }
        }

        Ok(token_ids)
    }

    /// Encode text and add special tokens (BOS/EOS).
    ///
    /// # Arguments
    ///
    /// * `text` - Text to encode
    /// * `add_bos` - Add beginning-of-sequence token
    /// * `add_eos` - Add end-of-sequence token
    pub fn encode_with_special(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<u32>, AprenderError> {
        let mut ids = Vec::new();

        if add_bos {
            if let Some(ref bos) = self.special_tokens.bos {
                if let Some(&id) = self.vocab.get(bos) {
                    ids.push(id);
                }
            }
        }

        ids.extend(self.encode(text)?);

        if add_eos {
            if let Some(ref eos) = self.special_tokens.eos {
                if let Some(&id) = self.vocab.get(eos) {
                    ids.push(id);
                }
            }
        }

        Ok(ids)
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    ///
    /// * `ids` - Token IDs to decode
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Decoded text
    /// * `Err(AprenderError)` - If decoding fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::BpeTokenizer;
    ///
    /// let corpus = vec!["hello"];
    /// let tokenizer = BpeTokenizer::train(&corpus, 50).expect("train");
    ///
    /// let ids = tokenizer.encode("hello").expect("encode");
    /// let text = tokenizer.decode(&ids).expect("decode");
    /// assert_eq!(text, "hello");
    /// ```
    pub fn decode(&self, ids: &[u32]) -> Result<String, AprenderError> {
        let mut result = String::new();
        let mut need_space = false;

        for &id in ids {
            // Skip special tokens in output
            if let Some(ref bos) = self.special_tokens.bos {
                if self.vocab.get(bos) == Some(&id) {
                    continue;
                }
            }
            if let Some(ref eos) = self.special_tokens.eos {
                if self.vocab.get(eos) == Some(&id) {
                    continue;
                }
            }
            if let Some(ref pad) = self.special_tokens.pad {
                if self.vocab.get(pad) == Some(&id) {
                    continue;
                }
            }

            let token = self
                .inverse_vocab
                .get(&id)
                .map_or_else(|| self.special_tokens.unk.clone(), Clone::clone);

            // Handle end-of-word marker
            if token.ends_with(&self.end_of_word) {
                if need_space {
                    result.push(' ');
                }
                let cleaned = token.trim_end_matches(self.end_of_word.as_str());
                result.push_str(cleaned);
                need_space = true;
            } else {
                if need_space && result.ends_with(' ') {
                    // Already has space
                } else if need_space {
                    // Continue building word
                }
                result.push_str(&token);
            }
        }

        Ok(result)
    }

    /// Get the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get a reference to the vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Get the merge rules.
    #[must_use]
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Check if a token exists in the vocabulary.
    #[must_use]
    pub fn contains(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Get the ID for a token.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get the token for an ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.inverse_vocab.get(&id).map(String::as_str)
    }
}

impl Tokenizer for BpeTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        // Return string tokens instead of IDs for Tokenizer trait
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let mut word_tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if !word_tokens.is_empty() {
                if let Some(last) = word_tokens.last_mut() {
                    last.push_str(&self.end_of_word);
                }
            }

            // Apply merges
            for (left, right) in &self.merges {
                let merged = format!("{left}{right}");
                let mut i = 0;
                while i < word_tokens.len().saturating_sub(1) {
                    if &word_tokens[i] == left && &word_tokens[i + 1] == right {
                        merged.clone_into(&mut word_tokens[i]);
                        word_tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            tokens.extend(word_tokens);
        }

        Ok(tokens)
    }
}

/// `WordPiece` tokenizer (used by BERT).
///
/// `WordPiece` is similar to BPE but uses a different scoring criterion:
/// it maximizes the likelihood of the training data rather than frequency.
/// Subwords (except the first) are prefixed with "##".
///
/// # Algorithm
///
/// 1. Initialize vocabulary with all characters
/// 2. Score pairs by: freq(ab) / (freq(a) * freq(b))
/// 3. Merge pair with highest score
/// 4. Repeat until vocabulary size reached
///
/// # Examples
///
/// ```
/// use aprender::text::tokenize::WordPieceTokenizer;
///
/// let corpus = vec!["playing", "played", "player", "plays"];
/// let tokenizer = WordPieceTokenizer::train(&corpus, 50).expect("train");
///
/// let tokens = tokenizer.encode("playing").expect("encode");
/// assert!(!tokens.is_empty());
/// ```
///
/// # References
///
/// - Wu et al. (2016): Google's Neural Machine Translation System
/// - Devlin et al. (2019): BERT: Pre-training of Deep Bidirectional Transformers
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    inverse_vocab: HashMap<u32, String>,
    /// Continuation prefix (default: "##")
    continuation_prefix: String,
    /// Unknown token
    unk_token: String,
    /// Maximum word length before splitting to unk
    max_word_len: usize,
}

impl WordPieceTokenizer {
    /// Train a `WordPiece` tokenizer on the given corpus.
    ///
    /// # Arguments
    ///
    /// * `corpus` - Slice of text documents to train on
    /// * `vocab_size` - Target vocabulary size
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::tokenize::WordPieceTokenizer;
    ///
    /// let corpus = vec!["unbelievable", "believable", "believe"];
    /// let tokenizer = WordPieceTokenizer::train(&corpus, 100).expect("train");
    /// ```
    #[allow(clippy::too_many_lines)]
    pub fn train(corpus: &[&str], vocab_size: usize) -> Result<Self, AprenderError> {
        if vocab_size < 10 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "vocab_size".to_string(),
                value: vocab_size.to_string(),
                constraint: "must be at least 10".to_string(),
            });
        }

        let continuation_prefix = "##".to_string();
        let unk_token = "[UNK]".to_string();

        // Initialize vocab with special tokens
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut next_id: u32 = 0;

        vocab.insert(unk_token.clone(), next_id);
        next_id += 1;
        vocab.insert("[PAD]".to_string(), next_id);
        next_id += 1;
        vocab.insert("[CLS]".to_string(), next_id);
        next_id += 1;
        vocab.insert("[SEP]".to_string(), next_id);
        next_id += 1;
        vocab.insert("[MASK]".to_string(), next_id);
        next_id += 1;

        // Count word frequencies
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for doc in corpus {
            for word in doc.split_whitespace() {
                *word_freqs.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }

        // Initialize with characters (first char as-is, rest with ##)
        let mut word_splits: HashMap<String, (usize, Vec<String>)> = HashMap::new();
        for (word, freq) in &word_freqs {
            let chars: Vec<char> = word.chars().collect();
            if chars.is_empty() {
                continue;
            }

            let mut tokens = vec![chars[0].to_string()];
            for c in chars.iter().skip(1) {
                tokens.push(format!("{continuation_prefix}{c}"));
            }

            // Add all tokens to vocab
            for token in &tokens {
                if !vocab.contains_key(token) {
                    vocab.insert(token.clone(), next_id);
                    next_id += 1;
                }
            }

            word_splits.insert(word.clone(), (*freq, tokens));
        }

        // Iteratively merge using WordPiece scoring
        while vocab.len() < vocab_size {
            // Count pair frequencies and individual frequencies
            let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
            let mut token_freqs: HashMap<String, usize> = HashMap::new();

            for (freq, splits) in word_splits.values() {
                for token in splits {
                    *token_freqs.entry(token.clone()).or_insert(0) += freq;
                }
                if splits.len() < 2 {
                    continue;
                }
                for window in splits.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            // Score pairs: freq(ab) / (freq(a) * freq(b))
            let best_pair = pair_freqs
                .iter()
                .map(|((a, b), &freq)| {
                    let freq_a = token_freqs.get(a).copied().unwrap_or(1);
                    let freq_b = token_freqs.get(b).copied().unwrap_or(1);
                    let score = freq as f64 / (freq_a as f64 * freq_b as f64);
                    ((a.clone(), b.clone()), score)
                })
                .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(pair, _)| pair);

            let Some((left, right)) = best_pair else {
                break;
            };

            // Merge: combine tokens (remove ## prefix from right if present)
            let right_suffix = &right[continuation_prefix.len()..];
            let merged = if right.starts_with(&continuation_prefix) {
                format!("{left}{right_suffix}")
            } else {
                format!("{left}{right}")
            };

            // Add merged token
            if !vocab.contains_key(&merged) {
                vocab.insert(merged.clone(), next_id);
                next_id += 1;
            }

            // Apply merge to all word splits
            for (_, splits) in word_splits.values_mut() {
                let mut i = 0;
                while i < splits.len().saturating_sub(1) {
                    if splits[i] == left && splits[i + 1] == right {
                        merged.clone_into(&mut splits[i]);
                        splits.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Ok(Self {
            vocab,
            inverse_vocab,
            continuation_prefix,
            unk_token,
            max_word_len: 100,
        })
    }

    /// Create from pre-built vocabulary.
    #[must_use]
    pub fn from_vocab(vocab: HashMap<String, u32>) -> Self {
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            inverse_vocab,
            continuation_prefix: "##".to_string(),
            unk_token: "[UNK]".to_string(),
            max_word_len: 100,
        }
    }

    /// Encode text to token IDs using greedy longest-match-first.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, AprenderError> {
        let mut ids = Vec::new();
        let unk_id = self.vocab.get(&self.unk_token).copied().unwrap_or(0);

        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            if word.len() > self.max_word_len {
                ids.push(unk_id);
                continue;
            }

            let mut word_ids = Vec::new();
            let mut start = 0;
            let chars: Vec<char> = word.chars().collect();

            while start < chars.len() {
                let mut end = chars.len();
                let mut found = false;

                while start < end {
                    let substr: String = chars[start..end].iter().collect();
                    let token = if start == 0 {
                        substr.clone()
                    } else {
                        {
                            let prefix = &self.continuation_prefix;
                            format!("{prefix}{substr}")
                        }
                    };

                    if let Some(&id) = self.vocab.get(&token) {
                        word_ids.push(id);
                        start = end;
                        found = true;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    // Character not in vocab, use UNK
                    word_ids.clear();
                    word_ids.push(unk_id);
                    break;
                }
            }

            ids.extend(word_ids);
        }

        Ok(ids)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, AprenderError> {
        let mut result = String::new();
        let mut need_space = true;

        for &id in ids {
            let token = self.inverse_vocab.get(&id).map_or(&self.unk_token, |t| t);

            // Skip special tokens
            if token.starts_with('[') && token.ends_with(']') {
                continue;
            }

            if token.starts_with(&self.continuation_prefix) {
                // Continuation token - no space, strip prefix
                result.push_str(&token[self.continuation_prefix.len()..]);
            } else {
                if !result.is_empty() && need_space {
                    result.push(' ');
                }
                result.push_str(token);
            }
            need_space = !token.starts_with(&self.continuation_prefix);
        }

        Ok(result)
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get reference to vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            if word.len() > self.max_word_len {
                tokens.push(self.unk_token.clone());
                continue;
            }

            let chars: Vec<char> = word.chars().collect();
            let mut start = 0;
            let mut word_tokens = Vec::new();

            while start < chars.len() {
                let mut end = chars.len();
                let mut found = false;

                while start < end {
                    let substr: String = chars[start..end].iter().collect();
                    let token = if start == 0 {
                        substr.clone()
                    } else {
                        {
                            let prefix = &self.continuation_prefix;
                            format!("{prefix}{substr}")
                        }
                    };

                    if self.vocab.contains_key(&token) {
                        word_tokens.push(token);
                        start = end;
                        found = true;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    word_tokens.clear();
                    word_tokens.push(self.unk_token.clone());
                    break;
                }
            }

            tokens.extend(word_tokens);
        }

        Ok(tokens)
    }
}

/// Unigram tokenizer (`SentencePiece`).
///
/// Unigram uses a probabilistic model where each token has a probability,
/// and the tokenization is chosen to maximize the total probability.
/// Training removes tokens that least affect the total loss.
///
/// # Algorithm
///
/// 1. Initialize with a large vocabulary (all substrings up to `max_len`)
/// 2. Compute loss = -sum(log P(token)) for each token
/// 3. Remove tokens that increase loss the least
/// 4. Repeat until target vocabulary size
///
/// # Examples
///
/// ```
/// use aprender::text::tokenize::UnigramTokenizer;
///
/// let corpus = vec!["hello world", "hello there"];
/// let tokenizer = UnigramTokenizer::train(&corpus, 100).expect("train");
///
/// let tokens = tokenizer.encode("hello").expect("encode");
/// assert!(!tokens.is_empty());
/// ```
///
/// # References
///
/// - Kudo (2018): Subword Regularization: Improving Neural Network Translation Models
/// - Kudo & Richardson (2018): `SentencePiece`
#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    /// Token to (ID, log probability) mapping
    vocab: HashMap<String, (u32, f64)>,
    /// ID to token mapping
    inverse_vocab: HashMap<u32, String>,
    /// Unknown token
    unk_token: String,
    /// BOS token
    bos_token: String,
    /// EOS token
    eos_token: String,
}

impl UnigramTokenizer {
    /// Train a Unigram tokenizer on the given corpus.
    ///
    /// # Arguments
    ///
    /// * `corpus` - Slice of text documents to train on
    /// * `vocab_size` - Target vocabulary size
    pub fn train(corpus: &[&str], vocab_size: usize) -> Result<Self, AprenderError> {
        if vocab_size < 10 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "vocab_size".to_string(),
                value: vocab_size.to_string(),
                constraint: "must be at least 10".to_string(),
            });
        }

        let unk_token = "<unk>".to_string();
        let bos_token = "<s>".to_string();
        let eos_token = "</s>".to_string();

        // Initialize vocab with special tokens and high counts
        let mut token_counts: HashMap<String, usize> = HashMap::new();
        token_counts.insert(unk_token.clone(), 1_000_000);
        token_counts.insert(bos_token.clone(), 1_000_000);
        token_counts.insert(eos_token.clone(), 1_000_000);

        // Add word marker
        let word_boundary = "▁".to_string(); // SentencePiece word boundary
        token_counts.insert(word_boundary.clone(), 1_000_000);

        // Count all character n-grams up to length 16
        let max_ngram = 16;
        for doc in corpus {
            // Replace spaces with word boundary
            let mut processed = String::new();
            for w in doc.split_whitespace() {
                processed.push_str(&word_boundary);
                processed.push_str(w);
            }

            let chars: Vec<char> = processed.chars().collect();
            for start in 0..chars.len() {
                for end in start + 1..=std::cmp::min(start + max_ngram, chars.len()) {
                    let ngram: String = chars[start..end].iter().collect();
                    *token_counts.entry(ngram).or_insert(0) += 1;
                }
            }
        }

        // Calculate initial vocabulary size
        let mut vocab_items: Vec<(String, usize)> = token_counts.into_iter().collect();

        // Sort by frequency (descending), keep top tokens
        vocab_items.sort_by(|a, b| b.1.cmp(&a.1));

        // Prune to target size (keep special tokens + most frequent)
        if vocab_items.len() > vocab_size {
            vocab_items.truncate(vocab_size);
        }

        // Calculate log probabilities
        let total: f64 = vocab_items.iter().map(|(_, c)| *c as f64).sum();
        let mut vocab: HashMap<String, (u32, f64)> = HashMap::new();
        let mut inverse_vocab: HashMap<u32, String> = HashMap::new();

        for (id, (token, count)) in vocab_items.iter().enumerate() {
            let log_prob = ((*count as f64) / total).ln();
            vocab.insert(token.clone(), (id as u32, log_prob));
            inverse_vocab.insert(id as u32, token.clone());
        }

        // Ensure special tokens are present
        let num_tokens = vocab.len() as u32;
        if !vocab.contains_key(&unk_token) {
            vocab.insert(unk_token.clone(), (num_tokens, -10.0));
            inverse_vocab.insert(num_tokens, unk_token.clone());
        }

        Ok(Self {
            vocab,
            inverse_vocab,
            unk_token,
            bos_token,
            eos_token,
        })
    }

    /// Create from pre-built vocabulary with probabilities.
    #[must_use]
    pub fn from_vocab(vocab: HashMap<String, (u32, f64)>) -> Self {
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, (id, _))| (*id, k.clone())).collect();
        Self {
            vocab,
            inverse_vocab,
            unk_token: "<unk>".to_string(),
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
        }
    }

    /// Encode text using Viterbi algorithm for optimal segmentation.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, AprenderError> {
        let word_boundary = "▁";
        let mut processed = String::new();
        for w in text.split_whitespace() {
            processed.push_str(word_boundary);
            processed.push_str(w);
        }

        if processed.is_empty() {
            return Ok(Vec::new());
        }

        let chars: Vec<char> = processed.chars().collect();
        let n = chars.len();

        // Viterbi: best[i] = (best_score, best_token_end, token)
        let mut best: Vec<(f64, usize, String)> =
            vec![(f64::NEG_INFINITY, 0, String::new()); n + 1];
        best[0] = (0.0, 0, String::new());

        for i in 0..n {
            if best[i].0 == f64::NEG_INFINITY {
                continue;
            }

            for j in i + 1..=std::cmp::min(i + 16, n) {
                let substr: String = chars[i..j].iter().collect();
                if let Some(&(_, log_prob)) = self.vocab.get(&substr) {
                    let score = best[i].0 + log_prob;
                    if score > best[j].0 {
                        best[j] = (score, i, substr);
                    }
                }
            }

            // Fallback: single character as UNK
            if best[i + 1].0 == f64::NEG_INFINITY {
                let char_str = chars[i].to_string();
                let log_prob = self.vocab.get(&char_str).map_or(-100.0, |(_, p)| *p);
                best[i + 1] = (best[i].0 + log_prob, i, char_str);
            }
        }

        // Backtrack to find tokens
        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let token = &best[pos].2;
            let prev = best[pos].1;

            if let Some(&(id, _)) = self.vocab.get(token) {
                tokens.push(id);
            } else {
                // Use UNK for unknown tokens
                if let Some(&(id, _)) = self.vocab.get(&self.unk_token) {
                    tokens.push(id);
                }
            }
            pos = prev;
        }

        tokens.reverse();
        Ok(tokens)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, AprenderError> {
        let word_boundary = '▁';
        let mut result = String::new();

        for &id in ids {
            let token = self.inverse_vocab.get(&id).map_or(&self.unk_token, |t| t);

            // Skip special tokens
            if token == &self.unk_token || token == &self.bos_token || token == &self.eos_token {
                continue;
            }

            // Replace word boundary with space
            for c in token.chars() {
                if c == word_boundary {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                } else {
                    result.push(c);
                }
            }
        }

        Ok(result)
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get reference to vocabulary (without probabilities).
    #[must_use]
    pub fn vocab_ids(&self) -> HashMap<String, u32> {
        self.vocab
            .iter()
            .map(|(k, (id, _))| (k.clone(), *id))
            .collect()
    }

    /// Get log probability of a token.
    #[must_use]
    pub fn log_prob(&self, token: &str) -> Option<f64> {
        self.vocab.get(token).map(|(_, p)| *p)
    }
}

impl Tokenizer for UnigramTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>, AprenderError> {
        let ids = self.encode(text)?;
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|id| self.inverse_vocab.get(id).cloned())
            .collect();
        Ok(tokens)
    }
}


#[cfg(test)]
mod tests;
