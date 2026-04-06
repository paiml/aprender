#[allow(clippy::wildcard_imports)]
use super::*;
use crate::text::Tokenizer;
use crate::AprenderError;
use std::collections::HashMap;

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
    pub(super) max_word_len: usize,
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
    pub(super) vocab: HashMap<String, (u32, f64)>,
    /// ID to token mapping
    pub(super) inverse_vocab: HashMap<u32, String>,
    /// Unknown token
    pub(super) unk_token: String,
    /// BOS token
    pub(super) bos_token: String,
    /// EOS token
    pub(super) eos_token: String,
}
