#[allow(clippy::wildcard_imports)]
use super::*;
use crate::text::Tokenizer;
use crate::AprenderError;
use std::collections::HashMap;

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
#[path = "tests.rs"]
mod tests;
