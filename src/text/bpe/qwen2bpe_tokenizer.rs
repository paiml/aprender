#[allow(clippy::wildcard_imports)]
use super::*;
use crate::error::{AprenderError, Result};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// ============================================================================
// Priority-queue BPE merge internals (GH-378)
// Port of HuggingFace tokenizers word.rs algorithm.
// ============================================================================

/// A pending merge in the priority queue.
/// Min-heap ordered: lowest rank (= highest priority) pops first.
#[derive(Debug, Eq, PartialEq)]
struct BpeMerge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl Ord for BpeMerge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap: lower rank = higher priority
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for BpeMerge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A symbol in the doubly-linked list used during BPE merging.
/// Merges update pointers in O(1) — no array shifting.
#[derive(Debug, Clone, Copy)]
struct BpeSymbol {
    id: u32,
    prev: i32, // -1 = no predecessor
    next: i32, // -1 = no successor
    len: u32,  // 0 = deleted
    src: u32,  // index into original tokens array (fallback for unknown IDs)
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer with given config
    #[must_use]
    pub fn new(config: BpeConfig) -> Self {
        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Self {
            config,
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            merge_id_map: HashMap::new(),
            merge_token_ids: HashMap::new(),
            merge_id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
            byte_encoder,
            byte_decoder,
        }
    }

    /// Create a new BPE tokenizer with pre-sized data structures.
    ///
    /// Avoids repeated HashMap rehashing when loading large vocabularies.
    /// For Qwen2 (151K vocab, 151K merges), this eliminates ~15 rehash
    /// cycles per HashMap during loading.
    #[must_use]
    pub(crate) fn with_capacity(config: BpeConfig, vocab_cap: usize, merge_cap: usize) -> Self {
        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Self {
            config,
            vocab: HashMap::with_capacity(vocab_cap),
            id_to_token: HashMap::with_capacity(vocab_cap),
            merges: Vec::with_capacity(merge_cap),
            merge_ranks: HashMap::new(), // populated by add_merge only, not fast path
            merge_id_map: HashMap::with_capacity(merge_cap),
            merge_token_ids: HashMap::new(),
            merge_id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
            byte_encoder,
            byte_decoder,
        }
    }

    /// Load tokenizer from a `HuggingFace` tokenizer.json file path.
    ///
    /// Parses the `HuggingFace` tokenizer.json format, extracting:
    /// - `model.vocab` (token-to-ID mapping)
    /// - `model.merges` (ordered BPE merge rules)
    /// - `added_tokens` (special tokens like `<|endoftext|>`, `<|im_start|>`)
    ///
    /// The byte encoder for UTF-8 byte-level BPE is built automatically.
    ///
    /// # Arguments
    /// * `path` - Path to a `HuggingFace` tokenizer.json file
    ///
    /// # Returns
    /// A fully loaded `BpeTokenizer` with vocabulary, merge rules, and special tokens.
    ///
    /// # Errors
    /// Returns error if the file cannot be read or the JSON is malformed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use aprender::text::bpe::BpeTokenizer;
    ///
    /// let tokenizer = BpeTokenizer::from_huggingface("path/to/tokenizer.json")
    ///     .expect("failed to load tokenizer");
    /// assert!(tokenizer.vocab_size() > 0);
    /// let ids = tokenizer.encode("Hello world");
    /// assert!(!ids.is_empty());
    /// ```
    pub fn from_huggingface<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json =
            std::fs::read_to_string(path.as_ref()).map_err(|e| AprenderError::FormatError {
                message: format!(
                    "Failed to read tokenizer file '{}': {e}",
                    path.as_ref().display()
                ),
            })?;
        Self::from_huggingface_json(&json)
    }

    /// Load tokenizer from a `HuggingFace` tokenizer.json string.
    ///
    /// This is the in-memory counterpart of [`from_huggingface`](Self::from_huggingface).
    /// Useful when the JSON has already been read into a string (e.g., from an HTTP response).
    ///
    /// # Arguments
    /// * `json` - JSON string in `HuggingFace` tokenizer.json format
    ///
    /// # Returns
    /// A fully loaded `BpeTokenizer`.
    ///
    /// # Errors
    /// Returns error if JSON parsing fails or the structure is invalid.
    pub fn from_huggingface_json(json: &str) -> Result<Self> {
        super::load_from_json(json)
    }

    /// Load tokenizer from legacy GPT-2/RoBERTa format (vocab.json + merges.txt).
    ///
    /// CodeBERT and other RoBERTa-family models use this format instead of the
    /// unified tokenizer.json. The vocab.json maps tokens to IDs, and merges.txt
    /// contains ordered BPE merge rules (one per line, `#version` header skipped).
    ///
    /// # Arguments
    /// * `vocab_path` - Path to vocab.json (`{"token": id, ...}`)
    /// * `merges_path` - Path to merges.txt (header + one `pair1 pair2` per line)
    ///
    /// # Returns
    /// A fully loaded `BpeTokenizer` with vocabulary and merge rules.
    ///
    /// # Errors
    /// Returns error if files cannot be read or JSON is malformed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use aprender::text::bpe::BpeTokenizer;
    ///
    /// let tokenizer = BpeTokenizer::from_vocab_merges(
    ///     "path/to/vocab.json",
    ///     "path/to/merges.txt",
    /// ).expect("failed to load tokenizer");
    /// assert!(tokenizer.vocab_size() > 0);
    /// ```
    pub fn from_vocab_merges<P: AsRef<std::path::Path>, Q: AsRef<std::path::Path>>(
        vocab_path: P,
        merges_path: Q,
    ) -> Result<Self> {
        let vocab_json =
            std::fs::read_to_string(vocab_path.as_ref()).map_err(|e| AprenderError::FormatError {
                message: format!(
                    "Failed to read vocab file '{}': {e}",
                    vocab_path.as_ref().display()
                ),
            })?;
        let merges_txt =
            std::fs::read_to_string(merges_path.as_ref()).map_err(|e| AprenderError::FormatError {
                message: format!(
                    "Failed to read merges file '{}': {e}",
                    merges_path.as_ref().display()
                ),
            })?;
        super::load_from_files(&vocab_json, &merges_txt)
    }

    /// Create tokenizer with GPT-2 base vocabulary (stub)
    ///
    /// # Note
    /// Real implementation requires loading vocabulary files.
    #[must_use]
    pub fn gpt2_base() -> Self {
        let config = BpeConfig::gpt2();
        let mut tokenizer = Self::new(config);

        // Add basic ASCII characters as initial vocab
        for i in 0..=255u8 {
            if let Some(&c) = tokenizer.byte_encoder.get(&i) {
                let token = c.to_string();
                let id = u32::from(i);
                tokenizer.vocab.insert(token.clone(), id);
                tokenizer.id_to_token.insert(id, token);
            }
        }

        // Add special tokens (source of truth: special-tokens-registry-v1.yaml)
        tokenizer.add_special_token("<|endoftext|>", crate::demo::SpecialTokens::gpt2().eos_id);

        tokenizer
    }

    /// Add a special token
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_string(), id);
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }

    /// Add a merge rule
    pub fn add_merge(&mut self, first: &str, second: &str) {
        let rank = self.merges.len();
        let rule = MergeRule::new(first, second);
        self.merge_ranks
            .insert((first.to_string(), second.to_string()), rank);
        self.merges.push(rule);

        // Build ID-based merge map for priority-queue algorithm (GH-378)
        let merged = format!("{first}{second}");
        let first_id = self.get_or_assign_id(first);
        let second_id = self.get_or_assign_id(second);
        let merged_id = self.get_or_assign_id(&merged);
        self.merge_id_map
            .insert((first_id, second_id), (rank as u32, merged_id));
    }

    /// Bulk-add a merge rule from owned strings, avoiding redundant clones.
    ///
    /// Skips `merge_ranks` population (only used by tests, never at encode time).
    /// The priority-queue algorithm (GH-378) uses `merge_id_map` exclusively.
    /// Saves 300K String clones on Qwen2-scale vocabularies.
    pub(crate) fn add_merge_owned(&mut self, first: String, second: String) {
        let rank = self.merges.len();

        // Build ID-based merge map (the only structure used at encode time)
        let merged = format!("{}{}", first, second);
        let first_id = self.get_or_assign_id(&first);
        let second_id = self.get_or_assign_id(&second);
        let merged_id = self.get_or_assign_id(&merged);
        self.merge_id_map
            .insert((first_id, second_id), (rank as u32, merged_id));

        // Move originals into MergeRule (no clones needed — merge_ranks skipped)
        self.merges.push(MergeRule { first, second });
    }

    /// Get vocab ID for a token, auto-assigning internally if absent.
    /// Checks public vocab first, then internal merge_token_ids.
    /// Never adds to public vocab — keeps vocab_size() accurate.
    fn get_or_assign_id(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }
        if let Some(&id) = self.merge_token_ids.get(token) {
            return id;
        }
        // Assign internal IDs starting above any possible vocab ID
        let id = (u32::MAX / 2) + self.merge_token_ids.len() as u32;
        self.merge_token_ids.insert(token.to_string(), id);
        self.merge_id_to_token.insert(id, token.to_string());
        id
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token ID for a token
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get token for an ID
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Check if token is a special token
    #[must_use]
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    /// Encode text to token IDs.
    ///
    /// # Arguments
    /// * `text` - Text to encode
    ///
    /// # Returns
    /// Vector of token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let mut ids = Vec::new();

        // PMAT-114: Handle special tokens FIRST before BPE tokenization
        // This ensures tokens like <|im_start|> are encoded as single tokens (151644)
        // rather than being split into characters (27, 91, 318, 4906, 91, 29)
        for segment in self.split_on_special_tokens(text) {
            if let Some(&special_id) = self.special_tokens.get(&segment) {
                ids.push(special_id);
            } else {
                self.encode_segment(&segment, &mut ids);
            }
        }

        ids
    }

    /// Encode a regular (non-special-token) text segment into token IDs.
    fn encode_segment(&self, segment: &str, ids: &mut Vec<u32>) {
        let segment_text =
            if self.config.add_prefix_space && !segment.starts_with(' ') && ids.is_empty() {
                format!(" {segment}")
            } else {
                segment.to_string()
            };

        for word in self.pre_tokenize(&segment_text) {
            let byte_word = self.bytes_to_bpe_tokens(&word);
            for token in self.bpe(&byte_word) {
                let id = self
                    .vocab
                    .get(&token)
                    .or_else(|| self.vocab.get(&self.config.unk_token));
                if let Some(&id) = id {
                    ids.push(id);
                }
            }
        }
    }

    /// Split text on special tokens while preserving them as separate segments.
    /// Returns vec of segments where special tokens are their own elements.
    pub(crate) fn split_on_special_tokens(&self, text: &str) -> Vec<String> {
        if self.special_tokens.is_empty() {
            return vec![text.to_string()];
        }

        // Sort special tokens by length (longest first) to avoid partial matches
        let mut sorted_tokens: Vec<_> = self.special_tokens.keys().collect();
        sorted_tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));

        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            match Self::find_earliest_special_token(remaining, &sorted_tokens) {
                Some((pos, token)) => {
                    if pos > 0 {
                        result.push(remaining[..pos].to_string());
                    }
                    result.push(token.clone());
                    remaining = &remaining[pos + token.len()..];
                }
                None => {
                    result.push(remaining.to_string());
                    break;
                }
            }
        }

        result
    }

    /// Find the earliest occurrence of any special token in `text`.
    fn find_earliest_special_token<'a>(
        text: &str,
        sorted_tokens: &[&'a String],
    ) -> Option<(usize, &'a String)> {
        let mut earliest: Option<(usize, &'a String)> = None;
        for token in sorted_tokens {
            if let Some(pos) = text.find(token.as_str()) {
                if earliest.map_or(true, |(prev_pos, _)| pos < prev_pos) {
                    earliest = Some((pos, token));
                }
            }
        }
        earliest
    }

    /// Decode token IDs to text.
    ///
    /// # Arguments
    /// * `ids` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        if ids.is_empty() {
            return String::new();
        }

        let mut text = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                // Skip special tokens in output
                if !self.special_tokens.contains_key(token) {
                    text.push_str(token);
                }
            }
        }

        // Convert byte tokens back to UTF-8
        self.bpe_tokens_to_bytes(&text)
    }

    /// Encode text to token IDs with error handling.
    ///
    /// # Errors
    /// Returns error if encoding fails.
    pub fn encode_checked(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.encode(text))
    }

    /// Decode token IDs to text with error handling.
    ///
    /// # Errors
    /// Returns error if decoding fails.
    pub fn decode_checked(&self, ids: &[u32]) -> Result<String> {
        Ok(self.decode(ids))
    }

    /// Pre-tokenize text into words
    pub(crate) fn pre_tokenize(&self, text: &str) -> Vec<String> {
        // Simple regex-like pattern: split on whitespace, keeping punctuation
        // Future: Use self.config for model-specific pre-tokenization rules
        let _ = &self.config;
        let mut words = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            if c.is_whitespace() {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                // Include the space as part of next word
                current.push(c);
            } else {
                current.push(c);
            }
        }

        if !current.is_empty() {
            words.push(current);
        }

        words
    }

    /// Convert string to byte-encoded tokens
    pub(crate) fn bytes_to_bpe_tokens(&self, word: &str) -> Vec<String> {
        word.bytes()
            .map(|b| {
                self.byte_encoder
                    .get(&b)
                    .map_or_else(|| format!("?{b}"), |&c| c.to_string())
            })
            .collect()
    }

    /// Convert byte-encoded tokens back to string
    pub(crate) fn bpe_tokens_to_bytes(&self, text: &str) -> String {
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Apply BPE merges to token list.
    ///
    /// Uses priority-queue + doubly-linked-list algorithm (GH-378):
    /// - O(n) initial pair scan → seed min-heap
    /// - O(log n) per merge pop, O(1) pointer update, O(1) new-pair enqueue
    /// - Lazy validation: stale queue entries skipped on pop
    ///
    /// Port of HuggingFace `tokenizers` word.rs `merge_all()`.
    pub(crate) fn bpe(&self, tokens: &[String]) -> Vec<String> {
        if tokens.len() <= 1 {
            return tokens.to_vec();
        }

        let n = tokens.len();

        // Convert strings → u32 IDs (check public vocab, then internal merge vocab)
        let ids: Vec<u32> = tokens
            .iter()
            .map(|t| {
                self.vocab
                    .get(t.as_str())
                    .or_else(|| self.merge_token_ids.get(t.as_str()))
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect();

        // Build doubly-linked symbol list
        let mut symbols: Vec<BpeSymbol> = Vec::with_capacity(n);
        for (i, &id) in ids.iter().enumerate() {
            symbols.push(BpeSymbol {
                id,
                prev: if i > 0 { i as i32 - 1 } else { -1 },
                next: if i < n - 1 { (i + 1) as i32 } else { -1 },
                len: 1,
                src: i as u32,
            });
        }

        // Seed priority queue with all adjacent pairs that have merge rules
        let mut queue = BinaryHeap::with_capacity(n);
        for i in 0..n - 1 {
            let pair = (symbols[i].id, symbols[i + 1].id);
            if let Some(&(rank, new_id)) = self.merge_id_map.get(&pair) {
                queue.push(BpeMerge {
                    pos: i,
                    rank,
                    new_id,
                });
            }
        }

        // Merge loop: pop lowest-rank merge, apply, enqueue new pairs
        while let Some(top) = queue.pop() {
            if !Self::is_valid_merge(&top, &symbols, &self.merge_id_map) {
                continue;
            }
            Self::apply_merge(&top, &mut symbols);
            Self::enqueue_neighbor_pairs(&symbols, &self.merge_id_map, top.pos, &mut queue);
        }

        // Convert surviving symbols back to strings
        symbols
            .iter()
            .filter(|s| s.len > 0)
            .map(|s| {
                self.id_to_token
                    .get(&s.id)
                    .or_else(|| self.merge_id_to_token.get(&s.id))
                    .cloned()
                    .unwrap_or_else(|| tokens[s.src as usize].clone())
            })
            .collect()
    }

    /// Check whether a queued merge is still valid (symbol alive, pair unchanged).
    fn is_valid_merge(
        top: &BpeMerge,
        symbols: &[BpeSymbol],
        merge_id_map: &HashMap<(u32, u32), (u32, u32)>,
    ) -> bool {
        if symbols[top.pos].len == 0 || symbols[top.pos].next < 0 {
            return false;
        }
        let right = symbols[symbols[top.pos].next as usize];
        let pair = (symbols[top.pos].id, right.id);
        merge_id_map
            .get(&pair)
            .map_or(false, |&(_, new_id)| new_id == top.new_id)
    }

    /// Apply a merge: left symbol absorbs right, right marked deleted.
    fn apply_merge(top: &BpeMerge, symbols: &mut [BpeSymbol]) {
        let next_pos = symbols[top.pos].next as usize;
        let right = symbols[next_pos];

        symbols[top.pos].id = top.new_id;
        symbols[top.pos].len += right.len;
        symbols[top.pos].next = right.next;
        symbols[next_pos].len = 0;

        if right.next >= 0 && (right.next as usize) < symbols.len() {
            symbols[right.next as usize].prev = top.pos as i32;
        }
    }

    /// Enqueue merge pairs for the left and right neighbors of a just-merged symbol.
    fn enqueue_neighbor_pairs(
        symbols: &[BpeSymbol],
        merge_id_map: &HashMap<(u32, u32), (u32, u32)>,
        pos: usize,
        queue: &mut BinaryHeap<BpeMerge>,
    ) {
        // Left neighbor + merged symbol
        if symbols[pos].prev >= 0 {
            let prev_pos = symbols[pos].prev as usize;
            if let Some(&(rank, new_id)) =
                merge_id_map.get(&(symbols[prev_pos].id, symbols[pos].id))
            {
                queue.push(BpeMerge {
                    pos: prev_pos,
                    rank,
                    new_id,
                });
            }
        }
        // Merged symbol + right neighbor
        if symbols[pos].next >= 0 {
            let next = symbols[pos].next as usize;
            if let Some(&(rank, new_id)) = merge_id_map.get(&(symbols[pos].id, symbols[next].id)) {
                queue.push(BpeMerge { pos, rank, new_id });
            }
        }
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new(BpeConfig::default())
    }
}

// ============================================================================
// Qwen2 BPE Tokenizer
// ============================================================================

/// Qwen2-specific BPE tokenizer with chat template support.
///
/// Extends the base BPE tokenizer with Qwen2's special tokens and
/// chat formatting conventions.
///
/// # Example
///
/// ```rust
/// use aprender::text::bpe::Qwen2BpeTokenizer;
///
/// let tokenizer = Qwen2BpeTokenizer::new();
///
/// // Check special tokens
/// assert!(tokenizer.is_eos(151645)); // <|im_end|>
///
/// // Format a chat message
/// let formatted = tokenizer.format_chat("user", "Hello, world!");
/// assert!(formatted.contains("<|im_start|>user"));
/// ```
#[derive(Debug, Clone)]
pub struct Qwen2BpeTokenizer {
    /// Base tokenizer
    pub(super) base: BpeTokenizer,
    /// Special token IDs
    pub(super) im_start_id: u32,
    pub(super) im_end_id: u32,
    pub(super) endoftext_id: u32,
}
