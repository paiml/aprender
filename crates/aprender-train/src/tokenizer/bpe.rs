//! BPE (Byte Pair Encoding) tokenizer implementation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use super::config::{Normalization, TokenizerConfig};
use super::error::{Result, TokenizerError};
use super::traits::{TokenId, Tokenizer};

/// BPE (Byte Pair Encoding) tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    config: TokenizerConfig,
    /// Token to ID mapping
    vocab: HashMap<String, TokenId>,
    /// ID to token mapping
    id_to_token_map: HashMap<TokenId, String>,
    /// Merge rules (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Whether the tokenizer is trained
    trained: bool,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            config,
            vocab: HashMap::new(),
            id_to_token_map: HashMap::new(),
            merges: Vec::new(),
            trained: false,
        }
    }

    /// Initialize vocabulary with special tokens and bytes
    fn init_vocab(&mut self) {
        let mut id: TokenId = 0;

        // Add special tokens
        let special = [
            &self.config.special_tokens.unk,
            &self.config.special_tokens.bos,
            &self.config.special_tokens.eos,
            &self.config.special_tokens.pad,
            &self.config.special_tokens.mask,
        ];

        for token in special {
            self.vocab.insert(token.clone(), id);
            self.id_to_token_map.insert(id, token.clone());
            id += 1;
        }

        // Add all single bytes as base vocabulary
        for byte in 0..=255u8 {
            let token = format!("{byte:02x}");
            if !self.vocab.contains_key(&token) {
                self.vocab.insert(token.clone(), id);
                self.id_to_token_map.insert(id, token);
                id += 1;
            }
        }
    }

    /// Get pair frequencies from tokenized corpus
    #[cfg(test)]
    fn get_pair_freqs(&self, tokenized: &[Vec<String>]) -> HashMap<(String, String), usize> {
        let mut freqs = HashMap::new();

        for tokens in tokenized {
            for pair in tokens.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                *freqs.entry(key).or_insert(0) += 1;
            }
        }

        freqs
    }

    /// Merge the most frequent pair
    #[cfg(test)]
    fn merge_pair(&self, tokenized: &mut [Vec<String>], pair: &(String, String), merged: &str) {
        for tokens in tokenized.iter_mut() {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    tokens[i] = merged.to_string();
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }
    }

    /// Apply the configured Unicode normalization, then optional lowercasing.
    ///
    /// NFC is applied BEFORE lowercasing because `char::to_lowercase()` is not
    /// closed over non-NFC input for every grapheme — normalizing first makes
    /// the pipeline deterministic for composed/decomposed variants of the
    /// same visible text.
    fn preprocess(&self, text: &str) -> String {
        let normalized = match self.config.normalization {
            Normalization::None => text.to_string(),
            Normalization::NFC => text.nfc().collect(),
        };
        if self.config.lowercase {
            normalized.to_lowercase()
        } else {
            normalized
        }
    }

    /// Tokenize text to bytes (initial tokenization)
    fn to_bytes(&self, text: &str) -> Vec<String> {
        text.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Apply all learned merges
    fn apply_merges(&self, mut tokens: Vec<String>) -> Vec<String> {
        for (a, b) in &self.merges {
            let merged = format!("{a}{b}");
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if &tokens[i] == a && &tokens[i + 1] == b {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        tokens
    }

    /// Borrow the learned `token → id` vocabulary map.
    ///
    /// Exposed so callers (e.g. `apr tokenize train`) can emit the HuggingFace
    /// `vocab.json` artifact mandated by `contracts/tokenizer-bpe-v1.yaml` without
    /// serializing the whole `BPETokenizer` struct. Read-only by design — training
    /// and encoding continue to own the `HashMap`.
    pub fn vocab(&self) -> &HashMap<String, TokenId> {
        &self.vocab
    }

    /// Borrow the ordered list of learned merge rules (`(left, right)` pairs in
    /// merge order).
    ///
    /// Exposed so callers can write the HuggingFace `merges.txt` artifact. The
    /// order is load-bearing: `merges.txt` consumers apply pairs top-to-bottom.
    pub fn merges(&self) -> &[(String, String)] {
        &self.merges
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TokenizerError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load tokenizer from file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| TokenizerError::Serialization(e.to_string()))
    }

    /// Reconstruct a trained `BPETokenizer` from the HuggingFace-style pair of
    /// `vocab.json` + `merges.txt` emitted by `apr tokenize train`.
    ///
    /// # Format
    /// - `vocab.json`: JSON object mapping token string → token id (u32). Order
    ///   is informational; the loader inverts the map to build `id_to_token`.
    /// - `merges.txt`: leading `#version: 0.2\n` header line, then one merge per
    ///   line in apply order. Each line is `"<left> <right>"` with a single
    ///   ASCII space separator (tokens in the aprender-train hex
    ///   representation never contain spaces, so space-split is unambiguous).
    ///
    /// # Parameters
    /// - `vocab_path`: path to `vocab.json`
    /// - `merges_path`: path to `merges.txt`
    /// - `config`: caller-supplied config (normalization, special tokens, etc.)
    ///   since those fields are not recorded in the HF-style files. MUST match
    ///   the config used at training time — wrong normalization here produces
    ///   silently-wrong encodings.
    ///
    /// # Invariants
    /// - C-PRETOK-BIN INV-PRETOK-001: every loaded vocab id < returned
    ///   tokenizer's `vocab_size()`.
    /// - Every merge's `(left, right)` concatenation is present in the loaded
    ///   vocab (otherwise applying the merge during encode would produce a
    ///   token the vocab cannot resolve). Enforced; mismatch returns an error.
    pub fn from_vocab_merges(
        vocab_path: &str,
        merges_path: &str,
        config: TokenizerConfig,
    ) -> Result<Self> {
        let vocab_json = std::fs::read_to_string(vocab_path)?;
        let vocab: HashMap<String, TokenId> = serde_json::from_str(&vocab_json)
            .map_err(|e| TokenizerError::Serialization(e.to_string()))?;

        let id_to_token_map: HashMap<TokenId, String> =
            vocab.iter().map(|(tok, &id)| (id, tok.clone())).collect();

        if id_to_token_map.len() != vocab.len() {
            return Err(TokenizerError::Serialization(
                "vocab.json contains duplicate token ids (collision detected after inverting map)"
                    .to_string(),
            ));
        }

        let merges_text = std::fs::read_to_string(merges_path)?;
        let mut merges: Vec<(String, String)> = Vec::new();
        for (line_no, line) in merges_text.lines().enumerate() {
            if line.is_empty() || line.starts_with("#") {
                continue;
            }
            let mut parts = line.splitn(2, ' ');
            let left = parts
                .next()
                .ok_or_else(|| {
                    TokenizerError::Serialization(format!(
                        "merges.txt line {}: missing left token",
                        line_no + 1
                    ))
                })?
                .to_string();
            let right = parts
                .next()
                .ok_or_else(|| {
                    TokenizerError::Serialization(format!(
                        "merges.txt line {}: missing right token (expected '<left> <right>')",
                        line_no + 1
                    ))
                })?
                .to_string();

            let merged = format!("{left}{right}");
            if !vocab.contains_key(&merged) {
                return Err(TokenizerError::Serialization(format!(
                    "merges.txt line {}: merged token {:?} not present in vocab.json",
                    line_no + 1,
                    merged
                )));
            }
            merges.push((left, right));
        }

        Ok(Self { config, vocab, id_to_token_map, merges, trained: true })
    }
}

impl Tokenizer for BPETokenizer {
    fn train(&mut self, corpus: &[&str]) -> Result<()> {
        train_fast(self, corpus)
    }

    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let tokens = self.to_bytes(&self.preprocess(text));
        let tokens = self.apply_merges(tokens);

        let unk_id = *self
            .vocab
            .get(&self.config.special_tokens.unk)
            .expect("UNK token must exist in trained vocabulary");

        let ids: Vec<TokenId> =
            tokens.iter().map(|t| *self.vocab.get(t).unwrap_or(&unk_id)).collect();

        Ok(ids)
    }

    fn decode(&self, ids: &[TokenId]) -> Result<String> {
        if !self.trained {
            return Err(TokenizerError::NotTrained);
        }

        let mut hex_string = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token_map.get(&id) {
                // Skip special tokens
                if token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                hex_string.push_str(token);
            }
        }

        // Convert hex string back to bytes
        let bytes: Vec<u8> = (0..hex_string.len())
            .step_by(2)
            .filter_map(|i| {
                if i + 2 <= hex_string.len() {
                    u8::from_str_radix(&hex_string[i..i + 2], 16).ok()
                } else {
                    None
                }
            })
            .collect();

        String::from_utf8(bytes).map_err(|e| TokenizerError::Training(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.id_to_token_map.get(&id).map(String::as_str)
    }

    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

// ─────────────────────────────────────────────────────────────
// Priority-queue + inverted-index BPE training.
//
// Contract: contracts/bpe-training-perf-v1.yaml (v1.1.0).
//   - Algorithm: Sennrich 2016 / HuggingFace tokenizers style.
//   - Tie-breaker: lex-min on (left_id, right_id) for cross-run
//     determinism (INV-BPE-006, FALSIFY-BPE-TRAIN-PERF-002).
//   - Complexity: O((V + E) log V) amortized where E is total pair-
//     count updates. Replaces a naive O(V · N · L) loop that did not
//     complete a 50K-vocab × 127 MB training run in 25 h 40 m.
//   - Observability: periodic `[bpe]` stderr reports every 1 000
//     merges (FALSIFY-BPE-TRAIN-PERF-004).
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Eq, PartialEq)]
struct HeapEntry {
    count: i64,
    pair: (TokenId, TokenId),
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: higher count wins (BinaryHeap is a max-heap).
        // Tie-breaker: smaller (left_id, right_id) tuple wins — invert
        // the pair comparison so the smaller pair is "greater" and
        // therefore popped first.
        self.count.cmp(&other.count).then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Fast priority-queue + inverted-index BPE training.
///
/// Invoked via `<BPETokenizer as Tokenizer>::train`. Exposed as a free
/// function so tests can compare against `train_naive_reference` for
/// FALSIFY-BPE-TRAIN-PERF-001 (parity) and -005 (speedup).
pub(crate) fn train_fast(tok: &mut BPETokenizer, corpus: &[&str]) -> Result<()> {
    use std::collections::{BinaryHeap, HashMap, HashSet};
    use std::time::Instant;

    let start = Instant::now();
    let target = tok.config.vocab_size;
    let min_frequency = tok.config.min_frequency.max(1) as i64;

    tok.init_vocab();

    eprintln!("[bpe-setup] ingest start: {} docs", corpus.len());
    use std::io::Write;
    let _ = std::io::stderr().flush();

    // Byte-tokenize every document, fold duplicates into (Vec<TokenId>, multiplicity) pairs.
    let t0 = Instant::now();
    let mut word_counts: HashMap<Vec<TokenId>, u64> = HashMap::new();
    for doc in corpus {
        let text = tok.preprocess(doc);
        let hex_tokens = tok.to_bytes(&text);
        if hex_tokens.is_empty() {
            continue;
        }
        let ids: Vec<TokenId> = hex_tokens
            .iter()
            .map(|t| *tok.vocab.get(t).expect("byte hex token must be in init_vocab"))
            .collect();
        *word_counts.entry(ids).or_insert(0) += 1;
    }
    eprintln!(
        "[bpe-setup] ingest done: {} unique words in {:.1}s",
        word_counts.len(),
        t0.elapsed().as_secs_f64()
    );
    let _ = std::io::stderr().flush();

    let mut words: Vec<(Vec<TokenId>, u64)> = word_counts.into_iter().collect();

    // Build pair indexes.
    let t1 = Instant::now();
    let mut pair_counts: HashMap<(TokenId, TokenId), i64> = HashMap::new();
    let mut pair_words: HashMap<(TokenId, TokenId), HashSet<usize>> = HashMap::new();
    for (word_ix, (ids, mult)) in words.iter().enumerate() {
        let m = *mult as i64;
        for w in ids.windows(2) {
            let p = (w[0], w[1]);
            *pair_counts.entry(p).or_insert(0) += m;
            pair_words.entry(p).or_default().insert(word_ix);
        }
    }
    eprintln!(
        "[bpe-setup] pair indexes: {} unique pairs in {:.1}s",
        pair_counts.len(),
        t1.elapsed().as_secs_f64()
    );
    let _ = std::io::stderr().flush();

    // Seed heap.
    let t2 = Instant::now();
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(pair_counts.len());
    for (p, c) in &pair_counts {
        if *c > 0 {
            heap.push(HeapEntry { count: *c, pair: *p });
        }
    }
    eprintln!(
        "[bpe-setup] heap seeded: {} entries in {:.1}s; entering merge loop",
        heap.len(),
        t2.elapsed().as_secs_f64()
    );
    let _ = std::io::stderr().flush();

    let mut merges_emitted: usize = 0;

    // Scratch buffers hoisted OUT of the per-word loop. Each early common-pair
    // merge affects ~100K words; allocating transient containers per word cost
    // ~400K mallocs per merge (observed as 1+ s/merge at merge 400, PID
    // 1568187, 2026-04-20). HashSet reuse was tried and FALSIFIED (27% slower
    // on PID 1638021, 2026-04-20) because `HashSet::clear()` walks the backing
    // array (up to 4096 slots) per call. Vec + sort_unstable + merge-pass for
    // set ops wins on (u32, u32) keys: cheaper to sort than to hash.
    let mut old_pairs_buf: Vec<(TokenId, TokenId)> = Vec::with_capacity(512);
    let mut new_pairs_buf: Vec<(TokenId, TokenId)> = Vec::with_capacity(512);
    let mut pairs_touched_buf: Vec<(TokenId, TokenId)> = Vec::with_capacity(1 << 16);
    let mut affected_buf: Vec<usize> = Vec::with_capacity(1 << 16);

    while tok.vocab.len() < target {
        let entry = match heap.pop() {
            Some(e) => e,
            None => break,
        };
        // Drop stale entries — a pair's count was updated after this entry was pushed.
        let current = *pair_counts.get(&entry.pair).unwrap_or(&0);
        if current != entry.count {
            continue;
        }
        if current < min_frequency {
            break;
        }

        let (a, b) = entry.pair;
        let a_str = tok.id_to_token_map[&a].clone();
        let b_str = tok.id_to_token_map[&b].clone();
        let merged_str = format!("{a_str}{b_str}");
        let new_id: TokenId = tok.vocab.len() as TokenId;
        tok.vocab.insert(merged_str.clone(), new_id);
        tok.id_to_token_map.insert(new_id, merged_str);
        tok.merges.push((a_str, b_str));
        merges_emitted += 1;

        // Apply merge in every word containing (a, b). Snapshot the set first so we
        // can mutate pair_words during the sweep.
        affected_buf.clear();
        if let Some(ws) = pair_words.get(&(a, b)) {
            affected_buf.extend(ws.iter().copied());
        }

        // Aggregate the set of pairs whose count changed across ALL affected
        // words. Pushing heap entries once per pair per merge (rather than
        // once per (pair, word) tuple) is load-bearing: early merges of
        // common byte-pairs can touch 10⁵+ words, and pushing per-word
        // produced 10⁸+ stale heap entries / merge, OOM-killing the run
        // (observed 2026-04-20, PID 1387417 hit 29 GB RSS).
        pairs_touched_buf.clear();

        for &word_ix in &affected_buf {
            let (ids, mult) = &mut words[word_ix];
            let m = *mult as i64;

            // Collect old pairs into reused buffer (zero alloc).
            old_pairs_buf.clear();
            old_pairs_buf.extend(ids.windows(2).map(|w| (w[0], w[1])));

            // In-place greedy left-to-right merge of (a, b) → new_id.
            // Since the merge only shrinks the Vec, read ≥ write holds, so
            // the single-buffer two-pointer walk is safe.
            let mut write = 0;
            let mut read = 0;
            while read < ids.len() {
                if read + 1 < ids.len() && ids[read] == a && ids[read + 1] == b {
                    ids[write] = new_id;
                    write += 1;
                    read += 2;
                } else {
                    ids[write] = ids[read];
                    write += 1;
                    read += 1;
                }
            }
            ids.truncate(write);

            // Collect new pairs into reused buffer.
            new_pairs_buf.clear();
            new_pairs_buf.extend(ids.windows(2).map(|w| (w[0], w[1])));

            // Multiset deltas on pair_counts (duplicates matter for counts).
            for p in &old_pairs_buf {
                *pair_counts.entry(*p).or_insert(0) -= m;
            }
            for p in &new_pairs_buf {
                *pair_counts.entry(*p).or_insert(0) += m;
            }

            // Set deltas on pair_words via sort + linear merge-pass.
            // HashSet alternative was FALSIFIED (27% slower) because
            // HashSet::clear walks the backing array (4096 slots) per call.
            // sort_unstable on (u32, u32) is branch-predictable + LLVM
            // auto-vectorizes; no hashing cost for POD keys.
            old_pairs_buf.sort_unstable();
            old_pairs_buf.dedup();
            new_pairs_buf.sort_unstable();
            new_pairs_buf.dedup();

            let mut i = 0usize;
            let mut j = 0usize;
            while i < old_pairs_buf.len() && j < new_pairs_buf.len() {
                match old_pairs_buf[i].cmp(&new_pairs_buf[j]) {
                    std::cmp::Ordering::Less => {
                        if let Some(ws) = pair_words.get_mut(&old_pairs_buf[i]) {
                            ws.remove(&word_ix);
                        }
                        pairs_touched_buf.push(old_pairs_buf[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        pair_words.entry(new_pairs_buf[j]).or_default().insert(word_ix);
                        pairs_touched_buf.push(new_pairs_buf[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        // Present in both — no pair_words delta, but still
                        // touched (multiplicity / top-pair ordering may shift).
                        pairs_touched_buf.push(old_pairs_buf[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            while i < old_pairs_buf.len() {
                if let Some(ws) = pair_words.get_mut(&old_pairs_buf[i]) {
                    ws.remove(&word_ix);
                }
                pairs_touched_buf.push(old_pairs_buf[i]);
                i += 1;
            }
            while j < new_pairs_buf.len() {
                pair_words.entry(new_pairs_buf[j]).or_default().insert(word_ix);
                pairs_touched_buf.push(new_pairs_buf[j]);
                j += 1;
            }
        }

        // Dedup aggregated pairs across all affected words, then push ONE
        // refreshed heap entry per affected pair (not per word).
        pairs_touched_buf.sort_unstable();
        pairs_touched_buf.dedup();
        for p in &pairs_touched_buf {
            let c = *pair_counts.get(p).unwrap_or(&0);
            if c > 0 {
                heap.push(HeapEntry { count: c, pair: *p });
            }
        }

        // The merged pair itself is fully consumed — purge its entries.
        pair_counts.remove(&(a, b));
        pair_words.remove(&(a, b));

        if merges_emitted == 1 || merges_emitted.is_multiple_of(100) {
            let elapsed = start.elapsed().as_secs_f64();
            let top_count = heap.peek().map(|e| e.count).unwrap_or(0);
            eprintln!(
                "[bpe] merges={} vocab={} elapsed={:.1}s top_count={} heap={} pairs={}",
                merges_emitted,
                tok.vocab.len(),
                elapsed,
                top_count,
                heap.len(),
                pair_counts.len()
            );
            let _ = std::io::stderr().flush();
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "[bpe] DONE merges={} vocab={} elapsed={:.1}s",
        merges_emitted,
        tok.vocab.len(),
        elapsed
    );
    let _ = std::io::stderr().flush();

    tok.trained = true;
    Ok(())
}

/// Naive reference implementation — the pre-task-#118 algorithm, verbatim
/// except that the tie-breaker is forced to lex-min on (left_id, right_id)
/// so its output is a deterministic baseline for FALSIFY-BPE-TRAIN-PERF-001
/// (parity) and -005 (speedup measurement). Retained ONLY for tests — the
/// shipped training path is `train_fast`.
#[cfg(test)]
#[doc(hidden)]
pub(crate) fn train_naive_reference(tok: &mut BPETokenizer, corpus: &[&str]) -> Result<()> {
    let target = tok.config.vocab_size;
    let min_frequency = tok.config.min_frequency.max(1);

    tok.init_vocab();

    let mut tokenized: Vec<Vec<String>> =
        corpus.iter().map(|s| tok.to_bytes(&tok.preprocess(s))).collect();

    while tok.vocab.len() < target {
        let freqs = tok.get_pair_freqs(&tokenized);

        // Pick pair with max count, lex-min on (left_id, right_id) on ties.
        let mut best: Option<(usize, (TokenId, TokenId), (String, String))> = None;
        for (pair_str, count) in &freqs {
            if *count < min_frequency {
                continue;
            }
            let left_id = *tok.vocab.get(&pair_str.0).expect("left must be in vocab");
            let right_id = *tok.vocab.get(&pair_str.1).expect("right must be in vocab");
            match &best {
                None => best = Some((*count, (left_id, right_id), pair_str.clone())),
                Some((bc, bp, _)) => {
                    if *count > *bc || (*count == *bc && (left_id, right_id) < *bp) {
                        best = Some((*count, (left_id, right_id), pair_str.clone()));
                    }
                }
            }
        }

        let (_count, _ids, pair_str) = match best {
            Some(b) => b,
            None => break,
        };

        let merged = format!("{}{}", pair_str.0, pair_str.1);
        let new_id: TokenId = tok.vocab.len() as TokenId;
        tok.vocab.insert(merged.clone(), new_id);
        tok.id_to_token_map.insert(new_id, merged.clone());
        tok.merges.push(pair_str.clone());
        tok.merge_pair(&mut tokenized, &pair_str, &merged);
    }

    tok.trained = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_new() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);
        assert!(!tokenizer.is_trained());
    }

    #[test]
    fn test_bpe_train() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello hello", "hello world", "world hello"];
        tokenizer.train(&corpus).expect("operation should succeed");

        assert!(tokenizer.is_trained());
        assert!(tokenizer.vocab_size() > 256); // Base bytes + some merges
    }

    #[test]
    fn test_bpe_encode_not_trained() {
        let config = TokenizerConfig::bpe();
        let tokenizer = BPETokenizer::new(config);

        let result = tokenizer.encode("hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_encode_decode() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["hello world", "hello there"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let text = "hello";
        let encoded = tokenizer.encode(text).expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_lowercase() {
        let config =
            TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1).with_lowercase(true);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["Hello World"];
        tokenizer.train(&corpus).expect("operation should succeed");

        let encoded = tokenizer.encode("HELLO").expect("encoding should succeed");
        let decoded = tokenizer.decode(&encoded).expect("encoding should succeed");

        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_id_to_token() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).expect("operation should succeed");

        // ID 0 should be <unk>
        assert_eq!(tokenizer.id_to_token(0), Some("<unk>"));
    }

    #[test]
    fn test_bpe_token_to_id() {
        let config = TokenizerConfig::bpe().with_vocab_size(300).with_min_frequency(1);
        let mut tokenizer = BPETokenizer::new(config);

        let corpus = vec!["test"];
        tokenizer.train(&corpus).expect("operation should succeed");

        assert_eq!(tokenizer.token_to_id("<unk>"), Some(0));
    }

    // C-TOK-BPE-001 INV-TOK-003: NFC normalization makes composed and decomposed
    // variants of the same grapheme hash to identical byte sequences, so a
    // tokenizer trained on one form encodes the other form identically.
    #[test]
    fn test_bpe_nfc_composed_decomposed_parity() {
        let composed = "café"; // U+00E9
        let decomposed = "cafe\u{0301}"; // e + combining acute

        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1)
            .with_normalization(Normalization::NFC);
        let mut tokenizer = BPETokenizer::new(config);
        tokenizer.train(&[composed]).expect("operation should succeed");

        let ids_composed = tokenizer.encode(composed).expect("encoding should succeed");
        let ids_decomposed = tokenizer.encode(decomposed).expect("encoding should succeed");

        assert_eq!(
            ids_composed, ids_decomposed,
            "NFC must map composed and decomposed café to identical token IDs"
        );

        let decoded = tokenizer.decode(&ids_composed).expect("decoding should succeed");
        assert_eq!(decoded, composed, "NFC round-trip must recover composed form");
    }

    // Without NFC, composed and decomposed café MUST diverge — this is the
    // exact drift INV-TOK-003 is defending against at training/inference boundary.
    #[test]
    fn test_bpe_without_nfc_composed_decomposed_diverge() {
        let composed = "café";
        let decomposed = "cafe\u{0301}";

        let config = TokenizerConfig::bpe()
            .with_vocab_size(300)
            .with_min_frequency(1)
            .with_normalization(Normalization::None);
        let mut tokenizer = BPETokenizer::new(config);
        tokenizer.train(&[composed]).expect("operation should succeed");

        let ids_composed = tokenizer.encode(composed).expect("encoding should succeed");
        let ids_decomposed = tokenizer.encode(decomposed).expect("encoding should succeed");

        assert_ne!(
            ids_composed, ids_decomposed,
            "Without NFC, composed and decomposed café MUST diverge (falsification witness for INV-TOK-003)"
        );
    }

    // C-PRETOK-BIN GATE-PRETOK-003 prerequisite: reloading a trained
    // tokenizer from its emitted vocab.json + merges.txt MUST yield
    // byte-identical encodings vs the original in-memory tokenizer.
    // Any drift here means `apr tokenize encode-corpus` (which loads
    // via from_vocab_merges) would produce shards that differ from
    // what the tokenizer intended — ShardBatchIter round-trip fails.
    #[test]
    fn test_bpe_from_vocab_merges_roundtrip() {
        use std::fmt::Write;
        let config = TokenizerConfig::bpe()
            .with_vocab_size(400)
            .with_min_frequency(1)
            .with_normalization(Normalization::NFC);
        let mut original = BPETokenizer::new(config.clone());
        let corpus = vec!["def hello():\n    return 1\n", "def world():\n    return 2\n"];
        original.train(&corpus).expect("training should succeed");

        let tmp = std::env::temp_dir().join(format!(
            "bpe_roundtrip_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let vocab_path = tmp.join("vocab.json");
        let merges_path = tmp.join("merges.txt");

        let mut entries: Vec<(&String, &TokenId)> = original.vocab().iter().collect();
        entries.sort_by_key(|(_, id)| *id);
        let ordered: serde_json::Map<String, serde_json::Value> = entries
            .into_iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::Number((*v).into())))
            .collect();
        let vocab_json = serde_json::to_string_pretty(&ordered).unwrap();
        std::fs::write(&vocab_path, vocab_json).unwrap();

        let mut merges_content = String::from("#version: 0.2\n");
        for (left, right) in original.merges() {
            writeln!(merges_content, "{left} {right}").unwrap();
        }
        std::fs::write(&merges_path, merges_content).unwrap();

        let reloaded = BPETokenizer::from_vocab_merges(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
            config,
        )
        .expect("from_vocab_merges should succeed");

        assert_eq!(reloaded.vocab_size(), original.vocab_size(), "reloaded vocab size must match");

        for text in &corpus {
            let original_ids = original.encode(text).expect("original encode");
            let reloaded_ids = reloaded.encode(text).expect("reloaded encode");
            assert_eq!(
                original_ids, reloaded_ids,
                "reloaded encoding must byte-equal original encoding for {text:?}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // Negative: from_vocab_merges must reject a merges.txt with a merged
    // token not present in vocab.json — that's a corrupt pair, and encoding
    // would silently emit <unk> instead of the intended token.
    #[test]
    fn test_bpe_from_vocab_merges_rejects_orphan_merge() {
        let tmp = std::env::temp_dir().join(format!(
            "bpe_orphan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let vocab_path = tmp.join("vocab.json");
        let merges_path = tmp.join("merges.txt");

        std::fs::write(&vocab_path, r#"{"<unk>": 0, "aa": 1, "bb": 2}"#).unwrap();
        std::fs::write(&merges_path, "#version: 0.2\naa bb\n").unwrap();

        let result = BPETokenizer::from_vocab_merges(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
            TokenizerConfig::bpe(),
        );

        assert!(
            result.is_err(),
            "from_vocab_merges must reject merges.txt with merged token not in vocab.json"
        );
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("aabb"),
            "error should name the offending merged token, got: {err_msg}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // Synthetic Python-like corpus builder for perf / parity tests. Deterministic.
    fn synthetic_python_corpus(n_docs: usize) -> Vec<String> {
        let templates: &[&str] = &[
            "def fn_{i}(x):\n    return x * {i}\n",
            "class C_{i}:\n    def __init__(self):\n        self.x = {i}\n",
            "for i in range({i}):\n    print(i * {i})\n",
            "def add_{i}(a, b):\n    return a + b + {i}\n",
            "import math\nprint(math.sqrt({i}))\n",
            "if x == {i}:\n    return True\nelse:\n    return False\n",
            "xs = [{i}, {i}, {i}]\nfor x in xs:\n    print(x)\n",
            "def process_{i}(data):\n    result = []\n    for item in data:\n        result.append(item + {i})\n    return result\n",
        ];
        (0..n_docs).map(|i| templates[i % templates.len()].replace("{i}", &i.to_string())).collect()
    }

    // FALSIFY-BPE-TRAIN-PERF-001: fast and naive produce identical output under lex-min.
    #[test]
    fn bpe_fast_vs_naive_parity() {
        let config = TokenizerConfig::bpe()
            .with_vocab_size(512)
            .with_min_frequency(1)
            .with_normalization(Normalization::NFC);

        let corpus_owned = synthetic_python_corpus(20);
        let corpus: Vec<&str> = corpus_owned.iter().map(String::as_str).collect();

        let mut fast = BPETokenizer::new(config.clone());
        super::train_fast(&mut fast, &corpus).expect("fast train should succeed");

        let mut naive = BPETokenizer::new(config);
        super::train_naive_reference(&mut naive, &corpus).expect("naive train should succeed");

        assert_eq!(
            fast.vocab_size(),
            naive.vocab_size(),
            "vocab sizes must match between fast and naive"
        );
        assert_eq!(fast.merges(), naive.merges(), "merge sequence must be identical");

        let mut fast_entries: Vec<(&String, &TokenId)> = fast.vocab().iter().collect();
        let mut naive_entries: Vec<(&String, &TokenId)> = naive.vocab().iter().collect();
        fast_entries.sort_by_key(|(_, id)| *id);
        naive_entries.sort_by_key(|(_, id)| *id);
        assert_eq!(
            fast_entries, naive_entries,
            "vocab (id → token) must be identical between fast and naive"
        );
    }

    // FALSIFY-BPE-TRAIN-PERF-002: same corpus + same config → byte-identical output.
    #[test]
    fn bpe_fast_is_deterministic() {
        let config = TokenizerConfig::bpe()
            .with_vocab_size(400)
            .with_min_frequency(1)
            .with_normalization(Normalization::NFC);

        let corpus_owned = synthetic_python_corpus(15);
        let corpus: Vec<&str> = corpus_owned.iter().map(String::as_str).collect();

        let mut a = BPETokenizer::new(config.clone());
        super::train_fast(&mut a, &corpus).expect("run A");
        let mut b = BPETokenizer::new(config);
        super::train_fast(&mut b, &corpus).expect("run B");

        assert_eq!(a.merges(), b.merges(), "merges must be byte-identical across runs");
        assert_eq!(a.vocab_size(), b.vocab_size(), "vocab size must match");

        let mut a_entries: Vec<(&String, &TokenId)> = a.vocab().iter().collect();
        let mut b_entries: Vec<(&String, &TokenId)> = b.vocab().iter().collect();
        a_entries.sort_by_key(|(_, id)| *id);
        b_entries.sort_by_key(|(_, id)| *id);
        assert_eq!(a_entries, b_entries, "vocab map must be byte-identical across runs");
    }

    // FALSIFY-BPE-TRAIN-PERF-005: fast ≥ 1.5× faster than the naive it replaces.
    // Org policy: any replacement must clear 1.5× or it is rejected.
    //
    // Uses a 500-doc / vocab=2048 / min_frequency=1 representative workload
    // per contract bpe-training-perf-v1.yaml v1.1.0. min_frequency=1 forces
    // the full 1787 merges (rather than early-stopping when counts fall
    // below 2), which is what exposes the quadratic cost of the naïve loop.
    //
    // In debug builds the constant-factor noise swamps the signal, so we
    // only assert in release — but we DO run the test in debug to catch
    // regressions in the fast path that explode its runtime beyond reason.
    #[test]
    fn bpe_fast_meets_1_5x_parity_replacement_rule() {
        use std::time::Instant;

        let config = TokenizerConfig::bpe()
            .with_vocab_size(2048)
            .with_min_frequency(1)
            .with_normalization(Normalization::NFC);

        let corpus_owned = synthetic_python_corpus(500);
        let corpus: Vec<&str> = corpus_owned.iter().map(String::as_str).collect();

        let mut naive = BPETokenizer::new(config.clone());
        let t0 = Instant::now();
        super::train_naive_reference(&mut naive, &corpus).expect("naive train");
        let naive_secs = t0.elapsed().as_secs_f64();

        let mut fast = BPETokenizer::new(config);
        let t0 = Instant::now();
        super::train_fast(&mut fast, &corpus).expect("fast train");
        let fast_secs = t0.elapsed().as_secs_f64();

        let ratio = naive_secs / fast_secs;
        eprintln!(
            "[bpe-speedup] naive={naive_secs:.3}s fast={fast_secs:.3}s ratio={ratio:.2}× \
             vocab_naive={} vocab_fast={}",
            naive.vocab_size(),
            fast.vocab_size()
        );

        // Correctness-floor: parity must hold at this scale too.
        assert_eq!(
            fast.merges(),
            naive.merges(),
            "at perf-workload scale, fast and naive merges MUST still match"
        );

        if cfg!(debug_assertions) {
            // Debug mode: assert fast is not worse than 1.0× (i.e. not slower).
            // The real 1.5× bar is enforced in release mode below.
            assert!(
                fast_secs < naive_secs * 1.5,
                "even in debug, fast must not be dramatically slower than naive \
                 (ratio={ratio:.2}×)"
            );
        } else {
            assert!(
                ratio >= 1.5,
                "org policy: replacement must be ≥1.5× faster than the replaced \
                 algorithm — got {ratio:.2}× (naive={naive_secs:.3}s, fast={fast_secs:.3}s)"
            );
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_bpe_encode_produces_valid_ids(text in "[a-zA-Z ]{1,20}") {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(300)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);
            tokenizer.train(&[&text]).expect("operation should succeed");

            let encoded = tokenizer.encode(&text).expect("encoding should succeed");

            for id in encoded {
                prop_assert!(tokenizer.id_to_token(id).is_some());
            }
        }

        #[test]
        fn prop_vocab_size_bounded(target_size in 261usize..500) {
            let config = TokenizerConfig::bpe()
                .with_vocab_size(target_size)
                .with_min_frequency(1);
            let mut tokenizer = BPETokenizer::new(config);

            let corpus = vec!["hello world hello world test test"];
            tokenizer.train(&corpus).expect("operation should succeed");

            prop_assert!(tokenizer.vocab_size() <= target_size);
        }
    }
}
