//! Canonical SPM-style BPE encoding for GGUF `tokenizer.ggml.model = "gemma4"` vocabularies
//! (APR-EMBED-001 EG-1).
//!
//! EmbeddingGemma 2 ships a 262,144-token `gemma4` vocabulary: ranked merges over raw UTF-8,
//! with SentencePiece's `▁` (U+2581) standing for a space and `<0xXX>` byte tokens as the
//! fallback. `GGUFModel::encode` used to run greedy longest-match on it, which is not the
//! model's tokenization, and `apr tokenize` said so with `greedy-fallback`.
//!
//! This module ports the reference implementation, llama.cpp (the pinned comparator,
//! `scripts/llama_pin.toml`), `LLAMA_VOCAB_PRE_TYPE_GEMMA4` in `src/llama-vocab.cpp`:
//! - special tokens are split out first, longest first (`tokenizer_st_partition`, shared with
//!   [`super::byte_level_bpe`]);
//! - every space of a raw fragment becomes `▁` (`llama_escape_whitespace`), with no space prefix;
//! - the fragment splits into runs of newlines and runs of everything else (`[^\n]+|[\n]+`);
//! - a newline run that is itself a token is one symbol (llama.cpp PR 21343);
//! - otherwise each UTF-8 character is a symbol, and merges apply by TEXT, lowest rank first,
//!   leftmost on ties, skipping a pair whose sides changed since it was queued;
//! - a final symbol that is not a token is spelled byte by byte as `<0xXX>` tokens, and a byte
//!   with no such token is dropped, as llama.cpp drops it.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use super::byte_level_bpe::{partition_specials, special_tokens, token_types, Fragment};
use super::types::GGUFValue;

/// The `tokenizer.ggml.model` values this module encodes.
const SPM_BPE_MODELS: [&str; 1] = ["gemma4"];

/// SentencePiece's visible space.
const SPM_SPACE: char = '\u{2581}';

/// Is this file's vocabulary an SPM-style BPE vocabulary (`tokenizer.ggml.model = "gemma4"`)?
#[must_use]
pub fn is_spm_bpe<S: std::hash::BuildHasher>(metadata: &HashMap<String, GGUFValue, S>) -> bool {
    matches!(
        metadata.get("tokenizer.ggml.model"),
        Some(GGUFValue::String(s)) if SPM_BPE_MODELS.contains(&s.as_str())
    )
}

/// Why a GGUF vocabulary cannot be encoded by [`SpmBpe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpmBpeRefusal {
    /// `tokenizer.ggml.model` is absent or names another tokenizer.
    NotSpmBpe(String),
    /// The file has no `tokenizer.ggml.merges` (llama.cpp refuses such a file too).
    MissingMerges,
}

impl std::fmt::Display for SpmBpeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSpmBpe(m) => write!(f, "tokenizer.ggml.model '{m}' is not SPM-style BPE"),
            Self::MissingMerges => write!(f, "the file has no tokenizer.ggml.merges"),
        }
    }
}

/// An SPM-style BPE tokenizer built from a GGUF's vocabulary, merges and token types.
#[derive(Debug)]
pub struct SpmBpe {
    token_to_id: HashMap<String, u32>,
    /// `(first, second) -> rank`, from `tokenizer.ggml.merges` in file order; first rule wins.
    ranks: HashMap<(String, String), u32>,
    /// The `<0xXX>` byte tokens' ids, by byte.
    byte_ids: [Option<u32>; 256],
    /// Special tokens, longest text first.
    specials: Vec<(String, u32)>,
}

impl SpmBpe {
    /// Build the tokenizer for a `gemma4`-model GGUF, cached per distinct (vocabulary, merges,
    /// token types) as [`super::byte_level_bpe::ByteLevelBpe::from_gguf`] is.
    ///
    /// # Errors
    /// [`SpmBpeRefusal`] when the file is not SPM-style BPE or lacks merges.
    pub fn from_gguf(
        metadata: &HashMap<String, GGUFValue>,
        vocab: &[String],
    ) -> Result<Arc<Self>, SpmBpeRefusal> {
        if !is_spm_bpe(metadata) {
            let model = match metadata.get("tokenizer.ggml.model") {
                Some(GGUFValue::String(s)) => s.clone(),
                _ => "absent".to_string(),
            };
            return Err(SpmBpeRefusal::NotSpmBpe(model));
        }
        let merges: Vec<&str> = match metadata.get("tokenizer.ggml.merges") {
            Some(GGUFValue::Array(values)) => values
                .iter()
                .filter_map(|v| match v {
                    GGUFValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect(),
            _ => return Err(SpmBpeRefusal::MissingMerges),
        };
        let token_types = token_types(metadata);

        let mut hasher = DefaultHasher::new();
        vocab.hash(&mut hasher);
        merges.hash(&mut hasher);
        token_types.hash(&mut hasher);
        let key = hasher.finish();

        static CACHE: OnceLock<Mutex<Vec<(u64, Arc<SpmBpe>)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        if let Some(hit) = cache.lock().ok().and_then(|c| {
            c.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| Arc::clone(v))
        }) {
            return Ok(hit);
        }
        let built = Arc::new(Self::build(vocab, &merges, &token_types));
        if let Ok(mut c) = cache.lock() {
            if c.len() >= 8 {
                c.remove(0);
            }
            c.push((key, Arc::clone(&built)));
        }
        Ok(built)
    }

    /// Build from explicit tables (no cache).
    #[must_use]
    pub fn build(vocab: &[String], merges: &[&str], token_types: &[i32]) -> Self {
        let token_to_id: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();

        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, rule) in merges.iter().enumerate() {
            // llama.cpp splits at the first space at or after BYTE 1 (`find(' ', 1)`), so a rule
            // may merge a first piece that is itself a space; a rule with no such space is
            // ("", ""). A space is ASCII, so the split point is always a char boundary.
            let split = rule.bytes().skip(1).position(|b| b == b' ').map(|p| p + 1);
            let (first, second) = match split {
                Some(at) => (&rule[..at], &rule[at + 1..]),
                None => ("", ""),
            };
            ranks
                .entry((first.to_string(), second.to_string()))
                .or_insert(rank as u32);
        }

        let mut byte_ids = [None; 256];
        for (b, id) in byte_ids.iter_mut().enumerate() {
            *id = token_to_id.get(byte_token(b as u8).as_str()).copied();
        }

        Self {
            token_to_id,
            ranks,
            byte_ids,
            specials: special_tokens(vocab, token_types),
        }
    }

    /// Encode `text` into token ids. Special tokens in the text are matched first, as
    /// llama.cpp does with `parse_special = true`; no BOS or EOS is added (the caller owns that).
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for fragment in partition_specials(text, &self.specials) {
            match fragment {
                Fragment::Special(id) => ids.push(id),
                Fragment::Raw(raw) => {
                    let escaped = raw.replace(' ', &SPM_SPACE.to_string());
                    for word in newline_runs(&escaped) {
                        self.encode_word(word, &mut ids);
                    }
                },
            }
        }
        ids
    }

    /// BPE over one newline-free (or newline-only) word: a port of
    /// `llm_tokenizer_bpe_session::tokenize` for `LLAMA_VOCAB_PRE_TYPE_GEMMA4`.
    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        let mut syms = if word.bytes().all(|b| b == b'\n') && self.token_to_id.contains_key(word) {
            vec![Sym {
                start: 0,
                len: word.len(),
                prev: None,
                next: None,
            }]
        } else {
            seed_symbols(word)
        };
        self.merge_all(word, &mut syms);
        let mut at = (!syms.is_empty()).then_some(0);
        while let Some(i) = at {
            let s = syms[i];
            if s.len > 0 {
                self.emit(&word[s.start..s.start + s.len], out);
            }
            at = s.next;
        }
    }

    /// The merge loop: pop the best pending pair, apply it if both sides still spell what they
    /// spelled when it was queued, and queue the two pairs it creates with its neighbours.
    fn merge_all(&self, word: &str, syms: &mut [Sym]) {
        let mut queue = BinaryHeap::new();
        for left in 1..syms.len() {
            self.enqueue(word, syms, left - 1, left, &mut queue);
        }
        while let Some(p) = queue.pop() {
            let (l, r) = (syms[p.left], syms[p.right]);
            if l.len == 0 || r.len == 0 || l.len + r.len != p.len || l.start + l.len != r.start {
                continue;
            }
            if word[l.start..l.start + p.len] != *p.text {
                continue;
            }
            syms[p.left].len += r.len;
            syms[p.right].len = 0;
            syms[p.left].next = r.next;
            if let Some(nn) = r.next {
                syms[nn].prev = Some(p.left);
            }
            if let Some(prev) = syms[p.left].prev {
                self.enqueue(word, syms, prev, p.left, &mut queue);
            }
            if let Some(nn) = syms[p.left].next {
                self.enqueue(word, syms, p.left, nn, &mut queue);
            }
        }
    }

    fn enqueue(
        &self,
        word: &str,
        syms: &[Sym],
        left: usize,
        right: usize,
        queue: &mut BinaryHeap<Pending>,
    ) {
        let (l, r) = (syms[left], syms[right]);
        let first = &word[l.start..l.start + l.len];
        let second = &word[r.start..r.start + r.len];
        if let Some(&rank) = self.ranks.get(&(first.to_string(), second.to_string())) {
            queue.push(Pending {
                rank,
                left,
                right,
                len: l.len + r.len,
                text: format!("{first}{second}"),
            });
        }
    }

    /// One final symbol: its token, or its bytes as `<0xXX>` tokens.
    fn emit(&self, piece: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.token_to_id.get(piece) {
            out.push(id);
            return;
        }
        out.extend(piece.bytes().filter_map(|b| self.byte_ids[usize::from(b)]));
    }
}

/// The SentencePiece byte token for `b`: `<0xXX>`, upper-case hex.
fn byte_token(b: u8) -> String {
    format!("<0x{b:02X}>")
}

/// `[^\n]+|[\n]+`: the fragment as alternating runs of newlines and of everything else.
fn newline_runs(text: &str) -> Vec<&str> {
    let mut runs = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 1..=bytes.len() {
        if i == bytes.len() || (bytes[i] == b'\n') != (bytes[start] == b'\n') {
            runs.push(&text[start..i]);
            start = i;
        }
    }
    runs
}

/// One symbol per UTF-8 character.
fn seed_symbols(word: &str) -> Vec<Sym> {
    let n = word.chars().count();
    word.char_indices()
        .enumerate()
        .map(|(i, (start, c))| Sym {
            start,
            len: c.len_utf8(),
            prev: i.checked_sub(1),
            next: Some(i + 1).filter(|&j| j < n),
        })
        .collect()
}

/// A symbol: a byte range of the word, linked to its live neighbours. `len == 0` once merged
/// into its left neighbour.
#[derive(Debug, Clone, Copy)]
struct Sym {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// A candidate merge. Ordered so that `BinaryHeap` (a max-heap) pops the LOWEST rank first
/// and, among equal ranks, the LEFTMOST pair: llama.cpp's `llm_bigram_bpe::comparator`.
#[derive(Debug, PartialEq, Eq)]
struct Pending {
    rank: u32,
    left: usize,
    right: usize,
    len: usize,
    text: String,
}

impl Ord for Pending {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.rank.cmp(&self.rank).then(other.left.cmp(&self.left))
    }
}

impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
#[path = "spm_bpe_tests.rs"]
mod tests;
