//! Canonical byte-level BPE encoding for GGUF `tokenizer.ggml.model = "gpt2"` vocabularies
//! (#3726).
//!
//! `GGUFModel::encode` used to run greedy longest-match over each whole segment: no
//! pre-tokenizer, no merge ranks. On a byte-level BPE vocabulary that produces ids the model
//! never saw in training. `" quorum"` became `Ġquo|rum` where the merges give `Ġqu|orum`;
//! `"   Title"` became `ĠĠĠ|Title` where the pre-tokenizer gives `ĠĠ|ĠTitle`; every byte
//! outside the printable glyph set became id 0. The model then copies the odd pieces out of
//! the prompt and joins canonical continuations onto them, so it quoted `quorum` back as
//! "quoorum" (#3693).
//!
//! This module ports the reference implementation, llama.cpp (the pinned comparator,
//! `scripts/llama_pin.toml`):
//! - the pre-tokenizer is `unicode_regex_split_custom_qwen2` / `_qwen35` (`src/unicode.cpp`),
//!   the hand-written form of the tokenizer.json regex, selected by `tokenizer.ggml.pre`;
//! - each piece's bytes map through GPT-2's `bytes_to_unicode` glyphs;
//! - merges apply lowest rank first, leftmost on ties (`llm_tokenizer_bpe_session`);
//! - special tokens (token type CONTROL, USER_DEFINED or UNKNOWN) are split out of the text
//!   first, longest first, as `tokenizer_st_partition` does with `parse_special`.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

use super::types::GGUFValue;

/// GGUF token types that `tokenizer_st_partition` treats as special with `parse_special`:
/// UNKNOWN (2), CONTROL (3), USER_DEFINED (4).
const SPECIAL_TOKEN_TYPES: [i32; 3] = [2, 3, 4];

/// The pre-tokenizers this module implements, by their `tokenizer.ggml.pre` name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PreTokenizer {
    /// `qwen2`: Qwen2, Qwen2.5, Qwen3 and Qwen3-MoE files.
    Qwen2,
    /// `qwen35`: Qwen3.5 dense and MoE. Letter runs also take combining marks (`\p{M}`).
    Qwen35,
}

impl PreTokenizer {
    /// The pre-tokenizer named by a GGUF's `tokenizer.ggml.pre`, or `None` when this module
    /// does not implement it.
    #[must_use]
    pub fn from_gguf_pre(pre: &str) -> Option<Self> {
        match pre {
            "qwen2" => Some(Self::Qwen2),
            "qwen35" => Some(Self::Qwen35),
            _ => None,
        }
    }

    /// #3742: the pre-tokenizer a model family's GGUF declares, for files that carry no name
    /// of their own (`.apr` files converted before `tokenizer.pre_type` was written). Every
    /// Qwen2, Qwen2.5, Qwen3 and Qwen3-MoE GGUF on the fleet says `qwen2` and every Qwen3.5 says
    /// `qwen35` (measured over the lambda and gx10 inventories).
    #[must_use]
    pub fn for_architecture(arch: &str) -> Option<Self> {
        match arch {
            "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" => Some(Self::Qwen2),
            "qwen35" | "qwen35moe" => Some(Self::Qwen35),
            _ => None,
        }
    }

    /// #3742: the pre-tokenizer whose HuggingFace `tokenizer.json` `Split` regex is `pattern`
    /// (the form llama.cpp quotes above each of its custom splitters).
    #[must_use]
    pub fn from_hf_regex(pattern: &str) -> Option<Self> {
        const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        match pattern {
            QWEN2 => Some(Self::Qwen2),
            QWEN35 => Some(Self::Qwen35),
            _ => None,
        }
    }

    /// Split `text` into pre-tokens: a line-for-line port of llama.cpp's
    /// `unicode_regex_split_custom_qwen2` (and `_qwen35`, which differs only in treating
    /// `\p{M}` as part of a letter run). Its regex, from tokenizer.json:
    ///
    /// ```text
    /// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    /// ```
    #[must_use]
    pub fn split(self, text: &str) -> Vec<&str> {
        let sp = Splitter {
            chars: text.char_indices().collect(),
            marks: self == Self::Qwen35,
        };
        let byte_at = |i: usize| sp.chars.get(i).map_or(text.len(), |&(b, _)| b);
        let mut pieces = Vec::new();
        let mut pos = 0usize;
        while pos < sp.chars.len() {
            let end = sp.next_end(pos);
            pieces.push(&text[byte_at(pos)..byte_at(end)]);
            pos = end;
        }
        pieces
    }
}

/// One text's codepoints, and the regex alternatives of llama.cpp's Qwen splitter as methods
/// tried in the splitter's order. Each returns where its match ends, or `None`.
struct Splitter {
    chars: Vec<(usize, char)>,
    /// `qwen35`: `\p{M}` belongs to letter runs.
    marks: bool,
}

impl Splitter {
    fn cpt(&self, i: usize) -> Option<char> {
        self.chars.get(i).map(|&(_, c)| c)
    }

    /// `None` is llama.cpp's out-of-range `unicode_cpt_flags{}`: every flag false and
    /// `as_uint() == 0`. Every in-range codepoint has a nonzero flag word.
    fn flags(&self, i: usize) -> Option<CptFlags> {
        self.cpt(i).map(CptFlags::of)
    }

    fn is_word(&self, f: CptFlags) -> bool {
        f.letter || (self.marks && f.mark)
    }

    fn is_other(&self, f: CptFlags) -> bool {
        !(f.whitespace || f.letter || f.number || (self.marks && f.mark))
    }

    fn word_at(&self, i: usize) -> bool {
        self.flags(i).is_some_and(|f| self.is_word(f))
    }

    fn other_at(&self, i: usize) -> bool {
        self.flags(i).is_some_and(|f| self.is_other(f))
    }

    /// The end of the piece starting at `pos` (always past `pos`).
    fn next_end(&self, pos: usize) -> usize {
        self.contraction(pos)
            .or_else(|| self.letters(pos))
            .or_else(|| self.number(pos))
            .or_else(|| self.punctuation(pos))
            .unwrap_or_else(|| self.whitespace(pos))
    }

    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
    fn contraction(&self, pos: usize) -> Option<usize> {
        if self.cpt(pos) != Some('\'') {
            return None;
        }
        let c1 = self.cpt(pos + 1)?.to_ascii_lowercase();
        if matches!(c1, 's' | 't' | 'm' | 'd') {
            return Some(pos + 2);
        }
        let c2 = self.cpt(pos + 2)?.to_ascii_lowercase();
        matches!((c1, c2), ('r' | 'v', 'e') | ('l', 'l')).then_some(pos + 3)
    }

    /// `[^\r\n\p{L}\p{N}]?\p{L}+` (qwen35: `[\p{L}\p{M}]+`)
    fn letters(&self, pos: usize) -> Option<usize> {
        let c = self.cpt(pos)?;
        let f = CptFlags::of(c);
        if c == '\r' || c == '\n' || f.number || !(self.is_word(f) || self.word_at(pos + 1)) {
            return None;
        }
        let mut end = pos + 1;
        while self.word_at(end) {
            end += 1;
        }
        Some(end)
    }

    /// `\p{N}`: one digit per piece.
    fn number(&self, pos: usize) -> Option<usize> {
        self.flags(pos)?.number.then_some(pos + 1)
    }

    /// `<space>?[^\s\p{L}\p{N}]+[\r\n]*` (qwen35 also excludes `\p{M}`). llama.cpp tests the
    /// NEXT codepoint when this one is a space, and an out-of-range next codepoint passes.
    fn punctuation(&self, pos: usize) -> Option<usize> {
        let space = self.cpt(pos)? == ' ';
        let probe = if space { pos + 1 } else { pos };
        if self.flags(probe).is_some_and(|f| !self.is_other(f)) {
            return None;
        }
        let mut end = probe;
        while self.other_at(end) {
            end += 1;
        }
        while matches!(self.cpt(end), Some('\r' | '\n')) {
            end += 1;
        }
        Some(end)
    }

    /// `\s*[\r\n]+`, then `\s+(?!\S)`, then `\s+`, then one unmatched codepoint.
    fn whitespace(&self, pos: usize) -> usize {
        let mut run = 0usize;
        let mut last_newline_end = None;
        while self.flags(pos + run).is_some_and(|f| f.whitespace) {
            if matches!(self.cpt(pos + run), Some('\r' | '\n')) {
                last_newline_end = Some(pos + run + 1);
            }
            run += 1;
        }
        match last_newline_end {
            Some(end) => end,
            // a run followed by more text gives its last codepoint to that text
            None if run > 1 && pos + run < self.chars.len() => pos + run - 1,
            None => pos + run.max(1),
        }
    }
}

/// The subset of llama.cpp's `unicode_cpt_flags` the Qwen splitters read.
#[derive(Debug, Clone, Copy)]
struct CptFlags {
    /// `\p{L}`
    letter: bool,
    /// `\p{N}`
    number: bool,
    /// `\p{M}`
    mark: bool,
    /// `\s`: llama.cpp's `unicode_set_whitespace` is exactly Unicode `White_Space`, which is
    /// what `char::is_whitespace` tests.
    whitespace: bool,
}

impl CptFlags {
    fn of(c: char) -> Self {
        let group = c.general_category_group();
        Self {
            letter: group == GeneralCategoryGroup::Letter,
            number: group == GeneralCategoryGroup::Number,
            mark: group == GeneralCategoryGroup::Mark,
            whitespace: c.is_whitespace(),
        }
    }
}

/// GPT-2's `bytes_to_unicode`: the glyph each byte is spelled with in a byte-level vocabulary.
/// Printable Latin-1 (`!`..`~`, `¡`..`¬`, `®`..`ÿ`) maps to itself; the other 68 bytes map to
/// U+0100 onward, in byte order.
fn byte_glyphs() -> &'static [char; 256] {
    static GLYPHS: OnceLock<[char; 256]> = OnceLock::new();
    GLYPHS.get_or_init(|| {
        let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
        let mut glyphs = ['\0'; 256];
        let mut next = 0u32;
        for b in 0..=255u8 {
            glyphs[usize::from(b)] = if printable(b) {
                char::from(b)
            } else {
                next += 1;
                char::from_u32(255 + next).expect("U+0100..U+0143 are valid scalars")
            };
        }
        glyphs
    })
}

/// A byte-level BPE tokenizer built from a GGUF's vocabulary, merges, token types and
/// pre-tokenizer name.
#[derive(Debug)]
pub struct ByteLevelBpe {
    pre: PreTokenizer,
    token_to_id: HashMap<String, u32>,
    /// id -> token text, for [`ByteLevelBpe::decode`] (#3742).
    id_to_token: Vec<String>,
    /// `(left, right) -> (rank, merged)`, from `tokenizer.ggml.merges` in file order.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// The 256 byte glyphs' ids, by byte.
    byte_ids: [Option<u32>; 256],
    /// Special tokens, longest text first.
    specials: Vec<(String, u32)>,
}

/// Why a GGUF vocabulary cannot be encoded by [`ByteLevelBpe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ByteLevelBpeRefusal {
    /// `tokenizer.ggml.pre` names a pre-tokenizer this module does not implement.
    UnknownPreTokenizer(String),
    /// The file has no `tokenizer.ggml.pre`.
    MissingPreTokenizer,
    /// The file has no `tokenizer.ggml.merges`.
    MissingMerges,
}

impl std::fmt::Display for ByteLevelBpeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPreTokenizer(p) => write!(f, "pre-tokenizer '{p}' is not implemented"),
            Self::MissingPreTokenizer => write!(f, "the file has no tokenizer.ggml.pre"),
            Self::MissingMerges => write!(f, "the file has no tokenizer.ggml.merges"),
        }
    }
}

fn string_array<'a>(
    metadata: &'a HashMap<String, GGUFValue>,
    key: &str,
) -> Option<impl Iterator<Item = &'a str>> {
    match metadata.get(key) {
        Some(GGUFValue::Array(values)) => Some(values.iter().filter_map(|v| match v {
            GGUFValue::String(s) => Some(s.as_str()),
            _ => None,
        })),
        _ => None,
    }
}

impl ByteLevelBpe {
    /// Build the tokenizer for a `gpt2`-model GGUF. The build is cached per distinct
    /// (vocabulary, merges, token types, pre-tokenizer), so repeated calls on one model pay
    /// for one hash of the tables, not for rebuilding them.
    ///
    /// # Errors
    /// [`ByteLevelBpeRefusal`] when the file lacks merges or a pre-tokenizer this module
    /// implements; the caller decides what to do instead.
    pub fn from_gguf(
        metadata: &HashMap<String, GGUFValue>,
        vocab: &[String],
    ) -> Result<Arc<Self>, ByteLevelBpeRefusal> {
        let pre_name = match metadata.get("tokenizer.ggml.pre") {
            Some(GGUFValue::String(s)) => s.as_str(),
            _ => return Err(ByteLevelBpeRefusal::MissingPreTokenizer),
        };
        let pre = PreTokenizer::from_gguf_pre(pre_name)
            .ok_or_else(|| ByteLevelBpeRefusal::UnknownPreTokenizer(pre_name.to_string()))?;
        let merges: Vec<&str> = string_array(metadata, "tokenizer.ggml.merges")
            .ok_or(ByteLevelBpeRefusal::MissingMerges)?
            .collect();
        let token_types: Vec<i32> = match metadata.get("tokenizer.ggml.token_type") {
            Some(GGUFValue::Array(values)) => values
                .iter()
                .map(|v| match v {
                    GGUFValue::Int32(t) => *t,
                    _ => 1,
                })
                .collect(),
            _ => Vec::new(),
        };

        Ok(Self::cached(pre, vocab, &merges, &token_types))
    }

    /// Build, or reuse, the tokenizer for these exact tables. The build is cached per distinct
    /// (pre-tokenizer, vocabulary, merges, token types), so repeated calls on one model pay
    /// for one hash of the tables, not for rebuilding them.
    #[must_use]
    pub fn cached(
        pre: PreTokenizer,
        vocab: &[String],
        merges: &[&str],
        token_types: &[i32],
    ) -> Arc<Self> {
        let mut hasher = DefaultHasher::new();
        pre.hash(&mut hasher);
        vocab.hash(&mut hasher);
        merges.hash(&mut hasher);
        token_types.hash(&mut hasher);
        let key = hasher.finish();

        static CACHE: OnceLock<Mutex<Vec<(u64, Arc<ByteLevelBpe>)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        if let Some(hit) = cache.lock().ok().and_then(|c| {
            c.iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| Arc::clone(v))
        }) {
            return hit;
        }
        let built = Arc::new(Self::build(pre, vocab, merges, token_types));
        if let Ok(mut c) = cache.lock() {
            if c.len() >= 8 {
                c.remove(0);
            }
            c.push((key, Arc::clone(&built)));
        }
        built
    }

    /// #3742: an identity for a (vocabulary, merges) pair, the two tables that decide byte-level
    /// BPE ids. Two files with the same fingerprint tokenize every text identically, whatever
    /// their weights: the tokenizer-parity gate uses it to find, for an `.apr`, a GGUF whose
    /// `llama-tokenize` ids are the reference.
    #[must_use]
    pub fn tables_fingerprint(vocab: &[String], merges: &[&str]) -> u64 {
        let mut hasher = DefaultHasher::new();
        vocab.hash(&mut hasher);
        merges.hash(&mut hasher);
        hasher.finish()
    }

    /// Build from explicit tables (no cache).
    #[must_use]
    pub fn build(
        pre: PreTokenizer,
        vocab: &[String],
        merges: &[&str],
        token_types: &[i32],
    ) -> Self {
        let token_to_id: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();

        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, rule) in merges.iter().enumerate() {
            let Some((left, right)) = rule.split_once(' ') else {
                continue;
            };
            let (Some(&l), Some(&r), Some(&m)) = (
                token_to_id.get(left),
                token_to_id.get(right),
                token_to_id.get(format!("{left}{right}").as_str()),
            ) else {
                continue;
            };
            // The first (lowest-rank) occurrence of a pair wins, as a rank lookup would.
            merge_map.entry((l, r)).or_insert((rank as u32, m));
        }

        let glyphs = byte_glyphs();
        let mut byte_ids = [None; 256];
        for (b, id) in byte_ids.iter_mut().enumerate() {
            *id = token_to_id.get(glyphs[b].to_string().as_str()).copied();
        }

        let mut specials: Vec<(String, u32)> = if token_types.len() == vocab.len() {
            vocab
                .iter()
                .zip(token_types)
                .enumerate()
                .filter(|(_, (_, t))| SPECIAL_TOKEN_TYPES.contains(t))
                .map(|(id, (text, _))| (text.clone(), id as u32))
                .collect()
        } else {
            // #3742: no usable token types (an `.apr` converted before they were written). In a
            // BPE vocabulary every ordinary token is a byte glyph or a merge's output; an added
            // token is neither. Restricted to `<...>`-shaped text, that rule reproduces
            // llama.cpp's CONTROL + USER_DEFINED set exactly on every Qwen family in the fleet
            // (33/22/26/26 tokens, none missing, none extra; without the shape test Qwen3.5
            // would also take 201 ordinary CJK tokens that no merge produces).
            let produced: std::collections::HashSet<String> = merges
                .iter()
                .filter_map(|m| m.split_once(' ').map(|(l, r)| format!("{l}{r}")))
                .collect();
            vocab
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    t.chars().count() > 2
                        && t.starts_with('<')
                        && t.ends_with('>')
                        && !produced.contains(t.as_str())
                })
                .map(|(id, t)| (t.clone(), id as u32))
                .collect()
        };
        specials.retain(|(t, _)| !t.is_empty());
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        Self {
            pre,
            token_to_id,
            id_to_token: vocab.to_vec(),
            merges: merge_map,
            byte_ids,
            specials,
        }
    }

    /// Encode `text` into token ids. Special tokens in the text are matched first, as
    /// llama.cpp does with `parse_special = true`; no BOS is added (the caller owns that).
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for fragment in self.partition_specials(text) {
            match fragment {
                Fragment::Special(id) => ids.push(id),
                Fragment::Raw(raw) => {
                    for piece in self.pre.split(raw) {
                        self.encode_piece(piece, &mut ids);
                    }
                },
            }
        }
        ids
    }

    /// Decode ids back to text: each ordinary token's glyphs map back to their bytes (the
    /// inverse of GPT-2's `bytes_to_unicode`); a special token is its own text, as llama.cpp
    /// renders it. `decode(encode(x)) == x` for any `x` (#3742: the `.apr` decoder this
    /// replaces lost non-ASCII bytes and whitespace glyphs).
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        static GLYPH_TO_BYTE: OnceLock<HashMap<char, u8>> = OnceLock::new();
        let glyph_to_byte = GLYPH_TO_BYTE.get_or_init(|| {
            byte_glyphs()
                .iter()
                .enumerate()
                .map(|(b, &g)| (g, b as u8))
                .collect()
        });
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(token) = self.id_to_token.get(id as usize) else {
                continue;
            };
            if self
                .specials
                .iter()
                .any(|(t, sid)| *sid == id && t == token)
            {
                bytes.extend_from_slice(token.as_bytes());
                continue;
            }
            for c in token.chars() {
                match glyph_to_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    },
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// `tokenizer_st_partition`: each special token, longest first, claims its occurrences
    /// in the raw fragments that remain.
    fn partition_specials<'t>(&self, text: &'t str) -> Vec<Fragment<'t>> {
        let mut fragments = vec![Fragment::Raw(text)];
        for (special, id) in &self.specials {
            if !text.contains(special.as_str()) {
                continue;
            }
            let mut next = Vec::with_capacity(fragments.len());
            for fragment in fragments {
                let Fragment::Raw(mut raw) = fragment else {
                    next.push(fragment);
                    continue;
                };
                while let Some(at) = raw.find(special.as_str()) {
                    if at > 0 {
                        next.push(Fragment::Raw(&raw[..at]));
                    }
                    next.push(Fragment::Special(*id));
                    raw = &raw[at + special.len()..];
                }
                if !raw.is_empty() {
                    next.push(Fragment::Raw(raw));
                }
            }
            fragments = next;
        }
        fragments
    }

    /// BPE over one pre-token's byte glyphs: a port of `llm_tokenizer_bpe_session`'s merge
    /// loop. Lowest rank first; among equal ranks, the leftmost pair.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let mut syms = self.seed_symbols(piece);
        if syms.is_empty() {
            return;
        }
        self.merge_all(&mut syms);
        let mut at = Some(0);
        while let Some(i) = at {
            if syms[i].id != u32::MAX {
                out.push(syms[i].id);
            }
            at = syms[i].next;
        }
    }

    /// One symbol per byte, spelled by its glyph.
    fn seed_symbols(&self, piece: &str) -> Vec<Sym> {
        let glyphs = byte_glyphs();
        piece
            .bytes()
            .enumerate()
            .map(|(i, b)| Sym {
                // A byte-level vocabulary carries all 256 glyphs; one that does not is
                // looked up by glyph text, and a miss stays visible as u32::MAX (never 0).
                id: self.byte_ids[usize::from(b)].unwrap_or_else(|| {
                    self.token_to_id
                        .get(glyphs[usize::from(b)].to_string().as_str())
                        .copied()
                        .unwrap_or(u32::MAX)
                }),
                prev: i.checked_sub(1),
                next: Some(i + 1).filter(|&j| j < piece.len()),
                alive: true,
            })
            .collect()
    }

    /// The merge loop: pop the best pending pair, apply it if it is still current, and queue
    /// the two pairs it creates with its neighbours.
    fn merge_all(&self, syms: &mut [Sym]) {
        let mut queue = BinaryHeap::new();
        for left in 0..syms.len() - 1 {
            self.enqueue(syms, left, left + 1, &mut queue);
        }
        while let Some(pending) = queue.pop() {
            if !Self::apply(syms, &pending) {
                continue;
            }
            if let Some(p) = syms[pending.left].prev {
                self.enqueue(syms, p, pending.left, &mut queue);
            }
            if let Some(nn) = syms[pending.left].next {
                self.enqueue(syms, pending.left, nn, &mut queue);
            }
        }
    }

    /// Merge `pending.right` into `pending.left` unless either side changed since the pair
    /// was queued (llama.cpp's stale-bigram check). Returns whether it merged.
    fn apply(syms: &mut [Sym], pending: &Pending) -> bool {
        let (l, r) = (syms[pending.left], syms[pending.right]);
        let current = l.alive
            && r.alive
            && l.next == Some(pending.right)
            && l.id == pending.left_id
            && r.id == pending.right_id;
        if !current {
            return false;
        }
        syms[pending.left].id = pending.merged;
        syms[pending.left].next = r.next;
        syms[pending.right].alive = false;
        if let Some(nn) = r.next {
            syms[nn].prev = Some(pending.left);
        }
        true
    }

    fn enqueue(&self, syms: &[Sym], left: usize, right: usize, queue: &mut BinaryHeap<Pending>) {
        let (left_id, right_id) = (syms[left].id, syms[right].id);
        if let Some(&(rank, merged)) = self.merges.get(&(left_id, right_id)) {
            queue.push(Pending {
                rank,
                left,
                right,
                left_id,
                right_id,
                merged,
            });
        }
    }
}

enum Fragment<'t> {
    Raw(&'t str),
    Special(u32),
}

#[derive(Debug, Clone, Copy)]
struct Sym {
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

/// A candidate merge. Ordered so that `BinaryHeap` (a max-heap) pops the LOWEST rank first
/// and, among equal ranks, the LEFTMOST pair: llama.cpp's `llm_bigram_bpe::comparator`.
#[derive(Debug, PartialEq, Eq)]
struct Pending {
    rank: u32,
    left: usize,
    right: usize,
    left_id: u32,
    right_id: u32,
    merged: u32,
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
#[path = "byte_level_bpe_tests.rs"]
mod tests;
