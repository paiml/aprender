//! APR-PERF-GATE-001 v2.2 §4.4.6 — `method: client_tokenizer`, made real.
//!
//! # The defect this closes
//!
//! §4.4.6 offers two counting methods and the repo could only produce one of
//! them honestly. `server_usage` takes the numbers out of the server's own
//! `usage` block — and two servers' `usage` blocks are two implementations'
//! opinions. Measured on one W1 corpus, one model file and one canonical
//! tokenizer, with both servers deterministic across every replicate:
//!
//! | counted by                    | `prompt_tokens` |
//! |-------------------------------|-----------------|
//! | the raw prompt text, no template | **505**      |
//! | `apr serve`'s `usage`            | **513**      |
//! | `llama-server`'s `usage`         | **534**      |
//!
//! Those three were measured on the corpus as it stood at `_meta.body_words =
//! 496`; the raw count is **512** now, because measuring it was also the first
//! thing that could see it sitting one token above §4.3.1's floor and the
//! corpus was retuned (see [`W1_RAW_PROMPT_TOKENS`](tokenizer_tests)). The
//! deltas are template overhead, not corpus length — `apr`'s 8-token ChatML
//! wrapper, `llama`'s embedded jinja template with Qwen's default system
//! message — so retuning moves all three together and changes nothing about the
//! argument below.
//!
//! The two servers never agree, the gap is not a constant offset and not a
//! constant ratio — it was −18, −21 and **+19** on four differently shaped
//! prompts, so its SIGN flips — and every token in that gap is prefill work one
//! side does and the other does not. A `tok/s` ratio built from two servers'
//! `usage` blocks divides two different numerators.
//!
//! `client_tokenizer` is the way out: **one** tokenizer, run **client-side**,
//! applied identically to both lanes, counting the text that actually went on
//! the wire and the text that actually came back. Before this module there was
//! no client-side counter anywhere in the crate — `git grep 'Tokenizer::from_file'`
//! over `src/llm/` and `src/perf_gate/` returned nothing — so `client_tokenizer`
//! was a string an operator could type on the command line and a digest they
//! could paste, validated only as *64 lowercase hex characters*.
//!
//! # The digest is COMPUTED, never declared
//!
//! [`ClientTokenizer::from_file`] reads the file and hashes **the bytes it just
//! read**. The pair is set from one expression, so a `ClientTokenizer` cannot
//! carry a digest belonging to a file it did not open, and
//! [`TokenAccounting::client_tokenizer`] derives the §4.4.6 block *from* the
//! counter rather than accepting one alongside it. `--tokenizer-sha256` is
//! consequently demoted from a declaration to an ASSERTION
//! ([`ClientTokenizer::assert_digest`]): it can refuse a run, it can no longer
//! supply a value.
//!
//! This follows `aprender-core`'s SetFit boundary
//! (`crates/aprender-core/src/setfit/tokenizer.rs`), which seals the same pair
//! for the same reason. What is deliberately NOT copied is the retention of
//! `source_bytes`: SetFit keeps them because a `.apr` artifact must be able to
//! REBUILD its tokenizer. A perf receipt only has to be able to detect a
//! substituted one, and a 7 MB buffer per band would be carried for nothing.
//!
//! # Truncation and padding are cleared, and that is load-bearing
//!
//! A `tokenizer.json` may declare `truncation` and `padding` in the file, and
//! `tokenizers` honours them. The checked-in fixture at
//! `crates/aprender-core/tests/fixtures/setfit/tokenizer.json` declares
//! `truncation.max_length = 128` and `padding = Fixed(128)`, so a counter that
//! merely opened it and called `encode` would report **128** for `""`, **128**
//! for `"hello world"` and **128** for a 568-token document: a constant, shaped
//! exactly like a measurement. Qwen's `tokenizer.json` declares neither, so this
//! defect would have been invisible on the model the campaign actually measures
//! and would have surfaced the first time somebody pointed the flag at a
//! tokenizer that does. Both are cleared in [`ClientTokenizer::from_bytes`] and
//! `padding_and_truncation_declared_in_the_file_do_not_become_the_count` fails
//! if either clear is removed.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::receipt::TokenizationBlock;

/// §4.4.6 — whether a [`ClientTokenizer`] count includes special tokens.
///
/// A FACT of this counter, not a declaration the operator makes about it.
/// [`ClientTokenizer::count`] encodes with `add_special_tokens = false`, so the
/// count is of the text itself and of nothing the template or the post-processor
/// would have added. That is what makes it comparable across two servers whose
/// templates differ — which, measured, they do.
pub const COUNTS_SPECIAL_TOKENS: bool = false;

/// §4.4.6 — whether a [`ClientTokenizer`] completion count includes the echoed
/// prompt. It does not: the prompt and the completion are counted from separate
/// strings.
pub const COUNTS_PROMPT_ECHO: bool = false;

/// Everything that can go wrong between a path and a token count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// The tokenizer file could not be read.
    Io {
        /// The path that was attempted.
        path: PathBuf,
        /// The OS error.
        reason: String,
    },
    /// The bytes are not a parseable `tokenizers` serialization, or truncation
    /// and padding could not be cleared from it.
    Load {
        /// Where the bytes came from.
        origin: String,
        /// What the library said.
        reason: String,
    },
    /// The tokenizer refused to encode a string.
    Encode {
        /// Where the tokenizer came from.
        origin: String,
        /// What the library said.
        reason: String,
    },
    /// `--tokenizer-sha256` did not match the digest of the file that was opened.
    DigestMismatch {
        /// What the operator asserted.
        expected: String,
        /// What the opened bytes actually hash to.
        computed: String,
        /// Which file was opened.
        origin: String,
    },
    /// `--tokenizer-sha256` was not 64 lowercase hex characters, so it could not
    /// be an assertion about anything.
    NotADigest(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, reason } => {
                write!(f, "cannot read tokenizer {}: {reason}", path.display())
            }
            Self::Load { origin, reason } => {
                write!(f, "cannot load tokenizer from {origin}: {reason}")
            }
            Self::Encode { origin, reason } => {
                write!(f, "tokenizer {origin} failed to encode: {reason}")
            }
            Self::DigestMismatch {
                expected,
                computed,
                origin,
            } => write!(
                f,
                "--tokenizer-sha256 {expected} does not match {origin}, which hashes to \
                 {computed}. §4.4.6's digest is COMPUTED from the file the run opens; the flag \
                 asserts it and cannot supply it."
            ),
            Self::NotADigest(s) => write!(
                f,
                "--tokenizer-sha256 {s:?} is not 64 lowercase hex characters, so it asserts nothing"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Lowercase-hex sha256 of a byte slice.
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// True for exactly 64 lowercase hex characters.
fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The model's own tokenizer, run client-side, counting both directions.
///
/// Built only from bytes this type read itself, and stamped with the digest of
/// exactly those bytes.
pub struct ClientTokenizer {
    /// Truncation and padding cleared — see the module docs.
    inner: tokenizers::Tokenizer,
    /// Where the bytes came from, for error messages and the operator's log.
    origin: String,
    /// How many bytes were hashed. A second, cheap identity signal: two
    /// serializations of the same vocabulary differ in length as well as digest.
    source_len: usize,
    /// Lowercase-hex sha256 of the bytes this tokenizer was built from.
    tokenizer_sha256: String,
}

impl std::fmt::Debug for ClientTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientTokenizer")
            .field("origin", &self.origin)
            .field("source_len", &self.source_len)
            .field("tokenizer_sha256", &self.tokenizer_sha256)
            .finish()
    }
}

impl ClientTokenizer {
    /// Open `path`, hash the bytes read, and build a counter over them.
    ///
    /// # Errors
    /// [`TokenizerError::Io`] when the file cannot be read;
    /// [`TokenizerError::Load`] when it is not a tokenizer.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let bytes = std::fs::read(path).map_err(|e| TokenizerError::Io {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;
        // The digest below is taken over `bytes` -- the buffer this call just
        // read -- so "the digest of the file the run opened" is true by
        // construction rather than by discipline.
        Self::from_bytes(&bytes, path.display().to_string())
    }

    /// Build a counter over `bytes`, stamped with their digest.
    ///
    /// # Errors
    /// [`TokenizerError::Load`] when the bytes are not a parseable tokenizer or
    /// truncation/padding cannot be cleared.
    pub fn from_bytes(bytes: &[u8], origin: impl Into<String>) -> Result<Self, TokenizerError> {
        let origin = origin.into();
        let mut inner =
            tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| TokenizerError::Load {
                origin: origin.clone(),
                reason: e.to_string(),
            })?;
        // LOAD-BEARING. A `tokenizer.json` that declares padding turns every
        // count into the padded width -- a constant that looks like a
        // measurement. See the module docs and the test named after this.
        inner
            .with_truncation(None)
            .map_err(|e| TokenizerError::Load {
                origin: origin.clone(),
                reason: format!("cannot clear the file's truncation: {e}"),
            })?;
        inner.with_padding(None);

        Ok(Self {
            inner,
            origin,
            source_len: bytes.len(),
            tokenizer_sha256: sha256_hex(bytes),
        })
    }

    /// Lowercase-hex sha256 of the bytes this counter was built from.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// Where those bytes came from.
    #[must_use]
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// How many bytes were hashed.
    #[must_use]
    pub fn source_len(&self) -> usize {
        self.source_len
    }

    /// Count the tokens in `text`, special tokens excluded.
    ///
    /// # Errors
    /// [`TokenizerError::Encode`] if the tokenizer refuses the string.
    pub fn count(&self, text: &str) -> Result<u32, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, COUNTS_SPECIAL_TOKENS)
            .map_err(|e| TokenizerError::Encode {
                origin: self.origin.clone(),
                reason: e.to_string(),
            })?;
        // A count wider than u32 is not reachable for any string a request
        // carries, and saturating is preferable to a panic in a measurement
        // path: the sample would be absurd and visible rather than fatal.
        Ok(u32::try_from(encoding.get_ids().len()).unwrap_or(u32::MAX))
    }

    /// Refuse the run when the operator's asserted digest is not this file's.
    ///
    /// This is the whole of `--tokenizer-sha256`'s new job. It used to be the
    /// only source of the receipt's digest, validated as 64 lowercase hex and
    /// nothing more, which is why it was free text.
    ///
    /// # Errors
    /// [`TokenizerError::NotADigest`] when `expected` is not 64 lowercase hex;
    /// [`TokenizerError::DigestMismatch`] when it is a digest and is the wrong one.
    pub fn assert_digest(&self, expected: &str) -> Result<(), TokenizerError> {
        if !is_sha256(expected) {
            return Err(TokenizerError::NotADigest(expected.to_string()));
        }
        if expected != self.tokenizer_sha256 {
            return Err(TokenizerError::DigestMismatch {
                expected: expected.to_string(),
                computed: self.tokenizer_sha256.clone(),
                origin: self.origin.clone(),
            });
        }
        Ok(())
    }

    /// The §4.4.6 block this counter — and only this counter — can produce.
    ///
    /// The digest comes from `self`, and `counts_special_tokens` /
    /// `counts_prompt_echo` are the counter's FACTS. Nothing here is a parameter,
    /// which is the point: the three fields that made up §4.4.6's declaration
    /// are now all derived from the file that was opened.
    #[must_use]
    pub fn tokenization_block(&self) -> TokenizationBlock {
        TokenizationBlock::ClientTokenizer {
            tokenizer_sha256: self.tokenizer_sha256.clone(),
            counts_special_tokens: COUNTS_SPECIAL_TOKENS,
            counts_prompt_echo: COUNTS_PROMPT_ECHO,
        }
    }
}

/// A §4.4.6 block and the counter that is allowed to have produced it.
///
/// The two travel together so a band cannot be handed a `client_tokenizer`
/// declaration with no counter behind it, nor a counter whose digest differs
/// from the one the receipt will carry. [`TokenizationBlock::require_counter`]
/// is the rule; this type is what makes it unavoidable, because `run_band` takes
/// a `TokenAccounting` rather than a bare block.
#[derive(Debug, Clone)]
pub struct TokenAccounting {
    block: TokenizationBlock,
    counter: Option<std::sync::Arc<ClientTokenizer>>,
}

impl TokenAccounting {
    /// Counts taken from the server's own `usage` fields.
    ///
    /// The two booleans stay parameters here because under `server_usage` they
    /// really are operator declarations about a remote implementation's
    /// semantics — §4.4.6 is titled "Token counting must be **declared**" and
    /// this is the branch where that is all anyone can do.
    #[must_use]
    pub fn server_usage(counts_special_tokens: bool, counts_prompt_echo: bool) -> Self {
        Self {
            block: TokenizationBlock::ServerUsage {
                counts_special_tokens,
                counts_prompt_echo,
            },
            counter: None,
        }
    }

    /// Counts computed client-side, with the block DERIVED from the counter.
    ///
    /// **This derivation, and not any guard, is what makes the receipt's §4.4.6
    /// digest honest.** There is no parameter through which a borrowed digest
    /// could enter: the block is computed from the counter that will do the
    /// counting, so a `client_tokenizer` band carrying a digest for a file it
    /// never opened is not a state this constructor can produce.
    ///
    /// That is worth stating precisely, because it is easy to credit the wrong
    /// mechanism. [`TokenizationBlock::require_counter`]'s digest-comparison arm
    /// is a real rule and it is exercised by tests, but on the CLI path it
    /// cannot fail: every production `TokenAccounting` comes from this
    /// constructor or from [`Self::server_usage`], and both are self-consistent
    /// by construction. The arm guards [`Self::from_parts`], which today has no
    /// caller outside `#[cfg(test)]`. Derivation is a poka-yoke; the guard is
    /// the backstop behind it for a caller that does not yet exist. Adding a
    /// contrived production caller to make the guard reachable would be
    /// inspection theatre standing where a mistake-proof already is.
    #[must_use]
    pub fn client_tokenizer(counter: ClientTokenizer) -> Self {
        Self {
            block: counter.tokenization_block(),
            counter: Some(std::sync::Arc::new(counter)),
        }
    }

    /// Pair a block that came from somewhere else with a counter, checking them
    /// against each other.
    ///
    /// # It has no non-test caller, and that is the point
    ///
    /// `git grep 'TokenAccounting::from_parts'` finds this definition and six
    /// call sites, all inside `#[cfg(test)]`. Every production `TokenAccounting`
    /// is built by [`Self::client_tokenizer`] or [`Self::server_usage`], neither
    /// of which can produce a mismatched pair, so the checks below are a
    /// backstop for a boundary nothing crosses yet rather than a guard standing
    /// between the CLI and a defect.
    ///
    /// It is kept, and kept checked, because the boundary is one a receipt
    /// reader will eventually want — reconstructing a `TokenAccounting` from a
    /// receipt's own §4.4.6 block plus a locally opened tokenizer is exactly the
    /// shape of "verify this receipt", and that caller must not be able to pair
    /// a declared digest with a different file. What it is NOT is the reason the
    /// producer is honest today; see [`Self::client_tokenizer`].
    ///
    /// # Errors
    /// Whatever [`TokenizationBlock::require_counter`] refuses, plus
    /// [`TokenizationBlock::validate`].
    pub fn from_parts(
        block: TokenizationBlock,
        counter: Option<ClientTokenizer>,
    ) -> Result<Self, String> {
        block.validate()?;
        block.require_counter(counter.as_ref().map(ClientTokenizer::tokenizer_sha256))?;
        Ok(Self {
            block,
            counter: counter.map(std::sync::Arc::new),
        })
    }

    /// The §4.4.6 block for the receipt.
    #[must_use]
    pub fn block(&self) -> &TokenizationBlock {
        &self.block
    }

    /// The client-side counter, when the declared method is `client_tokenizer`.
    #[must_use]
    pub fn counter(&self) -> Option<&ClientTokenizer> {
        self.counter.as_deref()
    }

    /// A shared handle to the counter, for spawned workers.
    ///
    /// `tokenizers::Tokenizer` is `Send + Sync`, so one counter serves every
    /// worker in a band; cloning it per worker would duplicate a 151k-entry
    /// vocabulary `c` times for no gain.
    #[must_use]
    pub fn counter_handle(&self) -> Option<std::sync::Arc<ClientTokenizer>> {
        self.counter.clone()
    }

    /// Re-check the pair. Cheap, and run again at the top of every band so a
    /// `TokenAccounting` that crossed a crate boundary is checked where it is
    /// spent, not only where it was built.
    ///
    /// This call — `PromptCounts::build`'s first statement — is on the real
    /// path, but under the two derived constructors it **cannot refuse**. A
    /// three-point mutation lattice, run against `--features llm --lib`:
    ///
    /// | mutation | result |
    /// |---|---|
    /// | delete this `validate()?` call | **GREEN, 6616/6616** |
    /// | break [`Self::client_tokenizer`]'s derivation (emit a constant digest) | RED, 3 |
    /// | both together | RED, **2** |
    ///
    /// The third row is the interesting one: `an_invalid_tokenization_block_
    /// refuses_the_run` goes green again when this call is removed, so this
    /// call is the sole catcher of exactly one test, and only in the presence of
    /// a simultaneous defect in the derivation. It is a cheap re-assertion at
    /// the point of spend, not the mechanism that keeps the digest true.
    /// [`Self::client_tokenizer`]'s derivation is.
    ///
    /// # Errors
    /// Whatever `validate` or `require_counter` refuses.
    pub fn validate(&self) -> Result<(), String> {
        self.block.validate()?;
        self.block.require_counter(
            self.counter
                .as_deref()
                .map(ClientTokenizer::tokenizer_sha256),
        )
    }
}

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tokenizer_tests;
