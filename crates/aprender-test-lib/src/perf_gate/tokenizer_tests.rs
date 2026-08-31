//! Tests for §4.4.6's client-side token counter.
//!
//! # What runs where
//!
//! Every test in the first two sections runs unconditionally, against a real
//! HuggingFace tokenizer already committed to this repository
//! (`crates/aprender-core/tests/fixtures/setfit/tokenizer.json`, the pinned
//! all-MiniLM-L6-v2 WordPiece vocabulary). Real load, real digest, real counts,
//! real guard — nothing is stubbed.
//!
//! [`the_w1_corpus_counts_512_tokens_per_prompt`] is the exception. It needs
//! Qwen2.5-Coder's 7 MB `tokenizer.json`, which is not in this repository and is
//! not present on two of the four hosts in the fleet, so it is reached through
//! `APR_PERF_GATE_TOKENIZER=<path>`. It is stated here, rather than left to a
//! comment, that this test does **not** run in CI: it is the one test that pins
//! the counter to the number the campaign actually measured, and if you are
//! changing anything in `tokenizer.rs` you are expected to run it. Setting
//! `APR_PERF_GATE_REQUIRE_W1_TOKENIZER=1` turns "no tokenizer" from a skip into
//! a failure, which is how a host that owns the file wires it into a lane.

use super::*;
use crate::perf_gate::receipt::TokenCountingMethod;
use std::path::PathBuf;

/// `crates/aprender-test-lib` — the root every fixture path below is relative to.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The pinned all-MiniLM-L6-v2 `tokenizer.json` already committed for SetFit.
///
/// Borrowed rather than duplicated, and chosen over the other three
/// `tokenizer.json` files in the tree for one specific reason: this one
/// **declares `truncation.max_length = 128` and `padding = Fixed(128)` in the
/// file**, so it is the only in-tree fixture that can catch a counter which
/// forgot to clear them.
fn minilm_path() -> PathBuf {
    manifest_dir().join("../aprender-core/tests/fixtures/setfit/tokenizer.json")
}

fn minilm_bytes() -> Vec<u8> {
    std::fs::read(minilm_path())
        .unwrap_or_else(|e| panic!("fixture {}: {e}", minilm_path().display()))
}

fn minilm() -> ClientTokenizer {
    ClientTokenizer::from_file(&minilm_path()).expect("the MiniLM fixture must load")
}

/// A SECOND, distinct tokenizer: the same bytes plus one trailing newline.
///
/// JSON tolerates trailing whitespace, so this loads and tokenizes identically
/// while hashing to a different digest. That is exactly the shape the digest
/// guard has to catch — a file that is "obviously the same tokenizer" and is not
/// the one the receipt names.
fn minilm_variant() -> ClientTokenizer {
    let mut bytes = minilm_bytes();
    bytes.push(b'\n');
    ClientTokenizer::from_bytes(&bytes, "minilm+newline").expect("still a tokenizer")
}

/// APR-PERF-GATE-001 v2.2 §4.3.1's W1 corpus, as committed.
fn w1_path() -> PathBuf {
    manifest_dir().join("../aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl")
}

/// The 256 raw prompt strings of W1, without the `_meta` header line.
fn w1_prompts() -> Vec<String> {
    let text = std::fs::read_to_string(w1_path())
        .unwrap_or_else(|e| panic!("W1 corpus {}: {e}", w1_path().display()));
    text.lines()
        .filter_map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).ok()?;
            Some(v.get("prompt")?.as_str()?.to_string())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The digest is computed, never declared
// ---------------------------------------------------------------------------

/// The digest a counter carries must be the digest of the bytes it opened.
///
/// Re-derived here from an independent read of the same path, so the two cannot
/// drift apart silently — the same shape as SetFit's
/// `tokenizer_bytes_hash_agrees_with_the_recorded_sha256`.
#[test]
fn the_digest_is_computed_from_the_bytes_that_were_opened() {
    let tk = minilm();
    let independent = sha256_hex(&minilm_bytes());
    assert_eq!(tk.tokenizer_sha256(), independent);
    assert_eq!(tk.tokenizer_sha256().len(), 64);
    assert_eq!(tk.source_len(), minilm_bytes().len());
}

/// Two byte sequences, two digests. A counter cannot inherit another's.
#[test]
fn a_different_file_gets_a_different_digest() {
    let a = minilm();
    let b = minilm_variant();
    assert_ne!(a.tokenizer_sha256(), b.tokenizer_sha256());
    assert_eq!(b.source_len(), a.source_len() + 1);
}

#[test]
fn a_missing_file_is_refused_rather_than_counted() {
    let err = ClientTokenizer::from_file(std::path::Path::new(
        "/nonexistent/perf-gate/tokenizer.json",
    ))
    .expect_err("a path that is not there cannot produce a digest");
    assert!(matches!(&err, TokenizerError::Io { .. }), "{err}");
}

#[test]
fn bytes_that_are_not_a_tokenizer_are_refused() {
    let err = ClientTokenizer::from_bytes(b"{\"not\": \"a tokenizer\"}", "inline")
        .expect_err("must not load");
    assert!(matches!(&err, TokenizerError::Load { .. }), "{err}");
}

// ---------------------------------------------------------------------------
// The counts are real
// ---------------------------------------------------------------------------

/// THE trap this module exists to avoid, on the one in-tree fixture that has it.
///
/// `setfit/tokenizer.json` declares `truncation.max_length = 128` and
/// `padding = Fixed(128)`. A counter that opens it and calls `encode` without
/// clearing both reports **128 for every input** — for the empty string, for two
/// words, and for a 568-token document alike. That is a constant wearing a
/// measurement's clothes, and it would be invisible on Qwen's `tokenizer.json`,
/// which declares neither.
///
/// Delete either `with_truncation(None)` or `with_padding(None)` in
/// `ClientTokenizer::from_bytes` and this test fails on its first assertion.
#[test]
fn padding_and_truncation_declared_in_the_file_do_not_become_the_count() {
    let tk = minilm();
    // Frozen against the same tokenizer driven by the reference Python
    // implementation with `no_truncation()` / `no_padding()`.
    assert_eq!(tk.count("").expect("empty encodes"), 0);
    assert_eq!(tk.count("hello world").expect("encodes"), 2);
    assert_eq!(tk.count("The quick brown fox.").expect("encodes"), 5);

    // The document is longer than the file's declared 128-token window, so an
    // uncleared truncation is not merely a padding artifact here: it silently
    // DISCARDS 447 tokens of prompt.
    //
    // 575 is MiniLM's count of W1 prompt 0, frozen against the same fixture
    // driven by the reference Python implementation. It is a function of the
    // COMMITTED CORPUS and moves when `--body-words` is retuned (it was 568 at
    // `body_words = 496`), which is stated here so the next retune knows to
    // re-freeze it rather than treat it as a regression.
    let long = w1_prompts().first().cloned().expect("W1 has prompts");
    assert_eq!(
        tk.count(&long).expect("encodes"),
        575,
        "a count of 128 means the file's own truncation/padding was honoured"
    );
}

/// Counting is a function of the text, not of the call order or of the
/// tokenizer's internal state.
#[test]
fn counting_is_deterministic_and_additive_in_nothing_but_the_text() {
    let tk = minilm();
    let a = tk.count("alpha beta gamma").expect("encodes");
    for _ in 0..8 {
        assert_eq!(tk.count("alpha beta gamma").expect("encodes"), a);
    }
    assert!(a > 0);
    assert!(tk.count("alpha beta gamma delta").expect("encodes") > a);
}

// ---------------------------------------------------------------------------
// `--tokenizer-sha256` is an ASSERTION
// ---------------------------------------------------------------------------

#[test]
fn an_asserted_digest_that_matches_the_opened_file_is_accepted() {
    let tk = minilm();
    let digest = tk.tokenizer_sha256().to_string();
    assert!(tk.assert_digest(&digest).is_ok());
}

/// The flag can refuse a run. It can no longer supply a value.
#[test]
fn an_asserted_digest_that_does_not_match_refuses_the_run() {
    let tk = minilm();
    let other = minilm_variant().tokenizer_sha256().to_string();
    let err = tk
        .assert_digest(&other)
        .expect_err("a digest for a file this run did not open must be refused");
    match &err {
        TokenizerError::DigestMismatch {
            expected, computed, ..
        } => {
            assert_eq!(expected, &other);
            assert_eq!(computed, tk.tokenizer_sha256());
        }
        other => panic!("expected DigestMismatch, got {other}"),
    }
    // The message has to say which way round it is, or an operator will
    // "fix" it by editing the file rather than the flag.
    assert!(err.to_string().contains("COMPUTED"), "{err}");
}

#[test]
fn a_digest_shaped_like_free_text_asserts_nothing_and_is_refused() {
    let tk = minilm();
    for bogus in ["deadbeef", "", &"C".repeat(64), &"g".repeat(64)] {
        let Err(err) = tk.assert_digest(bogus) else {
            panic!("{bogus:?} is not a digest and must be refused");
        };
        assert!(matches!(&err, TokenizerError::NotADigest(_)), "{err}");
    }
}

// ---------------------------------------------------------------------------
// The guard: a receipt cannot carry a digest for a file the run never opened
// ---------------------------------------------------------------------------

/// The §4.4.6 block a counter produces carries the counter's own digest and the
/// counter's own counting facts. Nothing here is a parameter.
#[test]
fn the_block_is_derived_from_the_counter_not_supplied_alongside_it() {
    let tk = minilm();
    let digest = tk.tokenizer_sha256().to_string();
    let accounting = TokenAccounting::client_tokenizer(tk);
    match accounting.block() {
        TokenizationBlock::ClientTokenizer {
            tokenizer_sha256,
            counts_special_tokens,
            counts_prompt_echo,
        } => {
            assert_eq!(tokenizer_sha256, &digest);
            assert_eq!(*counts_special_tokens, COUNTS_SPECIAL_TOKENS);
            assert_eq!(*counts_prompt_echo, COUNTS_PROMPT_ECHO);
        }
        other => panic!("expected ClientTokenizer, got {other:?}"),
    }
    assert!(accounting.validate().is_ok());
    assert!(accounting.counter().is_some());
}

/// THE GUARD. A hand-written block naming one tokenizer, paired with a counter
/// built from another, is refused — and `require_counter` is what refuses it.
///
/// Before this change `require_counter` took a `bool` supplied by its caller and
/// was called from nothing but its own two unit tests, so the entire question
/// "was this digest produced by a file the run opened?" had no answer anywhere.
#[test]
fn a_digest_for_a_file_this_run_never_opened_is_refused() {
    let borrowed = minilm().tokenizer_sha256().to_string();
    let counter = minilm_variant();
    assert_ne!(counter.tokenizer_sha256(), borrowed);

    let forged = TokenizationBlock::ClientTokenizer {
        tokenizer_sha256: borrowed.clone(),
        counts_special_tokens: COUNTS_SPECIAL_TOKENS,
        counts_prompt_echo: COUNTS_PROMPT_ECHO,
    };
    // The forged block is perfectly well-formed by the OLD rule: 64 lowercase
    // hex characters, and it is even a real tokenizer's digest.
    assert!(forged.validate().is_ok());

    let err = TokenAccounting::from_parts(forged, Some(counter))
        .expect_err("a borrowed digest must not reach a receipt");
    assert!(err.contains("did not open"), "{err}");
    assert!(err.contains(&borrowed), "{err}");
}

#[test]
fn a_client_tokenizer_declaration_with_no_counter_is_refused() {
    let block = TokenizationBlock::ClientTokenizer {
        tokenizer_sha256: "a".repeat(64),
        counts_special_tokens: false,
        counts_prompt_echo: false,
    };
    let err = TokenAccounting::from_parts(block, None).expect_err("no counter, no method");
    assert!(err.contains("no client TokenCounter"), "{err}");
}

#[test]
fn a_counter_under_a_server_usage_declaration_is_refused() {
    let err = TokenAccounting::from_parts(
        TokenizationBlock::ServerUsage {
            counts_special_tokens: true,
            counts_prompt_echo: false,
        },
        Some(minilm()),
    )
    .expect_err("a counter that will not be used must not be silently ignored");
    assert!(err.contains("server_usage"), "{err}");
}

#[test]
fn server_usage_without_a_counter_is_the_unchanged_path() {
    let a = TokenAccounting::server_usage(true, false);
    assert!(a.validate().is_ok());
    assert!(a.counter().is_none());
    assert_eq!(a.block().method(), TokenCountingMethod::ServerUsage);
}

/// A malformed digest is still refused when it arrives through `from_parts`,
/// before the counter comparison ever runs.
#[test]
fn from_parts_still_applies_the_old_digest_shape_rule() {
    let block = TokenizationBlock::ClientTokenizer {
        tokenizer_sha256: "not-a-digest".to_string(),
        counts_special_tokens: false,
        counts_prompt_echo: false,
    };
    let err = TokenAccounting::from_parts(block, Some(minilm())).expect_err("shape rule first");
    assert!(err.contains("64 lowercase hex"), "{err}");
}

// ---------------------------------------------------------------------------
// The W1 number the campaign measured
// ---------------------------------------------------------------------------

/// The raw W1 prompt text, counted with no template, is 512 tokens.
///
/// It was **505** when this counter first measured it, at
/// `_meta.body_words = 496`. 505 is inside §4.3.1's `512 ± 8` by exactly one
/// token — the floor is 504 — so the campaign's blocking workload sat 0.198%
/// above the value at which it fails its own band assertion. That had been
/// invisible because under `server_usage --stream` neither server emits a
/// `usage` block, so every observation was `prompt_tokens = 0` and
/// [`assert_prompt_tokens_in_band`](crate::llm::assert_prompt_tokens_in_band)
/// refused it as an instrumentation gap before ever comparing 505 to 504. This
/// module is the first thing in the repository able to make that comparison,
/// and the first measurement it made was one token from the edge.
///
/// The corpus was regenerated at `--body-words 503`, which measures **512** —
/// the target itself — on all 256 records, with 8 tokens of margin on each side
/// instead of 1 and 15. `_meta.token_count_note` already named that remedy:
/// "if it fails, retune --body-words and regenerate".
const W1_RAW_PROMPT_TOKENS: u32 = 512;

/// `tokenizer.json` as HuggingFace serves it for Qwen2.5-Coder, 7,031,645 bytes.
const QWEN_TOKENIZER_CANONICAL: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";

/// A newer `tokenizers` re-save of the same vocabulary, 11,421,892 bytes: same
/// `model.vocab`, same `added_tokens`, same 151,665 entries, and identical id
/// sequences — it differs only in how `model.merges` is serialized (`"a b"`
/// strings rather than `["a", "b"]` pairs). It is the ONLY copy present on two
/// of the four fleet hosts, so refusing it here would refuse the hosts rather
/// than the tokenizer.
const QWEN_TOKENIZER_RESAVE: &str =
    "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8";

/// §4.3.1's corpus is 256 prompts plus one `_meta` header line.
///
/// Always runs. The 512 test below reads this file, so a corpus that moved or
/// shrank has to fail here rather than turn the 512 assertion into a vacuous
/// loop over zero prompts.
#[test]
fn the_w1_corpus_is_present_and_has_256_prompts() {
    let prompts = w1_prompts();
    assert_eq!(
        prompts.len(),
        256,
        "W1 is 256 prompts ({})",
        w1_path().display()
    );
    assert!(prompts.iter().all(|p| !p.is_empty()));
    // Distinct by construction (the corpus says so in its own `_meta`), which
    // is what stops prefix caching from standing in for the scheduler.
    let distinct: std::collections::HashSet<&String> = prompts.iter().collect();
    assert_eq!(distinct.len(), 256);
}

/// Every W1 prompt is exactly 512 tokens of raw text, with no template applied.
///
/// This is the number the whole §4.4.6 argument rests on: the same corpus,
/// through the same model file, was reported as **513** by `apr serve`'s `usage`
/// and **534** by `llama-server`'s, because each server applies its own chat
/// template before counting. The raw count is what is on the wire, and it is
/// the only one of the three both lanes can be compared on.
///
/// It is also the number §4.3.1's band is asserted against, and the ONLY one
/// that can be: under `server_usage` with the streaming §4.5 requires, neither
/// server emits a `usage` block at all, so the band check sees
/// `prompt_tokens = 0` on every request and refuses. Tuning the corpus to this
/// count is therefore not a trade against the server-side numbers — it is
/// tuning it to the only measurement the check will ever see.
///
/// Needs Qwen2.5-Coder's `tokenizer.json`:
///
/// ```text
/// APR_PERF_GATE_TOKENIZER=/path/to/qwen2.5-coder/tokenizer.json \
///   cargo test -p aprender-test-lib --features llm w1_corpus_counts_512
/// ```
///
/// Set `APR_PERF_GATE_REQUIRE_W1_TOKENIZER=1` to make its absence a failure.
#[test]
fn the_w1_corpus_counts_512_tokens_per_prompt() {
    let Some(path) = std::env::var_os("APR_PERF_GATE_TOKENIZER") else {
        assert!(
            std::env::var("APR_PERF_GATE_REQUIRE_W1_TOKENIZER").as_deref() != Ok("1"),
            "APR_PERF_GATE_REQUIRE_W1_TOKENIZER=1 but APR_PERF_GATE_TOKENIZER is unset"
        );
        eprintln!(
            "SKIP the_w1_corpus_counts_512_tokens_per_prompt: set \
             APR_PERF_GATE_TOKENIZER=<qwen2.5-coder tokenizer.json> to run it"
        );
        return;
    };
    let path = PathBuf::from(path);
    let tk =
        ClientTokenizer::from_file(&path).expect("APR_PERF_GATE_TOKENIZER must be a tokenizer");

    // Guard against pointing this at some other model's tokenizer and reading a
    // green tick: only the two known serializations of Qwen2.5-Coder's
    // vocabulary are accepted.
    assert!(
        matches!(
            tk.tokenizer_sha256(),
            QWEN_TOKENIZER_CANONICAL | QWEN_TOKENIZER_RESAVE
        ),
        "{} hashes to {}, which is neither Qwen2.5-Coder serialization",
        path.display(),
        tk.tokenizer_sha256()
    );

    let prompts = w1_prompts();
    assert_eq!(prompts.len(), 256);
    let mut counts = Vec::with_capacity(prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let n = tk.count(prompt).expect("encodes");
        assert_eq!(
            n, W1_RAW_PROMPT_TOKENS,
            "W1 prompt {i} is not {W1_RAW_PROMPT_TOKENS} raw tokens"
        );
        counts.push(n);
    }

    // The count alone is not the finding. §4.3.1's band is `512 ± 8`, and the
    // question that went unasked for the whole life of this corpus is how far
    // from its EDGE the blocking workload sits. At body_words = 496 the answer
    // was one token. Assert the margin, so a future retune that lands legal but
    // marginal fails here rather than on a fleet host twelve minutes into a
    // spent run.
    // The band is READ FROM THE CORPUS, never retyped here: a second copy of
    // `512 ± 8` in this file is a second thing to keep in step with §4.3.1.
    let band = crate::llm::load_prompt_corpus(&w1_path())
        .expect("W1 parses")
        .band
        .expect("W1 declares target_prompt_tokens and tolerance_tokens");
    let observed: Vec<(usize, u32)> = counts.iter().copied().enumerate().collect();
    crate::llm::assert_prompt_tokens_in_band(band, &observed).expect("W1 is inside §4.3.1's band");
    let (min, max) = (
        counts.iter().copied().min().expect("256 prompts"),
        counts.iter().copied().max().expect("256 prompts"),
    );
    let floor_margin = min - band.lo();
    let ceiling_margin = band.hi() - max;
    assert!(
        floor_margin >= 4 && ceiling_margin >= 4,
        "W1 sits {floor_margin} token(s) above the floor of {band} and {ceiling_margin} below \
         its ceiling (min {min}, max {max}). A corpus one token from its own floor fails on any \
         tokenizer re-serialization; retune `scripts/gen_prompts_w1.py --body-words` and \
         regenerate"
    );

    eprintln!(
        "W1: 256/256 prompts at {W1_RAW_PROMPT_TOKENS} raw tokens ({floor_margin} above the \
         band floor, {ceiling_margin} below its ceiling), tokenizer {} ({} bytes)",
        tk.tokenizer_sha256(),
        tk.source_len()
    );
}
