//! ENC-02 parity tests: `MiniLmTokenizer` against the frozen fixtures.
//!
//! `tokenizer_cases.json` is the phase's CORPUS OF RECORD (01-04): it carries a
//! case for every text any other fixture uses, so it holds MORE cases than the
//! six required classes. The parity test therefore iterates **every** case in
//! the file and never hardcodes a case-name list — adding a fixture text
//! automatically widens this gate.
//!
//! Tokenizer parity has **no epsilon**. Every comparison here is integer
//! equality; a tolerance literal in this file would be a defect.

use super::*;

use std::path::PathBuf;

/// One frozen tokenizer case.
#[derive(serde::Deserialize)]
struct TokenizerCase {
    id: String,
    texts: Vec<String>,
    max_length: usize,
    input_ids: Vec<Vec<u32>>,
    token_type_ids: Vec<Vec<u32>>,
    attention_mask: Vec<Vec<u8>>,
    truncated: Vec<bool>,
    original_token_counts: Vec<usize>,
}

/// The whole corpus of record.
#[derive(serde::Deserialize)]
struct TokenizerCases {
    revision: String,
    tokenizer_sha256: String,
    cases: Vec<TokenizerCase>,
}

/// Resolve the frozen-fixture directory.
///
/// `APRENDER_SETFIT_FIXTURES` overrides; otherwise the in-crate path is used,
/// following the resolution style of `aprender-bench-tokenizer`.
fn fixtures_dir() -> PathBuf {
    if let Ok(p) = std::env::var("APRENDER_SETFIT_FIXTURES") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/setfit")
}

fn tokenizer_bytes() -> Vec<u8> {
    let path = fixtures_dir().join("tokenizer.json");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn load_cases() -> TokenizerCases {
    let path = fixtures_dir().join("tokenizer_cases.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn tokenizer() -> MiniLmTokenizer {
    MiniLmTokenizer::from_bytes(&tokenizer_bytes())
        .unwrap_or_else(|e| panic!("MiniLmTokenizer::from_bytes: {e}"))
}

/// Flatten a fixture's row-of-rows into the row-major layout `SentenceBatch` uses.
fn flatten<T: Copy>(rows: &[Vec<T>]) -> Vec<T> {
    rows.iter().flat_map(|r| r.iter().copied()).collect()
}

// ---------------------------------------------------------------------------
// Parity over the whole corpus of record
// ---------------------------------------------------------------------------

#[test]
fn tokenizer_parity_every_frozen_case_matches_exactly() {
    let fixture = load_cases();
    // The corpus of record grows as other fixtures are added; this asserts it is
    // a corpus rather than a token gesture, without pinning a case list.
    assert!(
        fixture.cases.len() > 4,
        "corpus of record has only {} cases",
        fixture.cases.len()
    );

    let tok = tokenizer();
    for case in &fixture.cases {
        assert_eq!(
            case.max_length, MAX_SEQUENCE_LENGTH,
            "case {} was frozen at a different max_length",
            case.id
        );
        let texts: Vec<&str> = case.texts.iter().map(String::as_str).collect();
        let batch = tok
            .encode_batch(&texts)
            .unwrap_or_else(|e| panic!("case {}: encode_batch: {e}", case.id));

        let expected_batch = case.input_ids.len();
        let expected_seq = case.input_ids[0].len();
        assert_eq!(batch.batch(), expected_batch, "case {} batch", case.id);
        assert_eq!(batch.seq(), expected_seq, "case {} seq", case.id);

        // Exact integer equality — no tolerance exists for tokenizer parity.
        assert_eq!(
            batch.input_ids(),
            flatten(&case.input_ids).as_slice(),
            "case {} input_ids",
            case.id
        );
        assert_eq!(
            batch.token_type_ids(),
            flatten(&case.token_type_ids).as_slice(),
            "case {} token_type_ids",
            case.id
        );
        assert_eq!(
            batch.attention_mask(),
            flatten(&case.attention_mask).as_slice(),
            "case {} attention_mask",
            case.id
        );

        assert_eq!(
            batch.truncation().len(),
            expected_batch,
            "case {} truncation arity",
            case.id
        );
        for (i, fact) in batch.truncation().iter().enumerate() {
            assert_eq!(
                fact.truncated, case.truncated[i],
                "case {} row {i} truncated",
                case.id
            );
            assert_eq!(
                fact.original_len, case.original_token_counts[i],
                "case {} row {i} original_len",
                case.id
            );
        }
    }
}

#[test]
fn tokenizer_parity_canonical_ids_are_not_slice_rows() {
    // The slice vocabulary is 97 rows; canonical MiniLM ids run to 30521. If the
    // tokenizer ever started emitting slice rows, every id would fall under 97.
    let fixture = load_cases();
    let tok = tokenizer();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.id == "single_short")
        .expect("single_short case present");
    let texts: Vec<&str> = case.texts.iter().map(String::as_str).collect();
    let batch = tok.encode_batch(&texts).expect("encode_batch");
    assert!(
        batch.input_ids().iter().any(|id| *id > 97),
        "ids look like slice rows, not canonical ids: {:?}",
        batch.input_ids()
    );
}

#[test]
fn tokenizer_parity_tokenizer_sha256_matches_the_frozen_pin() {
    let fixture = load_cases();
    let tok = tokenizer();
    assert_eq!(tok.tokenizer_sha256(), fixture.tokenizer_sha256);
    // The fixture's revision is the pinned upstream revision recorded in
    // upstream_manifest.json; keeping the assertion here ties the parity gate to
    // that pin rather than to an independently chosen value.
    assert_eq!(fixture.revision.len(), 40, "revision is a full commit sha");
}

#[test]
fn tokenizer_parity_rejects_bytes_that_are_not_a_tokenizer() {
    let err = MiniLmTokenizer::from_bytes(b"{ not a tokenizer }")
        .expect_err("malformed bytes must not load");
    assert!(
        matches!(err, SetFitError::TokenizerLoad { .. }),
        "expected TokenizerLoad, got {err}"
    );
}

// ---------------------------------------------------------------------------
// SentenceBatch shape, identity, provenance and rejections
// ---------------------------------------------------------------------------

#[test]
fn sentence_batch_carries_the_producing_tokenizer_identity() {
    let fixture = load_cases();
    let tok = tokenizer();
    let batch = tok
        .encode_batch(&["A quick brown fox jumps over the lazy dog."])
        .expect("encode_batch");
    assert_eq!(batch.tokenizer_sha256(), tok.tokenizer_sha256());
    assert_eq!(batch.tokenizer_sha256(), fixture.tokenizer_sha256);
}

#[test]
fn sentence_batch_single_input_seq_is_its_own_length() {
    let tok = tokenizer();
    let batch = tok
        .encode_batch(&["Padding must not change this sentence."])
        .expect("encode_batch");
    assert_eq!(batch.batch(), 1);
    // 10 tokens including [CLS]/[SEP] per the frozen invariance_single case.
    assert_eq!(batch.seq(), 10);
    assert_eq!(batch.input_ids().len(), 10);
    assert!(
        batch.attention_mask().iter().all(|m| *m == 1),
        "a single un-truncated input has no padding"
    );
}

#[test]
fn sentence_batch_reports_original_length_for_a_truncated_input() {
    let fixture = load_cases();
    let case = fixture
        .cases
        .iter()
        .find(|c| c.id == "truncation_long")
        .expect("truncation_long case present");
    let tok = tokenizer();
    let texts: Vec<&str> = case.texts.iter().map(String::as_str).collect();
    let batch = tok.encode_batch(&texts).expect("encode_batch");

    assert_eq!(batch.seq(), MAX_SEQUENCE_LENGTH);
    assert_eq!(batch.input_ids().len(), MAX_SEQUENCE_LENGTH);
    assert!(
        batch.truncation()[0].truncated,
        "row must report truncation"
    );
    assert_eq!(
        batch.truncation()[0].original_len,
        case.original_token_counts[0]
    );
    assert!(
        batch.truncation()[0].original_len > MAX_SEQUENCE_LENGTH,
        "a truncated row's original length must exceed the bound"
    );
}

#[test]
fn sentence_batch_provenance_is_the_sha256_of_each_input_in_order() {
    let texts = [
        "The cat sat on the mat.",
        "Stock markets fell sharply today.",
    ];
    let tok = tokenizer();
    let batch = tok.encode_batch(&texts).expect("encode_batch");
    assert_eq!(batch.provenance().len(), 2);
    for (i, text) in texts.iter().enumerate() {
        assert_eq!(batch.provenance()[i].index, i, "provenance order");
        assert_eq!(
            batch.provenance()[i].text_sha256,
            sha256_hex(text.as_bytes()),
            "provenance hash for input {i}"
        );
    }
    // Distinct inputs must not collide.
    assert_ne!(
        batch.provenance()[0].text_sha256,
        batch.provenance()[1].text_sha256
    );
}

#[test]
fn sentence_batch_empty_text_list_is_rejected_not_panicked() {
    let tok = tokenizer();
    let err = tok
        .encode_batch(&[])
        .expect_err("empty batch must be rejected");
    assert!(
        matches!(err, SetFitError::BatchInvalid { .. }),
        "expected BatchInvalid, got {err}"
    );
    assert!(
        err.to_string().contains("empty"),
        "error should say what was wrong: {err}"
    );
}

#[test]
fn sentence_batch_rows_are_row_major_and_arity_consistent() {
    let tok = tokenizer();
    let batch = tok
        .encode_batch(&[
            "Short text.",
            "A noticeably longer sentence that forces the batch to pad the shorter rows.",
        ])
        .expect("encode_batch");
    let n = batch.batch() * batch.seq();
    assert_eq!(batch.input_ids().len(), n);
    assert_eq!(batch.token_type_ids().len(), n);
    assert_eq!(batch.attention_mask().len(), n);
    assert_eq!(batch.truncation().len(), batch.batch());
    assert_eq!(batch.provenance().len(), batch.batch());
    // Row 0 is shorter, so it must carry padding; row 1 must not.
    let seq = batch.seq();
    assert!(batch.attention_mask()[..seq].contains(&0));
    assert!(!batch.attention_mask()[seq..].contains(&0));
}

#[test]
fn sentence_batch_mask_is_binary_and_padding_ids_are_zero() {
    let tok = tokenizer();
    let batch = tok
        .encode_batch(&["Tiny.", "Padding must not change this sentence."])
        .expect("encode_batch");
    for (i, m) in batch.attention_mask().iter().enumerate() {
        assert!(*m == 0 || *m == 1, "mask[{i}] = {m}");
        if *m == 0 {
            // pad_token_id is 0 for this tokenizer; a padded slot must be pad.
            assert_eq!(batch.input_ids()[i], 0, "padded slot {i} is not [PAD]");
            assert_eq!(batch.token_type_ids()[i], 0, "padded slot {i} type id");
        }
    }
}

// ---------------------------------------------------------------------------
// D-08 / W1 structural seal — asserted against the source, not against docs
// ---------------------------------------------------------------------------

#[test]
fn sentence_batch_has_no_public_fields_and_no_mutable_accessors() {
    let src = include_str!("tokenizer.rs");
    for field in [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "batch",
        "seq",
        "truncation",
        "provenance",
        "tokenizer_sha256",
    ] {
        let violation = format!("pub {field}:");
        assert!(
            !src.contains(&violation),
            "SentenceBatch field `{field}` is `pub` — W1 data-layer seal broken"
        );
    }
    assert!(
        !src.contains("_mut(&mut self)"),
        "a `_mut` accessor would let out-of-crate code mutate a received batch"
    );
    assert!(
        src.contains("pub(crate) fn from_bytes"),
        "D-08: MiniLmTokenizer::from_bytes must be pub(crate)"
    );
    assert!(
        !src.contains("pub fn from_bytes"),
        "D-08 seal broken: a bare `pub fn from_bytes` exists"
    );
}

// ---------------------------------------------------------------------------
// Retained source bytes (plan 03-08) — the reload half of tokenizer identity
// ---------------------------------------------------------------------------

/// The retained bytes and the recorded digest describe the SAME tokenizer.
///
/// Two facts are now stored where one used to be, so the failure mode this test
/// exists for is that they drift: a `from_bytes` that hashed one buffer and
/// retained another would rebuild a tokenizer whose identity check passes against
/// a hash it does not have. Re-deriving the digest from what `source_bytes()`
/// returns is the only comparison that can see that; comparing the field against
/// itself cannot.
#[test]
fn tokenizer_bytes_hash_agrees_with_the_recorded_sha256() {
    let bytes = tokenizer_bytes();
    assert!(
        !bytes.is_empty(),
        "the frozen tokenizer fixture must be non-empty"
    );
    let tokenizer = MiniLmTokenizer::from_bytes(&bytes).expect("the frozen tokenizer must load");

    assert_eq!(
        tokenizer.source_bytes(),
        bytes.as_slice(),
        "the retained bytes must be the bytes the tokenizer was built from, byte for byte",
    );
    assert_eq!(
        sha256_hex(tokenizer.source_bytes()),
        tokenizer.tokenizer_sha256(),
        "the digest re-derived from the RETAINED bytes must equal the RECORDED digest; a \
         disagreement means an artifact carrying both would describe two different tokenizers",
    );
}

/// The retained bytes rebuild a tokenizer that tokenizes identically.
///
/// Byte equality above says the buffer survived; this says the buffer is
/// SUFFICIENT. A reload path that carried the right bytes into a tokenizer
/// configured differently — a lost truncation bound, a lost padding mode — would
/// pass the hash check and still produce different ids, which is the failure a
/// hash-only artifact can never detect.
#[test]
fn tokenizer_bytes_rebuild_a_tokenizer_that_agrees_on_every_frozen_case() {
    let cases = load_cases();
    let original = MiniLmTokenizer::from_bytes(&tokenizer_bytes()).expect("original loads");
    let rebuilt = MiniLmTokenizer::from_bytes(original.source_bytes()).expect("rebuild loads");

    assert!(
        !cases.cases.is_empty(),
        "the corpus of record must have cases"
    );
    for case in &cases.cases {
        let texts: Vec<&str> = case.texts.iter().map(String::as_str).collect();
        let a = original.encode_batch(&texts).expect("original encodes");
        let b = rebuilt.encode_batch(&texts).expect("rebuild encodes");
        assert_eq!(a.input_ids(), b.input_ids(), "case {}: input ids", case.id);
        assert_eq!(
            a.token_type_ids(),
            b.token_type_ids(),
            "case {}: token type ids",
            case.id
        );
        assert_eq!(
            a.attention_mask(),
            b.attention_mask(),
            "case {}: attention mask",
            case.id
        );
        assert_eq!(
            a.truncation(),
            b.truncation(),
            "case {}: truncation facts",
            case.id
        );
        assert_eq!(
            a.tokenizer_sha256(),
            b.tokenizer_sha256(),
            "case {}: tokenizer identity",
            case.id
        );
    }
}

/// `PADDING_MODE` names the strategy the constructor actually configures.
///
/// The constant travels into a persistence artifact, so it is a CLAIM about the
/// tokenizer rather than decoration. The source assertion is what keeps the claim
/// tied to the call: a `with_padding` switched to a fixed width while the constant
/// kept saying `batch_longest` is exactly the drift an artifact reader could not
/// see.
#[test]
fn tokenizer_bytes_padding_mode_constant_matches_the_configured_strategy() {
    assert_eq!(PADDING_MODE, "batch_longest");
    let src = include_str!("tokenizer.rs");
    assert!(
        src.contains("strategy: tokenizers::PaddingStrategy::BatchLongest"),
        "PADDING_MODE claims batch-longest padding; the constructor must configure it",
    );
}
