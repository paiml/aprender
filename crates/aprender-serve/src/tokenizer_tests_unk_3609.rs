//! #3609: an unknown token is a property of a model family, not of BPE.
//!
//! LLaMA-style vocabularies declare `<unk>`. Byte-level ones (GPT-2, Qwen) have no need
//! for one, because every byte already has a token, and Qwen3.5 declares none at all.
//! These rows pin both directions of the fix: a model WITH an unknown token still
//! resolves it (the fix is not "ignore unk"), and a model WITHOUT one tokenizes instead
//! of being refused. With no unknown token, nothing is synthesised and nothing is
//! dropped: a byte with no token is refused by name at construction.
//!
//! The two `*.gguf-header` fixtures are REAL headers, cut down by
//! `tests/fixtures/gguf-header-slices/generate.py`: every key and scalar value verbatim,
//! only the per-token arrays sliced (see that directory's MANIFEST.json).

use crate::gguf::{GGUFModel, GGUFValue};
use crate::tokenizer::{vocabulary_unk_token, BPETokenizer};

const QWEN35_HEADER: &[u8] =
    include_bytes!("../tests/fixtures/gguf-header-slices/qwen3.5-0.8b.gguf-header");
const TINYLLAMA_HEADER: &[u8] =
    include_bytes!("../tests/fixtures/gguf-header-slices/tinyllama-1.1b-chat.gguf-header");

const UNK_ID_KEY: &str = "tokenizer.ggml.unknown_token_id";
const EOS_ID_KEY: &str = "tokenizer.ggml.eos_token_id";

fn header(bytes: &[u8]) -> GGUFModel {
    GGUFModel::from_bytes(bytes).expect("a real GGUF header parses")
}

fn u32_key(model: &GGUFModel, key: &str) -> Option<u32> {
    match model.metadata.get(key) {
        Some(GGUFValue::UInt32(v)) => Some(*v),
        _ => None,
    }
}

/// All 256 GPT-2 byte-level glyphs: the first 256 tokens of a GPT-2/Qwen vocabulary.
fn byte_level_vocab() -> Vec<String> {
    (0u8..=255)
        .map(|b| crate::gguf::utils::gpt2_byte_to_unicode(b).to_string())
        .collect()
}

fn id_of(tok: &BPETokenizer, token: &str) -> u32 {
    tok.get_token_id(token).expect("token in vocabulary")
}

// ---------------------------------------------------------------------------
// The real Qwen3.5 header: declares EOS, declares no unknown token, has no `<unk>`.
// ---------------------------------------------------------------------------

#[test]
fn qwen35_real_header_declares_eos_and_no_unknown_token() {
    let model = header(QWEN35_HEADER);
    assert_eq!(u32_key(&model, EOS_ID_KEY), Some(248_046), "the real EOS id, verbatim");
    assert!(
        !model.metadata.contains_key(UNK_ID_KEY),
        "Qwen3.5 declares no tokenizer.ggml.unknown_token_id at all"
    );
    let vocab = model.vocabulary().expect("the header carries a vocabulary");
    assert!(!vocab.iter().any(|t| t == "<unk>"), "and has no <unk> token");
}

/// THE row #3609 is about. Mutation: restore the unconditional
/// `token_to_id.get("<unk>").ok_or_else(..)` in `BPETokenizer::new` and this goes RED.
#[test]
fn qwen35_real_vocabulary_builds_a_tokenizer_with_no_unknown_token() {
    let vocab = header(QWEN35_HEADER).vocabulary().expect("vocabulary");
    let unk = vocabulary_unk_token(&vocab);
    assert_eq!(unk, None, "nothing named <unk>, so nothing is passed");
    let tok = BPETokenizer::new(vocab, vec![], unk)
        .expect("a real vocabulary without <unk> is not refused (#3609)");
    assert_eq!(tok.unk_token_id(), None, "absent means absent: no synthesised <unk>");
    // With no unknown token, a character the vocabulary lacks (CJK here) goes through
    // the per-byte glyph fallback and round-trips. `é` is NOT a row: the greedy path
    // matches it as the byte-0xE9 glyph token before any fallback runs. That is a
    // pre-existing greedy-path defect, independent of the unknown token, filed as #3677.
    for text in ["Hi there", "世界", "a\nb"] {
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "{text:?} encodes to something");
        assert_eq!(tok.decode(&ids).expect("decode"), text, "{text:?} round-trips");
    }
}

// ---------------------------------------------------------------------------
// The real TinyLlama header: declares unknown_token_id = 0 = `<unk>`. It must still
// resolve (the other direction: the fix is not "ignore unk").
// ---------------------------------------------------------------------------

#[test]
fn tinyllama_real_header_declares_unk_and_it_still_resolves() {
    let model = header(TINYLLAMA_HEADER);
    assert_eq!(u32_key(&model, UNK_ID_KEY), Some(0), "the real declaration, verbatim");
    let vocab = model.vocabulary().expect("vocabulary");
    assert_eq!(vocab[0], "<unk>", "id 0 is the declared <unk>");
    let unk = vocabulary_unk_token(&vocab);
    assert_eq!(unk, Some("<unk>"));
    let tok = BPETokenizer::new(vocab, vec![], unk).expect("tokenizer");
    assert_eq!(tok.unk_token_id(), Some(0), "the declared unknown token still resolves");
}

// ---------------------------------------------------------------------------
// The case table on small vocabularies.
// ---------------------------------------------------------------------------

#[test]
fn declared_unknown_token_is_still_emitted_for_an_unencodable_byte() {
    let tok = BPETokenizer::new(vec!["<unk>".into(), "a".into()], vec![], "<unk>")
        .expect("tokenizer");
    assert_eq!(tok.encode("é"), vec![0, 0], "unchanged: two bytes, two <unk>");
}

#[test]
fn a_named_unknown_token_missing_from_the_vocabulary_is_refused() {
    let err = BPETokenizer::new(vec!["a".into(), "b".into()], vec![], "<unk>")
        .expect_err("naming an unknown token the vocabulary lacks is an error");
    assert!(err.to_string().contains("'<unk>'"), "names the token: {err}");
}

#[test]
fn no_unknown_token_and_byte_level_glyphs_encode_every_byte() {
    let tok = BPETokenizer::new(byte_level_vocab(), vec![], None).expect("tokenizer");
    // '世' is E4 B8 96: not a token, so each byte takes its glyph.
    let glyphs: Vec<u32> = [0xE4u8, 0xB8, 0x96]
        .iter()
        .map(|&b| id_of(&tok, &crate::gguf::utils::gpt2_byte_to_unicode(b).to_string()))
        .collect();
    assert_eq!(tok.encode("世"), glyphs);
    assert_eq!(tok.decode(&tok.encode("世")).expect("decode"), "世");
}

#[test]
fn no_unknown_token_and_byte_fallback_tokens_encode_every_byte() {
    let vocab: Vec<String> = (0u8..=255).map(|b| format!("<0x{b:02X}>")).collect();
    let tok = BPETokenizer::new(vocab, vec![], None).expect("tokenizer");
    assert_eq!(
        tok.encode("é"),
        vec![id_of(&tok, "<0xC3>"), id_of(&tok, "<0xA9>")]
    );
}

/// The cop's row: absent means absent at encode time too. No unknown token, and one byte
/// with neither a `<0xNN>` token nor a glyph token, is refused by name. It is never
/// dropped and never emitted as nothing.
#[test]
fn no_unknown_token_and_a_byte_with_no_token_is_refused_by_name() {
    let glyph_a9 = crate::gguf::utils::gpt2_byte_to_unicode(0xA9).to_string();
    let vocab: Vec<String> = byte_level_vocab()
        .into_iter()
        .filter(|t| *t != glyph_a9)
        .collect();
    let err = BPETokenizer::new(vocab, vec![], None)
        .expect_err("a byte nothing can encode must refuse the tokenizer");
    assert!(err.to_string().contains("0xA9"), "names the byte: {err}");
}

#[test]
fn vocabulary_unk_token_names_only_an_unk_that_is_present() {
    assert_eq!(vocabulary_unk_token(&["<unk>".into(), "a".into()]), Some("<unk>"));
    assert_eq!(vocabulary_unk_token(&["a".into(), "b".into()]), None);
    assert_eq!(vocabulary_unk_token(&[]), None);
}
