//! #4979: serve encodes a GGUF prompt with the file's own byte-level BPE.
//!
//! Every GGUF serve route built `BPETokenizer::new(vocab, vec![], unk)`, so `encode()` fell
//! back to greedy longest-match, while `apr run` encodes through `GGUFModel::encode` and the
//! file's ranked merges. On Qwen3-30B-A3B-Instruct-2507, " TANGERINE" went in as
//! `ĠTA|NG|ER|INE` instead of `ĠT|ANGER|INE`, and the model answered "TA NGERINE". The
//! synthetic file below reproduces that exact disagreement: its vocabulary holds `ĠTA` and
//! `NG`, which greedy reaches first, but no merge ever forms them.

use crate::gguf::test_factory::GGUFBuilder;
use crate::gguf::GGUFModel;
use crate::tokenizer::{vocabulary_unk_token, BPETokenizer};

const QWEN35_HEADER: &[u8] =
    include_bytes!("fixtures/gguf-header-slices/qwen3.5-0.8b.gguf-header");

const EXTRA: &[&str] = &[
    "\u{0120}T",
    "\u{0120}TA",
    "AN",
    "IN",
    "ANG",
    "ER",
    "INE",
    "ANGER",
    "NG",
];
/// Rank order: `Ġ T`, then `A N`, `I N`, `AN G`, `E R`, `IN E`, `ANG ER`. Nothing forms
/// `ĠTA` or `NG`, and nothing joins `ĠT` to `ANGER`.
const MERGES: &[&str] = &[
    "\u{0120} T",
    "A N",
    "I N",
    "AN G",
    "E R",
    "IN E",
    "ANG ER",
];
const TEXT: &str = " TANGERINE";

/// The 256 GPT-2 byte-level glyphs (ids 0..256), then `EXTRA`.
fn vocab() -> Vec<String> {
    (0u8..=255)
        .map(|b| crate::gguf::utils::gpt2_byte_to_unicode(b).to_string())
        .chain(EXTRA.iter().map(|s| (*s).to_string()))
        .collect()
}

fn gguf(tokenizer_model: &str, pre: &str) -> GGUFModel {
    let vocab = vocab();
    let tokens: Vec<&str> = vocab.iter().map(String::as_str).collect();
    let bytes = GGUFBuilder::new()
        .architecture("qwen3moe")
        .add_string("tokenizer.ggml.model", tokenizer_model)
        .add_string("tokenizer.ggml.pre", pre)
        .add_string_array("tokenizer.ggml.tokens", &tokens)
        .add_string_array("tokenizer.ggml.merges", MERGES)
        .build();
    GGUFModel::from_bytes(&bytes).expect("synthetic GGUF header parses")
}

/// What every GGUF serve route built before #4979: the vocabulary and nothing else.
fn vocabulary_only(model: &GGUFModel) -> BPETokenizer {
    let vocab = model.vocabulary().expect("vocabulary");
    let unk = vocabulary_unk_token(&vocab);
    BPETokenizer::new(vocab, vec![], unk).expect("tokenizer")
}

fn ids(tok: &BPETokenizer, pieces: &[&str]) -> Vec<u32> {
    pieces
        .iter()
        .map(|p| tok.get_token_id(p).expect("piece in vocabulary"))
        .collect()
}

/// The defect, pinned so the next row's difference means something: on this vocabulary
/// greedy longest-match is NOT the file's tokenization.
#[test]
fn greedy_longest_match_splits_tangerine_as_serve_did_4979() {
    let model = gguf("gpt2", "qwen2");
    let greedy = vocabulary_only(&model);
    assert!(!greedy.has_byte_level_bpe());
    assert_eq!(
        greedy.encode(TEXT),
        ids(&greedy, &["\u{0120}TA", "NG", "ER", "INE"])
    );
}

/// THE row #4979 is about. Mutation: delete the `byte_level` delegation at the top of
/// `BPETokenizer::encode` and this goes RED with the greedy split above.
#[test]
fn serve_tokenizer_encodes_with_the_files_merges_4979() {
    let model = gguf("gpt2", "qwen2");
    let bpe = model
        .byte_level_bpe()
        .expect("a gpt2 file with a known pre-tokenizer and merges has a byte-level BPE");
    let tok = vocabulary_only(&model).with_byte_level_bpe(bpe);
    assert!(tok.has_byte_level_bpe());
    let canonical = ids(&tok, &["\u{0120}T", "ANGER", "INE"]);
    assert_eq!(tok.encode(TEXT), canonical);
    assert_eq!(
        model.encode(TEXT),
        Some(canonical),
        "serve and `apr run` hand the model the same ids"
    );
    assert_eq!(tok.decode(&tok.encode(TEXT)).expect("decode"), TEXT);
}

/// A file whose tokenizer is not byte-level keeps the tokenizer it has. The file carries a
/// known pre-tokenizer and merges, so only the `tokenizer.ggml.model` check refuses it.
/// Mutation: drop that check in `GGUFModel::byte_level_bpe` (`gguf/byte_level_bpe.rs`) and
/// this goes RED.
#[test]
fn a_file_that_is_not_byte_level_attaches_nothing_4979() {
    assert!(gguf("llama", "qwen2").byte_level_bpe().is_none());
}

/// A pre-tokenizer that is not implemented attaches nothing: serve keeps greedy, the same
/// fallback `apr run` takes, and says so once on stderr.
#[test]
fn an_unimplemented_pre_tokenizer_attaches_nothing_4979() {
    let model = gguf("gpt2", "llama-bpe");
    assert!(model.byte_level_bpe().is_none());
    assert_eq!(
        Some(vocabulary_only(&model).encode(TEXT)),
        model.encode(TEXT),
        "both verbs take the same greedy fallback"
    );
}

/// The real Qwen3.5 header (every key verbatim, per-token arrays sliced to 1024) yields a
/// byte-level BPE, and the attached tokenizer encodes as `GGUFModel::encode` does.
#[test]
fn qwen35_real_header_attaches_its_byte_level_bpe_4979() {
    let model = GGUFModel::from_bytes(QWEN35_HEADER).expect("a real GGUF header parses");
    let bpe = model
        .byte_level_bpe()
        .expect("the real Qwen3.5 header is byte-level with a known pre-tokenizer");
    let tok = vocabulary_only(&model).with_byte_level_bpe(bpe);
    for text in ["Hi there", "fn main() {\n    let x = 1;\n}", " TANGERINE-4417"] {
        assert_eq!(Some(tok.encode(text)), model.encode(text), "{text:?}");
    }
}
