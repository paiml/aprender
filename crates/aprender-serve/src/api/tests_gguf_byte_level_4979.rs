//! #4979: the GGUF serve routes attach the file's own byte-level BPE to the tokenizer.
//!
//! The CPU route, the qwen3moe and dense CUDA routes and the batch route all retain the
//! mapped file through [`AppState::with_mapped_gguf_model`]; that is where the BPE is
//! attached. The tokenizer-level rows (greedy vs merges) are in
//! `tokenizer_tests_byte_level_4979.rs`.

use super::*;
use crate::gguf::MappedGGUFModel;

const QWEN35_HEADER: &[u8] =
    include_bytes!("../fixtures/gguf-header-slices/qwen3.5-0.8b.gguf-header");

/// Mutation: delete the `attach_gguf_byte_level_bpe` call in `with_mapped_gguf_model` and
/// this goes RED: the state keeps its greedy tokenizer.
#[test]
fn with_mapped_gguf_model_attaches_the_files_byte_level_bpe_4979() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("qwen3.5-0.8b.gguf-header");
    std::fs::write(&path, QWEN35_HEADER).expect("write the header");
    let mapped = Arc::new(MappedGGUFModel::from_path(&path).expect("map the header"));

    let vocab = mapped.model.vocabulary().expect("vocabulary");
    let unk = crate::tokenizer::vocabulary_unk_token(&vocab);
    let greedy = BPETokenizer::new(vocab, vec![], unk).expect("tokenizer");
    assert!(!greedy.has_byte_level_bpe());
    let mut state = AppState::demo().expect("demo state");
    state.tokenizer = Some(Arc::new(greedy));

    let state = state.with_mapped_gguf_model(Arc::clone(&mapped));
    let tok = state.tokenizer.as_ref().expect("the tokenizer is kept");
    assert!(tok.has_byte_level_bpe(), "the BPE is attached");
    for text in ["Hi there", " TANGERINE-4417"] {
        let want = mapped.model.encode(text);
        assert_eq!(Some(tok.encode(text)), want, "{text:?}");
    }
}

/// A state with no tokenizer gets none: the attach never invents one.
#[test]
fn with_mapped_gguf_model_without_a_tokenizer_stays_without_one_4979() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("qwen3.5-0.8b.gguf-header");
    std::fs::write(&path, QWEN35_HEADER).expect("write the header");
    let mapped = Arc::new(MappedGGUFModel::from_path(&path).expect("map the header"));

    let mut state = AppState::demo().expect("demo state");
    state.tokenizer = None;
    let state = state.with_mapped_gguf_model(mapped);
    assert!(state.tokenizer.is_none());
}
