//! Falsifiers for aprender#4979: serve built its GGUF tokenizer from the vocabulary
//! alone, so `encode` fell back to greedy longest-match ("ĠTA NG ER INE") instead of
//! merge-rank BPE ("ĠT ANGER INE"), and the model copied the odd split back.

use crate::api::AppState;
use crate::tokenizer::BPETokenizer;

fn vocab() -> Vec<String> {
    // Byte-level glyph vocab is not needed: `<unk>` is declared, so unmerged bytes map to it.
    ["<unk>", "a", "b", "c", "ab", "bc"]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn merges() -> Vec<(String, String)> {
    // `b c` outranks `a b`: rank-ordered BPE gives [a, bc]; greedy longest-match gives [ab, c].
    vec![("b".into(), "c".into()), ("a".into(), "b".into())]
}

#[test]
fn vocab_only_tokenizer_is_greedy() {
    let t = BPETokenizer::new(vocab(), vec![], "<unk>").expect("tokenizer");
    assert_eq!(t.encode("abc"), vec![4, 3], "greedy longest-match: ab, c");
}

#[test]
fn apply_merges_switches_encode_to_merge_rank() {
    let t = BPETokenizer::new(vocab(), vec![], "<unk>")
        .expect("tokenizer")
        .apply_merges(merges())
        .expect("merges");
    assert_eq!(t.encode("abc"), vec![1, 5], "merge rank: a, bc");
}

#[test]
fn app_state_with_gguf_merges_changes_encode() {
    let state = AppState::with_cache(1);
    // Replace the demo tokenizer with the synthetic one, then apply merges through the
    // state method serve calls.
    let mut state = state;
    state.tokenizer = Some(std::sync::Arc::new(
        BPETokenizer::new(vocab(), vec![], "<unk>").expect("tokenizer"),
    ));
    let before = state.tokenizer.as_ref().expect("t").encode("abc");
    let state = state.with_gguf_merges(merges()).expect("merges");
    let after = state.tokenizer.as_ref().expect("t").encode("abc");
    assert_eq!(before, vec![4, 3]);
    assert_eq!(after, vec![1, 5]);
}

#[test]
fn app_state_with_empty_merges_is_a_no_op() {
    let mut state = AppState::with_cache(1);
    state.tokenizer = Some(std::sync::Arc::new(
        BPETokenizer::new(vocab(), vec![], "<unk>").expect("tokenizer"),
    ));
    let state = state.with_gguf_merges(vec![]).expect("noop");
    assert_eq!(state.tokenizer.as_ref().expect("t").encode("abc"), vec![4, 3]);
}
