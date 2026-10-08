//! APR-EMBED-001 EG-1: case tables for the SPM-style BPE port. Each row is worked through
//! llama.cpp's `llm_tokenizer_bpe_session::tokenize` for `LLAMA_VOCAB_PRE_TYPE_GEMMA4` by hand;
//! the end-to-end id parity against the pinned llama.cpp on the real files is
//! `scripts/tokenizer_parity.sh`.

use super::*;

const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const BYTE: i32 = 6;

/// A tokenizer over `tokens` (all NORMAL, except `<0xXX>` as BYTE and `<...>` as CONTROL).
fn bpe(tokens: &[&str], merges: &[&str]) -> (SpmBpe, Vec<String>) {
    let vocab: Vec<String> = tokens.iter().map(|t| (*t).to_string()).collect();
    let types: Vec<i32> = vocab
        .iter()
        .map(|t| {
            if t.starts_with("<0x") {
                BYTE
            } else if t.starts_with('<') && t.ends_with('>') {
                CONTROL
            } else {
                NORMAL
            }
        })
        .collect();
    (SpmBpe::build(&vocab, merges, &types), vocab)
}

/// The token texts `text` encodes to.
fn pieces(tok: &(SpmBpe, Vec<String>), text: &str) -> Vec<String> {
    tok.0
        .encode(text)
        .into_iter()
        .map(|id| tok.1[id as usize].clone())
        .collect()
}

#[test]
fn merges_apply_lowest_rank_first() {
    let tok = bpe(&["a", "b", "c", "ab", "bc", "abc"], &["b c", "a b", "a bc"]);
    assert_eq!(pieces(&tok, "abc"), ["abc"], "b c (0), then a bc (2)");
    let tok = bpe(&["a", "b", "c", "ab", "bc", "abc"], &["b c", "a b"]);
    assert_eq!(
        pieces(&tok, "abc"),
        ["a", "bc"],
        "b c wins over a b; no a bc rule, so the word stops there"
    );
    let tok = bpe(&["a", "b", "c", "ab", "bc", "abc"], &["a b", "b c"]);
    assert_eq!(pieces(&tok, "abc"), ["ab", "c"], "a b wins over b c");
}

#[test]
fn equal_ranks_merge_leftmost_and_stale_pairs_are_skipped() {
    let tok = bpe(&["a", "aa"], &["a a"]);
    assert_eq!(
        pieces(&tok, "aaa"),
        ["aa", "a"],
        "the leftmost pair merges first"
    );
    assert_eq!(
        pieces(&tok, "aaaa"),
        ["aa", "aa"],
        "the middle pair went stale when its left side merged"
    );
}

#[test]
fn merges_are_keyed_by_text_not_by_vocabulary() {
    let tok = bpe(&["a", "b", "ab"], &[]);
    assert_eq!(
        pieces(&tok, "ab"),
        ["a", "b"],
        "a token with no merge rule is never formed"
    );
}

#[test]
fn spaces_become_the_visible_space_with_no_prefix() {
    let tok = bpe(&["a", "b", "\u{2581}", "\u{2581}b"], &["\u{2581} b"]);
    assert_eq!(pieces(&tok, "a b"), ["a", "\u{2581}b"]);
    assert_eq!(
        pieces(&tok, "b"),
        ["b"],
        "no space is prepended to the first word"
    );
    assert_eq!(pieces(&tok, " "), ["\u{2581}"]);
}

#[test]
fn merge_rules_split_at_the_first_space_after_byte_zero() {
    let (tok, _) = bpe(&[], &["\u{2581} b", "  x", " x y", "ab", "a b", "a b"]);
    let rank = |a: &str, b: &str| tok.ranks.get(&(a.to_string(), b.to_string())).copied();
    assert_eq!(rank("\u{2581}", "b"), Some(0), "a multibyte first piece");
    assert_eq!(
        rank(" ", "x"),
        Some(1),
        "a first piece that is itself a space"
    );
    assert_eq!(rank(" x", "y"), Some(2));
    assert_eq!(
        rank("", ""),
        Some(3),
        "a rule with no space is (\"\", \"\")"
    );
    assert_eq!(
        rank("a", "b"),
        Some(4),
        "a duplicate rule keeps its first rank"
    );
}

#[test]
fn newline_runs_split_the_fragment() {
    assert_eq!(newline_runs("a\n\nb c\n"), ["a", "\n\n", "b c", "\n"]);
    assert_eq!(newline_runs("\n"), ["\n"]);
    assert!(newline_runs("").is_empty());
}

#[test]
fn a_newline_run_that_is_a_token_is_one_symbol() {
    let tok = bpe(&["a", "b", "\n", "\n\n"], &[]);
    assert_eq!(
        pieces(&tok, "a\n\nb"),
        ["a", "\n\n", "b"],
        "no merge rule needed"
    );
    let tok = bpe(&["a", "\n"], &[]);
    assert_eq!(
        pieces(&tok, "a\n\n"),
        ["a", "\n", "\n"],
        "not a token: per char"
    );
}

#[test]
fn words_never_merge_across_a_newline() {
    let tok = bpe(&["a", "\n", "a\n"], &["a \n"]);
    assert_eq!(pieces(&tok, "a\n"), ["a", "\n"]);
}

#[test]
fn unknown_symbols_fall_back_to_upper_case_byte_tokens() {
    let tok = bpe(&["a", "<0xC3>", "<0xA9>"], &[]);
    assert_eq!(pieces(&tok, "a\u{e9}"), ["a", "<0xC3>", "<0xA9>"]);
    let tok = bpe(&["a", "<0xC3>"], &[]);
    assert_eq!(
        pieces(&tok, "\u{e9}a"),
        ["<0xC3>", "a"],
        "a byte with no token is dropped"
    );
}

#[test]
fn special_tokens_split_out_first_and_never_merge() {
    let tok = bpe(
        &["<pad>", "<eos>", "<bos>", "a", "\u{2581}", "\u{2581}a"],
        &["\u{2581} a"],
    );
    assert_eq!(
        pieces(&tok, "<bos> a<eos>"),
        ["<bos>", "\u{2581}a", "<eos>"]
    );
    assert!(tok.0.encode("").is_empty());
}

fn metadata(model: Option<&str>, merges: Option<&[&str]>) -> HashMap<String, GGUFValue> {
    let mut m = HashMap::new();
    if let Some(model) = model {
        m.insert(
            "tokenizer.ggml.model".to_string(),
            GGUFValue::String(model.to_string()),
        );
    }
    if let Some(merges) = merges {
        m.insert(
            "tokenizer.ggml.merges".to_string(),
            GGUFValue::Array(
                merges
                    .iter()
                    .map(|r| GGUFValue::String((*r).to_string()))
                    .collect(),
            ),
        );
    }
    m
}

#[test]
fn only_gemma4_is_spm_bpe() {
    assert!(is_spm_bpe(&metadata(Some("gemma4"), None)));
    for other in ["llama", "gpt2", "bpe", "Gemma4", "gemma"] {
        assert!(!is_spm_bpe(&metadata(Some(other), None)), "{other}");
    }
    assert!(!is_spm_bpe(&metadata(None, None)));
}

#[test]
fn from_gguf_refuses_by_name_and_caches_by_tables() {
    let vocab: Vec<String> = ["a", "b", "ab"].iter().map(|t| (*t).to_string()).collect();
    assert_eq!(
        SpmBpe::from_gguf(&metadata(Some("llama"), Some(&["a b"])), &vocab).err(),
        Some(SpmBpeRefusal::NotSpmBpe("llama".to_string()))
    );
    assert_eq!(
        SpmBpe::from_gguf(&metadata(None, None), &vocab).err(),
        Some(SpmBpeRefusal::NotSpmBpe("absent".to_string()))
    );
    assert_eq!(
        SpmBpe::from_gguf(&metadata(Some("gemma4"), None), &vocab).err(),
        Some(SpmBpeRefusal::MissingMerges)
    );
    let md = metadata(Some("gemma4"), Some(&["a b"]));
    let first = SpmBpe::from_gguf(&md, &vocab).expect("gemma4 with merges");
    let again = SpmBpe::from_gguf(&md, &vocab).expect("gemma4 with merges");
    assert!(Arc::ptr_eq(&first, &again), "the same tables hit the cache");
    assert_eq!(first.encode("ab"), [2]);
}
