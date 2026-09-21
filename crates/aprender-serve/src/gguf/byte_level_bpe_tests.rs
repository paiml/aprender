//! #3726: case tables for the byte-level BPE port. The splitter rows are worked through
//! llama.cpp's `unicode_regex_split_custom_qwen2` by hand; the end-to-end id parity against
//! the pinned llama.cpp on real GGUF files is `scripts/tokenizer_parity.sh`.

use super::*;

fn split(pre: PreTokenizer, s: &str) -> Vec<&str> {
    pre.split(s)
}

#[test]
fn byte_glyphs_are_gpt2_bytes_to_unicode() {
    let g = byte_glyphs();
    let distinct: std::collections::HashSet<char> = g.iter().copied().collect();
    assert_eq!(distinct.len(), 256, "every byte needs its own glyph");
    assert_eq!(g[usize::from(b'A')], 'A');
    assert_eq!(g[usize::from(b'~')], '~');
    assert_eq!(g[usize::from(b' ')], '\u{0120}', "space is Ġ");
    assert_eq!(g[usize::from(b'\n')], '\u{010A}', "newline is Ċ");
    assert_eq!(
        g[0x00], '\u{0100}',
        "the first non-printable byte is U+0100"
    );
    assert_eq!(
        g[0xAD], '\u{0143}',
        "soft hyphen, the last non-printable byte, is U+0143"
    );
    assert_eq!(g[0xE9], '\u{00E9}', "printable Latin-1 maps to itself");
}

#[test]
fn qwen2_split_case_table() {
    let q = PreTokenizer::Qwen2;
    let rows: &[(&str, &[&str])] = &[
        ("Hello world", &["Hello", " world"]),
        // \s+(?!\S): a run of spaces gives its last one to the word after it
        ("   Title", &["  ", " Title"]),
        (
            "a/docs/roadmaps/roadmap.yaml",
            &["a", "/docs", "/roadmaps", "/roadmap", ".yaml"],
        ),
        // \p{N} is one digit per piece
        ("100644", &["1", "0", "0", "6", "4", "4"]),
        ("bed4ce6b", &["bed", "4", "ce", "6", "b"]),
        ("it's", &["it", "'s"]),
        ("WE'RE", &["WE", "'RE"]),
        // punctuation run with its trailing newlines
        ("!!!\n\nx", &["!!!\n\n", "x"]),
        (" !!!", &[" !!!"]),
        // \s*[\r\n]+ then \s+(?!\S) then a word with its space
        ("x  \n  y", &["x", "  \n", " ", " y"]),
        // trailing whitespace at the end of the text is one piece
        ("a  ", &["a", "  "]),
        ("a ", &["a", " "]),
        // a lone space before a digit stays alone
        ("x 1", &["x", " ", "1"]),
        // non-ASCII letters are letters; U+200B (a format char) is "other"
        ("naïve café", &["naïve", " café"]),
        ("a\u{200B}b", &["a", "\u{200B}b"]),
        ("────────", &["────────"]),
        ("日本語テキスト", &["日本語テキスト"]),
    ];
    for (input, want) in rows {
        assert_eq!(&split(q, input), want, "qwen2 split of {input:?}");
    }
}

#[test]
fn qwen35_takes_combining_marks_into_letter_runs_and_qwen2_does_not() {
    let decomposed = "e\u{301}t\u{e9}"; // e + COMBINING ACUTE, t, é
    assert_eq!(split(PreTokenizer::Qwen35, decomposed), vec![decomposed]);
    assert_eq!(
        split(PreTokenizer::Qwen2, decomposed),
        vec!["e", "\u{301}t\u{e9}"]
    );
}

#[test]
fn split_pieces_concatenate_to_the_input() {
    let text = "fn main() {\n    let x = \"quorum\"; // it's 100% ok\r\n\t\tdone  \n}\n  ";
    for pre in [PreTokenizer::Qwen2, PreTokenizer::Qwen35] {
        assert_eq!(
            pre.split(text).concat(),
            text,
            "{pre:?} must not drop or add text"
        );
    }
}

/// A synthetic byte-level vocabulary: the 256 glyphs (ids 0..256), then `extra`.
fn vocab_with(extra: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = byte_glyphs().iter().map(char::to_string).collect();
    v.extend(extra.iter().map(|s| (*s).to_string()));
    v
}

fn id_of(vocab: &[String], t: &str) -> u32 {
    vocab.iter().position(|v| v == t).expect("in vocab") as u32
}

#[test]
fn merges_apply_lowest_rank_first() {
    let vocab = vocab_with(&["ab", "bc", "abc"]);
    // "b c" outranks "a b": abc -> a|bc -> abc
    let bpe = ByteLevelBpe::build(PreTokenizer::Qwen2, &vocab, &["b c", "a b", "a bc"], &[]);
    assert_eq!(bpe.encode("abc"), vec![id_of(&vocab, "abc")]);
    // "a b" outranks "b c", and there is no "ab c": abc -> ab|c
    let bpe = ByteLevelBpe::build(PreTokenizer::Qwen2, &vocab, &["a b", "b c"], &[]);
    assert_eq!(
        bpe.encode("abc"),
        vec![id_of(&vocab, "ab"), id_of(&vocab, "c")]
    );
}

#[test]
fn equal_ranks_merge_leftmost_first() {
    let vocab = vocab_with(&["aa"]);
    let bpe = ByteLevelBpe::build(PreTokenizer::Qwen2, &vocab, &["a a"], &[]);
    assert_eq!(
        bpe.encode("aaa"),
        vec![id_of(&vocab, "aa"), id_of(&vocab, "a")]
    );
}

#[test]
fn merges_never_cross_pre_token_boundaries() {
    // " b" exists as a merge, but "a" and " b" are different pieces: no "a b" token forms.
    let vocab = vocab_with(&["\u{0120}b", "a\u{0120}b"]);
    let bpe = ByteLevelBpe::build(
        PreTokenizer::Qwen2,
        &vocab,
        &["\u{0120} b", "a \u{0120}b"],
        &[],
    );
    assert_eq!(
        bpe.encode("a b"),
        vec![id_of(&vocab, "a"), id_of(&vocab, "\u{0120}b")]
    );
}

#[test]
fn non_ascii_bytes_are_their_glyphs_never_id_zero() {
    let vocab = vocab_with(&[]);
    let bpe = ByteLevelBpe::build(PreTokenizer::Qwen35, &vocab, &[], &[]);
    let glyphs = byte_glyphs();
    // é = C3 A9; ─ = E2 94 80
    let want: Vec<u32> = "é─"
        .bytes()
        .map(|b| id_of(&vocab, &glyphs[usize::from(b)].to_string()))
        .collect();
    let got = bpe.encode("é─");
    assert_eq!(got, want);
    assert!(
        !got.contains(&0) || want.contains(&0),
        "the old encoder's id-0 fallback is gone"
    );
}

#[test]
fn special_tokens_are_partitioned_longest_first_by_token_type() {
    // types: 1 normal, 3 control, 4 user-defined
    let vocab = vocab_with(&["<|im_start|>", "<think>", "<|im", "user"]);
    let mut types = vec![1i32; vocab.len()];
    let im = id_of(&vocab, "<|im_start|>");
    let think = id_of(&vocab, "<think>");
    types[im as usize] = 3;
    types[think as usize] = 4;
    types[id_of(&vocab, "<|im") as usize] = 3;
    let bpe = ByteLevelBpe::build(
        PreTokenizer::Qwen35,
        &vocab,
        &["u s", "us e", "use r"],
        &types,
    );
    let got = bpe.encode("<|im_start|>user<think>");
    assert_eq!(got.first(), Some(&im), "the longer special wins over <|im");
    assert_eq!(
        got.last(),
        Some(&think),
        "a USER_DEFINED token is special too (the old <|..|> rule missed <think>)"
    );
}

#[test]
fn unknown_pre_tokenizer_is_refused_by_name() {
    let mut md = HashMap::new();
    md.insert(
        "tokenizer.ggml.pre".to_string(),
        GGUFValue::String("llama-bpe".to_string()),
    );
    md.insert(
        "tokenizer.ggml.merges".to_string(),
        GGUFValue::Array(vec![]),
    );
    let err =
        ByteLevelBpe::from_gguf(&md, &vocab_with(&[])).expect_err("llama-bpe is not implemented");
    assert_eq!(
        err,
        ByteLevelBpeRefusal::UnknownPreTokenizer("llama-bpe".to_string())
    );
    md.remove("tokenizer.ggml.pre");
    assert_eq!(
        ByteLevelBpe::from_gguf(&md, &vocab_with(&[])).expect_err("no pre"),
        ByteLevelBpeRefusal::MissingPreTokenizer
    );
}

#[test]
fn decode_inverts_encode_for_every_byte_and_special() {
    // #3742: the .apr decoder this replaces lost non-ASCII bytes and whitespace glyphs.
    let vocab = vocab_with(&["<|im_start|>"]);
    let mut types = vec![1i32; vocab.len()];
    let im = vocab.len() - 1;
    types[im] = 3;
    let bpe = ByteLevelBpe::build(PreTokenizer::Qwen35, &vocab, &[], &types);
    let text = "a  b\t\r\n<|im_start|>é─ 日本\u{200B}!!!  ";
    assert_eq!(bpe.decode(&bpe.encode(text)), text);
    let all_bytes: String =
        String::from_utf8_lossy(&(0x20..0x7f).collect::<Vec<u8>>()).into_owned();
    assert_eq!(bpe.decode(&bpe.encode(&all_bytes)), all_bytes);
}
