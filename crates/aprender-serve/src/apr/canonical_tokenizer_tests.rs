//! #3742: the APR and tokenizer.json loaders name the pre-tokenizer or say they cannot. The
//! id parity against the pinned llama.cpp on real `.apr` files is scripts/tokenizer_parity.sh.

use super::*;

/// The GPT-2 glyph of each byte, then `extra`.
fn byte_level_vocab(extra: &[&str]) -> Vec<String> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut next = 0u32;
    let mut v: Vec<String> = (0..=255u8)
        .map(|b| {
            if printable(b) {
                char::from(b).to_string()
            } else {
                next += 1;
                char::from_u32(255 + next).expect("valid").to_string()
            }
        })
        .collect();
    v.extend(extra.iter().map(|s| (*s).to_string()));
    v
}

const QWEN2_HF_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[test]
fn byte_level_is_detected_by_the_space_and_newline_glyphs() {
    assert!(is_byte_level(&byte_level_vocab(&[]), None));
    assert!(is_byte_level(&byte_level_vocab(&[]), Some("BPE")));
    assert!(
        !is_byte_level(&byte_level_vocab(&[]), Some("llama")),
        "SentencePiece is never byte-level"
    );
    let spm: Vec<String> = ["▁the", "▁a", "b"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert!(!is_byte_level(&spm, None));
}

#[test]
fn apr_pre_tokenizer_prefers_the_declared_name_then_the_architecture() {
    let mut meta = super::super::AprMetadata::default();
    meta.architecture = Some("qwen35".to_string());
    assert_eq!(
        apr_pre_tokenizer(&meta),
        Some((
            PreTokenizer::Qwen35,
            PreSource::Architecture("qwen35".to_string())
        ))
    );
    meta.extra
        .insert("tokenizer.pre_type".to_string(), serde_json::json!("qwen2"));
    assert_eq!(
        apr_pre_tokenizer(&meta),
        Some((
            PreTokenizer::Qwen2,
            PreSource::Declared("qwen2".to_string())
        )),
        "the file's own name wins over the inference"
    );
    meta.extra.insert(
        "tokenizer.pre_type".to_string(),
        serde_json::json!("llama-bpe"),
    );
    assert_eq!(
        apr_pre_tokenizer(&meta),
        None,
        "a declared but unimplemented name is not overridden"
    );
    meta.extra.remove("tokenizer.pre_type");
    meta.architecture = Some("llama".to_string());
    assert_eq!(apr_pre_tokenizer(&meta), None);
}

#[test]
fn hf_regex_names_the_pre_tokenizer_and_rejects_others() {
    assert_eq!(
        PreTokenizer::from_hf_regex(QWEN2_HF_REGEX),
        Some(PreTokenizer::Qwen2)
    );
    assert_eq!(PreTokenizer::from_hf_regex(r"\s+|\w+"), None);
    let seq = serde_json::json!({"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": QWEN2_HF_REGEX}, "behavior": "Isolated", "invert": false},
        {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
    ]});
    assert_eq!(split_regex(&seq).as_deref(), Some(QWEN2_HF_REGEX));
}

#[test]
fn tokenizer_json_builds_the_canonical_encoder_with_its_added_tokens() {
    // base vocab: glyphs + "ab"; added: <|im_start|> (special) and <tool_call> (not special)
    let base = byte_level_vocab(&["ab"]);
    let n = base.len() as u32;
    let json = serde_json::json!({"pre_tokenizer": {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": QWEN2_HF_REGEX}}]}});
    let merges = vec![("a".to_string(), "b".to_string())];
    let added = vec![
        ("<|im_start|>".to_string(), n, true),
        ("<tool_call>".to_string(), n + 1, false),
    ];
    let bpe = canonical_for_tokenizer_json(&json, &base, &merges, &added)
        .expect("qwen2 regex is implemented");
    let ab = n - 1;
    assert_eq!(bpe.encode("<|im_start|>ab<tool_call>"), vec![n, ab, n + 1]);

    let other =
        serde_json::json!({"pre_tokenizer": {"type": "Split", "pattern": {"Regex": r"\s+"}}});
    assert!(canonical_for_tokenizer_json(&other, &base, &merges, &added).is_none());
}

#[test]
fn apr_without_token_types_infers_the_specials_structurally() {
    // "<|im_start|>" is no merge's output (an added token); "<p>" is (an ordinary token).
    let vocab = byte_level_vocab(&["<p", "<p>", "<|im_start|>"]);
    let merges = vec![
        ("<".to_string(), "p".to_string()),
        ("<p".to_string(), ">".to_string()),
    ];
    let mut meta = super::super::AprMetadata::default();
    meta.architecture = Some("qwen2".to_string());
    let bpe = canonical_for_apr(&meta, &vocab, &merges).expect("byte-level qwen2");
    let id = |t: &str| vocab.iter().position(|v| v == t).expect("in vocab") as u32;
    assert_eq!(
        bpe.encode("<|im_start|>"),
        vec![id("<|im_start|>")],
        "an added token is special"
    );
    // "<p>" is ordinary text: the pre-tokenizer splits it into "<p" (a letter run with its
    // one-character prefix) and ">", as llama.cpp does. Partitioned as a special it would
    // have come out as the single id of "<p>".
    assert_eq!(bpe.encode("<p>"), vec![id("<p"), id(">")]);
}
