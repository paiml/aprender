use super::*;
use serde_json::json;

#[test]
fn both_merge_forms_are_read_and_added_tokens_take_their_ids() {
    for merges in [json!(["a b", "ab c"]), json!([["a", "b"], ["ab", "c"]])] {
        let tok = json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, "merges": merges},
            "added_tokens": [{"id": 6, "content": "<|im_start|>", "special": true}]
        });
        let t = embedded_tokenizer_tables(&tok).expect("tables");
        assert_eq!(t.merges, vec!["a b".to_string(), "ab c".to_string()], "{merges}");
        assert_eq!(
            t.vocabulary,
            ["a", "b", "c", "ab", "abc", "", "<|im_start|>"].map(String::from).to_vec(),
            "the added token sits at its own id; the gap at 5 keeps it there"
        );
    }
}

#[test]
fn no_model_vocab_or_a_hostile_id_is_none() {
    assert_eq!(embedded_tokenizer_tables(&json!({"model": {"merges": []}})), None);
    let huge = json!({"model": {"vocab": {"a": 4_000_000_000u64}, "merges": []}});
    assert_eq!(embedded_tokenizer_tables(&huge), None);
}

/// #3803 done_when 2: an `.apr` whose tokenizer was embedded from an ARRAY-merge
/// tokenizer.json encodes like the tokenizer.json itself (through entrenar's `HfTokenizer`,
/// the canonical path #3742 measured against llama.cpp), special tokens included.
#[cfg(feature = "realizar")]
#[test]
fn an_apr_embedded_from_an_array_merge_tokenizer_json_encodes_like_the_json() {
    use aprender::serialization::apr::AprWriter;

    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut next = 0u32;
    let mut vocab = serde_json::Map::new();
    for b in 0..=255u8 {
        let glyph = if printable(b) {
            char::from(b)
        } else {
            next += 1;
            char::from_u32(255 + next).expect("valid")
        };
        vocab.insert(glyph.to_string(), json!(b));
    }
    vocab.insert("ab".to_string(), json!(256));
    vocab.insert("\u{0120}ab".to_string(), json!(257));
    let tokenizer_json = json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"], ["\u{0120}", "ab"]]},
        "added_tokens": [{"id": 258, "content": "<|im_start|>", "special": true}],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"},
             "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]}
    });
    let tables = embedded_tokenizer_tables(&tokenizer_json).expect("tables");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("exported.apr");
    let mut w = AprWriter::new();
    w.set_metadata("architecture", json!("qwen2"));
    w.set_metadata("tokenizer.vocabulary", json!(tables.vocabulary));
    w.set_metadata("tokenizer.merges", json!(tables.merges));
    w.add_tensor_f32("probe", vec![1], &[0.0]);
    w.write(&path).expect("write apr");

    let embedded = realizar::apr::AprV2Model::load(&path)
        .expect("load apr")
        .load_embedded_bpe_tokenizer()
        .expect("embedded tokenizer");
    assert!(embedded.canonical.is_some(), "qwen2 byte-level: canonical");
    let reference = crate::tokenizer::HfTokenizer::from_json(&tokenizer_json.to_string()).expect("json");
    for text in ["<|im_start|>ab ab", "ab é\nab", "x ab<|im_start|>"] {
        assert_eq!(embedded.encode(text), reference.encode(text), "{text:?}");
    }
    assert_eq!(embedded.encode(" ab"), vec![257], "the array-form merges were embedded and ranked");
}
