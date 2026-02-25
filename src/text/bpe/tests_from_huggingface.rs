use super::*;
use std::io::Write;

// ============================================================================
// SSC-023: BpeTokenizer::from_huggingface / from_huggingface_json tests
// ============================================================================

/// Helper: Build a minimal HuggingFace tokenizer.json string with given vocab,
/// merges, and added tokens.
fn mock_tokenizer_json(
    vocab: &[(&str, u32)],
    merges: &[&str],
    added_tokens: &[(&str, u32, bool)],
) -> String {
    let vocab_entries: Vec<String> = vocab
        .iter()
        .map(|(tok, id)| {
            // Escape backslashes and quotes for JSON
            let escaped = tok.replace('\\', "\\\\").replace('"', "\\\"");
            format!("\"{escaped}\": {id}")
        })
        .collect();

    let merge_entries: Vec<String> = merges.iter().map(|m| format!("\"{m}\"")).collect();

    let added_entries: Vec<String> = added_tokens
        .iter()
        .map(|(content, id, special)| {
            format!(
                "{{\"id\": {id}, \"content\": \"{content}\", \"special\": {special}}}"
            )
        })
        .collect();

    format!(
        r#"{{
    "model": {{
        "vocab": {{ {} }},
        "merges": [{}]
    }},
    "added_tokens": [{}]
}}"#,
        vocab_entries.join(", "),
        merge_entries.join(", "),
        added_entries.join(", ")
    )
}

// ============================================================================
// from_huggingface_json: basic loading
// ============================================================================

#[test]
fn test_ssc023_from_huggingface_json_basic() {
    let json = mock_tokenizer_json(
        &[("hello", 0), ("world", 1), ("Ġthe", 2)],
        &["he llo", "wo rld"],
        &[],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    assert_eq!(tokenizer.vocab_size(), 3);
    assert_eq!(tokenizer.token_to_id("hello"), Some(0));
    assert_eq!(tokenizer.token_to_id("world"), Some(1));
    assert_eq!(tokenizer.token_to_id("Ġthe"), Some(2));
}

#[test]
fn test_ssc023_from_huggingface_json_merges_preserved() {
    let json = mock_tokenizer_json(
        &[("a", 0), ("b", 1), ("ab", 2), ("c", 3), ("abc", 4)],
        &["a b", "ab c"],
        &[],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    // Merges should be loaded in order (priority = index)
    assert_eq!(tokenizer.merges.len(), 2);
    assert_eq!(tokenizer.merges[0].first, "a");
    assert_eq!(tokenizer.merges[0].second, "b");
    assert_eq!(tokenizer.merges[1].first, "ab");
    assert_eq!(tokenizer.merges[1].second, "c");

    // Merge ranks should reflect priority
    assert_eq!(
        tokenizer
            .merge_ranks
            .get(&("a".to_string(), "b".to_string())),
        Some(&0)
    );
    assert_eq!(
        tokenizer
            .merge_ranks
            .get(&("ab".to_string(), "c".to_string())),
        Some(&1)
    );
}

#[test]
fn test_ssc023_from_huggingface_json_special_tokens() {
    let json = mock_tokenizer_json(
        &[
            ("hello", 0),
            ("<|endoftext|>", 151643),
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
        ],
        &[],
        &[
            ("<|endoftext|>", 151643, true),
            ("<|im_start|>", 151644, true),
            ("<|im_end|>", 151645, true),
        ],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    assert!(tokenizer.is_special_token("<|endoftext|>"));
    assert!(tokenizer.is_special_token("<|im_start|>"));
    assert!(tokenizer.is_special_token("<|im_end|>"));
    assert!(!tokenizer.is_special_token("hello"));

    assert_eq!(tokenizer.token_to_id("<|im_start|>"), Some(151644));
    assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(151645));
}

#[test]
fn test_ssc023_from_huggingface_json_added_non_special_tokens() {
    let json = mock_tokenizer_json(
        &[("hello", 0)],
        &[],
        &[("extra_token", 999, false), ("<special>", 1000, true)],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    // Non-special added token should be in vocab but NOT in special_tokens
    assert_eq!(tokenizer.token_to_id("extra_token"), Some(999));
    assert!(!tokenizer.is_special_token("extra_token"));

    // Special added token should be in both
    assert_eq!(tokenizer.token_to_id("<special>"), Some(1000));
    assert!(tokenizer.is_special_token("<special>"));
}

// ============================================================================
// from_huggingface_json: error cases
// ============================================================================

#[test]
fn test_ssc023_from_huggingface_json_empty_string() {
    let result = BpeTokenizer::from_huggingface_json("");
    assert!(result.is_err());
}

#[test]
fn test_ssc023_from_huggingface_json_invalid_json() {
    let result = BpeTokenizer::from_huggingface_json("not json at all");
    assert!(result.is_err());
}

#[test]
fn test_ssc023_from_huggingface_json_missing_model() {
    let result = BpeTokenizer::from_huggingface_json("{}");
    assert!(result.is_err());
}

#[test]
fn test_ssc023_from_huggingface_json_missing_vocab() {
    let result = BpeTokenizer::from_huggingface_json(
        r#"{"model": {"merges": []}, "added_tokens": []}"#,
    );
    // serde should fail because vocab is required in HfModel
    assert!(result.is_err());
}

// ============================================================================
// from_huggingface: file-based loading
// ============================================================================

#[test]
fn test_ssc023_from_huggingface_file_basic() {
    let json = mock_tokenizer_json(
        &[("hello", 0), ("world", 1)],
        &["he llo"],
        &[("<|endoftext|>", 50256, true)],
    );

    let dir = tempfile::tempdir().expect("create temp dir");
    let file_path = dir.path().join("tokenizer.json");
    let mut file = std::fs::File::create(&file_path).expect("create file");
    file.write_all(json.as_bytes()).expect("write file");

    let tokenizer =
        BpeTokenizer::from_huggingface(&file_path).expect("should load from file");

    assert_eq!(tokenizer.vocab_size(), 3); // hello + world + <|endoftext|>
    assert!(tokenizer.is_special_token("<|endoftext|>"));
}

#[test]
fn test_ssc023_from_huggingface_file_not_found() {
    let result = BpeTokenizer::from_huggingface("/nonexistent/path/tokenizer.json");
    assert!(result.is_err());
}

#[test]
fn test_ssc023_from_huggingface_file_invalid_json() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let file_path = dir.path().join("tokenizer.json");
    std::fs::write(&file_path, "not valid json").expect("write file");

    let result = BpeTokenizer::from_huggingface(&file_path);
    assert!(result.is_err());
}

// ============================================================================
// Encode/decode roundtrip with loaded tokenizer
// ============================================================================

#[test]
fn test_ssc023_encode_decode_roundtrip_loaded() {
    // Build a mini tokenizer with byte-level vocab + some merges
    // Add byte-level characters (the GPT-2 style byte encoder maps bytes to
    // printable Unicode chars). We'll add a few common ASCII tokens directly.
    let (byte_enc, _) = bytes_to_unicode();
    let mut byte_tokens: Vec<(String, u32)> = Vec::new();
    for b in 0..=255u8 {
        if let Some(&c) = byte_enc.get(&b) {
            byte_tokens.push((c.to_string(), u32::from(b)));
        }
    }

    // Build vocab entries from byte tokens
    let byte_refs: Vec<(&str, u32)> = byte_tokens
        .iter()
        .map(|(s, id)| (s.as_str(), *id))
        .collect();

    // Add merged tokens
    let mut all_vocab: Vec<(&str, u32)> = byte_refs;
    all_vocab.push(("He", 256));
    all_vocab.push(("ll", 257));
    all_vocab.push(("Hell", 258));
    all_vocab.push(("Hello", 259));

    let json = mock_tokenizer_json(
        &all_vocab,
        &["H e", "l l", "He ll", "Hell o"],
        &[],
    );

    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    // "Hello" should encode to [259] via merges: H+e→He, l+l→ll, He+ll→Hell, Hell+o→Hello
    let ids = tokenizer.encode("Hello");
    assert!(!ids.is_empty(), "encode should produce tokens");

    // Decode back
    let decoded = tokenizer.decode(&ids);
    assert_eq!(decoded, "Hello", "roundtrip decode should match original");
}

#[test]
fn test_ssc023_encode_empty_input() {
    let json = mock_tokenizer_json(&[("hello", 0)], &[], &[]);
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    let ids = tokenizer.encode("");
    assert!(ids.is_empty(), "encoding empty string should produce no tokens");
}

#[test]
fn test_ssc023_decode_empty_ids() {
    let json = mock_tokenizer_json(&[("hello", 0)], &[], &[]);
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    let text = tokenizer.decode(&[]);
    assert!(text.is_empty(), "decoding empty IDs should produce empty string");
}

// ============================================================================
// Deterministic encoding
// ============================================================================

#[test]
fn test_ssc023_deterministic_encoding() {
    let json = mock_tokenizer_json(
        &[("a", 0), ("b", 1), ("ab", 2), ("c", 3)],
        &["a b"],
        &[],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    // Same input must always produce same output
    let ids1 = tokenizer.encode("abc");
    let ids2 = tokenizer.encode("abc");
    assert_eq!(ids1, ids2, "encoding must be deterministic");

    // Running again after clone
    let tokenizer2 = tokenizer.clone();
    let ids3 = tokenizer2.encode("abc");
    assert_eq!(ids1, ids3, "cloned tokenizer must produce identical encoding");
}

// ============================================================================
// Special token handling in encode/decode
// ============================================================================

#[test]
fn test_ssc023_encode_with_special_tokens_in_text() {
    let json = mock_tokenizer_json(
        &[
            ("hello", 0),
            ("<|im_start|>", 100),
            ("<|im_end|>", 101),
        ],
        &[],
        &[
            ("<|im_start|>", 100, true),
            ("<|im_end|>", 101, true),
        ],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    let ids = tokenizer.encode("<|im_start|>hello<|im_end|>");

    // Should contain the special token IDs
    assert!(
        ids.contains(&100),
        "should encode <|im_start|> as single token"
    );
    assert!(
        ids.contains(&101),
        "should encode <|im_end|> as single token"
    );
}

#[test]
fn test_ssc023_decode_skips_special_tokens() {
    let json = mock_tokenizer_json(
        &[("hello", 0), ("<|endoftext|>", 50256)],
        &[],
        &[("<|endoftext|>", 50256, true)],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    let decoded = tokenizer.decode(&[50256, 0]);
    // Special tokens should be skipped during decode
    assert!(
        !decoded.contains("<|endoftext|>"),
        "special tokens should be skipped in decode output"
    );
}

// ============================================================================
// byte_encoder integrity
// ============================================================================

#[test]
fn test_ssc023_byte_encoder_built_correctly() {
    let json = mock_tokenizer_json(&[("hello", 0)], &[], &[]);
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    // byte_encoder should map all 256 bytes
    assert_eq!(
        tokenizer.byte_encoder.len(),
        256,
        "byte_encoder must cover all 256 byte values"
    );

    // byte_decoder should be the exact inverse
    assert_eq!(
        tokenizer.byte_decoder.len(),
        256,
        "byte_decoder must cover all 256 chars"
    );

    // Verify invertibility
    for b in 0..=255u8 {
        let c = tokenizer
            .byte_encoder
            .get(&b)
            .expect("byte_encoder should have entry for every byte");
        let b_back = tokenizer
            .byte_decoder
            .get(c)
            .expect("byte_decoder should have entry for every mapped char");
        assert_eq!(
            b, *b_back,
            "byte_encoder/decoder must be inverse for byte {b}"
        );
    }
}

// ============================================================================
// Qwen2-scale vocab size detection
// ============================================================================

#[test]
fn test_ssc023_config_detection_qwen2_vocab() {
    // Build a JSON with >150K vocab entries to trigger Qwen2 config detection
    let mut vocab_entries: Vec<String> = Vec::new();
    for i in 0..151_000u32 {
        vocab_entries.push(format!("\"tok{i}\": {i}"));
    }
    let json = format!(
        "{{\"model\": {{\"vocab\": {{ {} }}, \"merges\": []}}, \"added_tokens\": []}}",
        vocab_entries.join(", ")
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load large vocab");

    // Config should be auto-detected as Qwen2 (no prefix space, unk = <|endoftext|>)
    assert!(
        !tokenizer.config.add_prefix_space,
        "Qwen2 config should not add prefix space"
    );
    assert_eq!(
        tokenizer.config.vocab_size,
        crate::demo::Qwen2Config::VOCAB_SIZE,
        "Qwen2 config should have VOCAB_SIZE = 151936"
    );
}

// ============================================================================
// Unicode and edge cases
// ============================================================================

#[test]
fn test_ssc023_unicode_tokens_in_vocab() {
    let json = mock_tokenizer_json(
        &[("日本語", 0), ("世界", 1)],
        &[],
        &[],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load unicode vocab");

    assert_eq!(tokenizer.token_to_id("日本語"), Some(0));
    assert_eq!(tokenizer.token_to_id("世界"), Some(1));
    assert_eq!(tokenizer.id_to_token(0), Some("日本語"));
}

#[test]
fn test_ssc023_id_to_token_roundtrip() {
    let json = mock_tokenizer_json(
        &[("alpha", 10), ("beta", 20), ("gamma", 30)],
        &[],
        &[],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load from valid JSON");

    // Every vocab entry should have a working id_to_token inverse
    assert_eq!(tokenizer.id_to_token(10), Some("alpha"));
    assert_eq!(tokenizer.id_to_token(20), Some("beta"));
    assert_eq!(tokenizer.id_to_token(30), Some("gamma"));
    assert_eq!(tokenizer.id_to_token(999), None);
}

#[test]
fn test_ssc023_merge_rules_without_matching_vocab() {
    // Merge rules reference tokens not in vocab - should still load without error
    let json = mock_tokenizer_json(
        &[("x", 0)],
        &["a b", "c d"],
        &[],
    );
    let tokenizer =
        BpeTokenizer::from_huggingface_json(&json).expect("should load even with orphan merges");

    assert_eq!(tokenizer.merges.len(), 2);
}

#[test]
fn test_ssc023_from_huggingface_json_no_added_tokens_field() {
    // added_tokens is optional (serde(default))
    let json = r#"{"model": {"vocab": {"hello": 0}, "merges": []}}"#;
    let tokenizer =
        BpeTokenizer::from_huggingface_json(json).expect("should handle missing added_tokens");

    assert_eq!(tokenizer.vocab_size(), 1);
}

// ============================================================================
// Integration with Qwen2BpeTokenizer
// ============================================================================

#[test]
fn test_ssc023_qwen2_from_json_uses_from_huggingface_json_path() {
    // Verify that Qwen2BpeTokenizer::from_json and BpeTokenizer::from_huggingface_json
    // produce equivalent base tokenizers when given the same input
    let json = mock_tokenizer_json(
        &[
            ("hello", 0),
            ("<|endoftext|>", 151643),
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
        ],
        &[],
        &[
            ("<|endoftext|>", 151643, true),
            ("<|im_start|>", 151644, true),
            ("<|im_end|>", 151645, true),
        ],
    );

    let qwen2 =
        Qwen2BpeTokenizer::from_json(&json).expect("Qwen2BpeTokenizer::from_json should work");
    let base =
        BpeTokenizer::from_huggingface_json(&json).expect("from_huggingface_json should work");

    // Both should have the same vocabulary
    assert_eq!(qwen2.base.vocab_size(), base.vocab_size());
    assert_eq!(
        qwen2.base.token_to_id("hello"),
        base.token_to_id("hello")
    );
}
