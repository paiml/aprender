use super::*;

/// FALSIFY-BPE-UPSTREAM-002 (load_from_files vs load_from_json parity), superseded by #3742:
/// the SAME Qwen2 vocabulary must get the SAME answer from both loaders, and since #3742 that
/// answer is a refusal by name. `load_from_json` refuses it for its declared regex
/// pre-tokenizer; `load_from_files` (the `apr tokenize import-hf` layout, PR #1596) refuses it
/// for its GPT-2 byte glyphs, because that layout carries no pre-tokenizer. Host-dependent:
/// skipped where the extracted files are absent.
#[test]
fn falsify_bpe_load_from_files_matches_load_from_json_encode() {
    let vocab_path = "/tmp/qwen-0.5b-tokenizer-extracted/vocab.json";
    let merges_path = "/tmp/qwen-0.5b-tokenizer-extracted/merges.txt";
    // The invoking user's home, exactly as the sibling S1/S2 tokenizer tests
    // resolve it (crates/aprender-core/src/models/qwen2/tests.rs,
    // models/qwen2/falsification.rs). A literal /home/<author>/ here made this
    // bisection unrunnable on every machine but one, while still printing the
    // same "skipping" line (#2532).
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let json_path = format!("{home}/.cache/qwen2/tokenizer.json");
    if !std::path::Path::new(vocab_path).exists()
        || !std::path::Path::new(merges_path).exists()
        || !std::path::Path::new(&json_path).exists()
    {
        eprintln!("[falsify-bpe-upstream-002] skipping: host lacks tokenizer files");
        return;
    }

    let vocab_json = std::fs::read_to_string(vocab_path).expect("read vocab");
    let merges_txt = std::fs::read_to_string(merges_path).expect("read merges");
    let json = std::fs::read_to_string(&json_path).expect("read tokenizer.json");
    for (loader, result) in [
        ("load_from_files", load_from_files(&vocab_json, &merges_txt)),
        ("load_from_json", load_from_json(&json)),
    ] {
        let err = result.expect_err("a Qwen2 byte-level vocabulary is refused (#3742)");
        assert!(err.to_string().contains("#3742"), "{loader}: {err}");
    }
}

/// FALSIFY-BPE-UPSTREAM-001, superseded by #3742: this crate's `BpeTokenizer` must REFUSE a
/// real Qwen2 tokenizer.json rather than encode it. Its pre-tokenizer is a regex; this crate
/// only splits on whitespace, so every id it produced for Qwen text was not the model's (the
/// defect #3726 removed from the GGUF path). apr-cli encodes such vocabularies through
/// realizar's canonical byte-level BPE. This test loads the host's HF-cache copy (skipped if
/// absent) and asserts the refusal names the reason.
#[test]
fn falsify_bpe_qwen_encode_python_does_not_unk_99pct() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let path = format!("{home}/.cache/qwen2/tokenizer.json");
    if !std::path::Path::new(&path).exists() {
        eprintln!(
            "[falsify_bpe_qwen_encode_python_does_not_unk_99pct] skipping: \
             host lacks {path} (test is host-dependent)"
        );
        return;
    }
    let json = std::fs::read_to_string(&path).expect("read tokenizer.json");
    let err =
        load_from_json(&json).expect_err("a regex pre-tokenizer vocabulary is refused (#3742)");
    let msg = err.to_string();
    assert!(
        msg.contains("Split regex") && msg.contains("#3742"),
        "{msg}"
    );
}

#[test]
fn test_encode_without_prefix_space() {
    let config = BpeConfig {
        add_prefix_space: false,
        ..BpeConfig::default()
    };
    let tokenizer = BpeTokenizer::new(config);
    let tokens = tokenizer.encode("test");
    // With empty vocab, should be empty
    assert!(tokens.is_empty());
}

#[test]
fn test_decode_with_unknown_id() {
    let tokenizer = BpeTokenizer::gpt2_base();
    // ID that doesn't exist in vocab
    let decoded = tokenizer.decode(&[999999]);
    // Unknown ID should be skipped
    // Either empty or an empty string - both cases handled by is_empty()
    assert!(decoded.is_empty());
}

#[test]
fn test_decode_skips_special_tokens() {
    let tokenizer = BpeTokenizer::gpt2_base();
    // Decode with special token ID
    // endoftext + 'H' — should not contain the special token text
    let decoded = tokenizer.decode(&[50256, 72]);
    assert!(!decoded.contains("<|endoftext|>"));
}

#[test]
fn test_bytes_to_bpe_tokens_unknown() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    // Force a path through unknown byte handling
    let result = tokenizer.bytes_to_bpe_tokens("A");
    assert!(!result.is_empty());
}

#[test]
fn test_load_from_json_with_special_added_tokens() {
    let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": []
            },
            "added_tokens": [
                {"id": 100, "content": "<special>", "special": true},
                {"id": 101, "content": "normal", "special": false}
            ]
        }"#;
    let result = load_from_json(json);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert!(tokenizer.is_special_token("<special>"));
    assert!(!tokenizer.is_special_token("normal"));
    assert_eq!(tokenizer.token_to_id("normal"), Some(101));
}

#[test]
fn test_load_from_json_whisper_vocab_size() {
    // Create vocab with >50000 entries for whisper detection
    let mut vocab_entries: Vec<String> = Vec::new();
    for i in 0..51000 {
        vocab_entries.push(format!("\"tok{i}\": {i}"));
    }
    let vocab_str = vocab_entries.join(", ");
    let json = format!(
        "{{\"model\": {{\"vocab\": {{ {} }}, \"merges\": []}}, \"added_tokens\": []}}",
        vocab_str
    );
    let result = load_from_json(&json);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_json_qwen2_vocab_size() {
    // Create vocab with >150000 entries for qwen2 detection
    let mut vocab_entries: Vec<String> = Vec::new();
    for i in 0..151000 {
        vocab_entries.push(format!("\"tok{i}\": {i}"));
    }
    let vocab_str = vocab_entries.join(", ");
    let json = format!(
        "{{\"model\": {{\"vocab\": {{ {} }}, \"merges\": []}}, \"added_tokens\": []}}",
        vocab_str
    );
    let result = load_from_json(&json);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_whisper_vocab_size() {
    // Create vocab with >50000 entries
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0..51000u32 {
        vocab.insert(format!("tok{i}"), i);
    }
    let vocab_json = serde_json::to_string(&vocab).expect("serialize");
    let merges = "";
    let result = load_from_files(&vocab_json, merges);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_gpt2_vocab_size() {
    // Create vocab with >40000 but <50000 entries
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0..41000u32 {
        vocab.insert(format!("tok{i}"), i);
    }
    let vocab_json = serde_json::to_string(&vocab).expect("serialize");
    let merges = "";
    let result = load_from_files(&vocab_json, merges);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_qwen2_vocab_size() {
    // Create vocab with >150000 entries
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0..151000u32 {
        vocab.insert(format!("tok{i}"), i);
    }
    let vocab_json = serde_json::to_string(&vocab).expect("serialize");
    let merges = "";
    let result = load_from_files(&vocab_json, merges);
    assert!(result.is_ok());
}

#[test]
fn test_load_from_files_empty_lines() {
    let vocab = "{}";
    let merges = "\n\na b\n\n";
    let result = load_from_files(vocab, merges);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert_eq!(tokenizer.merges.len(), 1);
}

#[test]
fn test_load_from_files_invalid_json() {
    let vocab = "not valid json";
    let merges = "";
    let result = load_from_files(vocab, merges);
    assert!(result.is_err());
}

#[test]
fn test_qwen2_from_json() {
    let json = r#"{
            "model": {
                "vocab": {
                    "<|endoftext|>": 151643,
                    "<|im_start|>": 151644,
                    "<|im_end|>": 151645,
                    "hello": 0
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "special": true},
                {"id": 151644, "content": "<|im_start|>", "special": true},
                {"id": 151645, "content": "<|im_end|>", "special": true}
            ]
        }"#;
    let result = Qwen2BpeTokenizer::from_json(json);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    assert!(tokenizer.is_eos(151645));
    assert!(tokenizer.is_bos(151644));
}

#[test]
fn test_qwen2_from_json_default_ids() {
    // Test with vocab missing special tokens - should use defaults
    let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1},
                "merges": []
            },
            "added_tokens": []
        }"#;
    let result = Qwen2BpeTokenizer::from_json(json);
    assert!(result.is_ok());

    let tokenizer = result.expect("load failed");
    // Should use default IDs
    assert_eq!(tokenizer.im_start_id(), Qwen2BpeTokenizer::IM_START_ID);
    assert_eq!(tokenizer.im_end_id(), Qwen2BpeTokenizer::IM_END_ID);
}

#[test]
fn test_qwen2_from_file_not_found() {
    let result = Qwen2BpeTokenizer::from_file("/nonexistent/path/tokenizer.json");
    assert!(result.is_err());
}

#[test]
fn test_merge_rule_debug_clone() {
    let rule = MergeRule::new("a", "b");
    let cloned = rule.clone();
    assert_eq!(rule, cloned);

    // Test Debug
    let debug_str = format!("{:?}", rule);
    assert!(debug_str.contains("MergeRule"));
}

#[test]
fn test_bpe_tokenizer_debug_clone() {
    let tokenizer = BpeTokenizer::gpt2_base();
    let cloned = tokenizer.clone();
    assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());

    // Test Debug
    let debug_str = format!("{:?}", tokenizer);
    assert!(debug_str.contains("BpeTokenizer"));
}

#[test]
fn test_qwen2_tokenizer_debug_clone() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let cloned = tokenizer.clone();
    assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());

    // Test Debug
    let debug_str = format!("{:?}", tokenizer);
    assert!(debug_str.contains("Qwen2BpeTokenizer"));
}

#[test]
fn test_bpe_config_debug_clone() {
    let config = BpeConfig::default();
    let cloned = config.clone();
    assert_eq!(config.vocab_size, cloned.vocab_size);

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("BpeConfig"));
}

#[test]
fn test_encode_unk_token_fallback() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    // Add unk token to vocab
    tokenizer.add_special_token("<unk>", 0);

    // Encode text - unknown bytes should fall back to unk token
    let tokens = tokenizer.encode("x");
    // Should either be empty or have unk token
    assert!(tokens.is_empty() || tokens.contains(&0));
}

#[test]
fn test_pre_tokenize_multiple_spaces() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    let words = tokenizer.pre_tokenize("hello  world");
    // Should handle multiple spaces
    assert!(words.len() >= 2);
}

#[test]
fn test_pre_tokenize_leading_space() {
    let tokenizer = BpeTokenizer::new(BpeConfig::default());
    let words = tokenizer.pre_tokenize(" hello");
    assert!(!words.is_empty());
    // First word should start with space
    assert!(words[0].starts_with(' '));
}

#[test]
fn test_bpe_tokens_to_bytes_invalid_chars() {
    let tokenizer = BpeTokenizer::gpt2_base();
    // String with chars not in byte_decoder
    let result = tokenizer.bpe_tokens_to_bytes("αβγ");
    // Should handle gracefully (lossy conversion)
    // Result might be empty or partial
    let _ = result;
}

#[test]
fn test_qwen2_encode_special_tokens() {
    let tokenizer = Qwen2BpeTokenizer::new();
    let text = "<|im_start|>user";
    let tokens = tokenizer.encode(text);
    // Should contain the special token ID
    assert!(tokens.contains(&151644));
}

#[test]
fn test_bpe_merge_priority() {
    let mut tokenizer = BpeTokenizer::new(BpeConfig::default());
    // Add merges with specific priority order
    tokenizer.add_merge("x", "y"); // rank 0 (highest priority)
    tokenizer.add_merge("a", "b"); // rank 1

    // Test that lower rank (higher priority) merge is applied first
    let tokens = vec![
        "a".to_string(),
        "b".to_string(),
        "x".to_string(),
        "y".to_string(),
    ];
    let result = tokenizer.bpe(&tokens);
    // Both merges should be applied
    assert_eq!(result.len(), 2);
}

/// PMAT-751: with add_prefix_space=true (GPT-2), the prefix space must be applied to
/// EVERY non-special segment — including one that follows a special token — matching
/// HuggingFace ByteLevel. Pre-fix, encode_segment gated the prefix space on
/// `ids.is_empty()`, so a post-special segment lost its leading-space marker and
/// produced token IDs diverging from HF. Vocab-independent falsifier: the IMPLICIT
/// prefix space on a post-special segment must equal the EXPLICIT one.
#[test]
fn test_pmat751_prefix_space_after_special_token() {
    let t = BpeTokenizer::gpt2_base(); // add_prefix_space=true, <|endoftext|> registered
                                       // "b" after the special must be tokenized as if it were " b" (implicit prefix space).
    let implicit = t.encode("a<|endoftext|>b");
    let explicit = t.encode("a<|endoftext|> b");
    assert_eq!(
        implicit, explicit,
        "PMAT-751: post-special segment 'b' must get the same prefix space as explicit ' b' \
         (HF ByteLevel applies add_prefix_space per non-special chunk). implicit={implicit:?} explicit={explicit:?}"
    );
    // And the first segment still gets it (no regression): "a" begins with the space marker.
    let first = t.encode("a");
    let spaced = t.encode(" a");
    assert_eq!(
        first, spaced,
        "PMAT-751: first-segment prefix space regressed"
    );
}

/// #3742: the refusal is the enforcement. `BpeTokenizer::pre_tokenize` only splits on
/// whitespace, so every byte-level vocabulary is refused by name: a declared `Split` regex, a
/// `ByteLevel` of either kind, or GPT-2 byte glyphs (`Ġ` and `Ċ`) in the vocabulary, through
/// either loader. A vocabulary that is none of these still loads.
#[test]
fn load_from_json_refuses_every_byte_level_vocabulary_and_keeps_the_rest() {
    let body = |vocab: &str, pre: &str| {
        format!(
            r#"{{"model": {{"type": "BPE", "vocab": {vocab}, "merges": ["a b"]}},
                "added_tokens": [], "pre_tokenizer": {pre}}}"#
        )
    };
    let plain = r#"{"a": 0, "b": 1, "ab": 2}"#;
    let glyphs = r#"{"a": 0, "b": 1, "ab": 2, "\u0120": 3, "\u010a": 4}"#;
    let refused = |json: &str, why: &str| {
        let err = load_from_json(json).expect_err(why);
        assert!(err.to_string().contains("#3742"), "{why}: {err}");
    };
    refused(
        &body(
            plain,
            r#"{"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{N}"}, "behavior": "Isolated"}]}"#,
        ),
        "a Split regex is refused",
    );
    refused(
        &body(plain, r#"{"type": "ByteLevel", "add_prefix_space": false}"#),
        "GPT-2's ByteLevel regex is refused",
    );
    refused(
        &body(
            plain,
            r#"{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}"#,
        ),
        "a ByteLevel with no regex does not split at all: refused",
    );
    refused(
        &body(glyphs, "null"),
        "GPT-2 byte glyphs with no pre-tokenizer are refused",
    );
    refused(
        &body(glyphs, r#"{"type": "Whitespace"}"#),
        "GPT-2 byte glyphs under a whitespace pre-tokenizer are refused",
    );
    assert!(
        load_from_json(&body(plain, "null")).is_ok(),
        "a vocabulary that is not byte-level still loads"
    );

    // The vocab.json + merges.txt layout carries no pre-tokenizer: the glyphs decide.
    let err = load_from_files(glyphs, "a b\n").expect_err("byte-level vocab.json is refused");
    assert!(err.to_string().contains("#3742"), "{err}");
    assert!(
        load_from_files(plain, "a b\n").is_ok(),
        "a vocab.json that is not byte-level still loads"
    );
}
