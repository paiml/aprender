use super::*;

/// FALSIFY-BPE-UPSTREAM-002 (load_from_files vs load_from_json parity):
/// Loading the SAME Qwen2 vocab via vocab.json+merges.txt MUST produce
/// the same encode behavior as loading via tokenizer.json. The encoder
/// pipeline downstream is the same; only the load path differs.
///
/// CONTEXT: PR #1596 routes encode-corpus through `load_from_files`
/// when tokenizer.json is absent. PR #1596's LIVE smoke produced 99%
/// `<unk>` despite `load_from_json` producing 0% on the same vocab
/// (FALSIFY-BPE-UPSTREAM-001 above). This test bisects: if RED, the
/// gap is in `load_from_files`'s setup (likely missing added_tokens
/// / merge format / pretokenizer config).
#[test]
fn falsify_bpe_load_from_files_matches_load_from_json_encode() {
    let vocab_path = "/tmp/qwen-0.5b-tokenizer-extracted/vocab.json";
    let merges_path = "/tmp/qwen-0.5b-tokenizer-extracted/merges.txt";
    let json_path = "/home/noah/.cache/qwen2/tokenizer.json";
    if !std::path::Path::new(vocab_path).exists()
        || !std::path::Path::new(merges_path).exists()
        || !std::path::Path::new(json_path).exists()
    {
        eprintln!("[falsify-bpe-upstream-002] skipping: host lacks tokenizer files");
        return;
    }

    let vocab_json = std::fs::read_to_string(vocab_path).expect("read vocab");
    let merges_txt = std::fs::read_to_string(merges_path).expect("read merges");
    let from_files = load_from_files(&vocab_json, &merges_txt).expect("load_from_files ok");

    let json = std::fs::read_to_string(json_path).expect("read tokenizer.json");
    let from_json = load_from_json(&json).expect("load_from_json ok");

    let text = "def fibonacci(n):\n    return n\n";
    let ids_files = from_files.encode(text);
    let ids_json = from_json.encode(text);

    eprintln!("[upstream-002] from_files: {ids_files:?}");
    eprintln!("[upstream-002] from_json:  {ids_json:?}");
    eprintln!(
        "[upstream-002] from_files vocab_size={}, from_json vocab_size={}",
        from_files.vocab_size(),
        from_json.vocab_size()
    );

    // Find unk count in each — Qwen2 unk is <|endoftext|> id 151643.
    let unk_id = 151643_u32;
    let unk_in_files = ids_files.iter().filter(|&&id| id == unk_id).count();
    let unk_in_json = ids_json.iter().filter(|&&id| id == unk_id).count();
    let total_files = ids_files.len();
    let total_json = ids_json.len();
    eprintln!(
        "[upstream-002] from_files: {total_files} tokens, {unk_in_files} unks ({:.3}%)",
        if total_files > 0 {
            unk_in_files as f32 / total_files as f32 * 100.0
        } else {
            0.0
        }
    );
    eprintln!(
        "[upstream-002] from_json:  {total_json} tokens, {unk_in_json} unks ({:.3}%)",
        if total_json > 0 {
            unk_in_json as f32 / total_json as f32 * 100.0
        } else {
            0.0
        }
    );

    let ratio_files = unk_in_files as f32 / total_files.max(1) as f32;
    let ratio_json = unk_in_json as f32 / total_json.max(1) as f32;

    // Assert: load_from_files's unk_ratio is within 5% of load_from_json's.
    // load_from_json achieves ~0% on this input (verified by upstream-001).
    // If load_from_files diverges by >5%, the load path is broken.
    let divergence = (ratio_files - ratio_json).abs();
    assert!(
        divergence < 0.05,
        "FALSIFY-BPE-UPSTREAM-002: load_from_files unk_ratio={ratio_files:.4} \
         diverges from load_from_json unk_ratio={ratio_json:.4} by {divergence:.4} \
         (>0.05). The two loaders SHOULD produce equivalent encoders for the \
         same vocab. Fix scope: align load_from_files setup with load_from_json \
         (likely added_tokens registration, merge format, or pretokenizer config)."
    );
}

/// FALSIFY-BPE-UPSTREAM-001 (SHIP-TWO §60 root-cause #2):
/// `BpeTokenizer::encode` with a Qwen2-style vocab MUST NOT return
/// 99%+ `<unk>` tokens for Python source text. This test loads a
/// real Qwen2 tokenizer.json from the host's HF cache (skipped if
/// not present) and asserts the encode entropy is sensible.
///
/// CONTEXT: PR #1596's encode-corpus dispatch routes Qwen vocab to
/// this encoder, but the encoder itself produces 99% `<unk>`
/// (entropy 0.111 bits / 17.21 max). Diagnosis pending bisection.
/// See evidence/section-60-5g-2-redispatch-2026-05-09/README.md.
#[test]
fn falsify_bpe_qwen_encode_python_does_not_unk_99pct() {
    let path = "/home/noah/.cache/qwen2/tokenizer.json";
    if !std::path::Path::new(path).exists() {
        eprintln!(
            "[falsify_bpe_qwen_encode_python_does_not_unk_99pct] skipping: \
             host lacks {path} (test is host-dependent for upstream H1C \
             investigation)"
        );
        return;
    }
    let json = std::fs::read_to_string(path).expect("read tokenizer.json");
    let tokenizer = load_from_json(&json).expect("load_from_json succeeds on Qwen2");

    // Python source: a self-contained, well-formed snippet using only
    // tokens that any Qwen-tokenizer-trained Python corpus should
    // recognize. NOT a contrived edge case.
    let text = "def fibonacci(n):\n    if n < 2:\n        return n\n    \
                return fibonacci(n - 1) + fibonacci(n - 2)\n";
    let ids = tokenizer.encode(text);

    // Find the unk_token id (Qwen2: <|endoftext|>).
    let unk_id = tokenizer.token_to_id(&BpeConfig::qwen2().unk_token);
    let unk_count = if let Some(unk) = unk_id {
        ids.iter().filter(|&&id| id == unk).count()
    } else {
        0
    };
    let total = ids.len();
    let unk_ratio = if total > 0 {
        unk_count as f32 / total as f32
    } else {
        0.0
    };

    eprintln!(
        "[falsify-bpe-upstream-001] text_bytes={}, encoded_tokens={total}, \
         unk_count={unk_count}, unk_ratio={unk_ratio:.4}, unk_id={unk_id:?}",
        text.len()
    );

    // Industry-baseline: a working byte-level BPE encoder on Python
    // source produces unk_ratio < 0.05 (under 5% unk on common code).
    // We accept < 0.50 as the bare-minimum sanity bound; if RED, the
    // upstream encoder is broken.
    assert!(
        unk_ratio < 0.50,
        "FALSIFY-BPE-UPSTREAM-001: BpeTokenizer::encode on Qwen2 vocab \
         produced unk_ratio={unk_ratio} (>{}); 99% `<unk>` defect class. \
         encoded_tokens={total}, unk_count={unk_count}, unk_id={unk_id:?}. \
         See evidence/section-60-5g-2-redispatch-2026-05-09/ + the §60 \
         val_loss=0.0008 root-cause cascade.",
        0.50
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
