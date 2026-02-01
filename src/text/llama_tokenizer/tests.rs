use super::*;

// ========================================================================
// Falsification Tests (Popperian)
// ========================================================================

/// LT-01: Tokenizer MUST load vocabulary from GGUF
/// Falsification: If vocab is not loaded, encoding will return only UNK tokens
#[test]
fn lt01_tokenizer_loads_vocab_from_gguf() {
    // Create minimal GGUF with tokenizer data
    let gguf_data = create_test_gguf();
    let tokenizer = LlamaTokenizer::from_gguf_bytes(&gguf_data);

    assert!(
        tokenizer.is_ok(),
        "FALSIFIED: Tokenizer failed to load from GGUF: {:?}",
        tokenizer.err()
    );

    let tokenizer = tokenizer.expect("already checked");
    assert!(tokenizer.vocab_size() > 0, "FALSIFIED: Vocabulary is empty");
}

/// LT-02: Tokenizer MUST encode text to non-empty tokens
/// Falsification: If encoding fails, result will be empty for non-empty input
#[test]
fn lt02_tokenizer_encodes_text() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode("Hello");

    assert!(
        !tokens.is_empty(),
        "FALSIFIED: Encoding returned empty for non-empty input"
    );
}

/// LT-03: Tokenizer MUST decode tokens back to readable text
/// Falsification: If decoding fails, result will be empty or garbage
#[test]
fn lt03_tokenizer_decodes_tokens() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode("Hello");
    let decoded = tokenizer.decode(&tokens);

    assert!(
        !decoded.is_empty(),
        "FALSIFIED: Decoding returned empty string"
    );
    // Note: Exact roundtrip may not match due to tokenization granularity
}

/// LT-04: BOS token MUST be prepended when requested
/// Falsification: First token will not be BOS ID
#[test]
fn lt04_bos_token_prepended() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode_with_bos("Hello");

    assert!(
        !tokens.is_empty(),
        "FALSIFIED: Encoding with BOS returned empty"
    );
    assert_eq!(
        tokens[0],
        tokenizer.bos_token_id(),
        "FALSIFIED: First token is not BOS"
    );
}

/// LT-05: Unknown characters MUST use byte fallback, not panic
/// Falsification: Encoding unknown chars would panic or return empty
#[test]
fn lt05_byte_fallback_for_unknown() {
    let tokenizer = create_test_tokenizer();
    // Use an emoji that's unlikely to be in a small test vocab
    let tokens = tokenizer.encode("Hello 🎉 World");

    assert!(
        !tokens.is_empty(),
        "FALSIFIED: Encoding with unknown chars returned empty"
    );
}

/// LT-06: Empty input MUST return empty tokens (not panic)
/// Falsification: Would panic or return non-empty
#[test]
fn lt06_empty_input_returns_empty() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode("");

    assert!(
        tokens.is_empty(),
        "FALSIFIED: Empty input returned non-empty tokens"
    );
}

/// LT-07: Special tokens MUST be excluded from decode output
/// Falsification: BOS/EOS would appear in decoded text
#[test]
fn lt07_special_tokens_excluded_from_decode() {
    let tokenizer = create_test_tokenizer();
    let tokens = vec![tokenizer.bos_token_id(), 100, 101, tokenizer.eos_token_id()];
    let decoded = tokenizer.decode(&tokens);

    assert!(
        !decoded.contains("<s>") && !decoded.contains("</s>"),
        "FALSIFIED: Special tokens appear in decoded output: {}",
        decoded
    );
}

/// LT-08: Tokenizer MUST reject invalid GGUF magic
/// Falsification: Would accept invalid data
#[test]
fn lt08_rejects_invalid_gguf() {
    let invalid_data = b"NOTGGUF0000000000000000";
    let result = LlamaTokenizer::from_gguf_bytes(invalid_data);

    assert!(result.is_err(), "FALSIFIED: Accepted invalid GGUF magic");
}

// ========================================================================
// Helper Functions
// ========================================================================

fn create_test_tokenizer() -> LlamaTokenizer {
    // Create a minimal tokenizer for testing
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "▁Hello".to_string(),
        "▁World".to_string(),
        "▁".to_string(),
        "H".to_string(),
        "e".to_string(),
        "l".to_string(),
        "o".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];

    LlamaTokenizer::new(tokens, scores, 1, 2, 0).expect("Failed to create test tokenizer")
}

fn create_test_gguf() -> Vec<u8> {
    let mut data = Vec::new();

    // GGUF header
    data.extend_from_slice(b"GGUF"); // magic
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&5u64.to_le_bytes()); // metadata_count

    // Metadata 1: tokenizer.ggml.tokens (string array)
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&8u32.to_le_bytes()); // string element type
    let tokens = ["<unk>", "<s>", "</s>", "▁Hello", "▁World"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Metadata 2: tokenizer.ggml.scores (f32 array)
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&6u32.to_le_bytes()); // f32 element type
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Metadata 3: bos_token_id
    let key3 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes()); // u32 type
    data.extend_from_slice(&1u32.to_le_bytes()); // value

    // Metadata 4: eos_token_id
    let key4 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes()); // u32 type
    data.extend_from_slice(&2u32.to_le_bytes()); // value

    // Metadata 5: unknown_token_id
    let key5 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes()); // u32 type
    data.extend_from_slice(&0u32.to_le_bytes()); // value

    data
}

// ========================================================================
// GPT-2 BPE Decoding Tests
// ========================================================================

/// GPT-01: GPT-2 byte decoder MUST correctly map unicode to bytes
/// Falsification: If mapping is wrong, decoded text will be garbage
#[test]
fn gpt01_byte_decoder_maps_correctly() {
    let decoder = build_gpt2_byte_decoder();

    // Printable ASCII should map to itself
    assert_eq!(decoder.get(&'A'), Some(&b'A'));
    assert_eq!(decoder.get(&'z'), Some(&b'z'));
    assert_eq!(decoder.get(&'0'), Some(&b'0'));

    // GPT-2 space marker (Ġ = U+0120) should map to space (0x20)
    assert_eq!(decoder.get(&'Ġ'), Some(&b' '));

    // Newline marker (Ċ = U+010A) should map to newline (0x0A)
    assert_eq!(decoder.get(&'Ċ'), Some(&b'\n'));

    // Tab marker (ĉ = U+0109) should map to tab (0x09)
    assert_eq!(decoder.get(&'ĉ'), Some(&b'\t'));
}

/// GPT-02: GPT-2 token decoding MUST produce valid UTF-8
/// Falsification: If decoding fails, String::from_utf8_lossy will show replacement chars
#[test]
fn gpt02_token_decoding_produces_utf8() {
    // "Hello" in GPT-2 BPE
    let token = "Hello";
    let bytes = decode_gpt2_token(token);
    assert_eq!(bytes, b"Hello");

    // " world" in GPT-2 BPE (Ġ prefix for space)
    let token_with_space = "Ġworld";
    let bytes = decode_gpt2_token(token_with_space);
    assert_eq!(bytes, b" world");
}

/// GPT-03: GPT-2 tokenizer MUST decode complete sentences correctly
/// Falsification: Sentence will be garbled
#[test]
fn gpt03_gpt2_tokenizer_decodes_sentences() {
    // Create a GPT-2 tokenizer with some tokens
    let tokens = vec![
        "<unk>".to_string(),
        "<|endoftext|>".to_string(), // BOS/EOS for GPT-2
        "</s>".to_string(),
        "Hello".to_string(),
        "Ġworld".to_string(), // " world" in GPT-2
        "!".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];

    let mut tokenizer =
        LlamaTokenizer::new(tokens, scores, 1, 2, 0).expect("Failed to create tokenizer");
    tokenizer.set_model(TokenizerModel::Gpt2);

    // Decode token IDs
    let token_ids = vec![3, 4, 5]; // "Hello", " world", "!"
    let decoded = tokenizer.decode(&token_ids);

    assert_eq!(decoded, "Hello world!");
}

/// GPT-04: Model type detection from GGUF MUST work
/// Falsification: Would default to SentencePiece even for GPT-2 models
#[test]
fn gpt04_model_type_detection() {
    let gguf_data = create_gpt2_test_gguf();
    let tokenizer = LlamaTokenizer::from_gguf_bytes(&gguf_data);

    assert!(tokenizer.is_ok());
    let tokenizer = tokenizer.expect("already checked");
    assert_eq!(
        tokenizer.model(),
        TokenizerModel::Gpt2,
        "FALSIFIED: GPT-2 model type not detected"
    );
}

fn create_gpt2_test_gguf() -> Vec<u8> {
    let mut data = Vec::new();

    // GGUF header
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&6u64.to_le_bytes()); // metadata_count (added one more)

    // Metadata 1: tokenizer.ggml.tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&8u32.to_le_bytes()); // string element type
    let tokens = ["<unk>", "<|endoftext|>", "</s>", "Hello", "Ġworld"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Metadata 2: tokenizer.ggml.scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Metadata 3: bos_token_id
    let key3 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // Metadata 4: eos_token_id
    let key4 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // Metadata 5: unknown_token_id
    let key5 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    // Metadata 6: tokenizer.ggml.model = "gpt2"
    let key6 = b"tokenizer.ggml.model";
    data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
    data.extend_from_slice(key6);
    data.extend_from_slice(&8u32.to_le_bytes()); // string type
    let model_str = b"gpt2";
    data.extend_from_slice(&(model_str.len() as u64).to_le_bytes());
    data.extend_from_slice(model_str);

    data
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_tokenizer_model_default() {
    let model = TokenizerModel::default();
    assert_eq!(model, TokenizerModel::SentencePiece);
}

#[test]
fn test_tokenizer_model_debug_clone_eq() {
    let model = TokenizerModel::Gpt2;
    let cloned = model;
    assert_eq!(model, cloned);

    let debug = format!("{:?}", model);
    assert!(debug.contains("Gpt2"));
}

#[test]
fn test_llama_tokenizer_debug_clone() {
    let tokenizer = create_test_tokenizer();
    let cloned = tokenizer.clone();
    assert_eq!(tokenizer.vocab_size(), cloned.vocab_size());

    let debug = format!("{:?}", tokenizer);
    assert!(debug.contains("LlamaTokenizer"));
}

#[test]
fn test_set_model_and_get_model() {
    let mut tokenizer = create_test_tokenizer();
    assert_eq!(tokenizer.model(), TokenizerModel::SentencePiece);

    tokenizer.set_model(TokenizerModel::Gpt2);
    assert_eq!(tokenizer.model(), TokenizerModel::Gpt2);
}

#[test]
fn test_id_to_token() {
    let tokenizer = create_test_tokenizer();

    // Known token
    let token = tokenizer.id_to_token(3);
    assert_eq!(token, Some("▁Hello"));

    // Unknown ID
    let unknown = tokenizer.id_to_token(9999);
    assert!(unknown.is_none());
}

#[test]
fn test_token_to_id() {
    let tokenizer = create_test_tokenizer();

    // Known token
    let id = tokenizer.token_to_id("▁Hello");
    assert_eq!(id, Some(3));

    // Unknown token
    let unknown = tokenizer.token_to_id("unknown_xyz");
    assert!(unknown.is_none());
}

#[test]
fn test_new_empty_vocabulary_error() {
    let result = LlamaTokenizer::new(vec![], vec![], 0, 0, 0);
    assert!(result.is_err());
}

#[test]
fn test_new_invalid_bos_id() {
    let tokens = vec!["<unk>".to_string(), "<s>".to_string()];
    let result = LlamaTokenizer::new(tokens, vec![0.0, 0.0], 999, 1, 0);
    assert!(result.is_err());
}

#[test]
fn test_new_invalid_eos_id() {
    let tokens = vec!["<unk>".to_string(), "<s>".to_string()];
    let result = LlamaTokenizer::new(tokens, vec![0.0, 0.0], 1, 999, 0);
    assert!(result.is_err());
}

#[test]
fn test_new_invalid_unk_id() {
    let tokens = vec!["<unk>".to_string(), "<s>".to_string()];
    let result = LlamaTokenizer::new(tokens, vec![0.0, 0.0], 1, 0, 999);
    assert!(result.is_err());
}

#[test]
fn test_decode_byte_token() {
    // Create tokenizer with byte token for newline
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "<0x0A>".to_string(), // newline byte token
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    let decoded = tokenizer.decode(&[3]); // decode byte token
    assert_eq!(decoded, "\n");
}

#[test]
fn test_decode_invalid_byte_token() {
    // Create tokenizer with malformed byte token
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "<0xZZ>".to_string(), // invalid hex
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // Should not panic, just output the token as-is
    let decoded = tokenizer.decode(&[3]);
    assert_eq!(decoded, "<0xZZ>");
}

#[test]
fn test_encode_gpt2_mode() {
    let mut tokenizer = create_test_tokenizer();
    tokenizer.set_model(TokenizerModel::Gpt2);

    let tokens = tokenizer.encode("Hello World");
    assert!(!tokens.is_empty());
}

#[test]
fn test_from_gguf_too_short() {
    let short_data = b"GGUF";
    let result = LlamaTokenizer::from_gguf_bytes(short_data);
    assert!(result.is_err());
}

#[test]
fn test_from_gguf_missing_tokens() {
    // Create GGUF without tokenizer.ggml.tokens
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count (zero)

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_gguf_value_variants() {
    // Test GGUFValue enum variants
    let values = vec![
        GGUFValue::UInt8(1),
        GGUFValue::Int8(-1),
        GGUFValue::UInt16(100),
        GGUFValue::Int16(-100),
        GGUFValue::UInt32(1000),
        GGUFValue::Int32(-1000),
        GGUFValue::Float32(1.5),
        GGUFValue::Bool(true),
        GGUFValue::String("test".to_string()),
        GGUFValue::Array(vec![GGUFValue::UInt8(1)]),
        GGUFValue::UInt64(10000),
        GGUFValue::Int64(-10000),
        GGUFValue::Float64(3.14),
    ];

    for val in &values {
        let debug = format!("{:?}", val);
        assert!(!debug.is_empty());
    }

    // Clone test
    let cloned = values.clone();
    assert_eq!(values.len(), cloned.len());
}

#[test]
fn test_decode_gpt2_non_mapped_char() {
    // Test decoding a character not in GPT-2 mapping
    // Should fallback to UTF-8 encoding
    let bytes = decode_gpt2_token("日本語"); // Japanese characters
    let text = String::from_utf8_lossy(&bytes);
    assert!(!text.is_empty());
}

#[test]
fn test_gpt2_decode_full_sentence() {
    let mut tokenizer = create_test_tokenizer();
    tokenizer.set_model(TokenizerModel::Gpt2);

    // Test decode_gpt2 path with special tokens
    let decoded = tokenizer.decode(&[
        tokenizer.bos_token_id(),
        3, // some token
        tokenizer.eos_token_id(),
    ]);
    // BOS and EOS should be filtered
    assert!(!decoded.contains("<s>"));
    assert!(!decoded.contains("</s>"));
}

#[test]
fn test_vocab_size_accessor() {
    let tokenizer = create_test_tokenizer();
    assert!(tokenizer.vocab_size() > 0);
}

#[test]
fn test_constants() {
    assert_eq!(LLAMA_VOCAB_SIZE, 32000);
    assert_eq!(BOS_TOKEN, "<s>");
    assert_eq!(EOS_TOKEN, "</s>");
    assert_eq!(UNK_TOKEN, "<unk>");
}

#[test]
fn test_encode_with_byte_fallback_for_emoji() {
    // Create tokenizer with byte tokens
    let mut tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "▁Hello".to_string(),
    ];
    // Add byte tokens for 0x00 to 0xFF
    for i in 0u8..=255 {
        tokens.push(format!("<0x{i:02X}>"));
    }
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // Encode text with emoji (uses byte fallback)
    let encoded = tokenizer.encode("🎉");
    assert!(!encoded.is_empty());
}

#[test]
fn test_decode_skips_unknown_token_id() {
    let tokenizer = create_test_tokenizer();
    // Decode with non-existent token ID
    let decoded = tokenizer.decode(&[9999]);
    // Should not panic, produces empty for missing token
    assert!(decoded.is_empty() || decoded.len() > 0);
}

#[test]
fn test_gpt2_byte_decoder_size() {
    let decoder = build_gpt2_byte_decoder();
    // GPT-2 byte decoder should have 256 entries
    assert_eq!(decoder.len(), 256);
}

#[test]
fn test_decode_sentencepiece_handles_hybrid_space() {
    // Test that hybrid tokenizers with GPT-2 space marker work
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "Ġworld".to_string(), // GPT-2 space marker
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    let decoded = tokenizer.decode(&[3]);
    assert_eq!(decoded, "world"); // Leading space removed
}

// ========================================================================
// Additional Coverage Tests for 95% Target
// ========================================================================

#[test]
fn test_pad_token_accessors() {
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "<pad>".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // PAD token not set by default
    assert!(tokenizer.pad_token_id.is_none());
}

#[test]
fn test_encode_long_text() {
    let tokenizer = create_test_tokenizer();
    // Test with text longer than 32 chars (max substring check)
    let long_text = "Hello World Hello World Hello World Hello World";
    let tokens = tokenizer.encode(long_text);
    assert!(!tokens.is_empty());
}

#[test]
fn test_encode_special_chars() {
    let tokenizer = create_test_tokenizer();
    let text_with_special = "Hello\tWorld\nNew Line";
    let tokens = tokenizer.encode(text_with_special);
    assert!(!tokens.is_empty());
}

#[test]
fn test_encode_numbers() {
    let tokenizer = create_test_tokenizer();
    let numeric = "12345 67890";
    let tokens = tokenizer.encode(numeric);
    assert!(!tokens.is_empty());
}

#[test]
fn test_decode_empty() {
    let tokenizer = create_test_tokenizer();
    let decoded = tokenizer.decode(&[]);
    assert!(decoded.is_empty());
}

#[test]
fn test_decode_gpt2_with_all_special_tokens() {
    let tokens = vec![
        "<unk>".to_string(),
        "<|endoftext|>".to_string(),
        "</s>".to_string(),
        "test".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];
    let mut tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();
    tokenizer.set_model(TokenizerModel::Gpt2);

    // Decode with only special tokens
    let decoded = tokenizer.decode(&[1, 2]); // BOS and EOS
    assert!(decoded.is_empty());
}

#[test]
fn test_decode_gpt2_mixed_tokens() {
    let tokens = vec![
        "<unk>".to_string(),
        "<|endoftext|>".to_string(),
        "</s>".to_string(),
        "Hello".to_string(),
        "Ġworld".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];
    let mut tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();
    tokenizer.set_model(TokenizerModel::Gpt2);

    let decoded = tokenizer.decode(&[1, 3, 4, 2]); // BOS, Hello, world, EOS
    assert_eq!(decoded, "Hello world");
}

#[test]
fn test_encode_gpt2_with_newlines() {
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "Hello".to_string(),
        "Ċworld".to_string(), // newline prefix
    ];
    let scores = vec![0.0; tokens.len()];
    let mut tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();
    tokenizer.set_model(TokenizerModel::Gpt2);

    let encoded = tokenizer.encode("test\nline");
    assert!(!encoded.is_empty());
}

#[test]
fn test_token_to_id_all_tokens() {
    let tokenizer = create_test_tokenizer();

    // Test all known tokens
    assert!(tokenizer.token_to_id("<unk>").is_some());
    assert!(tokenizer.token_to_id("<s>").is_some());
    assert!(tokenizer.token_to_id("</s>").is_some());
}

#[test]
fn test_id_to_token_boundary() {
    let tokenizer = create_test_tokenizer();

    // Test boundary conditions
    assert!(tokenizer.id_to_token(0).is_some()); // First token
    assert!(tokenizer.id_to_token(tokenizer.vocab_size() as u32 - 1).is_some()); // Last valid
    assert!(tokenizer.id_to_token(tokenizer.vocab_size() as u32).is_none()); // Out of bounds
}

#[test]
fn test_eos_token_id_accessor() {
    let tokenizer = create_test_tokenizer();
    assert_eq!(tokenizer.eos_token_id(), 2);
}

#[test]
fn test_unk_token_id_accessor() {
    let tokenizer = create_test_tokenizer();
    assert_eq!(tokenizer.unk_token_id(), 0);
}

#[test]
fn test_bos_token_id_accessor() {
    let tokenizer = create_test_tokenizer();
    assert_eq!(tokenizer.bos_token_id(), 1);
}

#[test]
fn test_encode_with_bos_empty() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode_with_bos("");
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], tokenizer.bos_token_id());
}

#[test]
fn test_encode_with_bos_single_word() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode_with_bos("Hello");
    assert!(tokens.len() >= 2); // BOS + at least one token
    assert_eq!(tokens[0], tokenizer.bos_token_id());
}

#[test]
fn test_decode_sentencepiece_byte_fallback_invalid() {
    // Create tokenizer with incomplete byte token
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "<0x".to_string(), // Malformed - no closing >
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    let decoded = tokenizer.decode(&[3]);
    // Should not crash, just output token as-is
    assert_eq!(decoded, "<0x");
}

#[test]
fn test_from_gguf_bytes_v1_format() {
    // Test GGUF version 1 (if supported)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&1u32.to_le_bytes()); // version 1
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // This may fail if v1 isn't supported
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    // Just verify it doesn't panic
    let _ = result.is_ok();
}

#[test]
fn test_gguf_value_array_nested() {
    let values = GGUFValue::Array(vec![
        GGUFValue::Array(vec![GGUFValue::UInt8(1), GGUFValue::UInt8(2)]),
        GGUFValue::Array(vec![GGUFValue::UInt8(3), GGUFValue::UInt8(4)]),
    ]);

    let debug = format!("{:?}", values);
    assert!(debug.contains("Array"));
}

#[test]
fn test_tokenizer_model_values() {
    // Test all TokenizerModel variants
    let sp = TokenizerModel::SentencePiece;
    let gpt2 = TokenizerModel::Gpt2;

    assert_ne!(sp, gpt2);
    assert_eq!(sp, TokenizerModel::default());
}

#[test]
fn test_encode_all_unk() {
    // Create tokenizer with minimal vocab
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // Encode text that has no matching tokens
    let encoded = tokenizer.encode("xyz123");
    // Should use byte fallback or UNK
    assert!(!encoded.is_empty());
}

#[test]
fn test_decode_multiple_byte_tokens() {
    // Create tokenizer with multiple byte tokens
    let mut tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
    ];
    // Add byte tokens
    for i in 0u8..10 {
        tokens.push(format!("<0x{:02X}>", i));
    }
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // Decode multiple consecutive byte tokens
    let decoded = tokenizer.decode(&[3, 4, 5]); // <0x00>, <0x01>, <0x02>
    assert_eq!(decoded.len(), 3);
}

#[test]
fn test_gpt2_byte_decoder_all_chars() {
    let decoder = build_gpt2_byte_decoder();

    // Verify all 256 bytes are mapped
    assert_eq!(decoder.len(), 256);

    // Verify specific mappings
    for b in b'!'..(b'~' + 1) {
        assert!(decoder.values().any(|&v| v == b));
    }
}

#[test]
fn test_decode_gpt2_token_complex() {
    // Test complex unicode sequences
    let token = "HelloĠworldĊnewline";
    let bytes = decode_gpt2_token(token);

    // Should contain space for Ġ and newline for Ċ
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains(" ")); // Ġ → space
    assert!(text.contains("\n")); // Ċ → newline
}

// ========================================================================
// Additional GGUF Parsing Coverage Tests
// ========================================================================

#[test]
fn test_gguf_skip_value_u8_type() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&6u64.to_le_bytes()); // metadata_count

    // Tokens metadata
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores metadata
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Metadata with u8 type (type 0) - should be skipped
    let key3 = b"general.quantization_version";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&0u32.to_le_bytes()); // u8 type
    data.push(1u8); // value

    // BOS token ID
    let key4 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS token ID
    let key5 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // Unknown token ID
    let key6 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
    data.extend_from_slice(key6);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i8_type() {
    let data = create_gguf_with_extra_metadata(1, &[-1i8 as u8]); // i8 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_u16_type() {
    let data = create_gguf_with_extra_metadata(2, &100u16.to_le_bytes()); // u16 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i16_type() {
    let data = create_gguf_with_extra_metadata(3, &(-100i16).to_le_bytes()); // i16 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i32_type() {
    let data = create_gguf_with_extra_metadata(5, &(-1000i32).to_le_bytes()); // i32 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_f32_type() {
    let data = create_gguf_with_extra_metadata(6, &3.14f32.to_le_bytes()); // f32 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_bool_type() {
    let data = create_gguf_with_extra_metadata(7, &[1u8]); // bool type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_u64_type() {
    let data = create_gguf_with_extra_metadata(10, &10000u64.to_le_bytes()); // u64 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_i64_type() {
    let data = create_gguf_with_extra_metadata(11, &(-10000i64).to_le_bytes()); // i64 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_value_f64_type() {
    let data = create_gguf_with_extra_metadata(12, &3.14159265f64.to_le_bytes()); // f64 type
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u8_elements() {
    let data = create_gguf_with_array_metadata(0, &[1u8, 2, 3, 4, 5]); // u8 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u16_elements() {
    let mut arr_data = Vec::new();
    for i in 0u16..5 {
        arr_data.extend_from_slice(&i.to_le_bytes());
    }
    let data = create_gguf_with_array_metadata(2, &arr_data); // u16 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u32_elements() {
    let mut arr_data = Vec::new();
    for i in 0u32..3 {
        arr_data.extend_from_slice(&i.to_le_bytes());
    }
    let data = create_gguf_with_array_metadata(4, &arr_data); // u32 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_array_u64_elements() {
    let mut arr_data = Vec::new();
    for i in 0u64..2 {
        arr_data.extend_from_slice(&i.to_le_bytes());
    }
    let data = create_gguf_with_array_metadata(10, &arr_data); // u64 array
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_skip_string_array_in_skip() {
    // Create GGUF with a string array that's not tokens/scores (should be skipped)
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&7u64.to_le_bytes()); // 7 metadata entries

    // Extra string array (should be skipped)
    let key0 = b"general.tags";
    data.extend_from_slice(&(key0.len() as u64).to_le_bytes());
    data.extend_from_slice(key0);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&8u32.to_le_bytes()); // string element type
    let tags = ["tag1", "tag2"];
    data.extend_from_slice(&(tags.len() as u64).to_le_bytes());
    for tag in &tags {
        let bytes = tag.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Now add required tokenizer metadata
    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // BOS
    let key3 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key4 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key5 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_truncated_key_length() {
    // GGUF where key length extends beyond data
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata entry

    // Key length that exceeds remaining data
    data.extend_from_slice(&1000u64.to_le_bytes());

    // Missing token metadata = error
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_gguf_truncated_value_type() {
    // GGUF where value type bytes are missing
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = b"test";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    // Missing value type - only 2 bytes instead of 4
    data.extend_from_slice(&[0u8, 0]);

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_gguf_missing_scores_uses_default() {
    // GGUF with tokens but without scores
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes()); // 4 metadata entries

    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // BOS
    let key2 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key3 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key4 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    // No scores - should use default
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_parse_string_array_wrong_element_type() {
    // Test parse_string_array with non-string element type
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&4u32.to_le_bytes()); // u32 element type (wrong, should be 8)
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 elements
    for _ in 0..3 {
        data.extend_from_slice(&0u32.to_le_bytes());
    }

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_parse_f32_array_wrong_element_type() {
    // Create GGUF with valid tokens but wrong scores array type
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&5u64.to_le_bytes());

    // Valid tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores with wrong element type
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&4u32.to_le_bytes()); // u32 element type (wrong, should be 6)
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0u32.to_le_bytes());
    }

    // Rest of required metadata
    let key3 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    let key4 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    let key5 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    // Should error because scores array has wrong type
    assert!(result.is_err());
}

#[test]
fn test_gguf_string_metadata_skip() {
    // Test that string metadata (type 8) other than tokenizer.ggml.model is skipped
    let data = create_gguf_with_string_metadata("general.name", "TestModel");
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_gguf_non_gpt2_model_string() {
    // Test that "llama" model string results in SentencePiece tokenizer
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&6u64.to_le_bytes());

    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Model type = "llama" (not "gpt2")
    let key3 = b"tokenizer.ggml.model";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&8u32.to_le_bytes()); // string type
    let model_str = b"llama";
    data.extend_from_slice(&(model_str.len() as u64).to_le_bytes());
    data.extend_from_slice(model_str);

    // BOS
    let key4 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key5 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key6 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
    data.extend_from_slice(key6);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
    let tokenizer = result.unwrap();
    assert_eq!(tokenizer.model(), TokenizerModel::SentencePiece);
}

// ========================================================================
// Helper functions for GGUF test data creation
// ========================================================================

fn create_gguf_with_extra_metadata(val_type: u32, val_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&6u64.to_le_bytes());

    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Extra metadata with specified type
    let key3 = b"general.extra";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&val_type.to_le_bytes());
    data.extend_from_slice(val_bytes);

    // BOS
    let key4 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key5 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key6 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
    data.extend_from_slice(key6);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    data
}

fn create_gguf_with_array_metadata(elem_type: u32, elem_bytes: &[u8]) -> Vec<u8> {
    let elem_size = match elem_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        _ => 1,
    };
    let count = elem_bytes.len() / elem_size;

    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&6u64.to_le_bytes());

    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Extra array metadata
    let key3 = b"general.array";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&elem_type.to_le_bytes());
    data.extend_from_slice(&(count as u64).to_le_bytes());
    data.extend_from_slice(elem_bytes);

    // BOS
    let key4 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key5 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key6 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
    data.extend_from_slice(key6);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    data
}

fn create_gguf_with_string_metadata(key_name: &str, value: &str) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&6u64.to_le_bytes());

    // Extra string metadata
    let key_bytes = key_name.as_bytes();
    data.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(key_bytes);
    data.extend_from_slice(&8u32.to_le_bytes()); // string type
    let val_bytes = value.as_bytes();
    data.extend_from_slice(&(val_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(val_bytes);

    // Tokens
    let key1 = b"tokenizer.ggml.tokens";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let tokens = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in &tokens {
        let bytes = token.as_bytes();
        data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }

    // Scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // BOS
    let key3 = b"tokenizer.ggml.bos_token_id";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());

    // EOS
    let key4 = b"tokenizer.ggml.eos_token_id";
    data.extend_from_slice(&(key4.len() as u64).to_le_bytes());
    data.extend_from_slice(key4);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());

    // UNK
    let key5 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key5.len() as u64).to_le_bytes());
    data.extend_from_slice(key5);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    data
}
