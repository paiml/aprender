use super::*;

#[test]
fn test_decode_sentencepiece_no_leading_space() {
    // Token that doesn't start with space prefix
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "abc".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    let decoded = tokenizer.decode(&[3]);
    // "abc" has no leading space, so no trimming
    assert_eq!(decoded, "abc");
}

#[test]
fn test_decode_byte_token_tab() {
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "<0x09>".to_string(), // tab byte token
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    let decoded = tokenizer.decode(&[3]);
    assert_eq!(decoded, "\t");
}

#[test]
fn test_gguf_truncated_metadata_boundary() {
    // Test where offset + 8 exceeds data length during metadata loop
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes()); // Claim 10 entries but data ends

    // Only provide partial data for first entry key length
    data.extend_from_slice(&[0u8; 4]); // Partial - only 4 bytes of key len (need 8)

    // Should handle gracefully (break out of loop)
    let result = LlamaTokenizer::from_gguf_bytes(&data);
    // Will fail because tokens not found, but should not panic
    assert!(result.is_err());
}

#[test]
fn test_gguf_tokenizer_model_non_gpt2_string() {
    // tokenizer.ggml.model = "sentencepiece" (not "gpt2") should default to SentencePiece
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
    let tokens_arr = ["<unk>", "<s>", "</s>"];
    data.extend_from_slice(&(tokens_arr.len() as u64).to_le_bytes());
    for token in &tokens_arr {
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
    data.extend_from_slice(&(tokens_arr.len() as u64).to_le_bytes());
    for _ in &tokens_arr {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Model type = "sentencepiece"
    let key3 = b"tokenizer.ggml.model";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&8u32.to_le_bytes()); // string type
    let model_str = b"sentencepiece";
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

    // UNK - not provided (only 5 metadata, need 6th)
    // Actually we said 6, so add it
    let key6 = b"tokenizer.ggml.unknown_token_id";
    data.extend_from_slice(&(key6.len() as u64).to_le_bytes());
    data.extend_from_slice(key6);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    let result = LlamaTokenizer::from_gguf_bytes(&data);
    assert!(result.is_ok());
    let tokenizer = result.expect("should succeed");
    assert_eq!(tokenizer.model(), TokenizerModel::SentencePiece);
}

#[test]
fn test_encode_multibyte_unicode_byte_fallback() {
    // Create tokenizer with byte tokens to test multi-byte UTF-8 byte fallback
    let mut tokens_vec = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    // Add all byte tokens
    for i in 0u8..=255 {
        tokens_vec.push(format!("<0x{i:02X}>"));
    }
    let scores = vec![0.0; tokens_vec.len()];
    let tokenizer = LlamaTokenizer::new(tokens_vec, scores, 1, 2, 0).unwrap();

    // Chinese character takes 3 bytes in UTF-8
    let encoded = tokenizer.encode("\u{4e16}"); // "世"
                                                // Should produce 3 byte tokens (0xE4, 0xB8, 0x96) plus the leading ▁
    assert!(!encoded.is_empty());
}

// ========================================================================
// BUG-TOK-001: Multibyte UTF-8 Byte Token Decoding
// ========================================================================

/// BUG-TOK-001: Byte tokens >= 128 MUST combine into valid UTF-8
///
/// Falsification: Previous code used `byte as char` which produces garbage
/// for multibyte UTF-8 sequences. For example:
/// - <0xE4><0xB8><0x96> should decode to "世" (Chinese character)
/// - Old code produced "ä¸\u{0096}" (Latin Extended + cedilla + control)
#[test]
fn test_bug_tok_001_multibyte_utf8_byte_tokens() {
    // Create tokenizer with all byte tokens
    let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    for i in 0u8..=255 {
        tokens.push(format!("<0x{i:02X}>"));
    }
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // UTF-8 bytes for "世" (Chinese character for "world")
    // U+4E16 = 0xE4 0xB8 0x96 in UTF-8
    // Token IDs: 3 + 0xE4 = 3 + 228 = 231, 3 + 0xB8 = 187, 3 + 0x96 = 153
    let token_ids = vec![
        3 + 0xE4, // <0xE4>
        3 + 0xB8, // <0xB8>
        3 + 0x96, // <0x96>
    ];

    let decoded = tokenizer.decode(&token_ids);

    // BUG-TOK-001 FIX: Should decode to "世" not "ä¸\u{0096}"
    assert_eq!(
        decoded, "世",
        "FALSIFIED: Multibyte UTF-8 byte tokens decoded incorrectly. \
         Expected '世' but got '{decoded}'"
    );
}

/// BUG-TOK-001: Mixed byte tokens and regular tokens
#[test]
fn test_bug_tok_001_mixed_byte_and_regular_tokens() {
    let mut tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "Hello".to_string(), // ID 3
    ];
    // Add byte tokens starting at ID 4
    for i in 0u8..=255 {
        tokens.push(format!("<0x{i:02X}>"));
    }
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // "Hello世" = "Hello" + UTF-8(世) = "Hello" + 0xE4 0xB8 0x96
    let token_ids = vec![
        3,        // "Hello"
        4 + 0xE4, // <0xE4>
        4 + 0xB8, // <0xB8>
        4 + 0x96, // <0x96>
    ];

    let decoded = tokenizer.decode(&token_ids);
    assert_eq!(decoded, "Hello世");
}

/// BUG-TOK-001: Emoji decoding via byte tokens
#[test]
fn test_bug_tok_001_emoji_byte_tokens() {
    let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    for i in 0u8..=255 {
        tokens.push(format!("<0x{i:02X}>"));
    }
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // UTF-8 bytes for "🎉" (party popper emoji)
    // U+1F389 = 0xF0 0x9F 0x8E 0x89 in UTF-8
    let token_ids = vec![
        3 + 0xF0, // <0xF0>
        3 + 0x9F, // <0x9F>
        3 + 0x8E, // <0x8E>
        3 + 0x89, // <0x89>
    ];

    let decoded = tokenizer.decode(&token_ids);
    assert_eq!(decoded, "🎉");
}

/// BUG-TOK-001: Invalid UTF-8 byte sequence should use replacement char
#[test]
fn test_bug_tok_001_invalid_utf8_uses_replacement() {
    let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    for i in 0u8..=255 {
        tokens.push(format!("<0x{i:02X}>"));
    }
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // Invalid UTF-8: 0xE4 alone (incomplete sequence)
    let token_ids = vec![3 + 0xE4];

    let decoded = tokenizer.decode(&token_ids);
    // Should use replacement character (�) for invalid UTF-8
    assert!(
        decoded.contains('\u{FFFD}'),
        "Invalid UTF-8 should produce replacement character"
    );
}

#[test]
fn test_parse_string_array_too_short() {
    // Test parse_string_array with data too short for header
    let result = LlamaTokenizer::parse_string_array(&[0u8; 5], 0);
    assert!(result.is_err());
}

#[test]
fn test_parse_f32_array_too_short() {
    // Test parse_f32_array with data too short for header
    let result = LlamaTokenizer::parse_f32_array(&[0u8; 5], 0);
    assert!(result.is_err());
}

pub(crate) fn create_gguf_with_string_metadata(key_name: &str, value: &str) -> Vec<u8> {
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
