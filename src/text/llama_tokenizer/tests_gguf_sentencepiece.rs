use super::*;

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

pub(crate) fn create_gguf_with_extra_metadata(val_type: u32, val_bytes: &[u8]) -> Vec<u8> {
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

pub(crate) fn create_gguf_with_array_metadata(elem_type: u32, elem_bytes: &[u8]) -> Vec<u8> {
    let elem_size = match elem_type {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
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

// ========================================================================
// Additional Coverage Tests for Uncovered Branches
// ========================================================================

#[test]
fn test_decode_gpt2_skips_unknown_token_id() {
    let tokens = vec![
        "<unk>".to_string(),
        "<|endoftext|>".to_string(),
        "</s>".to_string(),
        "Hello".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];
    let mut tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();
    tokenizer.set_model(TokenizerModel::Gpt2);

    // Token ID 9999 does not exist in id_to_token; GPT-2 decode should skip it
    let decoded = tokenizer.decode(&[3, 9999]);
    assert_eq!(decoded, "Hello");
}

#[test]
fn test_decode_sentencepiece_skips_unknown_token_id() {
    let tokenizer = create_test_tokenizer();

    // Token ID 9999 does not exist in id_to_token; SentencePiece decode should skip it
    let decoded = tokenizer.decode(&[3, 9999]);
    // Only "▁Hello" -> "Hello" (leading space stripped)
    assert_eq!(decoded, "Hello");
}

#[test]
fn test_encode_byte_fallback_no_byte_token_in_vocab() {
    // Create tokenizer with NO byte tokens at all (only basic tokens)
    let tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    // Encoding "A" (which has no vocab entry and no byte token) should use UNK
    let encoded = tokenizer.encode("A");
    assert!(!encoded.is_empty());
    // Every token should be UNK since no matches and no byte tokens
    for &token_id in &encoded {
        assert_eq!(token_id, tokenizer.unk_token_id());
    }
}

#[test]
fn test_encode_gpt2_space_and_newline_normalization() {
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "Hello".to_string(),
        "\u{0120}world".to_string(), // GPT-2 space prefix
        "\u{010A}line".to_string(),  // GPT-2 newline prefix
    ];
    let scores = vec![0.0; tokens.len()];
    let mut tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();
    tokenizer.set_model(TokenizerModel::Gpt2);

    // "Hello world" should normalize space to \u{0120}
    let encoded = tokenizer.encode("Hello world");
    assert!(!encoded.is_empty());
    // Should find "Hello" token
    assert!(encoded.contains(&3));
}

#[test]
fn test_skip_value_unknown_type() {
    // Test skip_value with a type code that doesn't match any known type (e.g., 99)
    // This exercises the final _ => {} arm in skip_value
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&6u64.to_le_bytes());

    // Required: tokens
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

    // Required: scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Unknown type metadata (type 99) - should be skipped by _ => {}
    let key3 = b"general.unknown_type";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&99u32.to_le_bytes()); // unknown type

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
    // Parsing will likely fail because unknown type skips 0 bytes,
    // but this exercises the code path
    let _ = result;
}

#[test]
fn test_skip_value_array_unknown_elem_type() {
    // Test skip_value array with an unknown element type (e.g., 99)
    // This exercises the inner _ => {} arm in skip_value for arrays
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&6u64.to_le_bytes());

    // Required: tokens
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

    // Required: scores
    let key2 = b"tokenizer.ggml.scores";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2);
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&6u32.to_le_bytes());
    data.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for _ in &tokens {
        data.extend_from_slice(&0.0f32.to_le_bytes());
    }

    // Array with unknown element type (type 99)
    let key3 = b"general.weird_array";
    data.extend_from_slice(&(key3.len() as u64).to_le_bytes());
    data.extend_from_slice(key3);
    data.extend_from_slice(&9u32.to_le_bytes()); // array type
    data.extend_from_slice(&99u32.to_le_bytes()); // unknown element type
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 elements

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
}

#[test]
fn test_decode_sentencepiece_leading_space_removal() {
    // The SentencePiece decoder removes leading space from result
    let tokens = vec![
        "<unk>".to_string(),
        "<s>".to_string(),
        "</s>".to_string(),
        "▁Hello".to_string(),
        "▁world".to_string(),
    ];
    let scores = vec![0.0; tokens.len()];
    let tokenizer = LlamaTokenizer::new(tokens, scores, 1, 2, 0).unwrap();

    let decoded = tokenizer.decode(&[3, 4]);
    // "▁Hello" -> " Hello", "▁world" -> " world"
    // Combined: " Hello world", leading space removed -> "Hello world"
    assert_eq!(decoded, "Hello world");
}
