//! GGUF vocabulary tests extracted from inline tests
//!
//! Tests for vocabulary loading, encoding, and decoding functionality.

use realizar::gguf::{GGUFModel, OwnedQuantizedModel};

#[test]
fn test_vocabulary_from_metadata() {
    // Build GGUF with tokenizer.ggml.tokens array
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Key: "tokenizer.ggml.tokens"
    let key = "tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // value_type = Array
    data.extend_from_slice(&8u32.to_le_bytes()); // element_type = String
    data.extend_from_slice(&3u64.to_le_bytes()); // array_len = 3

    // Token 0: "<pad>"
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"<pad>");
    // Token 1: "hello"
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"hello");
    // Token 2: "world"
    data.extend_from_slice(&5u64.to_le_bytes());
    data.extend_from_slice(b"world");

    let model = GGUFModel::from_bytes(&data).expect("test");
    let vocab = model.vocabulary().expect("Should have vocabulary");

    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab[0], "<pad>");
    assert_eq!(vocab[1], "hello");
    assert_eq!(vocab[2], "world");
}

#[test]
fn test_decode_with_vocabulary() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());

    // Tokens: "The ", "capital ", "of ", "France"
    for token in ["The ", "capital ", "of ", "France"] {
        data.extend_from_slice(&(token.len() as u64).to_le_bytes());
        data.extend_from_slice(token.as_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("test");
    let decoded = model.decode(&[0, 1, 2, 3]);

    assert_eq!(decoded, "The capital of France");
}

#[test]
fn test_decode_unknown_token() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(b"a");
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(b"b");

    let model = GGUFModel::from_bytes(&data).expect("test");
    // Token ID 5 is out of range (vocab only has 0, 1)
    let decoded = model.decode(&[0, 5, 1]);

    assert_eq!(decoded, "a�b");
}

#[test]
fn test_vocabulary_none_when_missing() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // No metadata

    let model = GGUFModel::from_bytes(&data).expect("test");
    assert!(model.vocabulary().is_none());
}

#[test]
fn test_decode_fallback_ascii() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // No metadata

    let model = GGUFModel::from_bytes(&data).expect("test");
    // Falls back to ASCII: 72=H, 105=i
    let decoded = model.decode(&[72, 105]);

    assert_eq!(decoded, "Hi");
}

#[test]
fn test_encode_simple() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());

    // Tokens with SentencePiece-style ▁ prefix for word boundaries
    // Token 0: "▁Hello", Token 1: "▁world", Token 2: unused
    // With SentencePiece prepending, "Hello world" → "▁Hello▁world"
    for token in ["▁Hello", "▁world", "unused"] {
        data.extend_from_slice(&(token.len() as u64).to_le_bytes());
        data.extend_from_slice(token.as_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("test");
    let tokens = model.encode("Hello world").expect("test");

    assert_eq!(tokens, vec![0, 1]); // "▁Hello" + "▁world"
}

#[test]
fn test_encode_longest_match() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());

    // Tokens: "▁a", "▁ab", "▁abc" - should pick longest match
    // With SentencePiece prepending, "abc" → "▁abc"
    for token in ["▁a", "▁ab", "▁abc"] {
        data.extend_from_slice(&(token.len() as u64).to_le_bytes());
        data.extend_from_slice(token.as_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("test");
    let tokens = model.encode("abc").expect("test");

    assert_eq!(tokens, vec![2]); // Should pick "▁abc" (longest match)
}

#[test]
fn test_encode_roundtrip() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());

    // SentencePiece-style vocabulary with ▁ prefix for word boundaries
    // With SentencePiece prepending, "The capital..." → "▁The▁capital..."
    for token in ["▁The", "▁capital", "▁of", "▁France"] {
        data.extend_from_slice(&(token.len() as u64).to_le_bytes());
        data.extend_from_slice(token.as_bytes());
    }

    let model = GGUFModel::from_bytes(&data).expect("test");
    let text = "The capital of France";
    let tokens = model.encode(text).expect("test");
    let decoded = model.decode(&tokens);

    // Decoded text converts ▁ to spaces for human-readable output
    assert_eq!(decoded, " The capital of France");
}

/// Test that sample_topk produces varied outputs (non-deterministic)
#[test]
fn test_sample_topk_produces_varied_outputs() {
    // Create logits with multiple high-probability tokens
    let mut logits = vec![0.0f32; 100];
    logits[0] = 5.0;
    logits[1] = 4.8;
    logits[2] = 4.6;
    logits[3] = 4.4;
    logits[4] = 4.2;

    // Sample multiple times and collect results
    let mut samples = std::collections::HashSet::new();
    for _ in 0..50 {
        let token = OwnedQuantizedModel::sample_topk(&logits, 1.0, 5);
        samples.insert(token);
    }

    // With true randomness, we should get multiple different tokens
    // (with high probability, more than 1 unique token in 50 samples)
    assert!(
        samples.len() > 1,
        "Expected varied sampling, got only {} unique tokens: {:?}. \
        This indicates deterministic sampling instead of random.",
        samples.len(),
        samples
    );
}
