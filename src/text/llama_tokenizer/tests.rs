pub(crate) use super::*;
pub(crate) use tests_gguf_sentencepiece::{create_gguf_with_array_metadata, create_gguf_with_extra_metadata};
pub(crate) use construction::create_gguf_with_string_metadata;

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

pub(super) fn create_test_tokenizer() -> LlamaTokenizer {
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

pub(super) fn create_test_gguf() -> Vec<u8> {
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

pub(super) fn create_gpt2_test_gguf() -> Vec<u8> {
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

#[path = "tests_part_02.rs"]
mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
#[path = "tests_gguf_sentencepiece.rs"]
mod tests_gguf_sentencepiece;
#[path = "construction.rs"]
mod construction;
