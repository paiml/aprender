//! Coverage tests for loader.rs: decode/encode edge cases, rope_type paths, metadata accessors
//!
//! Targets missed lines in loader.rs:
//! - GPT-2 style decode (gpt2_unicode_to_byte path)
//! - encode with GPT-2 style tokenizer
//! - SentencePiece decode with word boundary markers
//! - Byte token decoding (<0xHH>)
//! - decode fallback for unknown token IDs
//! - rope_type with various architectures (deepseek2, falcon, bert, etc.)
//! - metadata accessor None-paths (wrong type in metadata)
//! - vocabulary with non-string array elements

use crate::gguf::test_factory::GGUFBuilder;
use crate::gguf::GGUFModel;

// ============================================================================
// Decode Tests - GPT-2 Style Tokenizer
// ============================================================================

#[test]
fn test_decode_gpt2_style() {
    // GPT-2 tokenizer uses Ġ (U+0120) for space
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("tokenizer.ggml.model", "gpt2")
        .add_string_array("tokenizer.ggml.tokens", &["Hello", "\u{0120}world"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0, 1]);
    assert!(decoded.contains("Hello"));
    assert!(decoded.contains("world"));
}

#[test]
fn test_decode_sentencepiece_word_boundary() {
    // SentencePiece uses ▁ (U+2581) for word boundaries
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["Hello", "▁world", "▁test"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0, 1, 2]);
    // ▁ should be replaced with space
    assert!(decoded.contains("Hello"));
    assert!(decoded.contains(" world"));
    assert!(decoded.contains(" test"));
}

#[test]
fn test_decode_byte_tokens_multi() {
    // Test multiple byte tokens forming a UTF-8 sequence
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string_array("tokenizer.ggml.tokens", &["<0x41>", "<0x42>", "<0x43>"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0, 1, 2]);
    assert_eq!(decoded, "ABC");
}

#[test]
fn test_decode_out_of_bounds_token_id() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string_array("tokenizer.ggml.tokens", &["hello", "world"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    // Token ID 99 is out of bounds for 2-element vocab
    let decoded = model.decode(&[99]);
    // Should use "unknown" replacement character
    assert!(!decoded.is_empty());
}

#[test]
fn test_decode_no_vocab_ascii_fallback() {
    let data = GGUFBuilder::new().architecture("llama").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    // Without vocab, should map token IDs to ASCII chars
    let decoded = model.decode(&[65, 66, 67]); // A, B, C
    assert_eq!(decoded, "ABC");
}

#[test]
fn test_decode_no_vocab_high_ids_capped() {
    let data = GGUFBuilder::new().architecture("llama").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    // Token IDs > 127 should be capped to 127 in ASCII fallback
    let decoded = model.decode(&[200, 300]);
    // Both capped to 127 = DEL, then char::from_u32 should handle
    assert!(!decoded.is_empty());
}

#[test]
fn test_decode_empty_token_ids() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string_array("tokenizer.ggml.tokens", &["a", "b"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[]);
    assert!(decoded.is_empty());
}

// ============================================================================
// Encode Tests - Edge Cases
// ============================================================================

#[test]
fn test_encode_gpt2_style() {
    let data = GGUFBuilder::new()
        .architecture("qwen2")
        .add_string("tokenizer.ggml.model", "gpt2")
        .add_string_array(
            "tokenizer.ggml.tokens",
            &["<unk>", "H", "ello", "\u{0120}world"],
        )
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let tokens = model.encode("Hello world");
    assert!(tokens.is_some());
    let tokens = tokens.expect("tokens");
    assert!(!tokens.is_empty());
}

#[test]
fn test_encode_empty_text() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["<unk>", "hello"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let tokens = model.encode("");
    // Empty text should return Some with either empty or minimal tokens
    assert!(tokens.is_some());
}

#[test]
fn test_encode_no_vocab_returns_none() {
    let data = GGUFBuilder::new().architecture("llama").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.encode("test").is_none());
}

#[test]
fn test_encode_unknown_chars_byte_fallback() {
    // When a character can't be matched, should fall back to byte tokens
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["<unk>", "<0x48>", "<0x69>", "▁"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let tokens = model.encode("Hi");
    assert!(tokens.is_some());
    let tokens = tokens.expect("tokens");
    assert!(!tokens.is_empty());
}

// ============================================================================
// rope_type Tests - Architecture Coverage
// ============================================================================

#[test]
fn test_rope_type_falcon() {
    let data = GGUFBuilder::new().architecture("falcon").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_bert() {
    let data = GGUFBuilder::new().architecture("bert").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_stablelm() {
    let data = GGUFBuilder::new().architecture("stablelm").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_deepseek2() {
    let data = GGUFBuilder::new().architecture("deepseek2").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_starcoder2() {
    let data = GGUFBuilder::new().architecture("starcoder2").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_gptneox() {
    let data = GGUFBuilder::new().architecture("gptneox").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_dbrx() {
    let data = GGUFBuilder::new().architecture("dbrx").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_olmo2() {
    let data = GGUFBuilder::new().architecture("olmo2").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_internlm2() {
    let data = GGUFBuilder::new().architecture("internlm2").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_exaone() {
    let data = GGUFBuilder::new().architecture("exaone").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_minicpm3() {
    let data = GGUFBuilder::new().architecture("minicpm3").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_nemotron() {
    let data = GGUFBuilder::new().architecture("nemotron").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_openelm() {
    let data = GGUFBuilder::new().architecture("openelm").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_plamo() {
    let data = GGUFBuilder::new().architecture("plamo").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_plamo2() {
    let data = GGUFBuilder::new().architecture("plamo2").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_codeshell() {
    let data = GGUFBuilder::new().architecture("codeshell").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_orion() {
    let data = GGUFBuilder::new().architecture("orion").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_nomic_bert() {
    let data = GGUFBuilder::new().architecture("nomic-bert").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_olmoe() {
    let data = GGUFBuilder::new().architecture("olmoe").build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX
}

#[test]
fn test_rope_type_unknown_arch_defaults_norm() {
    let data = GGUFBuilder::new()
        .architecture("custom_unknown_model")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM (default)
}

#[test]
fn test_rope_type_with_neox_scaling_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.rope.scaling.type", "neox")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // neox -> NEOX
}

#[test]
fn test_rope_type_with_unknown_scaling_type() {
    // Unknown scaling type should fall through to architecture-based inference
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.rope.scaling.type", "something_else")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    // llama is not in NEOX list, so should be NORM
    assert_eq!(model.rope_type(), Some(0));
}

#[test]
fn test_rope_type_no_architecture() {
    let data = GGUFBuilder::new().build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.rope_type().is_none());
}

// ============================================================================
// Metadata Accessor Edge Cases
// ============================================================================

#[test]
fn test_embedding_dim_wrong_type() {
    // Set embedding_length as a string instead of u32
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.embedding_length", "64")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.embedding_dim().is_none()); // Wrong type, should return None
}

#[test]
fn test_num_layers_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.block_count", "4")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.num_layers().is_none());
}

#[test]
fn test_num_heads_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.attention.head_count", "4")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.num_heads().is_none());
}

#[test]
fn test_context_length_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.context_length", "4096")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.context_length().is_none());
}

#[test]
fn test_num_kv_heads_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.attention.head_count_kv", "2")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.num_kv_heads().is_none());
}

#[test]
fn test_rope_freq_base_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.rope.freq_base", "10000")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.rope_freq_base().is_none());
}

#[test]
fn test_rms_epsilon_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("llama.attention.layer_norm_rms_epsilon", "1e-5")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.rms_epsilon().is_none());
}

#[test]
fn test_bos_token_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("tokenizer.ggml.bos_token_id", "1")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.bos_token_id().is_none());
}

#[test]
fn test_eos_token_wrong_type() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string("tokenizer.ggml.eos_token_id", "2")
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.eos_token_id().is_none());
}

// ============================================================================
// Vocabulary Edge Cases
// ============================================================================

#[test]
fn test_vocabulary_empty_array() {
    // Empty tokens array should return None
    let data = GGUFBuilder::new()
        .architecture("llama")
        .add_string_array("tokenizer.ggml.tokens", &[])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.vocabulary().is_none());
}

include!("vocabulary_single_gguf.rs");
