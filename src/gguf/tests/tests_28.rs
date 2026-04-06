//! GGUF Tests Part 28: T-COV-95 Deep Coverage Bridge - loader.rs remaining paths
//!
//! Targets:
//! - get_tensor_f32: Q2_K, F16, Q4_1, Q5_0, Q5_1 dequantization branches
//! - read_value: all metadata type branches (UInt8, Int8, UInt16, Int16, Bool, UInt64, Int64, Float64, Array)
//! - decode: vocabulary-based (SentencePiece, GPT-2, byte tokens)
//! - encode: greedy tokenization with vocabulary
//! - from_bytes: error paths (truncated, wrong version)
//! - rope_type: explicit scaling type metadata
//!
//! Refs PMAT-802: Protocol T-COV-95

use crate::gguf::test_factory::*;
use crate::gguf::GGUFModel;

// ============================================================================
// get_tensor_f32: Q2_K dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q2_k() {
    let n = 256; // one super-block
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q2_k_tensor("test.weight", &[n as u64], &create_q2_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q2_K get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

#[test]
fn test_get_tensor_f32_q2_k_multi_block() {
    let n = 512; // 2 super-blocks
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q2_k_tensor("test.weight", &[n as u64], &create_q2_k_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: F16 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_f16() {
    let n = 32;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f16_tensor("test.weight", &[n as u64], &create_f16_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "F16 get_tensor_f32 failed: {:?}",
        values.err()
    );
    let v = values.expect("test value should be present");
    assert_eq!(v.len(), n);
}

#[test]
fn test_get_tensor_f32_f16_values_correct() {
    let n = 4;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f16_tensor("test.weight", &[n as u64], &create_f16_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    // Values should be approximately 0.0, 0.01, 0.02, 0.03
    assert!((values[0] - 0.0).abs() < 0.01);
    assert!((values[1] - 0.01).abs() < 0.01);
}

// ============================================================================
// get_tensor_f32: Q4_1 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q4_1() {
    let n = 32; // one block
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_1_tensor("test.weight", &[n as u64], &create_q4_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q4_1 get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

#[test]
fn test_get_tensor_f32_q4_1_multi_block() {
    let n = 128; // 4 blocks
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q4_1_tensor("test.weight", &[n as u64], &create_q4_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: Q5_0 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q5_0() {
    let n = 32;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_0_tensor("test.weight", &[n as u64], &create_q5_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q5_0 get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

#[test]
fn test_get_tensor_f32_q5_0_multi_block() {
    let n = 128;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_0_tensor("test.weight", &[n as u64], &create_q5_0_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
}

// ============================================================================
// get_tensor_f32: Q5_1 dequantization branch
// ============================================================================

#[test]
fn test_get_tensor_f32_q5_1() {
    let n = 32;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_1_tensor("test.weight", &[n as u64], &create_q5_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model.get_tensor_f32("test.weight", &data);
    assert!(
        values.is_ok(),
        "Q5_1 get_tensor_f32 failed: {:?}",
        values.err()
    );
    assert_eq!(values.expect("test value should be present").len(), n);
}

#[test]
fn test_get_tensor_f32_q5_1_multi_block() {
    let n = 128;
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_q5_1_tensor("test.weight", &[n as u64], &create_q5_1_data(n))
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let values = model
        .get_tensor_f32("test.weight", &data)
        .expect("test value should be present");
    assert_eq!(values.len(), n);
}

// ============================================================================
// read_value: metadata type branches
// ============================================================================

#[test]
fn test_metadata_u8() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u8("test.u8_val", 42)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.u8_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i8() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i8("test.i8_val", -42)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.i8_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_u16() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u16("test.u16_val", 1234)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.u16_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i16() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i16("test.i16_val", -1234)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.i16_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i32() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i32("test.i32_val", -100_000)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.i32_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_bool_true() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_bool("test.bool_val", true)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.bool_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_bool_false() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_bool("test.bool_val", false)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.bool_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_u64() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u64("test.u64_val", 123_456_789_000)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.u64_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_i64() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_i64("test.i64_val", -123_456_789_000)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.i64_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_f64() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_f64("test.f64_val", 3.14159265358979)
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.f64_val");
    assert!(val.is_some());
}

#[test]
fn test_metadata_string_array() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string_array("test.array", &["hello", "world", "test"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    let val = model.metadata.get("test.array");
    assert!(val.is_some());
}

#[test]
fn test_metadata_all_types_in_one_model() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_u8("t.u8", 1)
        .add_i8("t.i8", -1)
        .add_u16("t.u16", 100)
        .add_i16("t.i16", -100)
        .add_i32("t.i32", -50000)
        .add_bool("t.bool", true)
        .add_u64("t.u64", 999)
        .add_i64("t.i64", -999)
        .add_f64("t.f64", 2.718)
        .add_string_array("t.arr", &["a", "b"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    // 4 (arch+dim+layers+heads) + 10 = 14
    assert_eq!(model.metadata.len(), 14);
}

// ============================================================================
// decode: vocabulary-based (SentencePiece style)
// ============================================================================

#[test]
fn test_decode_sentencepiece_style() {
    let data = GGUFBuilder::new()
        .architecture("llama")
        .hidden_dim("llama", 32)
        .num_layers("llama", 1)
        .num_heads("llama", 1)
        .add_string("tokenizer.ggml.model", "llama")
        .add_string_array("tokenizer.ggml.tokens", &["▁Hello", "▁world", "!", "▁test"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert!(model.vocabulary().is_some());

    let text = model.decode(&[0, 1, 2]);
    assert!(text.contains("Hello"));
    assert!(text.contains("world"));
    assert!(text.contains("!"));
}

#[test]
fn test_decode_gpt2_style() {
    let data = GGUFBuilder::new()
        .architecture("qwen2")
        .hidden_dim("qwen2", 32)
        .num_layers("qwen2", 1)
        .num_heads("qwen2", 1)
        .add_string("tokenizer.ggml.model", "gpt2")
        .add_string_array("tokenizer.ggml.tokens", &["H", "e", "l", "o"])
        .build();
    let model = GGUFModel::from_bytes(&data).expect("test value should be present");
    assert!(model.vocabulary().is_some());

    let text = model.decode(&[0, 1, 2, 2, 3]);
    // GPT-2 unicode decode maps ASCII chars directly
    assert!(!text.is_empty());
}

include!("decode_byte.rs");
