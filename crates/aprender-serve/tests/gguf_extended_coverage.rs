//! Extended coverage tests for realizar/src/gguf.rs
//!
//! This module provides coverage for GGUF structs and functions
//! that are not behind the GPU feature flag.

use realizar::gguf::{
    ArchConstraints, GGUFConfig, GGUFHeader, GGUFValue, InferenceScratchBuffer, TensorInfo,
    GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K,
    GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

// ============================================================================
// Test 1-15: GGUFValue variants
// ============================================================================

#[test]
fn test_gguf_value_uint8() {
    let value = GGUFValue::UInt8(255);
    if let GGUFValue::UInt8(v) = value {
        assert_eq!(v, 255);
    } else {
        panic!("Expected UInt8");
    }
}

#[test]
fn test_gguf_value_int8() {
    let value = GGUFValue::Int8(-128);
    if let GGUFValue::Int8(v) = value {
        assert_eq!(v, -128);
    } else {
        panic!("Expected Int8");
    }
}

#[test]
fn test_gguf_value_uint16() {
    let value = GGUFValue::UInt16(65535);
    if let GGUFValue::UInt16(v) = value {
        assert_eq!(v, 65535);
    } else {
        panic!("Expected UInt16");
    }
}

#[test]
fn test_gguf_value_int16() {
    let value = GGUFValue::Int16(-32768);
    if let GGUFValue::Int16(v) = value {
        assert_eq!(v, -32768);
    } else {
        panic!("Expected Int16");
    }
}

#[test]
fn test_gguf_value_uint32() {
    let value = GGUFValue::UInt32(4294967295);
    if let GGUFValue::UInt32(v) = value {
        assert_eq!(v, 4294967295);
    } else {
        panic!("Expected UInt32");
    }
}

#[test]
fn test_gguf_value_int32() {
    let value = GGUFValue::Int32(-2147483648);
    if let GGUFValue::Int32(v) = value {
        assert_eq!(v, -2147483648);
    } else {
        panic!("Expected Int32");
    }
}

#[test]
fn test_gguf_value_uint64() {
    let value = GGUFValue::UInt64(18446744073709551615);
    if let GGUFValue::UInt64(v) = value {
        assert_eq!(v, 18446744073709551615);
    } else {
        panic!("Expected UInt64");
    }
}

#[test]
fn test_gguf_value_int64() {
    let value = GGUFValue::Int64(-9223372036854775808);
    if let GGUFValue::Int64(v) = value {
        assert_eq!(v, -9223372036854775808);
    } else {
        panic!("Expected Int64");
    }
}

#[test]
fn test_gguf_value_float32() {
    let value = GGUFValue::Float32(std::f32::consts::PI);
    if let GGUFValue::Float32(v) = value {
        assert!((v - std::f32::consts::PI).abs() < 0.0001);
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_gguf_value_float64() {
    let value = GGUFValue::Float64(std::f64::consts::E);
    if let GGUFValue::Float64(v) = value {
        assert!((v - std::f64::consts::E).abs() < 0.00001);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_gguf_value_bool_true() {
    let value = GGUFValue::Bool(true);
    if let GGUFValue::Bool(v) = value {
        assert!(v);
    } else {
        panic!("Expected Bool");
    }
}

#[test]
fn test_gguf_value_bool_false() {
    let value = GGUFValue::Bool(false);
    if let GGUFValue::Bool(v) = value {
        assert!(!v);
    } else {
        panic!("Expected Bool");
    }
}

#[test]
fn test_gguf_value_string() {
    let value = GGUFValue::String("test_string".to_string());
    if let GGUFValue::String(v) = value {
        assert_eq!(v, "test_string");
    } else {
        panic!("Expected String");
    }
}

#[test]
fn test_gguf_value_array() {
    let value = GGUFValue::Array(vec![
        GGUFValue::Int32(1),
        GGUFValue::Int32(2),
        GGUFValue::Int32(3),
    ]);
    if let GGUFValue::Array(arr) = value {
        assert_eq!(arr.len(), 3);
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_gguf_value_clone() {
    let value = GGUFValue::Int32(42);
    let cloned = value;
    if let GGUFValue::Int32(v) = cloned {
        assert_eq!(v, 42);
    }
}

// ============================================================================
// Test 16-30: GGUFHeader tests
// ============================================================================

#[test]
fn test_gguf_header_with_magic() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: GGUF_VERSION_V3,
        tensor_count: 100,
        metadata_count: 50,
    };
    assert_eq!(header.magic, GGUF_MAGIC);
    assert_eq!(header.version, GGUF_VERSION_V3);
}

#[test]
fn test_gguf_header_zero_counts() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 0,
        metadata_count: 0,
    };
    assert_eq!(header.tensor_count, 0);
    assert_eq!(header.metadata_count, 0);
}

#[test]
fn test_gguf_header_max_counts() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: u64::MAX,
        metadata_count: u64::MAX,
    };
    assert_eq!(header.tensor_count, u64::MAX);
    assert_eq!(header.metadata_count, u64::MAX);
}

#[test]
fn test_gguf_header_clone() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 200,
        metadata_count: 100,
    };
    let cloned = header.clone();
    assert_eq!(cloned.magic, header.magic);
    assert_eq!(cloned.tensor_count, header.tensor_count);
}

#[test]
fn test_gguf_header_debug() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 50,
        metadata_count: 25,
    };
    let debug = format!("{header:?}");
    assert!(debug.contains("GGUFHeader"));
}

#[test]
fn test_gguf_header_eq() {
    let h1 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 10,
        metadata_count: 5,
    };
    let h2 = h1.clone();
    assert_eq!(h1, h2);
}

#[test]
fn test_gguf_header_ne_version() {
    let h1 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 10,
        metadata_count: 5,
    };
    let h2 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 2,
        tensor_count: 10,
        metadata_count: 5,
    };
    assert_ne!(h1, h2);
}

#[test]
fn test_gguf_header_ne_tensor_count() {
    let h1 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 10,
        metadata_count: 5,
    };
    let h2 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 20,
        metadata_count: 5,
    };
    assert_ne!(h1, h2);
}

// ============================================================================
// Test 31-45: TensorInfo tests
// ============================================================================

#[test]
fn test_tensor_info_f32() {
    let info = TensorInfo {
        name: "model.embed_tokens.weight".to_string(),
        n_dims: 2,
        dims: vec![32000, 4096],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    assert_eq!(info.qtype, GGUF_TYPE_F32);
    assert_eq!(info.n_dims, 2);
}

#[test]
fn test_tensor_info_f16() {
    let info = TensorInfo {
        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
        n_dims: 2,
        dims: vec![4096, 4096],
        qtype: GGUF_TYPE_F16,
        offset: 0,
    };
    assert_eq!(info.qtype, GGUF_TYPE_F16);
}

#[test]
fn test_tensor_info_q4_0() {
    let info = TensorInfo {
        name: "model.layers.0.mlp.up_proj.weight".to_string(),
        n_dims: 2,
        dims: vec![11008, 4096],
        qtype: GGUF_TYPE_Q4_0,
        offset: 1024,
    };
    assert_eq!(info.qtype, GGUF_TYPE_Q4_0);
}

#[test]
fn test_tensor_info_q4_k() {
    let info = TensorInfo {
        name: "tensor.q4k".to_string(),
        n_dims: 2,
        dims: vec![256, 256],
        qtype: GGUF_TYPE_Q4_K,
        offset: 2048,
    };
    assert_eq!(info.qtype, GGUF_TYPE_Q4_K);
}

#[test]
fn test_tensor_info_q8_0() {
    let info = TensorInfo {
        name: "tensor.q8".to_string(),
        n_dims: 1,
        dims: vec![1024],
        qtype: GGUF_TYPE_Q8_0,
        offset: 4096,
    };
    assert_eq!(info.qtype, GGUF_TYPE_Q8_0);
}

#[test]
fn test_tensor_info_1d() {
    let info = TensorInfo {
        name: "bias".to_string(),
        n_dims: 1,
        dims: vec![4096],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    assert_eq!(info.n_dims, 1);
    assert_eq!(info.dims.len(), 1);
}

#[test]
fn test_tensor_info_3d() {
    let info = TensorInfo {
        name: "conv.weight".to_string(),
        n_dims: 3,
        dims: vec![64, 3, 7],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    assert_eq!(info.n_dims, 3);
    assert_eq!(info.dims.len(), 3);
}

#[test]
fn test_tensor_info_4d() {
    let info = TensorInfo {
        name: "conv2d.weight".to_string(),
        n_dims: 4,
        dims: vec![64, 3, 7, 7],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    assert_eq!(info.n_dims, 4);
}

#[test]
fn test_tensor_info_clone() {
    let info = TensorInfo {
        name: "test.weight".to_string(),
        n_dims: 2,
        dims: vec![100, 200],
        qtype: GGUF_TYPE_Q4_K,
        offset: 1024,
    };
    let cloned = info.clone();
    assert_eq!(cloned.name, info.name);
    assert_eq!(cloned.dims, info.dims);
}

#[test]
fn test_tensor_info_debug() {
    let info = TensorInfo {
        name: "debug.tensor".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    let debug = format!("{info:?}");
    assert!(debug.contains("debug.tensor"));
}

#[test]
fn test_tensor_info_large_offset() {
    let info = TensorInfo {
        name: "large.offset".to_string(),
        n_dims: 2,
        dims: vec![100, 100],
        qtype: GGUF_TYPE_F32,
        offset: u64::MAX,
    };
    assert_eq!(info.offset, u64::MAX);
}

#[test]
fn test_tensor_info_empty_dims() {
    let info = TensorInfo {
        name: "scalar".to_string(),
        n_dims: 0,
        dims: vec![],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    assert_eq!(info.n_dims, 0);
    assert!(info.dims.is_empty());
}

// ============================================================================
// Test 46-60: GGUFConfig tests
// ============================================================================

#[test]
fn test_gguf_config_llama() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 32,
        vocab_size: 32000,
        intermediate_dim: 11008,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim / config.num_heads, 128); // head_dim
    assert_eq!(config.num_heads, config.num_kv_heads); // not GQA
}

#[test]
fn test_gguf_config_llama2() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        vocab_size: 32000,
        intermediate_dim: 11008,
        context_length: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_ne!(config.num_heads, config.num_kv_heads); // is GQA
}

#[test]
fn test_gguf_config_phi() {
    let config = GGUFConfig {
        architecture: "phi".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2560,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 51200,
        intermediate_dim: 10240,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config.architecture, "phi");
}

#[test]
fn test_gguf_config_mistral() {
    let config = GGUFConfig {
        architecture: "mistral".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 14336,
        context_length: 8192,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config.architecture, "mistral");
    assert_eq!(config.context_length, 8192);
}

#[test]
fn test_gguf_config_head_dim_calculation() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 512,
        num_layers: 8,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 10000,
        intermediate_dim: 2048,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config.hidden_dim / config.num_heads, 64); // head_dim
}

#[test]
fn test_gguf_config_gqa_ratio() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 1024,
        num_layers: 4,
        num_heads: 16,
        num_kv_heads: 4, // 4:1 GQA ratio
        vocab_size: 10000,
        intermediate_dim: 4096,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_ne!(config.num_heads, config.num_kv_heads); // is GQA
    assert_eq!(config.num_heads / config.num_kv_heads, 4);
}

#[test]
fn test_gguf_config_clone() {
    let config = GGUFConfig {
        architecture: "cloneable".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    let cloned = config.clone();
    assert_eq!(cloned.architecture, config.architecture);
    assert_eq!(cloned.hidden_dim, config.hidden_dim);
}

#[test]
fn test_gguf_config_debug() {
    let config = GGUFConfig {
        architecture: "debug".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 500,
        intermediate_dim: 256,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    let debug = format!("{config:?}");
    assert!(debug.contains("GGUFConfig"));
}

#[test]
fn test_gguf_config_rope_theta_values() {
    let config_10k = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config_10k.rope_theta, 10000.0);

    let config_1m = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 512,
        context_length: 256,
        rope_theta: 1000000.0, // Extended context
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config_1m.rope_theta, 1000000.0);
}

// ============================================================================
// Test 61-75: InferenceScratchBuffer tests
// ============================================================================

fn make_test_config(hidden_dim: usize, num_heads: usize) -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers: 2,
        num_heads,
        num_kv_heads: num_heads,
        vocab_size: 1000,
        intermediate_dim: hidden_dim * 4,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    }
}

#[test]
fn test_inference_scratch_buffer_from_config() {
    let config = make_test_config(1024, 16);
    let buffer = InferenceScratchBuffer::from_config(&config);
    assert!(!buffer.hidden.is_empty());
}

#[test]
fn test_inference_scratch_buffer_small() {
    let config = make_test_config(256, 4);
    let buffer = InferenceScratchBuffer::from_config(&config);
    assert_eq!(buffer.hidden.len(), 256);
}

#[test]
fn test_inference_scratch_buffer_large() {
    let config = make_test_config(8192, 64);
    let buffer = InferenceScratchBuffer::from_config(&config);
    assert_eq!(buffer.hidden.len(), 8192);
}

#[test]
fn test_inference_scratch_buffer_reset() {
    let config = make_test_config(512, 8);
    let mut buffer = InferenceScratchBuffer::from_config(&config);
    buffer.hidden[0] = 1.0;
    buffer.reset();
    assert_eq!(buffer.hidden[0], 0.0);
}

#[test]
fn test_inference_scratch_buffer_debug() {
    let config = make_test_config(512, 8);
    let buffer = InferenceScratchBuffer::from_config(&config);
    let debug = format!("{buffer:?}");
    assert!(debug.contains("InferenceScratchBuffer"));
}

// ============================================================================
// Test 76-90: Constants tests
// ============================================================================

#[test]
fn test_gguf_magic_value() {
    assert_eq!(GGUF_MAGIC, 0x46554747); // "GGUF" in little-endian
}

#[test]
fn test_gguf_alignment_value() {
    assert_eq!(GGUF_ALIGNMENT, 32);
}

#[test]
fn test_gguf_version_v3() {
    assert_eq!(GGUF_VERSION_V3, 3);
}

#[test]
fn test_gguf_type_f32_value() {
    assert_eq!(GGUF_TYPE_F32, 0);
}

#[test]
fn test_gguf_type_f16_value() {
    assert_eq!(GGUF_TYPE_F16, 1);
}

#[test]
fn test_gguf_type_q4_0_value() {
    assert_eq!(GGUF_TYPE_Q4_0, 2);
}

#[test]
fn test_gguf_type_q8_0_value() {
    assert_eq!(GGUF_TYPE_Q8_0, 8);
}

#[test]
fn test_gguf_type_q4_k_value() {
    assert_eq!(GGUF_TYPE_Q4_K, 12);
}

#[test]
fn test_gguf_type_q5_k_value() {
    assert_eq!(GGUF_TYPE_Q5_K, 13);
}

#[test]
fn test_gguf_type_q6_k_value() {
    assert_eq!(GGUF_TYPE_Q6_K, 14);
}

// ============================================================================
// Test 91-100: Edge cases and additional coverage
// ============================================================================

#[test]
fn test_gguf_value_array_empty() {
    let value = GGUFValue::Array(vec![]);
    if let GGUFValue::Array(arr) = value {
        assert!(arr.is_empty());
    }
}

#[test]
fn test_gguf_value_string_empty() {
    let value = GGUFValue::String(String::new());
    if let GGUFValue::String(s) = value {
        assert!(s.is_empty());
    }
}

#[test]
fn test_gguf_value_string_unicode() {
    let value = GGUFValue::String("你好世界".to_string());
    if let GGUFValue::String(s) = value {
        assert_eq!(s, "你好世界");
    }
}

#[test]
fn test_tensor_info_unicode_name() {
    let info = TensorInfo {
        name: "模型.权重".to_string(),
        n_dims: 2,
        dims: vec![100, 200],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    assert_eq!(info.name, "模型.权重");
}

#[test]
fn test_tensor_info_long_name() {
    let long_name = "a".repeat(1000);
    let info = TensorInfo {
        name: long_name,
        n_dims: 2,
        dims: vec![10, 10],
        qtype: GGUF_TYPE_F32,
        offset: 0,
    };
    assert_eq!(info.name.len(), 1000);
}

#[test]
fn test_gguf_config_very_small_model() {
    let config = GGUFConfig {
        architecture: "tiny".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 16,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 100,
        intermediate_dim: 32,
        context_length: 16,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config.hidden_dim / config.num_heads, 16); // head_dim
}

#[test]
fn test_gguf_config_very_large_model() {
    let config = GGUFConfig {
        architecture: "huge".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 16384,
        num_layers: 128,
        num_heads: 128,
        num_kv_heads: 16,
        vocab_size: 128000,
        intermediate_dim: 65536,
        context_length: 131072,
        rope_theta: 10000000.0,
        eps: 1e-6,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };
    assert_eq!(config.hidden_dim / config.num_heads, 128); // head_dim
    assert_ne!(config.num_heads, config.num_kv_heads); // is GQA
}

#[test]
fn test_gguf_header_version_2() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 2,
        tensor_count: 10,
        metadata_count: 5,
    };
    assert_eq!(header.version, 2);
}

#[test]
fn test_tensor_info_all_qtypes() {
    let qtypes = [
        GGUF_TYPE_F32,
        GGUF_TYPE_F16,
        GGUF_TYPE_Q4_0,
        GGUF_TYPE_Q8_0,
        GGUF_TYPE_Q4_K,
        GGUF_TYPE_Q5_K,
        GGUF_TYPE_Q6_K,
    ];
    for qtype in qtypes {
        let info = TensorInfo {
            name: format!("tensor.qtype_{qtype}"),
            n_dims: 2,
            dims: vec![10, 10],
            qtype,
            offset: 0,
        };
        assert_eq!(info.qtype, qtype);
    }
}

#[test]
fn test_gguf_config_different_eps_values() {
    let eps_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
    for eps in eps_values {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: ArchConstraints::from_architecture("llama"),
            hidden_dim: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 512,
            context_length: 256,
            rope_theta: 10000.0,
            eps,
            rope_type: 0,
            bos_token_id: None,
            eos_token_id: None,
            explicit_head_dim: None,
        };
        assert!((config.eps - eps).abs() < 1e-10);
    }
}
