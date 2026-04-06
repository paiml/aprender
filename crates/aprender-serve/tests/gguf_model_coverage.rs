//! EXTREME TDD: GGUFConfig and GGUFModel Coverage Tests
//!
//! Comprehensive coverage tests for:
//! - GGUFModel parsing and metadata extraction
//! - GGUFConfig construction and validation
//! - MappedGGUFModel memory-mapped loading
//! - GGUFHeader structure
//! - TensorInfo structure
//! - GGUFValue enum variants
//! - Edge cases and error paths

use std::io::Write as _;

use realizar::gguf::{
    ArchConstraints, GGUFConfig, GGUFHeader, GGUFModel, GGUFValue, MappedGGUFModel, TensorInfo,
    GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1,
    GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
    GGUF_VERSION_V3,
};
use tempfile::NamedTempFile;

// ============================================================================
// HELPER FUNCTIONS FOR BUILDING TEST GGUF DATA
// ============================================================================

/// Build a minimal valid GGUF header
fn build_minimal_gguf_header() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
    data
}

/// Add string metadata to GGUF data
fn add_string_meta(data: &mut Vec<u8>, key: &str, value: &str) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // String type
    data.extend_from_slice(&(value.len() as u64).to_le_bytes());
    data.extend_from_slice(value.as_bytes());
}

/// Add u32 metadata to GGUF data
fn add_u32_meta(data: &mut Vec<u8>, key: &str, value: u32) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // UInt32 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add f32 metadata to GGUF data
fn add_f32_meta(data: &mut Vec<u8>, key: &str, value: f32) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&6u32.to_le_bytes()); // Float32 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add u8 metadata to GGUF data
fn add_u8_meta(data: &mut Vec<u8>, key: &str, value: u8) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // UInt8 type
    data.push(value);
}

/// Add i8 metadata to GGUF data
fn add_i8_meta(data: &mut Vec<u8>, key: &str, value: i8) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // Int8 type
    data.push(value as u8);
}

/// Add u16 metadata to GGUF data
fn add_u16_meta(data: &mut Vec<u8>, key: &str, value: u16) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes()); // UInt16 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add i16 metadata to GGUF data
fn add_i16_meta(data: &mut Vec<u8>, key: &str, value: i16) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // Int16 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add i32 metadata to GGUF data
fn add_i32_meta(data: &mut Vec<u8>, key: &str, value: i32) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&5u32.to_le_bytes()); // Int32 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add bool metadata to GGUF data
fn add_bool_meta(data: &mut Vec<u8>, key: &str, value: bool) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&7u32.to_le_bytes()); // Bool type
    data.push(u8::from(value));
}

/// Add u64 metadata to GGUF data
fn add_u64_meta(data: &mut Vec<u8>, key: &str, value: u64) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&10u32.to_le_bytes()); // UInt64 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add i64 metadata to GGUF data
fn add_i64_meta(data: &mut Vec<u8>, key: &str, value: i64) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&11u32.to_le_bytes()); // Int64 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add f64 metadata to GGUF data
fn add_f64_meta(data: &mut Vec<u8>, key: &str, value: f64) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&12u32.to_le_bytes()); // Float64 type
    data.extend_from_slice(&value.to_le_bytes());
}

/// Add array of u32 metadata to GGUF data
fn add_u32_array_meta(data: &mut Vec<u8>, key: &str, values: &[u32]) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // Array type
    data.extend_from_slice(&4u32.to_le_bytes()); // Element type: UInt32
    data.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
}

/// Add array of strings metadata to GGUF data
fn add_string_array_meta(data: &mut Vec<u8>, key: &str, values: &[&str]) {
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // Array type
    data.extend_from_slice(&8u32.to_le_bytes()); // Element type: String
    data.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &v in values {
        data.extend_from_slice(&(v.len() as u64).to_le_bytes());
        data.extend_from_slice(v.as_bytes());
    }
}

/// Add tensor info to GGUF data (GGML order - dims are reversed)
fn add_tensor_info(data: &mut Vec<u8>, name: &str, dims: &[u64], qtype: u32, offset: u64) {
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &dim in dims.iter().rev() {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&qtype.to_le_bytes());
    data.extend_from_slice(&offset.to_le_bytes());
}

/// Build GGUF with architecture metadata
fn build_gguf_with_arch(arch: &str, hidden: usize, layers: usize, heads: usize) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&4u64.to_le_bytes()); // metadata_count

    add_string_meta(&mut data, "general.architecture", arch);
    add_u32_meta(
        &mut data,
        &format!("{arch}.embedding_length"),
        hidden as u32,
    );
    add_u32_meta(&mut data, &format!("{arch}.block_count"), layers as u32);
    add_u32_meta(
        &mut data,
        &format!("{arch}.attention.head_count"),
        heads as u32,
    );

    data
}

// ============================================================================
// GGUF HEADER TESTS
// ============================================================================

#[test]
fn test_cov_gguf_header_struct_fields() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 100,
        metadata_count: 50,
    };

    assert_eq!(header.magic, 0x4655_4747);
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 100);
    assert_eq!(header.metadata_count, 50);
}

#[test]
fn test_cov_gguf_header_clone() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 10,
        metadata_count: 5,
    };

    let cloned = header.clone();

    assert_eq!(cloned.magic, header.magic);
    assert_eq!(cloned.version, header.version);
    assert_eq!(cloned.tensor_count, header.tensor_count);
    assert_eq!(cloned.metadata_count, header.metadata_count);
}

#[test]
fn test_cov_gguf_header_debug() {
    let header = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 1,
        metadata_count: 1,
    };

    let debug_str = format!("{header:?}");
    assert!(debug_str.contains("GGUFHeader"));
    assert!(debug_str.contains("magic"));
    assert!(debug_str.contains("version"));
}

#[test]
fn test_cov_gguf_header_partial_eq() {
    let header1 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 10,
        metadata_count: 5,
    };

    let header2 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 3,
        tensor_count: 10,
        metadata_count: 5,
    };

    let header3 = GGUFHeader {
        magic: GGUF_MAGIC,
        version: 2, // Different version
        tensor_count: 10,
        metadata_count: 5,
    };

    assert_eq!(header1, header2);
    assert_ne!(header1, header3);
}

// ============================================================================
// TENSOR INFO TESTS
// ============================================================================

#[test]
fn test_cov_tensor_info_struct_fields() {
    let tensor = TensorInfo {
        name: "test.weight".to_string(),
        n_dims: 2,
        dims: vec![1024, 512],
        qtype: GGUF_TYPE_Q4_K,
        offset: 0,
    };

    assert_eq!(tensor.name, "test.weight");
    assert_eq!(tensor.n_dims, 2);
    assert_eq!(tensor.dims, vec![1024, 512]);
    assert_eq!(tensor.qtype, GGUF_TYPE_Q4_K);
    assert_eq!(tensor.offset, 0);
}

#[test]
fn test_cov_tensor_info_clone() {
    let tensor = TensorInfo {
        name: "blk.0.attn_q.weight".to_string(),
        n_dims: 2,
        dims: vec![2048, 2048],
        qtype: GGUF_TYPE_F32,
        offset: 1024,
    };

    let cloned = tensor.clone();

    assert_eq!(cloned.name, tensor.name);
    assert_eq!(cloned.n_dims, tensor.n_dims);
    assert_eq!(cloned.dims, tensor.dims);
    assert_eq!(cloned.qtype, tensor.qtype);
    assert_eq!(cloned.offset, tensor.offset);
}

#[test]
fn test_cov_tensor_info_debug() {
    let tensor = TensorInfo {
        name: "token_embd.weight".to_string(),
        n_dims: 2,
        dims: vec![32000, 2048],
        qtype: GGUF_TYPE_Q8_0,
        offset: 0,
    };

    let debug_str = format!("{tensor:?}");
    assert!(debug_str.contains("TensorInfo"));
    assert!(debug_str.contains("token_embd.weight"));
}

#[test]
fn test_cov_tensor_info_partial_eq() {
    let tensor1 = TensorInfo {
        name: "test".to_string(),
        n_dims: 1,
        dims: vec![100],
        qtype: 0,
        offset: 0,
    };

    let tensor2 = tensor1.clone();

    let tensor3 = TensorInfo {
        name: "other".to_string(),
        n_dims: 1,
        dims: vec![100],
        qtype: 0,
        offset: 0,
    };

    assert_eq!(tensor1, tensor2);
    assert_ne!(tensor1, tensor3);
}

#[test]
fn test_cov_tensor_info_various_qtypes() {
    let qtypes = [
        (GGUF_TYPE_F32, "F32"),
        (GGUF_TYPE_F16, "F16"),
        (GGUF_TYPE_Q4_0, "Q4_0"),
        (GGUF_TYPE_Q4_1, "Q4_1"),
        (GGUF_TYPE_Q5_0, "Q5_0"),
        (GGUF_TYPE_Q5_1, "Q5_1"),
        (GGUF_TYPE_Q8_0, "Q8_0"),
        (GGUF_TYPE_Q4_K, "Q4_K"),
        (GGUF_TYPE_Q5_K, "Q5_K"),
        (GGUF_TYPE_Q6_K, "Q6_K"),
    ];

    for (qtype, name) in qtypes {
        let tensor = TensorInfo {
            name: format!("{name}_tensor"),
            n_dims: 2,
            dims: vec![256, 256],
            qtype,
            offset: 0,
        };
        assert_eq!(tensor.qtype, qtype);
    }
}

// ============================================================================
// GGUF MODEL TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_from_bytes_minimal() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse minimal");

    assert_eq!(model.header.magic, GGUF_MAGIC);
    assert_eq!(model.header.version, GGUF_VERSION_V3);
    assert_eq!(model.header.tensor_count, 0);
    assert_eq!(model.header.metadata_count, 0);
    assert!(model.metadata.is_empty());
    assert!(model.tensors.is_empty());
}

#[test]
fn test_cov_gguf_model_from_bytes_with_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&3u64.to_le_bytes()); // metadata_count

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 2048);
    add_bool_meta(&mut data, "llama.use_parallel_residual", true);

    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.metadata.len(), 3);
    assert!(model.metadata.contains_key("general.architecture"));
    assert!(model.metadata.contains_key("llama.embedding_length"));
    assert!(model.metadata.contains_key("llama.use_parallel_residual"));
}

#[test]
fn test_cov_gguf_model_from_bytes_invalid_magic() {
    let mut data = Vec::new();
    data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // Invalid magic
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_model_from_bytes_too_short() {
    let data = vec![0u8; 4]; // Too short for header
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_model_clone() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let cloned = model.clone();

    assert_eq!(cloned.header, model.header);
    assert_eq!(cloned.metadata.len(), model.metadata.len());
    assert_eq!(cloned.tensors.len(), model.tensors.len());
    assert_eq!(cloned.tensor_data_start, model.tensor_data_start);
}

#[test]
fn test_cov_gguf_model_debug() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    let debug_str = format!("{model:?}");
    assert!(debug_str.contains("GGUFModel"));
    assert!(debug_str.contains("header"));
}

// ============================================================================
// GGUF MODEL METADATA ACCESSOR TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_architecture_some() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.architecture(), Some("llama"));
}

#[test]
fn test_cov_gguf_model_architecture_none() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.architecture(), None);
}

#[test]
fn test_cov_gguf_model_embedding_dim_some() {
    let data = build_gguf_with_arch("phi2", 2560, 32, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.embedding_dim(), Some(2560));
}

#[test]
fn test_cov_gguf_model_embedding_dim_none_no_arch() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    // Without architecture, embedding_dim returns None
    assert_eq!(model.embedding_dim(), None);
}

#[test]
fn test_cov_gguf_model_num_layers_some() {
    let data = build_gguf_with_arch("qwen2", 896, 24, 14);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.num_layers(), Some(24));
}

#[test]
fn test_cov_gguf_model_num_layers_none() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // Only architecture

    add_string_meta(&mut data, "general.architecture", "test");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    // Architecture present but no block_count
    assert_eq!(model.num_layers(), None);
}

#[test]
fn test_cov_gguf_model_num_heads_some() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.num_heads(), Some(32));
}

#[test]
fn test_cov_gguf_model_num_heads_none() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "test");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.num_heads(), None);
}

#[test]
fn test_cov_gguf_model_num_kv_heads_some() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.attention.head_count_kv", 8);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.num_kv_heads(), Some(8));
}

#[test]
fn test_cov_gguf_model_num_kv_heads_none() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    // Our test helper doesn't add head_count_kv
    assert_eq!(model.num_kv_heads(), None);
}

#[test]
fn test_cov_gguf_model_context_length_some() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.context_length", 4096);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.context_length(), Some(4096));
}

#[test]
fn test_cov_gguf_model_context_length_none() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.context_length(), None);
}

#[test]
fn test_cov_gguf_model_rope_freq_base_some() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "qwen2");
    add_f32_meta(&mut data, "qwen2.rope.freq_base", 1_000_000.0);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_freq_base(), Some(1_000_000.0));
}

#[test]
fn test_cov_gguf_model_rope_freq_base_none() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_freq_base(), None);
}

#[test]
fn test_cov_gguf_model_rms_epsilon_some() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "qwen2");
    add_f32_meta(&mut data, "qwen2.attention.layer_norm_rms_epsilon", 1e-6);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let eps = model.rms_epsilon().expect("eps");
    assert!((eps - 1e-6).abs() < 1e-10);
}

#[test]
fn test_cov_gguf_model_rms_epsilon_none() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rms_epsilon(), None);
}

// ============================================================================
// GGUF MODEL ROPE TYPE TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_rope_type_neox_from_scaling() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "test");
    add_string_meta(&mut data, "test.rope.scaling.type", "yarn");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX style
}

#[test]
fn test_cov_gguf_model_rope_type_norm_from_scaling() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "test");
    add_string_meta(&mut data, "test.rope.scaling.type", "none");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(0)); // NORM style
}

#[test]
fn test_cov_gguf_model_rope_type_linear_scaling() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "test");
    add_string_meta(&mut data, "test.rope.scaling.type", "linear");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(0)); // Linear -> NORM style
}

#[test]
fn test_cov_gguf_model_rope_type_neox_scaling() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "test");
    add_string_meta(&mut data, "test.rope.scaling.type", "neox");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.rope_type(), Some(2)); // NEOX style
}

#[test]
fn test_cov_gguf_model_rope_type_inferred_qwen() {
    let data = build_gguf_with_arch("qwen2", 896, 24, 14);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_type(), Some(2)); // qwen2 -> NEOX
}

#[test]
fn test_cov_gguf_model_rope_type_inferred_phi() {
    let data = build_gguf_with_arch("phi2", 2560, 32, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_type(), Some(2)); // phi2 -> NEOX
}

#[test]
fn test_cov_gguf_model_rope_type_inferred_llama() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_type(), Some(0)); // llama -> NORM
}

#[test]
fn test_cov_gguf_model_rope_type_inferred_gemma() {
    let data = build_gguf_with_arch("gemma", 2048, 18, 8);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_type(), Some(2)); // gemma -> NEOX
}

#[test]
fn test_cov_gguf_model_rope_type_inferred_falcon() {
    let data = build_gguf_with_arch("falcon", 4544, 32, 71);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_type(), Some(2)); // falcon -> NEOX
}

#[test]
fn test_cov_gguf_model_rope_type_inferred_stablelm() {
    let data = build_gguf_with_arch("stablelm", 2048, 24, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_type(), Some(2)); // stablelm -> NEOX
}

#[test]
fn test_cov_gguf_model_rope_type_default_unknown() {
    // Unknown architecture defaults to NORM
    let data = build_gguf_with_arch("unknown_arch", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.rope_type(), Some(0)); // Unknown -> NORM (default)
}

// ============================================================================
// GGUF MODEL TOKENIZER METADATA TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_bos_token_id_some() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_u32_meta(&mut data, "tokenizer.ggml.bos_token_id", 1);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.bos_token_id(), Some(1));
}

#[test]
fn test_cov_gguf_model_bos_token_id_none() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.bos_token_id(), None);
}

#[test]
fn test_cov_gguf_model_eos_token_id_some() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_u32_meta(&mut data, "tokenizer.ggml.eos_token_id", 2);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.eos_token_id(), Some(2));
}

#[test]
fn test_cov_gguf_model_eos_token_id_none() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.eos_token_id(), None);
}

#[test]
fn test_cov_gguf_model_vocabulary_some() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_array_meta(
        &mut data,
        "tokenizer.ggml.tokens",
        &["<s>", "</s>", "hello"],
    );

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let vocab = model.vocabulary().expect("vocabulary");

    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab[0], "<s>");
    assert_eq!(vocab[1], "</s>");
    assert_eq!(vocab[2], "hello");
}

#[test]
fn test_cov_gguf_model_vocabulary_none() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.vocabulary(), None);
}

#[test]
fn test_cov_gguf_model_vocabulary_empty_array_returns_none() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Empty string array
    add_string_array_meta(&mut data, "tokenizer.ggml.tokens", &[]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    // Empty array returns None
    assert_eq!(model.vocabulary(), None);
}

// ============================================================================
// GGUF MODEL DECODE TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_decode_basic() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_array_meta(&mut data, "tokenizer.ggml.tokens", &["hello", " ", "world"]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0, 1, 2]);

    assert_eq!(decoded, "hello world");
}

#[test]
fn test_cov_gguf_model_decode_sentencepiece_marker() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // SentencePiece uses ▁ for word boundaries
    add_string_array_meta(
        &mut data,
        "tokenizer.ggml.tokens",
        &["\u{2581}hello", "\u{2581}world"],
    );

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0, 1]);

    assert_eq!(decoded, " hello world");
}

#[test]
fn test_cov_gguf_model_decode_byte_token() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Byte token format: <0xHH>
    add_string_array_meta(&mut data, "tokenizer.ggml.tokens", &["<0x41>"]); // 0x41 = 'A'

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0]);

    assert_eq!(decoded, "A");
}

#[test]
fn test_cov_gguf_model_decode_unknown_token() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_array_meta(&mut data, "tokenizer.ggml.tokens", &["a", "b"]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    // Token ID 99 is out of bounds
    let decoded = model.decode(&[0, 99, 1]);

    // Unknown tokens become replacement character
    assert!(decoded.contains('\u{FFFD}'));
}

#[test]
fn test_cov_gguf_model_decode_no_vocabulary_fallback() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    // Without vocabulary, decode falls back to showing token IDs
    let decoded = model.decode(&[65, 66, 67]);

    // Should contain token IDs as text
    assert!(!decoded.is_empty());
}

#[test]
fn test_cov_gguf_model_decode_gpt2_style() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "tokenizer.ggml.model", "gpt2");
    // GPT-2 uses Ġ for space
    add_string_array_meta(
        &mut data,
        "tokenizer.ggml.tokens",
        &["hello", "\u{0120}world"],
    );

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let decoded = model.decode(&[0, 1]);

    assert_eq!(decoded, "hello world");
}

// ============================================================================
// GGUF CONFIG TESTS
// ============================================================================

#[test]
fn test_cov_gguf_config_from_gguf_basic() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 2048);
    assert_eq!(config.num_layers, 22);
    assert_eq!(config.num_heads, 32);
}

#[test]
fn test_cov_gguf_config_from_gguf_missing_architecture() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = GGUFConfig::from_gguf(&model);

    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_config_from_gguf_missing_embedding_length() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "llama");

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = GGUFConfig::from_gguf(&model);

    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_config_from_gguf_missing_block_count() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 2048);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let result = GGUFConfig::from_gguf(&model);

    assert!(result.is_err());
}

#[test]
fn test_cov_gguf_config_defaults_num_heads() {
    // When num_heads is missing, defaults to hidden_dim / 64
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "test");
    add_u32_meta(&mut data, "test.embedding_length", 2048);
    add_u32_meta(&mut data, "test.block_count", 22);
    // No head_count - should default to 2048/64 = 32

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.num_heads, 32);
}

#[test]
fn test_cov_gguf_config_defaults_context_length() {
    let data = build_gguf_with_arch("test", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.context_length, 2048); // Default
}

#[test]
fn test_cov_gguf_config_defaults_rope_theta() {
    let data = build_gguf_with_arch("test", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert!((config.rope_theta - 10000.0).abs() < 1.0); // Default
}

#[test]
fn test_cov_gguf_config_defaults_eps() {
    let data = build_gguf_with_arch("test", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert!((config.eps - 1e-5).abs() < 1e-8); // Default
}

#[test]
fn test_cov_gguf_config_defaults_num_kv_heads() {
    // num_kv_heads defaults to num_heads when not specified
    let data = build_gguf_with_arch("test", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.num_kv_heads, 16); // Same as num_heads
}

#[test]
fn test_cov_gguf_config_defaults_vocab_size() {
    // vocab_size defaults to 32000 when no token_embd tensor
    let data = build_gguf_with_arch("test", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.vocab_size, 32000); // Default
}

#[test]
fn test_cov_gguf_config_defaults_intermediate_dim() {
    // intermediate_dim defaults to hidden_dim * 4 when no ffn_up tensor
    let data = build_gguf_with_arch("test", 1024, 12, 16);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.intermediate_dim, 4096); // 1024 * 4
}

#[test]
fn test_cov_gguf_config_with_all_options() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&8u64.to_le_bytes());

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 4096);
    add_u32_meta(&mut data, "llama.block_count", 32);
    add_u32_meta(&mut data, "llama.attention.head_count", 32);
    add_u32_meta(&mut data, "llama.attention.head_count_kv", 8);
    add_u32_meta(&mut data, "llama.context_length", 8192);
    add_f32_meta(&mut data, "llama.rope.freq_base", 500_000.0);
    add_f32_meta(&mut data, "llama.attention.layer_norm_rms_epsilon", 1e-5);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.context_length, 8192);
    assert!((config.rope_theta - 500_000.0).abs() < 1.0);
    assert!((config.eps - 1e-5).abs() < 1e-8);
}

#[test]
fn test_cov_gguf_config_vocab_from_tensor() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 metadata

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 512);
    add_u32_meta(&mut data, "llama.block_count", 4);

    // token_embd.weight tensor - dims in GGML order (reversed)
    // We want [50000, 512] after reversal, so we provide [512, 50000]
    add_tensor_info(
        &mut data,
        "token_embd.weight",
        &[50000, 512],
        GGUF_TYPE_F32,
        0,
    );

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.vocab_size, 50000);
}

#[test]
fn test_cov_gguf_config_intermediate_from_tensor() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 metadata

    add_string_meta(&mut data, "general.architecture", "llama");
    add_u32_meta(&mut data, "llama.embedding_length", 512);
    add_u32_meta(&mut data, "llama.block_count", 4);

    // blk.0.ffn_up.weight tensor - we want [8192, 512] after reversal
    add_tensor_info(
        &mut data,
        "blk.0.ffn_up.weight",
        &[8192, 512],
        GGUF_TYPE_F32,
        0,
    );

    let model = GGUFModel::from_bytes(&data).expect("parse");
    let config = GGUFConfig::from_gguf(&model).expect("config");

    assert_eq!(config.intermediate_dim, 8192);
}

// ============================================================================
// GGUF VALUE ENUM TESTS
// ============================================================================

#[test]
fn test_cov_gguf_value_u8() {
    let val = GGUFValue::UInt8(255);
    if let GGUFValue::UInt8(v) = val {
        assert_eq!(v, 255);
    } else {
        panic!("Expected UInt8");
    }
}

#[test]
fn test_cov_gguf_value_i8() {
    let val = GGUFValue::Int8(-128);
    if let GGUFValue::Int8(v) = val {
        assert_eq!(v, -128);
    } else {
        panic!("Expected Int8");
    }
}

#[test]
fn test_cov_gguf_value_u16() {
    let val = GGUFValue::UInt16(65535);
    if let GGUFValue::UInt16(v) = val {
        assert_eq!(v, 65535);
    } else {
        panic!("Expected UInt16");
    }
}

#[test]
fn test_cov_gguf_value_i16() {
    let val = GGUFValue::Int16(-32768);
    if let GGUFValue::Int16(v) = val {
        assert_eq!(v, -32768);
    } else {
        panic!("Expected Int16");
    }
}

#[test]
fn test_cov_gguf_value_u32() {
    let val = GGUFValue::UInt32(u32::MAX);
    if let GGUFValue::UInt32(v) = val {
        assert_eq!(v, u32::MAX);
    } else {
        panic!("Expected UInt32");
    }
}

#[test]
fn test_cov_gguf_value_i32() {
    let val = GGUFValue::Int32(i32::MIN);
    if let GGUFValue::Int32(v) = val {
        assert_eq!(v, i32::MIN);
    } else {
        panic!("Expected Int32");
    }
}

#[test]
fn test_cov_gguf_value_f32() {
    let val = GGUFValue::Float32(1.23456);
    if let GGUFValue::Float32(v) = val {
        assert!((v - 1.23456).abs() < 0.0001);
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_cov_gguf_value_bool_true() {
    let val = GGUFValue::Bool(true);
    if let GGUFValue::Bool(v) = val {
        assert!(v);
    } else {
        panic!("Expected Bool");
    }
}

#[test]
fn test_cov_gguf_value_bool_false() {
    let val = GGUFValue::Bool(false);
    if let GGUFValue::Bool(v) = val {
        assert!(!v);
    } else {
        panic!("Expected Bool");
    }
}

#[test]
fn test_cov_gguf_value_string() {
    let val = GGUFValue::String("test_string".to_string());
    if let GGUFValue::String(s) = val {
        assert_eq!(s, "test_string");
    } else {
        panic!("Expected String");
    }
}

#[test]
fn test_cov_gguf_value_array_u32() {
    let val = GGUFValue::Array(vec![
        GGUFValue::UInt32(1),
        GGUFValue::UInt32(2),
        GGUFValue::UInt32(3),
    ]);
    if let GGUFValue::Array(arr) = val {
        assert_eq!(arr.len(), 3);
        if let GGUFValue::UInt32(v) = arr[0] {
            assert_eq!(v, 1);
        }
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_cov_gguf_value_array_string() {
    let val = GGUFValue::Array(vec![
        GGUFValue::String("a".to_string()),
        GGUFValue::String("b".to_string()),
    ]);
    if let GGUFValue::Array(arr) = val {
        assert_eq!(arr.len(), 2);
    } else {
        panic!("Expected Array");
    }
}

#[test]
fn test_cov_gguf_value_u64() {
    let val = GGUFValue::UInt64(u64::MAX);
    if let GGUFValue::UInt64(v) = val {
        assert_eq!(v, u64::MAX);
    } else {
        panic!("Expected UInt64");
    }
}

#[test]
fn test_cov_gguf_value_i64() {
    let val = GGUFValue::Int64(i64::MIN);
    if let GGUFValue::Int64(v) = val {
        assert_eq!(v, i64::MIN);
    } else {
        panic!("Expected Int64");
    }
}

#[test]
fn test_cov_gguf_value_f64() {
    let val = GGUFValue::Float64(1.234567890123);
    if let GGUFValue::Float64(v) = val {
        assert!((v - 1.234567890123).abs() < 0.0000001);
    } else {
        panic!("Expected Float64");
    }
}

// ============================================================================
// METADATA PARSING TESTS (all value types)
// ============================================================================

#[test]
fn test_cov_metadata_parse_u8() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_u8_meta(&mut data, "test.u8", 42);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::UInt8(v)) = model.metadata.get("test.u8") {
        assert_eq!(*v, 42);
    } else {
        panic!("Expected UInt8");
    }
}

#[test]
fn test_cov_metadata_parse_i8() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_i8_meta(&mut data, "test.i8", -42);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Int8(v)) = model.metadata.get("test.i8") {
        assert_eq!(*v, -42);
    } else {
        panic!("Expected Int8");
    }
}

#[test]
fn test_cov_metadata_parse_u16() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_u16_meta(&mut data, "test.u16", 1234);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::UInt16(v)) = model.metadata.get("test.u16") {
        assert_eq!(*v, 1234);
    } else {
        panic!("Expected UInt16");
    }
}

#[test]
fn test_cov_metadata_parse_i16() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_i16_meta(&mut data, "test.i16", -1234);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Int16(v)) = model.metadata.get("test.i16") {
        assert_eq!(*v, -1234);
    } else {
        panic!("Expected Int16");
    }
}

#[test]
fn test_cov_metadata_parse_i32() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_i32_meta(&mut data, "test.i32", -123456);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Int32(v)) = model.metadata.get("test.i32") {
        assert_eq!(*v, -123456);
    } else {
        panic!("Expected Int32");
    }
}

#[test]
fn test_cov_metadata_parse_u64() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_u64_meta(&mut data, "test.u64", 9_999_999_999);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::UInt64(v)) = model.metadata.get("test.u64") {
        assert_eq!(*v, 9_999_999_999);
    } else {
        panic!("Expected UInt64");
    }
}

#[test]
fn test_cov_metadata_parse_i64() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_i64_meta(&mut data, "test.i64", -9_999_999_999);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Int64(v)) = model.metadata.get("test.i64") {
        assert_eq!(*v, -9_999_999_999);
    } else {
        panic!("Expected Int64");
    }
}

#[test]
fn test_cov_metadata_parse_f64() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_f64_meta(&mut data, "test.f64", 1.23456789012345);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float64(v)) = model.metadata.get("test.f64") {
        assert!((v - 1.23456789012345).abs() < 1e-10);
    } else {
        panic!("Expected Float64");
    }
}

#[test]
fn test_cov_metadata_parse_bool_true() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_bool_meta(&mut data, "test.bool", true);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Bool(v)) = model.metadata.get("test.bool") {
        assert!(*v);
    } else {
        panic!("Expected Bool");
    }
}

#[test]
fn test_cov_metadata_parse_array_u32() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_u32_array_meta(&mut data, "test.array", &[10, 20, 30]);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test.array") {
        assert_eq!(arr.len(), 3);
    } else {
        panic!("Expected Array");
    }
}

// ============================================================================
// MAPPED GGUF MODEL TESTS
// ============================================================================

#[test]
fn test_cov_mapped_gguf_model_from_path() {
    let data = build_gguf_with_arch("llama", 2048, 22, 32);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    assert_eq!(mapped.model.architecture(), Some("llama"));
}

#[test]
fn test_cov_mapped_gguf_model_data() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // data() should return the full memory-mapped file
    assert_eq!(mapped.data().len(), data.len());
}

#[test]
fn test_cov_mapped_gguf_model_file_size() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    assert_eq!(mapped.file_size(), data.len());
}

#[test]
fn test_cov_mapped_gguf_model_tensor_slice_valid() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // Valid slice
    let slice = mapped.tensor_slice(0, 10);
    assert!(slice.is_some());
    assert_eq!(slice.unwrap().len(), 10);
}

#[test]
fn test_cov_mapped_gguf_model_tensor_slice_out_of_bounds() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // Out of bounds slice - offset + size > file_size
    let slice = mapped.tensor_slice(mapped.file_size(), 1);
    assert!(slice.is_none());
}

#[test]
fn test_cov_mapped_gguf_model_tensor_slice_overflow() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // Overflow check - very large size that would overflow
    let slice = mapped.tensor_slice(0, usize::MAX);
    assert!(slice.is_none());
}

#[test]
fn test_cov_mapped_gguf_model_file_not_found() {
    let result = MappedGGUFModel::from_path("/nonexistent/path/model.gguf");
    assert!(result.is_err());
}

#[test]
#[cfg(unix)]
fn test_cov_mapped_gguf_model_advise_sequential() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // Should not panic
    mapped.advise_sequential();
}

#[test]
#[cfg(unix)]
fn test_cov_mapped_gguf_model_advise_random() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // Should not panic
    mapped.advise_random();
}

#[test]
#[cfg(unix)]
fn test_cov_mapped_gguf_model_advise_willneed() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // Should not panic
    mapped.advise_willneed();
}

#[test]
#[cfg(unix)]
fn test_cov_mapped_gguf_model_lock_memory() {
    let data = build_gguf_with_arch("test", 512, 4, 8);
    let mut temp = NamedTempFile::new().expect("create temp");
    temp.write_all(&data).expect("write");

    let mapped = MappedGGUFModel::from_path(temp.path()).expect("load");

    // May return true or false depending on privileges, should not panic
    let _ = mapped.lock_memory();
}

// ============================================================================
// GGUF CONSTANTS TESTS
// ============================================================================

#[test]
fn test_cov_gguf_constants_magic() {
    // Verify magic bytes spell "GGUF"
    let magic_bytes = GGUF_MAGIC.to_le_bytes();
    assert_eq!(&magic_bytes, b"GGUF");
}

#[test]
fn test_cov_gguf_constants_version() {
    assert_eq!(GGUF_VERSION_V3, 3);
}

#[test]
fn test_cov_gguf_constants_alignment() {
    assert_eq!(GGUF_ALIGNMENT, 32);
}

#[test]
fn test_cov_gguf_constants_qtypes() {
    assert_eq!(GGUF_TYPE_F32, 0);
    assert_eq!(GGUF_TYPE_F16, 1);
    assert_eq!(GGUF_TYPE_Q4_0, 2);
    assert_eq!(GGUF_TYPE_Q4_1, 3);
    assert_eq!(GGUF_TYPE_Q5_0, 6);
    assert_eq!(GGUF_TYPE_Q5_1, 7);
    assert_eq!(GGUF_TYPE_Q8_0, 8);
    assert_eq!(GGUF_TYPE_Q4_K, 12);
    assert_eq!(GGUF_TYPE_Q5_K, 13);
    assert_eq!(GGUF_TYPE_Q6_K, 14);
}

// ============================================================================
// TENSOR DATA START ALIGNMENT TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_tensor_data_start_aligned() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    // tensor_data_start should be 32-byte aligned
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

#[test]
fn test_cov_gguf_model_tensor_data_start_with_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&5u64.to_le_bytes()); // 5 metadata entries

    // Add several metadata entries of varying sizes
    add_string_meta(&mut data, "key1", "value1");
    add_u32_meta(&mut data, "key2", 123);
    add_string_meta(&mut data, "key3", "longer_value_here");
    add_f32_meta(&mut data, "key4", 1.5);
    add_bool_meta(&mut data, "key5", true);

    let model = GGUFModel::from_bytes(&data).expect("parse");

    // tensor_data_start should still be 32-byte aligned
    assert_eq!(model.tensor_data_start % GGUF_ALIGNMENT, 0);
}

// ============================================================================
// EDGE CASES AND ERROR PATHS
// ============================================================================

#[test]
fn test_cov_gguf_model_many_metadata_entries() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes()); // 10 metadata entries

    for i in 0..10 {
        add_u32_meta(&mut data, &format!("key_{i}"), i as u32);
    }

    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.metadata.len(), 10);
}

#[test]
fn test_cov_gguf_model_multiple_tensors() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&3u64.to_le_bytes()); // 3 tensors
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata

    add_string_meta(&mut data, "general.architecture", "test");

    add_tensor_info(&mut data, "tensor_a", &[100, 100], GGUF_TYPE_F32, 0);
    add_tensor_info(&mut data, "tensor_b", &[50, 200], GGUF_TYPE_Q4_0, 40000);
    add_tensor_info(&mut data, "tensor_c", &[10], GGUF_TYPE_Q8_0, 50000);

    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.tensors.len(), 3);
    assert_eq!(model.tensors[0].name, "tensor_a");
    assert_eq!(model.tensors[1].name, "tensor_b");
    assert_eq!(model.tensors[2].name, "tensor_c");
}

#[test]
fn test_cov_gguf_model_tensor_1d() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    add_tensor_info(&mut data, "bias", &[512], GGUF_TYPE_F32, 0);

    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.tensors[0].n_dims, 1);
    assert_eq!(model.tensors[0].dims, vec![512]);
}

#[test]
fn test_cov_gguf_model_tensor_3d() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    add_tensor_info(&mut data, "conv", &[3, 64, 64], GGUF_TYPE_F32, 0);

    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert_eq!(model.tensors[0].n_dims, 3);
    // Dims are reversed when parsed
    assert_eq!(model.tensors[0].dims.len(), 3);
}

#[test]
fn test_cov_gguf_model_empty_string_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_meta(&mut data, "empty_value", "");

    let model = GGUFModel::from_bytes(&data).expect("parse");

    if let Some(GGUFValue::String(s)) = model.metadata.get("empty_value") {
        assert!(s.is_empty());
    } else {
        panic!("Expected empty string");
    }
}

#[test]
fn test_cov_gguf_model_long_key_name() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let long_key = "a".repeat(100);
    add_u32_meta(&mut data, &long_key, 42);

    let model = GGUFModel::from_bytes(&data).expect("parse");

    assert!(model.metadata.contains_key(&long_key));
}

#[test]
fn test_cov_gguf_model_unicode_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_meta(&mut data, "unicode_test", "Hello");

    let model = GGUFModel::from_bytes(&data).expect("parse");

    if let Some(GGUFValue::String(s)) = model.metadata.get("unicode_test") {
        assert_eq!(s, "Hello");
    } else {
        panic!("Expected unicode string");
    }
}

// ============================================================================
// GGUF CONFIG STRUCT DIRECT TESTS
// ============================================================================

#[test]
fn test_cov_gguf_config_struct_fields() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
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

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.intermediate_dim, 11008);
    assert_eq!(config.context_length, 4096);
    assert!((config.rope_theta - 10000.0).abs() < 1.0);
    assert!((config.eps - 1e-5).abs() < 1e-8);
    assert_eq!(config.rope_type, 0);
}

#[test]
fn test_cov_gguf_config_gqa_model() {
    // Grouped Query Attention model (fewer KV heads than Q heads)
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8, // GQA: 4x fewer KV heads
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

    // Verify GQA ratio
    assert_eq!(config.num_heads / config.num_kv_heads, 4);
}

#[test]
fn test_cov_gguf_config_mha_model() {
    // Multi-Head Attention model (same number of KV heads as Q heads)
    let config = GGUFConfig {
        architecture: "gpt2".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 12, // MHA: same as num_heads
        vocab_size: 50257,
        intermediate_dim: 3072,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    // Verify MHA
    assert_eq!(config.num_heads, config.num_kv_heads);
}

#[test]
fn test_cov_gguf_config_head_dim_calculation() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
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

    // head_dim = hidden_dim / num_heads
    let head_dim = config.hidden_dim / config.num_heads;
    assert_eq!(head_dim, 128);
}

// ============================================================================
// GGUF CONFIG CLONE AND DEBUG TESTS
// ============================================================================

#[test]
fn test_cov_gguf_config_clone() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2048,
        num_layers: 22,
        num_heads: 32,
        num_kv_heads: 4,
        vocab_size: 32000,
        intermediate_dim: 5632,
        context_length: 2048,
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
    assert_eq!(cloned.num_layers, config.num_layers);
}

#[test]
fn test_cov_gguf_config_debug() {
    let config = GGUFConfig {
        architecture: "phi2".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 2560,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 32,
        vocab_size: 51200,
        intermediate_dim: 10240,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 2,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("GGUFConfig"));
}

// ============================================================================
// GGUF MODEL ENCODE TESTS
// ============================================================================

#[test]
fn test_cov_gguf_model_encode_with_vocab() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    add_string_array_meta(
        &mut data,
        "tokenizer.ggml.tokens",
        &["hello", "world", "test"],
    );

    let model = GGUFModel::from_bytes(&data).expect("parse");

    // encode should return Some when vocabulary is present
    let encoded = model.encode("hello");
    assert!(encoded.is_some());
    let tokens = encoded.unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_cov_gguf_model_encode_no_vocab() {
    let data = build_minimal_gguf_header();
    let model = GGUFModel::from_bytes(&data).expect("parse");

    // Without vocabulary, encode returns None
    let encoded = model.encode("test");
    assert!(encoded.is_none());
}

// ============================================================================
// OWNED QUANTIZED KV CACHE TESTS
// ============================================================================

use realizar::gguf::{OwnedQuantizedKVCache, QuantizedGenerateConfig};

#[test]
fn test_cov_kv_cache_new() {
    let cache = OwnedQuantizedKVCache::new(4, 512, 128);

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_len(), 128);
}

#[test]
fn test_cov_kv_cache_from_config() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 1024,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let cache = OwnedQuantizedKVCache::from_config(&config, 64);
    assert!(cache.is_empty());
    assert_eq!(cache.max_len(), 64);
}

#[test]
fn test_cov_kv_cache_append_and_advance() {
    let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];

    cache.append(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    let cached_k = cache.get_k(0);
    assert_eq!(cached_k.len(), 64);
    assert_eq!(cached_k[0], 1.0);

    let cached_v = cache.get_v(0);
    assert_eq!(cached_v.len(), 64);
    assert_eq!(cached_v[0], 2.0);
}

#[test]
fn test_cov_kv_cache_append_kv_batch() {
    let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

    let k_batch = vec![1.0f32; 64 * 3];
    let v_batch = vec![2.0f32; 64 * 3];

    cache.append_kv(0, &k_batch, &v_batch);
    cache.advance_by(3);

    assert_eq!(cache.len(), 3);
    let cached_k = cache.get_k(0);
    assert_eq!(cached_k.len(), 64 * 3);
}

#[test]
fn test_cov_kv_cache_rollback() {
    let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

    for _ in 0..5 {
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);
        cache.append(1, &k, &v);
        cache.advance();
    }
    assert_eq!(cache.len(), 5);

    cache.rollback_to(3, 64);
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.get_k(0).len(), 64 * 3);
}

#[test]
fn test_cov_kv_cache_rollback_no_op() {
    let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

    for _ in 0..3 {
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);
        cache.advance();
    }

    let orig_len = cache.len();
    cache.rollback_to(5, 64);
    assert_eq!(cache.len(), orig_len);
}

#[test]
fn test_cov_kv_cache_snapshot_len() {
    let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

    for _ in 0..4 {
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        cache.append(0, &k, &v);
        cache.advance();
    }

    assert_eq!(cache.snapshot_len(), 4);
}

#[test]
fn test_cov_kv_cache_reset() {
    let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    cache.append(0, &k, &v);
    cache.advance();

    assert!(!cache.is_empty());

    cache.reset();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cov_kv_cache_get_invalid_layer() {
    let cache = OwnedQuantizedKVCache::new(2, 64, 10);

    let k = cache.get_k(100);
    assert!(k.is_empty());

    let v = cache.get_v(100);
    assert!(v.is_empty());
}

#[test]
fn test_cov_kv_cache_default() {
    let cache = OwnedQuantizedKVCache::default();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_len(), 0);
}

#[test]
fn test_cov_kv_cache_clone() {
    let mut cache = OwnedQuantizedKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    cache.append(0, &k, &v);
    cache.advance();

    let cloned = cache.clone();
    assert_eq!(cloned.len(), cache.len());
    assert_eq!(cloned.max_len(), cache.max_len());
}

#[test]
fn test_cov_kv_cache_debug() {
    let cache = OwnedQuantizedKVCache::new(2, 64, 10);
    let debug_str = format!("{cache:?}");
    assert!(debug_str.contains("OwnedQuantizedKVCache"));
}

// ============================================================================
// QUANTIZED GENERATE CONFIG TESTS
// ============================================================================

#[test]
fn test_cov_generate_config_default() {
    let config = QuantizedGenerateConfig::default();

    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_cov_generate_config_deterministic() {
    let config = QuantizedGenerateConfig::deterministic(128);

    assert_eq!(config.max_tokens, 128);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_cov_generate_config_custom() {
    let config = QuantizedGenerateConfig {
        max_tokens: 256,
        temperature: 0.8,
        top_k: 50,
        stop_tokens: vec![1, 2],
        trace: false,
        ..Default::default()
    };

    assert_eq!(config.max_tokens, 256);
    assert!((config.temperature - 0.8).abs() < 0.001);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.stop_tokens, vec![1, 2]);
}

#[test]
fn test_cov_generate_config_clone() {
    let config = QuantizedGenerateConfig {
        max_tokens: 100,
        temperature: 0.5,
        top_k: 40,
        stop_tokens: vec![1, 2, 3],
        trace: false,
        ..Default::default()
    };

    let cloned = config.clone();
    assert_eq!(cloned.max_tokens, config.max_tokens);
    assert_eq!(cloned.temperature, config.temperature);
}

#[test]
fn test_cov_generate_config_debug() {
    let config = QuantizedGenerateConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("QuantizedGenerateConfig"));
}

// ============================================================================
// CONTIGUOUS KV CACHE TESTS
// ============================================================================

use realizar::gguf::ContiguousKVCache;

#[test]
fn test_cov_contiguous_cache_new() {
    let cache = ContiguousKVCache::new(4, 128, 64);

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_len(), 64);
    assert!(cache.is_contiguous());
    assert!(cache.is_cache_aligned());
}

#[test]
fn test_cov_contiguous_cache_from_config() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        num_layers: 8,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 1024,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let cache = ContiguousKVCache::from_config(&config, 32);
    assert!(cache.is_empty());
    assert_eq!(cache.max_len(), 32);
}

#[test]
fn test_cov_contiguous_cache_append_and_advance() {
    let mut cache = ContiguousKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];

    cache.append(0, &k, &v);
    cache.advance();

    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    let cached_k = cache.get_k(0);
    assert_eq!(cached_k.len(), 64);
    assert_eq!(cached_k[0], 1.0);
}

#[test]
fn test_cov_contiguous_cache_get_k_mut() {
    let mut cache = ContiguousKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    cache.append(0, &k, &v);
    cache.advance();

    let k_mut = cache.get_k_mut(0);
    assert_eq!(k_mut.len(), 64);
    k_mut[0] = 99.0;

    let k_read = cache.get_k(0);
    assert_eq!(k_read[0], 99.0);
}

#[test]
fn test_cov_contiguous_cache_get_v_mut() {
    let mut cache = ContiguousKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    cache.append(0, &k, &v);
    cache.advance();

    let v_mut = cache.get_v_mut(0);
    assert_eq!(v_mut.len(), 64);
    v_mut[0] = 88.0;

    let v_read = cache.get_v(0);
    assert_eq!(v_read[0], 88.0);
}

#[test]
fn test_cov_contiguous_cache_reset() {
    let mut cache = ContiguousKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    cache.append(0, &k, &v);
    cache.advance();

    cache.reset();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cov_contiguous_cache_reset_and_zero() {
    let mut cache = ContiguousKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    cache.append(0, &k, &v);
    cache.advance();

    cache.reset_and_zero();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cov_contiguous_cache_memory_bytes() {
    let cache = ContiguousKVCache::new(4, 128, 64);
    let bytes = cache.memory_bytes();

    assert!(bytes > 0);
    assert_eq!(bytes % 4, 0);
}

#[test]
fn test_cov_contiguous_cache_layer_stride() {
    let cache = ContiguousKVCache::new(4, 128, 64);
    let stride = cache.layer_stride();

    assert!(stride >= 64 * 128);
    assert_eq!(stride % 16, 0);
}

#[test]
fn test_cov_contiguous_cache_prefetch() {
    let mut cache = ContiguousKVCache::new(2, 64, 10);

    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    cache.append(0, &k, &v);
    cache.advance();

    cache.prefetch_k(0);
    cache.prefetch_v(0);
    cache.prefetch_k(100);
    cache.prefetch_v(100);
}

#[test]
fn test_cov_contiguous_cache_get_invalid_layer() {
    let cache = ContiguousKVCache::new(2, 64, 10);

    let k = cache.get_k(100);
    assert!(k.is_empty());

    let v = cache.get_v(100);
    assert!(v.is_empty());
}

#[test]
fn test_cov_contiguous_cache_get_mut_invalid_layer() {
    let mut cache = ContiguousKVCache::new(2, 64, 10);

    let k = cache.get_k_mut(100);
    assert!(k.is_empty());

    let v = cache.get_v_mut(100);
    assert!(v.is_empty());
}

// ============================================================================
// DISPATCH METRICS TESTS
// ============================================================================

use realizar::gguf::DispatchMetrics;
use std::time::Duration;

#[test]
fn test_cov_dispatch_metrics_new() {
    let metrics = DispatchMetrics::new();

    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 0);
    assert_eq!(metrics.total_dispatches(), 0);
}

#[test]
fn test_cov_dispatch_metrics_record_cpu_dispatch() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();

    assert_eq!(metrics.cpu_dispatches(), 3);
    assert_eq!(metrics.gpu_dispatches(), 0);
    assert_eq!(metrics.total_dispatches(), 3);
}

#[test]
fn test_cov_dispatch_metrics_record_gpu_dispatch() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_dispatch();
    metrics.record_gpu_dispatch();

    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 2);
    assert_eq!(metrics.total_dispatches(), 2);
}

#[test]
fn test_cov_dispatch_metrics_gpu_ratio_zero() {
    let metrics = DispatchMetrics::new();
    assert_eq!(metrics.gpu_ratio(), 0.0);
}

#[test]
fn test_cov_dispatch_metrics_gpu_ratio_half() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_dispatch();
    metrics.record_gpu_dispatch();

    assert!((metrics.gpu_ratio() - 0.5).abs() < 0.001);
}

#[test]
fn test_cov_dispatch_metrics_gpu_ratio_all_gpu() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_dispatch();
    metrics.record_gpu_dispatch();
    metrics.record_gpu_dispatch();

    assert!((metrics.gpu_ratio() - 1.0).abs() < 0.001);
}

#[test]
fn test_cov_dispatch_metrics_record_cpu_latency() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    assert!((metrics.cpu_latency_mean_us() - 200.0).abs() < 1.0);
}

#[test]
fn test_cov_dispatch_metrics_record_gpu_latency() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(50));
    metrics.record_gpu_latency(Duration::from_micros(150));

    assert!((metrics.gpu_latency_mean_us() - 100.0).abs() < 1.0);
}

#[test]
fn test_cov_dispatch_metrics_mean_latency_zero() {
    let metrics = DispatchMetrics::new();

    assert_eq!(metrics.cpu_latency_mean_us(), 0.0);
    assert_eq!(metrics.gpu_latency_mean_us(), 0.0);
}

#[test]
fn test_cov_dispatch_metrics_histogram_buckets() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(50));
    metrics.record_cpu_latency(Duration::from_micros(250));
    metrics.record_cpu_latency(Duration::from_micros(750));
    metrics.record_cpu_latency(Duration::from_micros(2000));
    metrics.record_cpu_latency(Duration::from_micros(10000));

    let buckets = metrics.cpu_latency_buckets();
    assert_eq!(buckets[0], 1);
    assert_eq!(buckets[1], 1);
    assert_eq!(buckets[2], 1);
    assert_eq!(buckets[3], 1);
    assert_eq!(buckets[4], 1);
}

#[test]
fn test_cov_dispatch_metrics_gpu_histogram() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(50));
    metrics.record_gpu_latency(Duration::from_micros(250));

    let buckets = metrics.gpu_latency_buckets();
    assert_eq!(buckets[0], 1);
    assert_eq!(buckets[1], 1);
}

#[test]
fn test_cov_dispatch_metrics_min_max_cpu() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(500));
    metrics.record_cpu_latency(Duration::from_micros(200));

    assert_eq!(metrics.cpu_latency_min_us(), 100);
    assert_eq!(metrics.cpu_latency_max_us(), 500);
}

#[test]
fn test_cov_dispatch_metrics_min_max_gpu() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(50));
    metrics.record_gpu_latency(Duration::from_micros(300));
    metrics.record_gpu_latency(Duration::from_micros(100));

    assert_eq!(metrics.gpu_latency_min_us(), 50);
    assert_eq!(metrics.gpu_latency_max_us(), 300);
}

#[test]
fn test_cov_dispatch_metrics_variance() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(100));

    assert_eq!(metrics.cpu_latency_variance_us(), 0.0);
}

#[test]
fn test_cov_dispatch_metrics_stddev() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(100));

    assert_eq!(metrics.cpu_latency_stddev_us(), 0.0);
}

#[test]
fn test_cov_dispatch_metrics_bucket_boundaries() {
    let boundaries = DispatchMetrics::BUCKET_BOUNDARIES;
    assert_eq!(boundaries.len(), 4);
    assert_eq!(boundaries[0], 100);
    assert_eq!(boundaries[1], 500);
    assert_eq!(boundaries[2], 1000);
    assert_eq!(boundaries[3], 5000);
}

#[test]
fn test_cov_dispatch_metrics_debug() {
    let metrics = DispatchMetrics::new();
    let debug_str = format!("{metrics:?}");
    assert!(debug_str.contains("DispatchMetrics"));
}

#[test]
fn test_cov_dispatch_metrics_default() {
    let metrics = DispatchMetrics::default();

    assert_eq!(metrics.cpu_dispatches(), 0);
    assert_eq!(metrics.gpu_dispatches(), 0);
}

#[test]
fn test_cov_dispatch_metrics_latency_count() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_gpu_latency(Duration::from_micros(300));

    assert_eq!(metrics.cpu_latency_count(), 2);
    assert_eq!(metrics.gpu_latency_count(), 1);
}

#[test]
fn test_cov_dispatch_metrics_sum_latency() {
    let metrics = DispatchMetrics::new();

    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));

    assert_eq!(metrics.cpu_latency_sum_us(), 300);
}

#[test]
fn test_cov_dispatch_metrics_gpu_variance() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));

    assert_eq!(metrics.gpu_latency_variance_us(), 0.0);
}

#[test]
fn test_cov_dispatch_metrics_gpu_stddev() {
    let metrics = DispatchMetrics::new();

    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));

    assert_eq!(metrics.gpu_latency_stddev_us(), 0.0);
}

#[test]
fn test_cov_dispatch_metrics_percentiles() {
    let metrics = DispatchMetrics::new();

    for _ in 0..10 {
        metrics.record_cpu_latency(Duration::from_micros(50));
    }

    let p50 = metrics.cpu_latency_p50_us();
    let p95 = metrics.cpu_latency_p95_us();
    let p99 = metrics.cpu_latency_p99_us();

    assert!((0.0..=100.0).contains(&p50));
    assert!((0.0..=100.0).contains(&p95));
    assert!((0.0..=100.0).contains(&p99));
}

#[test]
fn test_cov_dispatch_metrics_gpu_percentiles() {
    let metrics = DispatchMetrics::new();

    for _ in 0..10 {
        metrics.record_gpu_latency(Duration::from_micros(50));
    }

    let p50 = metrics.gpu_latency_p50_us();
    let p95 = metrics.gpu_latency_p95_us();
    let p99 = metrics.gpu_latency_p99_us();

    assert!((0.0..=100.0).contains(&p50));
    assert!((0.0..=100.0).contains(&p95));
    assert!((0.0..=100.0).contains(&p99));
}

#[test]
fn test_cov_dispatch_metrics_bucket_boundaries_strings() {
    let metrics = DispatchMetrics::new();
    let boundaries = metrics.bucket_boundaries_us();

    assert_eq!(boundaries.len(), 5);
    assert_eq!(boundaries[0], "0-100");
    assert_eq!(boundaries[4], "5000+");
}

// ============================================================================
// OWNED INFERENCE SCRATCH BUFFER TESTS
// ============================================================================

use realizar::gguf::OwnedInferenceScratchBuffer;

#[test]
fn test_cov_scratch_buffer_from_config() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 1024,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let scratch = OwnedInferenceScratchBuffer::from_config(&config);

    assert!(!scratch.qkv.is_empty());
    assert!(!scratch.attn_out.is_empty());
    assert!(!scratch.ffn_up.is_empty());
    assert!(!scratch.ffn_gate.is_empty());
    assert!(!scratch.ffn_down.is_empty());
    assert!(!scratch.logits.is_empty());
}

#[test]
fn test_cov_scratch_buffer_reset() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 1000,
        intermediate_dim: 1024,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let mut scratch = OwnedInferenceScratchBuffer::from_config(&config);
    scratch.reset();

    assert!(scratch.qkv.is_empty());
    assert!(scratch.attn_out.is_empty());
    assert!(scratch.ffn_up.is_empty());
}

#[test]
fn test_cov_scratch_buffer_debug() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 500,
        intermediate_dim: 512,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let scratch = OwnedInferenceScratchBuffer::from_config(&config);
    let debug_str = format!("{scratch:?}");
    assert!(debug_str.contains("OwnedInferenceScratchBuffer"));
}

// ============================================================================
// GGUF VALUE TESTS
// ============================================================================

#[test]
fn test_cov_gguf_value_clone() {
    let val = GGUFValue::String("test".to_string());
    let cloned = val;

    if let GGUFValue::String(s) = cloned {
        assert_eq!(s, "test");
    } else {
        panic!("Clone should preserve type");
    }
}

#[test]
fn test_cov_gguf_value_partial_eq() {
    let val1 = GGUFValue::UInt32(42);
    let val2 = GGUFValue::UInt32(42);
    let val3 = GGUFValue::UInt32(99);

    assert_eq!(val1, val2);
    assert_ne!(val1, val3);
}

#[test]
fn test_cov_gguf_value_debug_all_variants() {
    let values = vec![
        GGUFValue::UInt8(1),
        GGUFValue::Int8(-1),
        GGUFValue::UInt16(1),
        GGUFValue::Int16(-1),
        GGUFValue::UInt32(1),
        GGUFValue::Int32(-1),
        GGUFValue::Float32(1.0),
        GGUFValue::Bool(true),
        GGUFValue::String("test".to_string()),
        GGUFValue::UInt64(1),
        GGUFValue::Int64(-1),
        GGUFValue::Float64(1.0),
        GGUFValue::Array(vec![GGUFValue::UInt32(1)]),
    ];

    for val in values {
        let debug_str = format!("{val:?}");
        assert!(!debug_str.is_empty());
    }
}

// ============================================================================
// SMALL BUFFER TYPE TESTS
// ============================================================================

use realizar::gguf::{
    AttentionBuffer, HiddenBuffer, TokenBuffer, ATTENTION_BUFFER_INLINE_CAP, BUFFER_HW_SIZE,
    BUFFER_LW_SIZE, BUFFER_MAX_SIZE, HIDDEN_BUFFER_INLINE_CAP, TOKEN_BUFFER_INLINE_CAP,
};

#[test]
fn test_cov_token_buffer_inline_cap() {
    assert_eq!(TOKEN_BUFFER_INLINE_CAP, 32);

    let mut buffer: TokenBuffer = smallvec::SmallVec::new();
    for i in 0..TOKEN_BUFFER_INLINE_CAP {
        buffer.push(i as u32);
    }

    assert_eq!(buffer.len(), TOKEN_BUFFER_INLINE_CAP);
}

#[test]
fn test_cov_attention_buffer_inline_cap() {
    assert_eq!(ATTENTION_BUFFER_INLINE_CAP, 64);

    let mut buffer: AttentionBuffer = smallvec::SmallVec::new();
    for i in 0..ATTENTION_BUFFER_INLINE_CAP {
        buffer.push(i as f32);
    }

    assert_eq!(buffer.len(), ATTENTION_BUFFER_INLINE_CAP);
}

#[test]
fn test_cov_hidden_buffer_inline_cap() {
    assert_eq!(HIDDEN_BUFFER_INLINE_CAP, 128);

    let mut buffer: HiddenBuffer = smallvec::SmallVec::new();
    for i in 0..HIDDEN_BUFFER_INLINE_CAP {
        buffer.push(i as f32);
    }

    assert_eq!(buffer.len(), HIDDEN_BUFFER_INLINE_CAP);
}

#[test]
fn test_cov_buffer_watermarks() {
    assert_eq!(BUFFER_LW_SIZE, 1024);
    assert_eq!(BUFFER_HW_SIZE, 8 * 1024);
    assert_eq!(BUFFER_MAX_SIZE, 32 * 1024);

    // Constants are validated by the eq assertions above
}
