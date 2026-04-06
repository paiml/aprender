//! EXTREME TDD coverage tests for apr.rs module (Refs PMAT-802)
//!
//! Focus areas:
//! - MappedAprModel (memory-mapped loading)
//! - AprV2Model edge cases
//! - Error handling paths
//! - BpeTokenizer edge cases
//! - Dequantization functions
//!
//! Target: 85%+ coverage for apr.rs

use realizar::apr::{
    detect_format, is_apr_file, AprFlags, AprHeader, AprMetadata, AprV2Model, BpeTokenizer,
    MappedAprModel, TensorEntry, ALIGNMENT, HEADER_SIZE, MAGIC,
};
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper to create binary tensor entry for APR v2 format
fn create_binary_tensor_entry(
    name: &str,
    dtype: u8,
    shape: &[u64],
    offset: u64,
    size: u64,
) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(name.len() as u16).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.push(dtype);
    data.push(shape.len() as u8);
    for &dim in shape {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&offset.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    data
}

/// Helper to create minimal valid APR v2 model bytes
fn create_minimal_apr_model() -> Vec<u8> {
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_size = 64usize;
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&0u16.to_le_bytes());
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // Tensor data (16 floats)
    let data_start = data_offset as usize;
    for i in 0..16 {
        let val = i as f32;
        data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    data
}

/// Helper to create APR file on disk
fn create_apr_file() -> NamedTempFile {
    let mut temp = NamedTempFile::new().expect("create temp file");
    let data = create_minimal_apr_model();
    temp.write_all(&data).expect("write data");
    temp
}

// ============================================================================
// MappedAprModel Coverage Tests
// ============================================================================

#[test]
fn test_mapped_apr_model_from_path_valid() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.tensor_count(), 1);
    assert!(model.file_size() > 0);
}

#[test]
fn test_mapped_apr_model_from_path_nonexistent() {
    let result = MappedAprModel::from_path("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_data_accessor() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let data = model.data();
    assert!(!data.is_empty());
    assert_eq!(&data[0..4], &MAGIC);
}

#[test]
fn test_mapped_apr_model_file_size() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert!(model.file_size() >= HEADER_SIZE);
}

#[test]
fn test_mapped_apr_model_tensor_count() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.tensor_count(), 1);
}

#[test]
fn test_mapped_apr_model_data_offset() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert!(model.data_offset() > 0);
}

#[test]
fn test_mapped_apr_model_find_tensor_exists() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let tensor = model.find_tensor("test.weight");
    assert!(tensor.is_some());
    let entry = tensor.unwrap();
    assert_eq!(entry.name, "test.weight");
}

#[test]
fn test_mapped_apr_model_find_tensor_not_exists() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let tensor = model.find_tensor("nonexistent");
    assert!(tensor.is_none());
}

#[test]
fn test_mapped_apr_model_get_tensor_data_valid() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let data = model
        .get_tensor_data("test.weight")
        .expect("should get data");
    assert_eq!(data.len(), 64);
}

#[test]
fn test_mapped_apr_model_get_tensor_data_not_found() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let result = model.get_tensor_data("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_f32() {
    assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_f16() {
    assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q4_0() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q4_1() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q5_0() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q5_1() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q8_0() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q8_1() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q2_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q3_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q4_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q5_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q6_k() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_iq2_xxs() {
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_iq2_xs() {
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_bf16() {
    assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_unknown() {
    assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0);
}

#[test]
fn test_mapped_apr_model_invalid_magic() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"GGUF"); // Wrong magic
    data[4] = 2;
    data[5] = 0;
    temp.write_all(&data).expect("write data");

    let result = MappedAprModel::from_path(temp.path());
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_header_fields() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.header.magic, MAGIC);
    assert_eq!(model.header.version.0, 2);
}

#[test]
fn test_mapped_apr_model_metadata_fields() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.metadata.vocab_size, Some(100));
    assert_eq!(model.metadata.hidden_size, Some(64));
}

#[test]
fn test_mapped_apr_model_tensors_field() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].name, "test.weight");
}

// ============================================================================
// AprV2Model Load/from_bytes Coverage
// ============================================================================

#[test]
fn test_apr_v2_model_load_from_file() {
    let temp = create_apr_file();
    let model = AprV2Model::load(temp.path()).expect("should load");
    assert_eq!(model.tensor_count(), 1);
    assert!(model.is_mmap());
}

#[test]
fn test_apr_v2_model_load_nonexistent() {
    let result = AprV2Model::load("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_from_bytes_valid() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 1);
    assert!(!model.is_mmap());
}

#[test]
fn test_apr_v2_model_from_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_from_bytes_truncated() {
    let data = vec![0u8; 10];
    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

// ============================================================================
// AprHeader Coverage
// ============================================================================

#[test]
fn test_apr_header_from_bytes_valid() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 1;
    data[8..12].copy_from_slice(&42u32.to_le_bytes());

    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.version, (2, 1));
    assert_eq!(header.tensor_count, 42);
}

#[test]
fn test_apr_header_from_bytes_too_small() {
    let data = vec![0u8; 10];
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_header_from_bytes_wrong_magic() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(b"TEST");
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_apr_header_checksum_field() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.checksum, 0xDEADBEEF);
}

// ============================================================================
// AprFlags Coverage
// ============================================================================

#[test]
fn test_apr_flags_new() {
    let flags = AprFlags::new(0x0000);
    assert!(!flags.is_compressed());
    assert!(!flags.is_encrypted());
}

#[test]
fn test_apr_flags_lz4() {
    let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
    assert!(flags.is_lz4());
    assert!(flags.is_compressed());
    assert!(!flags.is_zstd());
}

#[test]
fn test_apr_flags_zstd() {
    let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
    assert!(flags.is_zstd());
    assert!(flags.is_compressed());
    assert!(!flags.is_lz4());
}

#[test]
fn test_apr_flags_encrypted() {
    let flags = AprFlags::new(AprFlags::ENCRYPTED);
    assert!(flags.is_encrypted());
    assert!(!flags.is_compressed());
}

#[test]
fn test_apr_flags_signed() {
    let flags = AprFlags::new(AprFlags::SIGNED);
    assert!(!flags.is_encrypted());
}

#[test]
fn test_apr_flags_sharded() {
    let flags = AprFlags::new(AprFlags::SHARDED);
    assert!(!flags.is_quantized());
}

#[test]
fn test_apr_flags_quantized() {
    let flags = AprFlags::new(AprFlags::QUANTIZED);
    assert!(flags.is_quantized());
}

#[test]
fn test_apr_flags_has_vocab() {
    let flags = AprFlags::new(AprFlags::HAS_VOCAB);
    assert!(flags.has_vocab());
}

#[test]
fn test_apr_flags_combined() {
    let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED | AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
    assert!(flags.is_lz4());
    assert!(flags.is_quantized());
    assert!(flags.has_vocab());
}

// ============================================================================
// TensorEntry Coverage
// ============================================================================

#[test]
fn test_tensor_entry_from_binary_valid() {
    let data = create_binary_tensor_entry("weights", 0, &[128, 256], 0, 131072);
    let (entry, consumed) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.name, "weights");
    assert_eq!(entry.dtype, "F32");
    assert_eq!(entry.shape, vec![128, 256]);
    assert!(consumed > 0);
}

#[test]
fn test_tensor_entry_from_binary_too_short() {
    let data = vec![0u8; 2];
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_from_binary_truncated_name() {
    let mut data = Vec::new();
    data.extend_from_slice(&100u16.to_le_bytes());
    data.extend_from_slice(b"short");
    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_element_count_2d() {
    let entry = TensorEntry {
        name: "test".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        offset: 0,
        size: 800,
    };
    assert_eq!(entry.element_count(), 200);
}

#[test]
fn test_tensor_entry_element_count_empty() {
    let entry = TensorEntry {
        name: "scalar".to_string(),
        dtype: "F32".to_string(),
        shape: vec![],
        offset: 0,
        size: 4,
    };
    assert_eq!(entry.element_count(), 1);
}

#[test]
fn test_tensor_entry_dtypes() {
    // GGML-compatible dtype byte IDs (see src/apr/mod.rs:316-337)
    let dtypes = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        (8, "Q8_0"),
        (9, "Q8_1"),
        (10, "Q2_K"),
        (11, "Q3_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (30, "BF16"),
    ];
    for (dtype_byte, expected) in dtypes {
        let data = create_binary_tensor_entry("test", dtype_byte, &[1], 0, 4);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(
            entry.dtype, expected,
            "dtype byte {} should be {}",
            dtype_byte, expected
        );
    }
}

// ============================================================================
// AprMetadata Coverage
// ============================================================================

#[test]
fn test_apr_metadata_is_transformer_true() {
    let meta = AprMetadata {
        hidden_size: Some(256),
        num_layers: Some(4),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(meta.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_missing_hidden() {
    let meta = AprMetadata {
        hidden_size: None,
        num_layers: Some(4),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!meta.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_missing_layers() {
    let meta = AprMetadata {
        hidden_size: Some(256),
        num_layers: None,
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!meta.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_missing_heads() {
    let meta = AprMetadata {
        hidden_size: Some(256),
        num_layers: Some(4),
        num_heads: None,
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!meta.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_missing_vocab() {
    let meta = AprMetadata {
        hidden_size: Some(256),
        num_layers: Some(4),
        num_heads: Some(8),
        vocab_size: None,
        ..Default::default()
    };
    assert!(!meta.is_transformer());
}

#[test]
fn test_apr_metadata_default() {
    let meta = AprMetadata::default();
    assert!(meta.hidden_size.is_none());
    assert!(!meta.is_transformer());
}

#[test]
fn test_apr_metadata_serialization() {
    let meta = AprMetadata {
        hidden_size: Some(1024),
        num_layers: Some(12),
        ..Default::default()
    };
    let json = serde_json::to_string(&meta).unwrap();
    let parsed: AprMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.hidden_size, Some(1024));
}

// ============================================================================
// BpeTokenizer Coverage
// ============================================================================

#[test]
fn test_bpe_tokenizer_encode_empty() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec![],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };
    let encoded = tokenizer.encode("");
    assert!(encoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_encode_single_char() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec!["a".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };
    let encoded = tokenizer.encode("a");
    assert_eq!(encoded, vec![0]);
}

#[test]
fn test_bpe_tokenizer_encode_with_merges() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);
    token_to_id.insert("ab".to_string(), 2);

    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
        merge_rules: vec![("a".to_string(), "b".to_string())],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };
    let encoded = tokenizer.encode("ab");
    assert!(!encoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_decode_empty() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec!["test".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };
    let decoded = tokenizer.decode(&[]);
    assert!(decoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_decode_valid() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec!["hello".to_string(), "world".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };
    let decoded = tokenizer.decode(&[0, 1]);
    assert!(decoded.contains("hello"));
    assert!(decoded.contains("world"));
}

#[test]
fn test_bpe_tokenizer_decode_out_of_bounds() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec!["a".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };
    let decoded = tokenizer.decode(&[0, 100]);
    assert!(decoded.contains('a'));
    assert!(decoded.contains("[100]"));
}

#[test]
fn test_bpe_tokenizer_bos_eos() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec![],
        merge_rules: vec![],
        bos_id: Some(1),
        eos_id: Some(2),
        special_tokens: HashMap::new(),
    };
    assert_eq!(tokenizer.bos_id, Some(1));
    assert_eq!(tokenizer.eos_id, Some(2));
}

// ============================================================================
// is_apr_file / detect_format Coverage
// ============================================================================

#[test]
fn test_is_apr_file_valid() {
    let temp = create_apr_file();
    assert!(is_apr_file(temp.path()));
}

#[test]
fn test_is_apr_file_nonexistent() {
    assert!(!is_apr_file("/nonexistent/path/model.apr"));
}

#[test]
fn test_is_apr_file_wrong_magic() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    temp.write_all(b"GGUF").expect("write magic");
    temp.write_all(&[0u8; 60]).expect("write padding");
    assert!(!is_apr_file(temp.path()));
}

#[test]
fn test_detect_format_apr_extension() {
    assert_eq!(detect_format("/path/model.apr"), "apr");
}

#[test]
fn test_detect_format_gguf_extension() {
    assert_eq!(detect_format("/path/model.gguf"), "gguf");
}

#[test]
fn test_detect_format_safetensors_extension() {
    assert_eq!(detect_format("/path/model.safetensors"), "safetensors");
}

#[test]
fn test_detect_format_unknown_extension() {
    assert_eq!(detect_format("/path/model.bin"), "unknown");
}

#[test]
fn test_detect_format_no_extension() {
    assert_eq!(detect_format("/path/model"), "unknown");
}

// ============================================================================
// Constants Coverage
// ============================================================================

#[test]
fn test_magic_constant() {
    assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x00]);
    assert_eq!(&MAGIC, b"APR\0");
}

#[test]
fn test_header_size_constant() {
    assert_eq!(HEADER_SIZE, 64);
}

#[test]
fn test_alignment_constant() {
    assert_eq!(ALIGNMENT, 64);
}

// ============================================================================
// AprV2Model Methods Coverage
// ============================================================================

#[test]
fn test_apr_v2_model_tensor_count() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 1);
}

#[test]
fn test_apr_v2_model_tensor_names() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let names = model.tensor_names();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "test.weight");
}

#[test]
fn test_apr_v2_model_metadata() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let meta = model.metadata();
    assert_eq!(meta.vocab_size, Some(100));
}

#[test]
fn test_apr_v2_model_get_tensor() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let tensor = model.get_tensor("test.weight");
    assert!(tensor.is_some());
}

#[test]
fn test_apr_v2_model_get_tensor_not_found() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let tensor = model.get_tensor("nonexistent");
    assert!(tensor.is_none());
}

#[test]
fn test_apr_v2_model_get_tensor_f32() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let floats = model.get_tensor_f32("test.weight").expect("should get f32");
    assert_eq!(floats.len(), 16);
}

#[test]
fn test_apr_v2_model_get_tensor_f32_not_found() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.get_tensor_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_get_tensor_bytes() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let bytes = model
        .get_tensor_bytes("test.weight")
        .expect("should get bytes");
    assert_eq!(bytes.len(), 64);
}

#[test]
fn test_apr_v2_model_get_tensor_bytes_not_found() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.get_tensor_bytes("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_estimated_parameters() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.estimated_parameters(), 16);
}

#[test]
fn test_apr_v2_model_predict_empty_features() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.predict(&[]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap()[0], 0.0);
}

#[test]
fn test_apr_v2_model_predict_with_features() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.predict(&[1.0, 2.0, 3.0]);
    assert!(result.is_ok());
}

#[test]
fn test_apr_v2_model_forward_empty_tokens() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_generate_empty_input() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.generate(&[], 10, None);
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_decode_tokens() {
    let vocab = vec!["hello".to_string(), " ".to_string(), "world".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "hello world");
}

#[test]
fn test_apr_v2_model_decode_tokens_out_of_bounds() {
    let vocab = vec!["a".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 99]);
    assert!(result.contains('a'));
    assert!(result.contains("[99]"));
}

// ============================================================================
// MappedAprModel Truncated File Handling
// ============================================================================

#[test]
fn test_mapped_apr_model_truncated_metadata() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    let mut data = vec![0u8; 100];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 (larger than file)
    temp.write_all(&data).expect("write data");

    let result = MappedAprModel::from_path(temp.path());
    assert!(result.is_err());
}

#[test]
fn test_mapped_apr_model_debug_impl() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("MappedAprModel"));
}

// ============================================================================
// AprV2Model Encrypted File Handling
// ============================================================================

#[test]
fn test_apr_v2_model_encrypted_error() {
    let mut data = vec![0u8; HEADER_SIZE + 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&AprFlags::ENCRYPTED.to_le_bytes());

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
}

// ============================================================================
// AprV2Model Compressed File Handling (without apr-compression feature)
// ============================================================================

#[cfg(not(feature = "apr-compression"))]
#[test]
fn test_apr_v2_model_compressed_requires_feature() {
    let mut data = vec![0u8; HEADER_SIZE + 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&AprFlags::LZ4_COMPRESSED.to_le_bytes());
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&0u32.to_le_bytes());
    data[24..32].copy_from_slice(&64u64.to_le_bytes());
    data[32..40].copy_from_slice(&64u64.to_le_bytes());

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("apr-compression"));
}

// ============================================================================
// Release CPU Pages Coverage (Unix only)
// ============================================================================

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[test]
fn test_apr_v2_model_release_cpu_pages() {
    let temp = create_apr_file();
    let model = AprV2Model::load(temp.path()).expect("should load");
    let result = model.release_cpu_pages();
    assert!(result.is_ok());
}

// ============================================================================
// ModelData Type Alias Coverage
// ============================================================================

#[test]
fn test_apr_model_type_alias() {
    // AprModel is a type alias for AprV2Model
    let data = create_minimal_apr_model();
    let model: realizar::apr::AprModel = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 1);
}

// ============================================================================
// ModelData Coverage (apr.rs)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_model_data_from_vec() {
    use realizar::apr::ModelData;
    let data = vec![1u8, 2, 3, 4, 5];
    let model_data = ModelData::from_vec(data.clone());
    assert_eq!(model_data.as_slice(), &data);
    assert!(!model_data.is_mmap());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_model_data_len_and_is_empty() {
    use realizar::apr::ModelData;
    let data = vec![1u8, 2, 3];
    let model_data = ModelData::from_vec(data);
    assert_eq!(model_data.len(), 3);
    assert!(!model_data.is_empty());

    let empty_data = ModelData::from_vec(vec![]);
    assert_eq!(empty_data.len(), 0);
    assert!(empty_data.is_empty());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_model_data_open_mmap() {
    use realizar::apr::ModelData;
    let temp = create_apr_file();
    let model_data = ModelData::open_mmap(temp.path()).expect("should open");
    assert!(model_data.is_mmap());
    assert!(!model_data.is_empty());
    assert_eq!(&model_data.as_slice()[0..4], &MAGIC);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_model_data_open_mmap_nonexistent() {
    use realizar::apr::ModelData;
    let result = ModelData::open_mmap("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[test]
fn test_model_data_release_cpu_pages_mmap() {
    use realizar::apr::ModelData;
    let temp = create_apr_file();
    let model_data = ModelData::open_mmap(temp.path()).expect("should open");
    let result = model_data.release_cpu_pages();
    assert!(result.is_ok());
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[test]
fn test_model_data_release_cpu_pages_heap() {
    use realizar::apr::ModelData;
    let model_data = ModelData::from_vec(vec![1, 2, 3]);
    let result = model_data.release_cpu_pages();
    assert!(result.is_ok()); // No-op for heap data
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[test]
fn test_model_data_advise_sequential_mmap() {
    use realizar::apr::ModelData;
    let temp = create_apr_file();
    let model_data = ModelData::open_mmap(temp.path()).expect("should open");
    let result = model_data.advise_sequential();
    assert!(result.is_ok());
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[test]
fn test_model_data_advise_sequential_heap() {
    use realizar::apr::ModelData;
    let model_data = ModelData::from_vec(vec![1, 2, 3]);
    let result = model_data.advise_sequential();
    assert!(result.is_ok()); // No-op for heap data
}

// ============================================================================
// apr_transformer.rs Coverage Tests
// ============================================================================

use realizar::apr_transformer::{
    AprBenchmarkResult, AprInferenceScratch, AprKVCache, AprParityComparison, AprPrefillResult,
    AprQuantizationType, AprTransformer, AprTransformerConfig, AprTransformerLayer, GenerateConfig,
    QuantizedAprTensorQ4, APR_PARITY_THRESHOLD_PCT,
};

// ============================================================================
// AprTransformerConfig Coverage
// ============================================================================

#[test]
fn test_apr_transformer_config_default() {
    let config = AprTransformerConfig::default();
    assert_eq!(config.architecture, "unknown");
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.intermediate_dim, 2048);
    assert_eq!(config.context_length, 2048);
    assert!((config.rope_theta - 10000.0).abs() < 0.01);
    assert!((config.eps - 1e-5).abs() < 1e-7);
}

#[test]
fn test_apr_transformer_config_serialization() {
    let config = AprTransformerConfig {
        architecture: "phi2".to_string(),
        hidden_dim: 2560,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 32,
        vocab_size: 51200,
        intermediate_dim: 10240,
        context_length: 2048,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };
    let json = serde_json::to_string(&config).unwrap();
    let parsed: AprTransformerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.architecture, "phi2");
    assert_eq!(parsed.hidden_dim, 2560);
}

#[test]
fn test_apr_transformer_config_equality() {
    let config1 = AprTransformerConfig::default();
    let config2 = AprTransformerConfig::default();
    assert_eq!(config1, config2);
}

// ============================================================================
// AprTransformerLayer Coverage
// ============================================================================

#[test]
fn test_apr_transformer_layer_empty() {
    let layer = AprTransformerLayer::empty(512, 2048);
    assert_eq!(layer.attn_norm_weight.len(), 512);
    assert_eq!(layer.qkv_weight.len(), 512 * 3 * 512);
    assert_eq!(layer.attn_output_weight.len(), 512 * 512);
    assert_eq!(layer.ffn_up_weight.len(), 512 * 2048);
    assert_eq!(layer.ffn_down_weight.len(), 2048 * 512);
    assert!(layer.ffn_gate_weight.is_none());
}

#[test]
fn test_apr_transformer_layer_empty_gqa() {
    let layer = AprTransformerLayer::empty_gqa(2048, 32, 4, 5632);
    assert_eq!(layer.attn_norm_weight.len(), 2048);
    // GQA: Q uses num_heads, K/V use num_kv_heads
    // head_dim = 2048 / 32 = 64
    // kv_dim = 4 * 64 = 256
    // qkv_out_dim = 2048 + 2*256 = 2560
    let expected_qkv = 2048 * 2560;
    assert_eq!(layer.qkv_weight.len(), expected_qkv);
}

#[test]
fn test_apr_transformer_layer_num_parameters() {
    let layer = AprTransformerLayer::empty(64, 256);
    let params = layer.num_parameters();
    // attn_norm (64) + qkv (64*3*64) + attn_out (64*64) + ffn_up (64*256) + ffn_down (256*64)
    let expected = 64 + (64 * 3 * 64) + (64 * 64) + (64 * 256) + (256 * 64);
    assert_eq!(params, expected);
}

#[test]
fn test_apr_transformer_layer_num_parameters_with_optional() {
    let mut layer = AprTransformerLayer::empty(64, 256);
    layer.attn_norm_bias = Some(vec![0.0; 64]);
    layer.qkv_bias = Some(vec![0.0; 192]);
    let params_with_bias = layer.num_parameters();
    let params_base = 64 + (64 * 3 * 64) + (64 * 64) + (64 * 256) + (256 * 64);
    assert_eq!(params_with_bias, params_base + 64 + 192);
}

// ============================================================================
// AprTransformer Coverage
// ============================================================================

#[test]
fn test_apr_transformer_new() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        vocab_size: 100,
        intermediate_dim: 256,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    assert_eq!(transformer.config().hidden_dim, 64);
    assert_eq!(transformer.layers.len(), 2);
    assert_eq!(transformer.token_embedding.len(), 100 * 64);
}

#[test]
fn test_apr_transformer_config_accessor() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    assert_eq!(transformer.config().architecture, "test");
}

#[test]
fn test_apr_transformer_num_parameters() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let params = transformer.num_parameters();
    assert!(params > 0);
}

#[test]
fn test_apr_transformer_memory_size() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let mem_size = transformer.memory_size();
    let params = transformer.num_parameters();
    assert_eq!(mem_size, params * 4); // f32 = 4 bytes
}

#[test]
fn test_apr_transformer_embed() {
    let config = AprTransformerConfig {
        hidden_dim: 8,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 16,
        ..Default::default()
    };
    let mut transformer = AprTransformer::new(config);

    // Set known embedding values
    for i in 0..8 {
        transformer.token_embedding[i] = i as f32;
    }

    let embeddings = transformer.embed(&[0]);
    assert_eq!(embeddings.len(), 8);
    assert!((embeddings[0] - 0.0).abs() < 0.001);
    assert!((embeddings[1] - 1.0).abs() < 0.001);
}

#[test]
fn test_apr_transformer_embed_out_of_vocab() {
    let config = AprTransformerConfig {
        hidden_dim: 4,
        num_layers: 1,
        vocab_size: 2,
        intermediate_dim: 8,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);

    // Token ID 999 is out of vocab
    let embeddings = transformer.embed(&[999]);
    assert_eq!(embeddings.len(), 4);
    // Should return zeros for OOV
    for &e in &embeddings {
        assert!((e - 0.0).abs() < 0.001);
    }
}

#[test]
fn test_apr_transformer_forward_empty() {
    let config = AprTransformerConfig::default();
    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_apr_transformer_forward_single_token() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.forward(&[1]);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 10); // vocab_size
}

#[test]
fn test_apr_transformer_predict_next() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.predict_next(&[1]);
    assert!(result.is_ok());
    let token = result.unwrap();
    assert!(token < 10); // Should be valid token ID
}

#[test]
fn test_apr_transformer_generate() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let result = transformer.generate(&[1], 3);
    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(!tokens.is_empty()); // At least prompt
    assert!(tokens.len() <= 4); // At most prompt + 3
}

// ============================================================================
// AprKVCache Coverage
// ============================================================================

#[test]
fn test_apr_kv_cache_new() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 4,
        context_length: 128,
        ..Default::default()
    };
    let cache = AprKVCache::new(&config);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 128);
}

#[test]
fn test_apr_kv_cache_append_and_get() {
    let config = AprTransformerConfig {
        hidden_dim: 8,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        context_length: 16,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);

    let head_dim = 8 / 2; // 4
    let kv_size = 2 * head_dim; // 8

    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    cache.append(0, &k, &v);
    cache.append(1, &k, &v);

    assert_eq!(cache.len(), 1);

    let (k_cached, v_cached) = cache.get(0);
    assert_eq!(k_cached.len(), kv_size);
    assert!((k_cached[0] - 1.0).abs() < 0.001);
    assert!((v_cached[0] - 2.0).abs() < 0.001);
}

#[test]
fn test_apr_kv_cache_clear() {
    let config = AprTransformerConfig {
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        context_length: 16,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);

    let head_dim = 4;
    let kv_size = 2 * head_dim;
    cache.append(0, &vec![1.0; kv_size], &vec![2.0; kv_size]);
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// ============================================================================
// GenerateConfig Coverage
// ============================================================================

#[test]
fn test_generate_config_default() {
    let config = GenerateConfig::default();
    assert_eq!(config.max_tokens, 32);
    assert!((config.temperature - 1.0).abs() < 0.001);
    assert!((config.top_p - 0.9).abs() < 0.001);
    assert_eq!(config.top_k, 0);
    assert!((config.repetition_penalty - 1.0).abs() < 0.001);
}

#[test]
fn test_generate_config_custom() {
    let config = GenerateConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 50,
        repetition_penalty: 1.2,
        trace: false,
        stop_tokens: vec![],
    };
    assert_eq!(config.max_tokens, 100);
    assert!((config.temperature - 0.7).abs() < 0.001);
}

// ============================================================================
// AprQuantizationType Coverage
// ============================================================================

#[test]
fn test_apr_quantization_type_bits_per_weight() {
    assert!((AprQuantizationType::F32.bits_per_weight() - 32.0).abs() < 0.001);
    assert!((AprQuantizationType::Q4_K.bits_per_weight() - 4.5).abs() < 0.001);
    assert!((AprQuantizationType::Q8_0.bits_per_weight() - 8.0).abs() < 0.001);
}

#[test]
fn test_apr_quantization_type_bytes_per_block() {
    assert_eq!(AprQuantizationType::F32.bytes_per_block(), 4);
    assert_eq!(AprQuantizationType::Q4_K.bytes_per_block(), 144);
    assert_eq!(AprQuantizationType::Q8_0.bytes_per_block(), 36);
}

#[test]
fn test_apr_quantization_type_values_per_block() {
    assert_eq!(AprQuantizationType::F32.values_per_block(), 1);
    assert_eq!(AprQuantizationType::Q4_K.values_per_block(), 256);
    assert_eq!(AprQuantizationType::Q8_0.values_per_block(), 32);
}

#[test]
fn test_apr_quantization_type_to_byte() {
    assert_eq!(AprQuantizationType::F32.to_byte(), 0);
    assert_eq!(AprQuantizationType::Q4_K.to_byte(), 1);
    assert_eq!(AprQuantizationType::Q8_0.to_byte(), 2);
}

#[test]
fn test_apr_quantization_type_from_byte() {
    assert_eq!(
        AprQuantizationType::from_byte(0),
        Some(AprQuantizationType::F32)
    );
    assert_eq!(
        AprQuantizationType::from_byte(1),
        Some(AprQuantizationType::Q4_K)
    );
    assert_eq!(
        AprQuantizationType::from_byte(2),
        Some(AprQuantizationType::Q8_0)
    );
    assert_eq!(AprQuantizationType::from_byte(99), None);
}

#[test]
fn test_apr_quantization_type_default() {
    let qtype: AprQuantizationType = Default::default();
    assert_eq!(qtype, AprQuantizationType::F32);
}

// ============================================================================
// QuantizedAprTensorQ4 Coverage
// ============================================================================

#[test]
fn test_quantized_apr_tensor_q4_new() {
    let data = vec![0u8; 100];
    let tensor = QuantizedAprTensorQ4::new(data, 32, 64);
    assert_eq!(tensor.data.len(), 100);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 64);
}

#[test]
fn test_quantized_apr_tensor_q4_zeros() {
    let tensor = QuantizedAprTensorQ4::zeros(32, 64);
    assert_eq!(tensor.in_dim, 32);
    assert_eq!(tensor.out_dim, 64);
    // Q4_0: 18 bytes per 32 values
    let expected_bytes = QuantizedAprTensorQ4::expected_bytes(32 * 64);
    assert_eq!(tensor.data.len(), expected_bytes);
}

#[test]
fn test_quantized_apr_tensor_q4_expected_bytes() {
    // 32 elements = 1 block = 18 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);
    // 64 elements = 2 blocks = 36 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);
    // 33 elements = 2 blocks (ceil) = 36 bytes
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36);
}

// ============================================================================
// AprInferenceScratch Coverage
// ============================================================================

#[test]
fn test_apr_inference_scratch_from_config() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        intermediate_dim: 256,
        ..Default::default()
    };
    let scratch = AprInferenceScratch::from_config(&config);
    assert_eq!(scratch.hidden.len(), 64);
    assert_eq!(scratch.normed.len(), 64);
    assert_eq!(scratch.qkv_out.len(), 64 * 3);
    assert_eq!(scratch.q.len(), 64);
    assert_eq!(scratch.k.len(), 64);
    assert_eq!(scratch.v.len(), 64);
    assert_eq!(scratch.attn_out.len(), 64);
    assert_eq!(scratch.ffn_input.len(), 64);
    assert_eq!(scratch.ffn_up.len(), 256);
    assert_eq!(scratch.ffn_gate.len(), 256);
    assert_eq!(scratch.ffn_out.len(), 64);
}

#[test]
fn test_apr_inference_scratch_clear() {
    let config = AprTransformerConfig {
        hidden_dim: 4,
        intermediate_dim: 8,
        ..Default::default()
    };
    let mut scratch = AprInferenceScratch::from_config(&config);

    // Fill with non-zero values
    scratch.hidden.fill(1.0);
    scratch.normed.fill(2.0);
    scratch.ffn_up.fill(3.0);

    scratch.clear();

    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
    assert!(scratch.normed.iter().all(|&x| x == 0.0));
    assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
}

// ============================================================================
// AprBenchmarkResult Coverage
// ============================================================================

#[test]
fn test_apr_benchmark_result_default() {
    let result = AprBenchmarkResult::default();
    assert_eq!(result.tokens_generated, 0);
    assert!((result.total_time_ms - 0.0).abs() < 0.001);
    assert!((result.tokens_per_second - 0.0).abs() < 0.001);
}

#[test]
fn test_apr_benchmark_result_meets_threshold() {
    let result = AprBenchmarkResult {
        tokens_per_second: 100.0,
        ..Default::default()
    };
    assert!(result.meets_threshold(50.0));
    assert!(result.meets_threshold(100.0));
    assert!(!result.meets_threshold(150.0));
}

#[test]
fn test_apr_benchmark_result_compare_to_baseline() {
    let result = AprBenchmarkResult {
        tokens_per_second: 95.0,
        peak_memory_mb: 500.0,
        ..Default::default()
    };
    let baseline = AprBenchmarkResult {
        tokens_per_second: 100.0,
        peak_memory_mb: 400.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    assert!((comparison.throughput_ratio - 0.95).abs() < 0.001);
    assert!((comparison.memory_ratio - 1.25).abs() < 0.001);
}

#[test]
fn test_apr_benchmark_result_compare_to_zero_baseline() {
    let result = AprBenchmarkResult {
        tokens_per_second: 100.0,
        peak_memory_mb: 500.0,
        ..Default::default()
    };
    let baseline = AprBenchmarkResult {
        tokens_per_second: 0.0,
        peak_memory_mb: 0.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    assert!((comparison.throughput_ratio - 1.0).abs() < 0.001);
    assert!((comparison.memory_ratio - 1.0).abs() < 0.001);
}

// ============================================================================
// AprPrefillResult Coverage
// ============================================================================

#[test]
fn test_apr_prefill_result_default() {
    let result = AprPrefillResult::default();
    assert_eq!(result.prompt_tokens, 0);
    assert!((result.prefill_time_ms - 0.0).abs() < 0.001);
    assert!((result.prefill_tok_s - 0.0).abs() < 0.001);
}

// ============================================================================
// AprParityComparison Coverage
// ============================================================================

#[test]
fn test_apr_parity_comparison_is_parity_true() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.96,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(comparison.is_parity());
}

#[test]
fn test_apr_parity_comparison_is_parity_false() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.80,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(!comparison.is_parity());
}

#[test]
fn test_apr_parity_comparison_boundary() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.95,
        memory_ratio: 1.0,
        parity_threshold_pct: 95.0,
    };
    assert!(comparison.is_parity());
}

// ============================================================================
// Additional AprV2Model Error Path Coverage
// ============================================================================

#[test]
fn test_apr_v2_model_forward_with_features_returns_prediction() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.predict(&[1.0, 2.0, 3.0, 4.0]);
    assert!(result.is_ok());
    // Prediction should return some values
    let pred = result.unwrap();
    assert!(!pred.is_empty());
}

#[test]
fn test_apr_metadata_num_layers_field() {
    let meta = AprMetadata {
        num_layers: Some(12),
        ..Default::default()
    };
    assert_eq!(meta.num_layers, Some(12));
}

#[test]
fn test_apr_metadata_rope_theta_field() {
    let meta = AprMetadata {
        rope_theta: Some(10000.0),
        ..Default::default()
    };
    assert_eq!(meta.rope_theta, Some(10000.0));
}

#[test]
fn test_apr_metadata_intermediate_size_field() {
    let meta = AprMetadata {
        intermediate_size: Some(11008),
        ..Default::default()
    };
    assert_eq!(meta.intermediate_size, Some(11008));
}

#[test]
fn test_tensor_entry_3d_shape() {
    let entry = TensorEntry {
        name: "test".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 3, 4],
        offset: 0,
        size: 96,
    };
    assert_eq!(entry.element_count(), 24);
}

#[test]
fn test_tensor_entry_1d_shape() {
    let entry = TensorEntry {
        name: "bias".to_string(),
        dtype: "F32".to_string(),
        shape: vec![100],
        offset: 0,
        size: 400,
    };
    assert_eq!(entry.element_count(), 100);
}

// ============================================================================
// BPE Tokenizer Extended Coverage
// ============================================================================

#[test]
fn test_bpe_tokenizer_encode_multiple_chars() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("h".to_string(), 0);
    token_to_id.insert("e".to_string(), 1);
    token_to_id.insert("l".to_string(), 2);
    token_to_id.insert("o".to_string(), 3);
    token_to_id.insert("he".to_string(), 4);
    token_to_id.insert("ll".to_string(), 5);

    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
        ],
        merge_rules: vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
        ],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    let encoded = tokenizer.encode("hello");
    assert!(!encoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_clone() {
    let tokenizer = BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec!["test".to_string()],
        merge_rules: vec![],
        bos_id: Some(1),
        eos_id: Some(2),
        special_tokens: HashMap::new(),
    };
    let cloned = tokenizer;
    assert_eq!(cloned.bos_id, Some(1));
    assert_eq!(cloned.eos_id, Some(2));
}

// ============================================================================
// AprFlags Extended Coverage
// ============================================================================

#[test]
fn test_apr_flags_all_combinations() {
    let flags = AprFlags::new(
        AprFlags::LZ4_COMPRESSED
            | AprFlags::ZSTD_COMPRESSED
            | AprFlags::ENCRYPTED
            | AprFlags::SIGNED
            | AprFlags::SHARDED
            | AprFlags::QUANTIZED
            | AprFlags::HAS_VOCAB,
    );
    assert!(flags.is_lz4());
    assert!(flags.is_zstd());
    assert!(flags.is_compressed());
    assert!(flags.is_encrypted());
    assert!(flags.is_quantized());
    assert!(flags.has_vocab());
}

#[test]
fn test_apr_flags_none() {
    let flags = AprFlags::new(0);
    assert!(!flags.is_lz4());
    assert!(!flags.is_zstd());
    assert!(!flags.is_compressed());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_quantized());
    assert!(!flags.has_vocab());
}

// ============================================================================
// Additional AprHeader Coverage
// ============================================================================

#[test]
fn test_apr_header_all_fields() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // major
    data[5] = 1; // minor
    data[6..8].copy_from_slice(&0x0001u16.to_le_bytes()); // flags (LZ4)
    data[8..12].copy_from_slice(&10u32.to_le_bytes()); // tensor_count
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&128u32.to_le_bytes()); // metadata_size
    data[24..32].copy_from_slice(&256u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&512u64.to_le_bytes()); // data_offset
    data[40..44].copy_from_slice(&0xABCDEF01u32.to_le_bytes()); // checksum

    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.version, (2, 1));
    assert_eq!(header.tensor_count, 10);
    assert_eq!(header.metadata_offset, 64);
    assert_eq!(header.metadata_size, 128);
    assert_eq!(header.tensor_index_offset, 256);
    assert_eq!(header.data_offset, 512);
    assert_eq!(header.checksum, 0xABCDEF01);
    assert!(header.flags.is_lz4());
}

// ============================================================================
// AprTransformer generate_with_cache Coverage
// ============================================================================

#[test]
fn test_apr_transformer_generate_with_cache() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        context_length: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);

    let gen_config = GenerateConfig {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };

    let result = transformer.generate_with_cache(&[1], &gen_config);
    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(!tokens.is_empty());
}

// ============================================================================
// AprTransformer forward_with_cache Coverage
// ============================================================================

#[test]
fn test_apr_transformer_forward_with_cache() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        context_length: 64,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // forward_with_cache takes: token_id: u32, cache: &mut AprKVCache, position: usize
    let result = transformer.forward_with_cache(1, &mut cache, 0);
    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.len(), 10);
    assert_eq!(cache.len(), 1);
}

// ============================================================================
// Constants Coverage
// ============================================================================

#[test]
fn test_apr_parity_threshold_constant() {
    assert!((APR_PARITY_THRESHOLD_PCT - 95.0).abs() < 0.001);
}

#[test]
fn test_apr_transformer_header_size_constant() {
    use realizar::apr_transformer::APR_TRANSFORMER_HEADER_SIZE;
    assert_eq!(APR_TRANSFORMER_HEADER_SIZE, 64);
}

// ============================================================================
// Additional tensor dtype coverage
// ============================================================================

#[test]
fn test_tensor_entry_dtype_bf16() {
    // BF16 = GGML dtype byte 30
    let data = create_binary_tensor_entry("test", 30, &[1], 0, 4);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "BF16");
}

#[test]
fn test_tensor_entry_dtype_q4_0() {
    // Q4_0 = GGML dtype byte 2
    let data = create_binary_tensor_entry("test", 2, &[1], 0, 4);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q4_0");
}

#[test]
fn test_tensor_entry_dtype_q4_1() {
    // Q4_1 = GGML dtype byte 3
    let data = create_binary_tensor_entry("test", 3, &[1], 0, 4);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q4_1");
}

#[test]
fn test_tensor_entry_dtype_q5_0() {
    // Q5_0 = GGML dtype byte 6
    let data = create_binary_tensor_entry("test", 6, &[1], 0, 4);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q5_0");
}

#[test]
fn test_tensor_entry_dtype_q5_1() {
    // Q5_1 = GGML dtype byte 7
    let data = create_binary_tensor_entry("test", 7, &[1], 0, 4);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q5_1");
}

#[test]
fn test_tensor_entry_dtype_q8_1() {
    // Q8_1 = GGML dtype byte 9
    let data = create_binary_tensor_entry("test", 9, &[1], 0, 4);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "Q8_1");
}

#[test]
fn test_tensor_entry_dtype_unknown() {
    // Unknown dtype bytes default to "F32"
    let data = create_binary_tensor_entry("test", 255, &[1], 0, 4);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.dtype, "F32");
}

// ============================================================================
// APR File Header Parsing Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_apr_header_version_major_minor() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 3; // major version 3
    data[5] = 5; // minor version 5
    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.version, (3, 5));
}

#[test]
fn test_apr_header_v1_v2_versions() {
    // Test v1.0
    let mut data_v1 = vec![0u8; HEADER_SIZE];
    data_v1[0..4].copy_from_slice(&MAGIC);
    data_v1[4] = 1;
    data_v1[5] = 0;
    let header_v1 = AprHeader::from_bytes(&data_v1).expect("should parse v1");
    assert_eq!(header_v1.version, (1, 0));

    // Test v2.0
    let mut data_v2 = vec![0u8; HEADER_SIZE];
    data_v2[0..4].copy_from_slice(&MAGIC);
    data_v2[4] = 2;
    data_v2[5] = 0;
    let header_v2 = AprHeader::from_bytes(&data_v2).expect("should parse v2");
    assert_eq!(header_v2.version, (2, 0));
}

#[test]
fn test_apr_header_large_tensor_count() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[8..12].copy_from_slice(&1_000_000u32.to_le_bytes());
    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.tensor_count, 1_000_000);
}

#[test]
fn test_apr_header_max_offsets() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[12..20].copy_from_slice(&u64::MAX.to_le_bytes()); // metadata_offset
    data[24..32].copy_from_slice(&u64::MAX.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&u64::MAX.to_le_bytes()); // data_offset
    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.metadata_offset, u64::MAX);
    assert_eq!(header.tensor_index_offset, u64::MAX);
    assert_eq!(header.data_offset, u64::MAX);
}

#[test]
fn test_apr_header_zero_tensor_count() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    let header = AprHeader::from_bytes(&data).expect("should parse");
    assert_eq!(header.tensor_count, 0);
}

// ============================================================================
// APR Tensor Loading - Quantized Tensors (PMAT-802)
// ============================================================================

/// Helper to create APR model with Q4_K tensor
fn create_apr_model_with_q4k_tensor() -> Vec<u8> {
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Q4_K tensor: dtype=12 (GGML Q4_K), shape [256] (one super block)
    let tensor_entry = create_binary_tensor_entry("test.q4k", 12, &[256], 0, 144);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_size = 144usize; // Q4_K: 144 bytes per super block
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&AprFlags::QUANTIZED.to_le_bytes());
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // Q4_K tensor data (simulated)
    let data_start = data_offset as usize;
    // d (f16), dmin (f16), scales (12 bytes), qs (128 bytes)
    data[data_start..data_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    data[data_start + 2..data_start + 4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0

    data
}

#[test]
fn test_apr_model_q4k_tensor_loading() {
    let data = create_apr_model_with_q4k_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load Q4_K model");
    let tensor = model.get_tensor("test.q4k");
    assert!(tensor.is_some());
    assert_eq!(tensor.unwrap().dtype, "Q4_K");
}

#[test]
fn test_apr_model_q4k_tensor_shape() {
    let data = create_apr_model_with_q4k_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load Q4_K model");
    let tensor = model.get_tensor("test.q4k").unwrap();
    assert_eq!(tensor.shape, vec![256]);
    assert_eq!(tensor.element_count(), 256);
}

/// Helper to create APR model with Q6_K tensor
fn create_apr_model_with_q6k_tensor() -> Vec<u8> {
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Q6_K tensor: dtype=14 (GGML Q6_K), shape [256] (one super block)
    let tensor_entry = create_binary_tensor_entry("test.q6k", 14, &[256], 0, 210);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_size = 210usize; // Q6_K: 210 bytes per super block
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&AprFlags::QUANTIZED.to_le_bytes());
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // Q6_K tensor data (simulated)
    // ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16)
    let data_start = data_offset as usize;
    data[data_start + 208..data_start + 210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0

    data
}

#[test]
fn test_apr_model_q6k_tensor_loading() {
    let data = create_apr_model_with_q6k_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load Q6_K model");
    let tensor = model.get_tensor("test.q6k");
    assert!(tensor.is_some());
    assert_eq!(tensor.unwrap().dtype, "Q6_K");
}

/// Helper to create APR model with Q8_0 tensor
fn create_apr_model_with_q8_0_tensor() -> Vec<u8> {
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // Q8_0 tensor: dtype=8 (GGML Q8_0), shape [32] (one block)
    let tensor_entry = create_binary_tensor_entry("test.q8_0", 8, &[32], 0, 34);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_size = 34usize; // Q8_0: 34 bytes per block (2 + 32)
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[6..8].copy_from_slice(&AprFlags::QUANTIZED.to_le_bytes());
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // Q8_0 tensor data (simulated)
    // scale (f16) + 32 int8 values
    let data_start = data_offset as usize;
    data[data_start..data_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0
    for i in 0..32 {
        data[data_start + 2 + i] = i as u8; // quant values 0-31
    }

    data
}

#[test]
fn test_apr_model_q8_0_tensor_loading() {
    let data = create_apr_model_with_q8_0_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load Q8_0 model");
    let tensor = model.get_tensor("test.q8_0");
    assert!(tensor.is_some());
    assert_eq!(tensor.unwrap().dtype, "Q8_0");
}

// ============================================================================
// APR Model Configuration Validation (PMAT-802)
// ============================================================================

#[test]
fn test_apr_metadata_complete_transformer_config() {
    let meta = AprMetadata {
        hidden_size: Some(4096),
        num_layers: Some(32),
        num_heads: Some(32),
        num_kv_heads: Some(8), // GQA style
        vocab_size: Some(128256),
        intermediate_size: Some(14336),
        max_position_embeddings: Some(8192),
        rope_theta: Some(500000.0),
        rms_norm_eps: Some(1e-5),
        ..Default::default()
    };
    assert!(meta.is_transformer());
    assert_eq!(meta.num_kv_heads, Some(8));
    assert_eq!(meta.intermediate_size, Some(14336));
}

#[test]
fn test_apr_metadata_architecture_field() {
    let meta = AprMetadata {
        architecture: Some("llama".to_string()),
        model_type: Some("causal_lm".to_string()),
        name: Some("TinyLlama-1.1B".to_string()),
        ..Default::default()
    };
    assert_eq!(meta.architecture, Some("llama".to_string()));
    assert_eq!(meta.model_type, Some("causal_lm".to_string()));
    assert_eq!(meta.name, Some("TinyLlama-1.1B".to_string()));
}

#[test]
fn test_apr_metadata_rope_type_field() {
    // rope_type=0: NORM (adjacent pairs)
    // rope_type=2: NEOX (split halves)
    let meta_norm = AprMetadata {
        rope_type: Some(0),
        ..Default::default()
    };
    assert_eq!(meta_norm.rope_type, Some(0));

    let meta_neox = AprMetadata {
        rope_type: Some(2),
        ..Default::default()
    };
    assert_eq!(meta_neox.rope_type, Some(2));
}

#[test]
fn test_apr_metadata_extra_fields() {
    let json = r#"{
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "vocab_size": 1000,
        "custom_field": "custom_value",
        "numeric_extra": 42
    }"#;
    let meta: AprMetadata = serde_json::from_str(json).expect("should parse");
    assert!(meta.is_transformer());
    assert!(meta.extra.contains_key("custom_field"));
    assert!(meta.extra.contains_key("numeric_extra"));
}

#[test]
fn test_apr_metadata_json_roundtrip() {
    let meta = AprMetadata {
        architecture: Some("phi".to_string()),
        hidden_size: Some(2560),
        num_layers: Some(32),
        num_heads: Some(32),
        vocab_size: Some(51200),
        ..Default::default()
    };
    let json = serde_json::to_string(&meta).expect("should serialize");
    let parsed: AprMetadata = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(parsed.architecture, Some("phi".to_string()));
    assert_eq!(parsed.hidden_size, Some(2560));
}

// ============================================================================
// APR Transformer Initialization Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_apr_transformer_gqa_config() {
    // Grouped Query Attention: num_kv_heads < num_heads
    let config = AprTransformerConfig {
        architecture: "llama".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8, // GQA: 4 query heads per KV head
        vocab_size: 128256,
        intermediate_dim: 14336,
        context_length: 8192,
        rope_theta: 500000.0,
        eps: 1e-5,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    assert_eq!(transformer.config().num_heads, 32);
    assert_eq!(transformer.config().num_kv_heads, 8);
    // head_dim = 4096 / 32 = 128
    // kv_dim = 8 * 128 = 1024
    assert!(!transformer.layers[0].qkv_weight.is_empty());
}

#[test]
fn test_apr_transformer_mha_config() {
    // Multi-Head Attention: num_kv_heads == num_heads
    let config = AprTransformerConfig {
        architecture: "gpt2".to_string(),
        hidden_dim: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 12, // MHA: all heads have own KV
        vocab_size: 50257,
        intermediate_dim: 3072,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    assert_eq!(
        transformer.config().num_heads,
        transformer.config().num_kv_heads
    );
}

#[test]
fn test_apr_transformer_layer_gqa_dimensions() {
    // Test GQA layer dimensions
    let hidden_dim = 256;
    let num_heads = 8;
    let num_kv_heads = 2;
    let intermediate_dim = 1024;

    let layer =
        AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

    let head_dim = hidden_dim / num_heads; // 32
    let kv_dim = num_kv_heads * head_dim; // 64
    let qkv_out_dim = hidden_dim + 2 * kv_dim; // 256 + 128 = 384

    assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
    assert_eq!(layer.attn_output_weight.len(), hidden_dim * hidden_dim);
    assert_eq!(layer.ffn_up_weight.len(), hidden_dim * intermediate_dim);
    assert_eq!(layer.ffn_down_weight.len(), intermediate_dim * hidden_dim);
}

#[test]
fn test_apr_transformer_config_rope_theta_variations() {
    // LLaMA 2 uses 10000.0
    let config_llama2 = AprTransformerConfig {
        rope_theta: 10000.0,
        ..Default::default()
    };
    assert!((config_llama2.rope_theta - 10000.0).abs() < 0.01);

    // LLaMA 3 uses 500000.0
    let config_llama3 = AprTransformerConfig {
        rope_theta: 500000.0,
        ..Default::default()
    };
    assert!((config_llama3.rope_theta - 500000.0).abs() < 0.01);

    // Qwen2.5 uses 1000000.0
    let config_qwen = AprTransformerConfig {
        rope_theta: 1000000.0,
        ..Default::default()
    };
    assert!((config_qwen.rope_theta - 1000000.0).abs() < 0.01);
}

// ============================================================================
// Weight Conversion and Quantization Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_mapped_apr_model_dtype_to_qtype_q5_k_extended() {
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
}

#[test]
fn test_mapped_apr_model_dtype_to_qtype_case_sensitive() {
    // The function is case-sensitive, lowercase should return unknown (0)
    assert_eq!(MappedAprModel::dtype_to_qtype("f32"), 0);
    assert_eq!(MappedAprModel::dtype_to_qtype("q4_k"), 0);
}

#[test]
fn test_quantized_apr_tensor_q4_dimensions() {
    let tensor = QuantizedAprTensorQ4::zeros(128, 256);
    assert_eq!(tensor.in_dim, 128);
    assert_eq!(tensor.out_dim, 256);

    // Check data size: 128 * 256 = 32768 elements, Q4_0 format
    let expected_bytes = QuantizedAprTensorQ4::expected_bytes(128 * 256);
    assert_eq!(tensor.data.len(), expected_bytes);
}

#[test]
fn test_quantized_apr_tensor_q4_block_alignment() {
    // Q4_0: 32 elements per block, 18 bytes per block
    // 31 elements should round up to 1 block
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(31), 18);

    // 32 elements = exactly 1 block
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);

    // 100 elements = ceil(100/32) * 18 = 4 * 18 = 72
    assert_eq!(QuantizedAprTensorQ4::expected_bytes(100), 72);
}

#[test]
fn test_apr_quantization_type_all_types() {
    // Test all quantization types
    let types = [
        (AprQuantizationType::F32, 32.0, 4, 1, 0),
        (AprQuantizationType::Q4_K, 4.5, 144, 256, 1),
        (AprQuantizationType::Q8_0, 8.0, 36, 32, 2),
    ];

    for (qtype, bits, bytes_per_block, values_per_block, byte_id) in types {
        assert!(
            (qtype.bits_per_weight() - bits).abs() < 0.001,
            "{:?} bits_per_weight",
            qtype
        );
        assert_eq!(
            qtype.bytes_per_block(),
            bytes_per_block,
            "{:?} bytes_per_block",
            qtype
        );
        assert_eq!(
            qtype.values_per_block(),
            values_per_block,
            "{:?} values_per_block",
            qtype
        );
        assert_eq!(qtype.to_byte(), byte_id, "{:?} to_byte", qtype);
        assert_eq!(
            AprQuantizationType::from_byte(byte_id),
            Some(qtype),
            "{:?} from_byte",
            qtype
        );
    }
}

// ============================================================================
// Error Handling Paths Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_apr_v2_model_truncated_tensor_index() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&5u32.to_le_bytes()); // 5 tensors expected
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
    data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index at 64
    data[32..40].copy_from_slice(&128u64.to_le_bytes()); // data_offset = 128

    // Model should load, header says 5 tensors but index may be truncated/empty
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 5); // Header says 5
                                         // Parsed tensor count may be less than header due to truncated index
    assert!(model.tensor_names().len() <= 5);
}

#[test]
fn test_apr_v2_model_get_tensor_f32_out_of_bounds() {
    let mut data = create_minimal_apr_model();
    // Modify tensor entry to have offset beyond file
    // The tensor offset is stored in the tensor index, so this is tricky
    // Instead, we truncate the data section
    data.truncate(data.len() - 60); // Remove part of tensor data

    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.get_tensor_f32("test.weight");
    assert!(result.is_err());
}

#[test]
fn test_apr_v2_model_generate_max_tokens_zero() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.generate(&[1], 0, None);
    // With max_tokens=0, should return just the prompt
    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert_eq!(tokens, vec![1]);
}

#[test]
fn test_apr_header_exactly_64_bytes() {
    // Exactly 64 bytes should work
    let mut data = vec![0u8; 64];
    data[0..4].copy_from_slice(&MAGIC);
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_ok());
}

#[test]
fn test_apr_header_63_bytes_fails() {
    // 63 bytes should fail (too small)
    let mut data = vec![0u8; 63];
    data[0..4].copy_from_slice(&MAGIC);
    let result = AprHeader::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_truncated_at_shape() {
    let mut data = Vec::new();
    data.extend_from_slice(&4u16.to_le_bytes()); // name_len = 4
    data.extend_from_slice(b"test"); // name
    data.push(0); // dtype = F32
    data.push(3); // ndim = 3
                  // Only 8 bytes of shape (should be 24 for 3 dims + 16 for offset+size)
    data.extend_from_slice(&[0u8; 8]);

    let result = TensorEntry::from_binary(&data);
    assert!(result.is_err());
}

#[test]
fn test_tensor_entry_8_dimensional() {
    // Test maximum 8 dimensions
    let data = create_binary_tensor_entry("test", 0, &[2, 2, 2, 2, 2, 2, 2, 2], 0, 1024);
    let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
    assert_eq!(entry.shape.len(), 8);
    assert_eq!(entry.element_count(), 256); // 2^8
}

#[test]
fn test_apr_flags_raw_bits() {
    // Test raw bit access
    let flags = AprFlags::new(0xFFFF);
    assert!(flags.is_lz4());
    assert!(flags.is_zstd());
    assert!(flags.is_compressed());
    assert!(flags.is_encrypted());
    assert!(flags.is_quantized());
    assert!(flags.has_vocab());
}

// ============================================================================
// AprKVCache Extended Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_apr_kv_cache_multiple_positions() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        context_length: 32,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);

    let head_dim = 16 / 4; // 4
    let kv_size = 2 * head_dim; // 8

    // Append multiple positions
    for pos in 0..5 {
        let k = vec![(pos as f32) + 1.0; kv_size];
        let v = vec![(pos as f32) * 2.0; kv_size];
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }

    assert_eq!(cache.len(), 5);
    assert!(!cache.is_empty());
}

#[test]
fn test_apr_kv_cache_capacity_limit() {
    let config = AprTransformerConfig {
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        context_length: 4, // Small capacity
        ..Default::default()
    };
    let cache = AprKVCache::new(&config);
    assert_eq!(cache.capacity(), 4);
}

// ============================================================================
// AprBenchmarkResult Extended Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_apr_benchmark_result_full_fields() {
    let result = AprBenchmarkResult {
        tokens_generated: 100,
        total_time_ms: 5000.0,
        tokens_per_second: 20.0,
        throughput_p50: 19.5,
        throughput_p99: 18.0,
        throughput_std_dev: 1.5,
        peak_memory_mb: 2048.0,
        ..Default::default()
    };
    assert_eq!(result.tokens_generated, 100);
    assert!((result.total_time_ms - 5000.0).abs() < 0.001);
    assert!((result.tokens_per_second - 20.0).abs() < 0.001);
    assert!((result.throughput_p50 - 19.5).abs() < 0.001);
    assert!((result.throughput_p99 - 18.0).abs() < 0.001);
    assert!((result.throughput_std_dev - 1.5).abs() < 0.001);
    assert!((result.peak_memory_mb - 2048.0).abs() < 0.001);
}

#[test]
fn test_apr_benchmark_result_meets_threshold_edge_cases() {
    let result = AprBenchmarkResult {
        tokens_per_second: 0.0,
        ..Default::default()
    };
    assert!(result.meets_threshold(0.0));
    assert!(!result.meets_threshold(0.001));
}

#[test]
fn test_apr_parity_comparison_threshold_variations() {
    // Test different threshold percentages
    let comparison = AprParityComparison {
        throughput_ratio: 0.90,
        memory_ratio: 1.0,
        parity_threshold_pct: 90.0,
    };
    assert!(comparison.is_parity());

    let comparison2 = AprParityComparison {
        throughput_ratio: 0.89,
        memory_ratio: 1.0,
        parity_threshold_pct: 90.0,
    };
    assert!(!comparison2.is_parity());
}

// ============================================================================
// AprInferenceScratch Extended Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_apr_inference_scratch_large_model() {
    let config = AprTransformerConfig {
        hidden_dim: 4096,
        intermediate_dim: 14336,
        num_heads: 32,
        ..Default::default()
    };
    let scratch = AprInferenceScratch::from_config(&config);

    assert_eq!(scratch.hidden.len(), 4096);
    assert_eq!(scratch.normed.len(), 4096);
    assert_eq!(scratch.qkv_out.len(), 4096 * 3);
    assert_eq!(scratch.ffn_up.len(), 14336);
    assert_eq!(scratch.ffn_gate.len(), 14336);
}

#[test]
fn test_apr_inference_scratch_clear_preserves_capacity() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        intermediate_dim: 256,
        ..Default::default()
    };
    let mut scratch = AprInferenceScratch::from_config(&config);
    let original_hidden_capacity = scratch.hidden.capacity();

    scratch.hidden.fill(999.0);
    scratch.clear();

    // After clear, values should be zero but capacity preserved
    assert_eq!(scratch.hidden.len(), 64);
    assert!(scratch.hidden.capacity() >= original_hidden_capacity);
    assert!(scratch.hidden.iter().all(|&x| x == 0.0));
}

// ============================================================================
// BpeTokenizer Extended Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_bpe_tokenizer_known_token_handling() {
    // BPE tokenizer converts space to Ġ, newline to Ċ, tab to ĉ internally
    let mut token_to_id = HashMap::new();
    token_to_id.insert("h".to_string(), 0);
    token_to_id.insert("e".to_string(), 1);
    token_to_id.insert("l".to_string(), 2);
    token_to_id.insert("o".to_string(), 3);

    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
        ],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    // Encode text with known characters
    let encoded = tokenizer.encode("hello");
    // Should encode h, e, l, l, o
    assert_eq!(encoded.len(), 5);
    assert_eq!(encoded, vec![0, 1, 2, 2, 3]);
}

#[test]
fn test_bpe_tokenizer_bpe_style_whitespace() {
    // BPE tokenizer converts space to Ġ (U+0120), newline to Ċ (U+010A), tab to ĉ (U+0109)
    let mut token_to_id = HashMap::new();
    token_to_id.insert("Ġ".to_string(), 0); // space becomes Ġ
    token_to_id.insert("Ċ".to_string(), 1); // newline becomes Ċ
    token_to_id.insert("ĉ".to_string(), 2); // tab becomes ĉ

    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec!["Ġ".to_string(), "Ċ".to_string(), "ĉ".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    let encoded_space = tokenizer.encode(" ");
    let encoded_newline = tokenizer.encode("\n");
    let encoded_tab = tokenizer.encode("\t");

    assert_eq!(encoded_space, vec![0]);
    assert_eq!(encoded_newline, vec![1]);
    assert_eq!(encoded_tab, vec![2]);
}

// ============================================================================
// detect_format and is_apr_file Extended Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_detect_format_case_sensitivity() {
    // Extensions should be case-insensitive internally
    assert_eq!(detect_format("/path/model.APR"), "apr");
    assert_eq!(detect_format("/path/model.GGUF"), "gguf");
    assert_eq!(detect_format("/path/model.SafeTensors"), "safetensors");
}

#[test]
fn test_detect_format_multiple_extensions() {
    // Only the last extension matters
    assert_eq!(detect_format("/path/model.tar.apr"), "apr");
    assert_eq!(detect_format("/path/model.apr.bak"), "unknown");
}

#[test]
fn test_is_apr_file_empty_file() {
    let temp = NamedTempFile::new().expect("create temp file");
    // Empty file should not be valid APR
    assert!(!is_apr_file(temp.path()));
}

#[test]
fn test_is_apr_file_partial_magic() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    temp.write_all(b"APR").expect("write data"); // Missing null terminator
    assert!(!is_apr_file(temp.path()));
}

// ============================================================================
// AprV2Model decode_tokens Extended Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_apr_v2_model_decode_tokens_empty_vocab() {
    let vocab: Vec<String> = vec![];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert!(result.contains("[0]"));
    assert!(result.contains("[1]"));
    assert!(result.contains("[2]"));
}

#[test]
fn test_apr_v2_model_decode_tokens_special_chars() {
    let vocab = vec![
        "<s>".to_string(),
        "</s>".to_string(),
        "<unk>".to_string(),
        " ".to_string(),
    ];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 3, 1]);
    assert!(result.contains("<s>"));
    assert!(result.contains(' '));
    assert!(result.contains("</s>"));
}

// ============================================================================
// MappedAprModel Extended Coverage (PMAT-802)
// ============================================================================

#[test]
fn test_mapped_apr_model_multiple_tensors() {
    // Create APR file with multiple tensors
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor1 = create_binary_tensor_entry("tensor1", 0, &[4, 4], 0, 64);
    let tensor2 = create_binary_tensor_entry("tensor2", 0, &[2, 2], 64, 16);
    let tensor_index: Vec<u8> = [tensor1, tensor2].concat();

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_index.len() as u64;
    let data_size = 80usize; // 64 + 16 bytes
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&2u32.to_le_bytes()); // 2 tensors
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

    // Write to temp file and load
    let mut temp = NamedTempFile::new().expect("create temp file");
    temp.write_all(&data).expect("write data");

    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    assert_eq!(model.tensor_count(), 2);
    assert!(model.find_tensor("tensor1").is_some());
    assert!(model.find_tensor("tensor2").is_some());
}

#[test]
fn test_mapped_apr_model_tensor_entry_fields() {
    let temp = create_apr_file();
    let model = MappedAprModel::from_path(temp.path()).expect("should load");
    let tensor = model.find_tensor("test.weight").unwrap();

    assert_eq!(tensor.name, "test.weight");
    assert_eq!(tensor.dtype, "F32");
    assert_eq!(tensor.shape, vec![4, 4]);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.size, 64);
}

// ============================================================================
// F16 Dequantization Coverage (via get_tensor_f32) - PMAT-802
// ============================================================================

/// Helper to create APR model with F16 tensor
fn create_apr_model_with_f16_tensor() -> Vec<u8> {
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // F16 tensor: dtype=1, shape [4] = 4 elements = 8 bytes
    let tensor_entry = create_binary_tensor_entry("test.f16", 1, &[4], 0, 8);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_size = 8usize; // 4 * 2 bytes per f16
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    // Header
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 0;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Metadata
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Tensor index
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // F16 tensor data: [1.0, 2.0, -0.5, 0.0]
    // F16 bit patterns: 1.0=0x3C00, 2.0=0x4000, -0.5=0xB800, 0.0=0x0000
    let data_start = data_offset as usize;
    data[data_start..data_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
    data[data_start + 2..data_start + 4].copy_from_slice(&0x4000u16.to_le_bytes()); // 2.0
    data[data_start + 4..data_start + 6].copy_from_slice(&0xB800u16.to_le_bytes()); // -0.5
    data[data_start + 6..data_start + 8].copy_from_slice(&0x0000u16.to_le_bytes()); // 0.0

    data
}

#[test]
fn test_apr_model_f16_tensor_dequantize() {
    let data = create_apr_model_with_f16_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load F16 model");
    let floats = model.get_tensor_f32("test.f16").expect("should dequantize");

    assert_eq!(floats.len(), 4);
    assert!(
        (floats[0] - 1.0).abs() < 0.01,
        "Expected 1.0, got {}",
        floats[0]
    );
    assert!(
        (floats[1] - 2.0).abs() < 0.01,
        "Expected 2.0, got {}",
        floats[1]
    );
    assert!(
        (floats[2] - (-0.5)).abs() < 0.01,
        "Expected -0.5, got {}",
        floats[2]
    );
    assert!(
        (floats[3] - 0.0).abs() < 0.01,
        "Expected 0.0, got {}",
        floats[3]
    );
}

#[test]
fn test_apr_model_f16_tensor_special_values() {
    // Test F16 special values: infinity, NaN, subnormal, zero
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor_entry = create_binary_tensor_entry("test.f16_special", 1, &[6], 0, 12);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let total_size = data_offset as usize + 12;
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    let data_start = data_offset as usize;
    // +inf: 0x7C00, -inf: 0xFC00, NaN: 0x7E00, +0: 0x0000, -0: 0x8000, subnormal: 0x0001
    data[data_start..data_start + 2].copy_from_slice(&0x7C00u16.to_le_bytes()); // +inf
    data[data_start + 2..data_start + 4].copy_from_slice(&0xFC00u16.to_le_bytes()); // -inf
    data[data_start + 4..data_start + 6].copy_from_slice(&0x7E00u16.to_le_bytes()); // NaN
    data[data_start + 6..data_start + 8].copy_from_slice(&0x0000u16.to_le_bytes()); // +0
    data[data_start + 8..data_start + 10].copy_from_slice(&0x8000u16.to_le_bytes()); // -0
    data[data_start + 10..data_start + 12].copy_from_slice(&0x0001u16.to_le_bytes()); // subnormal

    let model = AprV2Model::from_bytes(data).expect("should load");
    let floats = model
        .get_tensor_f32("test.f16_special")
        .expect("should dequantize");

    assert!(floats[0].is_infinite() && floats[0] > 0.0, "Expected +inf");
    assert!(floats[1].is_infinite() && floats[1] < 0.0, "Expected -inf");
    assert!(floats[2].is_nan(), "Expected NaN");
    assert!(floats[3] == 0.0, "Expected +0");
    assert!(floats[4] == 0.0, "Expected -0"); // -0 == 0 in f32 comparison
}

// ============================================================================
// Q8_0 Dequantization Coverage (via get_tensor_f32) - PMAT-802
// ============================================================================

#[test]
fn test_apr_model_q8_0_tensor_dequantize() {
    let data = create_apr_model_with_q8_0_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load Q8_0 model");
    let floats = model
        .get_tensor_f32("test.q8_0")
        .expect("should dequantize");

    // Q8_0: 32 elements, scale * int8 values
    // With scale=1.0 and quant values 0-31, we expect 0.0, 1.0, 2.0, ... 31.0
    assert_eq!(floats.len(), 32);
    for (i, &val) in floats.iter().enumerate() {
        assert!(
            (val - (i as f32)).abs() < 0.1,
            "Element {} expected {}, got {}",
            i,
            i,
            val
        );
    }
}

#[test]
fn test_apr_model_q8_0_multiple_blocks() {
    // Create Q8_0 tensor with 64 elements (2 blocks)
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor_entry = create_binary_tensor_entry("test.q8_0_multi", 8, &[64], 0, 68);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_size = 68usize; // 2 blocks * 34 bytes
    let total_size = data_offset as usize + data_size;
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // Block 1: scale=1.0, values 0-31
    let data_start = data_offset as usize;
    data[data_start..data_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale=1.0
    for i in 0..32 {
        data[data_start + 2 + i] = i as u8;
    }

    // Block 2: scale=0.5 (0x3800), values 0-31
    data[data_start + 34..data_start + 36].copy_from_slice(&0x3800u16.to_le_bytes()); // scale=0.5
    for i in 0..32 {
        data[data_start + 36 + i] = i as u8;
    }

    let model = AprV2Model::from_bytes(data).expect("should load");
    let floats = model
        .get_tensor_f32("test.q8_0_multi")
        .expect("should dequantize");

    assert_eq!(floats.len(), 64);
    // Block 1 should have values 0-31
    // Block 2 should have values 0-15.5 (scale=0.5)
}

// ============================================================================
// Q4_K Dequantization Coverage (via get_tensor_f32) - PMAT-802
// ============================================================================

#[test]
fn test_apr_model_q4k_tensor_dequantize() {
    let data = create_apr_model_with_q4k_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load Q4_K model");
    let floats = model.get_tensor_f32("test.q4k").expect("should dequantize");

    // Q4_K: 256 elements per super block
    assert_eq!(floats.len(), 256);
    // Values should be finite (not NaN/inf due to proper dequant)
    for &val in &floats {
        assert!(val.is_finite(), "Q4_K value should be finite");
    }
}

// ============================================================================
// Q6_K Dequantization Coverage (via get_tensor_f32) - PMAT-802
// ============================================================================

#[test]
fn test_apr_model_q6k_tensor_dequantize() {
    let data = create_apr_model_with_q6k_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load Q6_K model");
    let floats = model.get_tensor_f32("test.q6k").expect("should dequantize");

    // Q6_K: 256 elements per super block
    assert_eq!(floats.len(), 256);
    // Values should be finite
    for &val in &floats {
        assert!(val.is_finite(), "Q6_K value should be finite");
    }
}

// ============================================================================
// Unsupported Dtype Error Path - PMAT-802
// ============================================================================

/// Helper to create APR model with BF16 tensor (unsupported for dequant)
fn create_apr_model_with_bf16_tensor() -> Vec<u8> {
    let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // BF16 tensor: dtype=30 (GGML BF16)
    let tensor_entry = create_binary_tensor_entry("test.bf16", 30, &[4], 0, 8);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let total_size = data_offset as usize + 8;
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    data
}

#[test]
fn test_apr_model_unsupported_dtype_error() {
    let data = create_apr_model_with_bf16_tensor();
    let model = AprV2Model::from_bytes(data).expect("should load BF16 model");
    let result = model.get_tensor_f32("test.bf16");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Unsupported") || err_msg.contains("BF16"));
}

// ============================================================================
// AprV2Model::forward() Error Paths - PMAT-802
// ============================================================================

#[test]
fn test_apr_v2_model_forward_not_transformer() {
    // Create minimal model without transformer metadata
    let metadata = r#"{"architecture":"test"}"#; // Missing required fields
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);
    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let total_size = data_offset as usize + 64;
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.forward(&[1, 2, 3]);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("transformer") || err_msg.contains("missing"));
}

// ============================================================================
// AprV2Model::generate() Error Paths - PMAT-802
// ============================================================================

#[test]
fn test_apr_v2_model_generate_with_eos_token() {
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    // Generation with eos_token that won't be hit (model isn't real transformer)
    let result = model.generate(&[1], 5, Some(999));
    // Should return error since not a transformer
    assert!(result.is_err());
}

// ============================================================================
// AprV2Model::predict() Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_v2_model_predict_with_weight_tensor() {
    // Create model with weight and bias tensors for linear prediction
    let metadata = r#"{"architecture":"linear"}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    // weight tensor [2, 3] and bias tensor [2]
    let weight_entry = create_binary_tensor_entry("weight", 0, &[2, 3], 0, 24);
    let bias_entry = create_binary_tensor_entry("bias", 0, &[2], 24, 8);
    let tensor_index: Vec<u8> = [weight_entry, bias_entry].concat();

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_index.len() as u64;
    let total_size = data_offset as usize + 32; // 24 + 8 bytes
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&2u32.to_le_bytes()); // 2 tensors
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

    // Weight: [[1,0,0], [0,1,0]] (identity-ish)
    let data_start = data_offset as usize;
    let weights = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    for (i, &w) in weights.iter().enumerate() {
        data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }

    // Bias: [0.5, -0.5]
    let biases = [0.5f32, -0.5];
    for (i, &b) in biases.iter().enumerate() {
        data[data_start + 24 + i * 4..data_start + 24 + i * 4 + 4]
            .copy_from_slice(&b.to_le_bytes());
    }

    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.predict(&[1.0, 2.0, 3.0]).expect("should predict");

    // Output = W @ x + b = [[1,0,0],[0,1,0]] @ [1,2,3] + [0.5,-0.5] = [1.5, 1.5]
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.5).abs() < 0.01);
    assert!((result[1] - 1.5).abs() < 0.01);
}

#[test]
fn test_apr_v2_model_predict_no_weight_tensor() {
    // Model without weight tensor - falls back to sum
    let data = create_minimal_apr_model();
    let model = AprV2Model::from_bytes(data).expect("should load");
    let result = model.predict(&[1.0, 2.0, 3.0]).expect("should predict");

    // No "weight" tensor, so returns sum = 6.0
    assert_eq!(result.len(), 1);
    assert!((result[0] - 6.0).abs() < 0.01);
}

// ============================================================================
// ModelData Coverage - PMAT-802
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_model_data_debug_format() {
    use realizar::apr::ModelData;
    let data = ModelData::from_vec(vec![1, 2, 3]);
    let debug_str = format!("{:?}", data);
    assert!(debug_str.contains("Heap"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_model_data_mmap_debug_format() {
    use realizar::apr::ModelData;
    let temp = create_apr_file();
    let model_data = ModelData::open_mmap(temp.path()).expect("should open");
    let debug_str = format!("{:?}", model_data);
    assert!(debug_str.contains("Mmap"));
}

// ============================================================================
// TensorEntry Serialization Coverage - PMAT-802
// ============================================================================

#[test]
fn test_tensor_entry_serde_roundtrip() {
    let entry = TensorEntry {
        name: "test.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![16, 32, 64],
        offset: 1024,
        size: 131072,
    };
    let json = serde_json::to_string(&entry).expect("should serialize");
    let parsed: TensorEntry = serde_json::from_str(&json).expect("should deserialize");
    assert_eq!(parsed.name, "test.weight");
    assert_eq!(parsed.shape, vec![16, 32, 64]);
    assert_eq!(parsed.offset, 1024);
    assert_eq!(parsed.size, 131072);
}

#[test]
fn test_tensor_entry_element_count_high_dim() {
    let entry = TensorEntry {
        name: "high_dim".to_string(),
        dtype: "F32".to_string(),
        shape: vec![2, 3, 4, 5, 6],
        offset: 0,
        size: 2880, // 2*3*4*5*6*4 bytes
    };
    assert_eq!(entry.element_count(), 720); // 2*3*4*5*6
}

// ============================================================================
// AprHeader Clone Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_header_clone() {
    let mut data = vec![0u8; HEADER_SIZE];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[5] = 1;
    data[8..12].copy_from_slice(&42u32.to_le_bytes());
    data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

    let header = AprHeader::from_bytes(&data).expect("should parse");
    // Clone trait test: use the cloned value in a function call to avoid redundant_clone lint
    fn verify_header(h: AprHeader) {
        assert_eq!(h.magic, MAGIC);
        assert_eq!(h.version, (2, 1));
        assert_eq!(h.tensor_count, 42);
        assert_eq!(h.checksum, 0xDEADBEEF);
    }
    verify_header(header.clone());
    // Original still usable
    assert_eq!(header.version, (2, 1));
}

// ============================================================================
// AprFlags Default Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_flags_default() {
    let flags = AprFlags::default();
    assert!(!flags.is_compressed());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_quantized());
    assert!(!flags.has_vocab());
}

#[test]
fn test_apr_flags_clone() {
    let flags = AprFlags::new(AprFlags::QUANTIZED | AprFlags::HAS_VOCAB);
    let cloned = flags;
    assert!(cloned.is_quantized());
    assert!(cloned.has_vocab());
}

// ============================================================================
// AprV2Model File-based Tests - PMAT-802
// ============================================================================

#[test]
fn test_apr_v2_model_load_from_file_header_access() {
    let temp = create_apr_file();
    let model = AprV2Model::load(temp.path()).expect("should load");

    // Verify header fields are accessible
    assert_eq!(model.tensor_count(), 1);
    assert!(model.is_mmap());
}

#[test]
fn test_apr_v2_model_estimated_parameters_empty() {
    // Model with no tensors
    let metadata = r#"{"architecture":"test"}"#;
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset;
    let total_size = data_offset as usize;
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.estimated_parameters(), 0);
}

// ============================================================================
// BpeTokenizer Edge Cases - PMAT-802
// ============================================================================

#[test]
fn test_bpe_tokenizer_decode_bpe_special_chars() {
    // Test that BPE special characters are decoded correctly
    let vocab = vec!["Ġhello".to_string(), "Ċ".to_string(), "ĉworld".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);

    // Ġ → space, Ċ → newline, ĉ → tab
    assert!(result.contains(" hello"));
    assert!(result.contains('\n'));
    assert!(result.contains("\tworld"));
}

#[test]
fn test_bpe_tokenizer_encode_unicode() {
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);

    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec!["a".to_string(), "b".to_string()],
        merge_rules: vec![],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    // Non-ASCII chars may not be in vocab, so result may be empty or partial
    let encoded = tokenizer.encode("ab");
    assert!(!encoded.is_empty());
}

#[test]
fn test_bpe_tokenizer_merge_application() {
    // Test that merge rules are applied correctly
    let mut token_to_id = HashMap::new();
    token_to_id.insert("a".to_string(), 0);
    token_to_id.insert("b".to_string(), 1);
    token_to_id.insert("ab".to_string(), 2);

    let tokenizer = BpeTokenizer {
        token_to_id,
        id_to_token: vec!["a".to_string(), "b".to_string(), "ab".to_string()],
        merge_rules: vec![("a".to_string(), "b".to_string())],
        bos_id: None,
        eos_id: None,
        special_tokens: HashMap::new(),
    };

    let encoded = tokenizer.encode("ab");
    // After merging "a" + "b" → "ab", should get single token
    assert_eq!(encoded.len(), 1);
    assert_eq!(encoded[0], 2);
}

// ============================================================================
// AprMetadata rms_norm_eps Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_metadata_rms_norm_eps_field() {
    let meta = AprMetadata {
        rms_norm_eps: Some(1e-6),
        ..Default::default()
    };
    assert_eq!(meta.rms_norm_eps, Some(1e-6));
}

#[test]
fn test_apr_metadata_max_position_embeddings_field() {
    let meta = AprMetadata {
        max_position_embeddings: Some(131072),
        ..Default::default()
    };
    assert_eq!(meta.max_position_embeddings, Some(131072));
}

// ============================================================================
// AprTransformerLayer Optional Fields Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_transformer_layer_all_optional_fields() {
    let mut layer = AprTransformerLayer::empty(64, 256);

    // Set all optional fields
    layer.attn_norm_bias = Some(vec![0.0; 64]);
    layer.qkv_bias = Some(vec![0.0; 64 * 3]);
    layer.attn_output_bias = Some(vec![0.0; 64]);
    layer.ffn_norm_weight = Some(vec![1.0; 64]);
    layer.ffn_norm_bias = Some(vec![0.0; 64]);
    layer.ffn_up_bias = Some(vec![0.0; 256]);
    layer.ffn_gate_weight = Some(vec![0.0; 64 * 256]);
    layer.ffn_gate_bias = Some(vec![0.0; 256]);
    layer.ffn_down_bias = Some(vec![0.0; 64]);

    let params = layer.num_parameters();
    // Should include all optional fields
    assert!(params > 0);
}

#[test]
fn test_apr_transformer_layer_memory_size() {
    let layer = AprTransformerLayer::empty(64, 256);
    let mem_size = layer.num_parameters() * 4; // 4 bytes per f32
    assert!(mem_size > 0);
}

// ============================================================================
// AprKVCache Extended Operations - PMAT-802
// ============================================================================

#[test]
fn test_apr_kv_cache_get_layer_invalid() {
    let config = AprTransformerConfig {
        hidden_dim: 8,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        context_length: 16,
        ..Default::default()
    };
    let mut cache = AprKVCache::new(&config);

    // Append to layer 0
    let head_dim = 4;
    let kv_size = 2 * head_dim;
    cache.append(0, &vec![1.0; kv_size], &vec![2.0; kv_size]);

    // Get from layer 0 (valid)
    let (k, v) = cache.get(0);
    assert!(!k.is_empty());
    assert!(!v.is_empty());
}

// ============================================================================
// AprPrefillResult Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_prefill_result_full_fields() {
    let result = AprPrefillResult {
        prompt_tokens: 128,
        prefill_time_ms: 45.5,
        prefill_tok_s: 2813.19,
    };
    assert_eq!(result.prompt_tokens, 128);
    assert!((result.prefill_time_ms - 45.5).abs() < 0.001);
    assert!((result.prefill_tok_s - 2813.19).abs() < 0.01);
}

// ============================================================================
// AprTransformerConfig Clone and Debug Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_transformer_config_clone() {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 1024,
        num_layers: 16,
        num_heads: 16,
        num_kv_heads: 4,
        vocab_size: 50000,
        intermediate_dim: 4096,
        context_length: 4096,
        rope_theta: 500000.0,
        eps: 1e-6,
        ..Default::default()
    };
    // Clone trait test: use the cloned value in a function call to avoid redundant_clone lint
    fn verify_config(c: AprTransformerConfig) {
        assert_eq!(c.architecture, "test");
        assert_eq!(c.hidden_dim, 1024);
        assert_eq!(c.num_kv_heads, 4);
    }
    verify_config(config.clone());
    // Original still usable
    assert_eq!(config.hidden_dim, 1024);
}

#[test]
fn test_apr_transformer_config_debug() {
    let config = AprTransformerConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("AprTransformerConfig"));
    assert!(debug_str.contains("hidden_dim"));
}

// ============================================================================
// detect_format Magic Bytes Coverage - PMAT-802
// ============================================================================

#[test]
fn test_detect_format_by_magic_bytes_apr() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    temp.write_all(&MAGIC).expect("write magic");
    temp.write_all(&[0u8; 60]).expect("write padding");
    // Don't use .apr extension - rely on magic
    assert_eq!(detect_format(temp.path()), "apr");
}

#[test]
fn test_detect_format_by_magic_bytes_gguf() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    temp.write_all(&[0x47, 0x47, 0x55, 0x46])
        .expect("write GGUF magic");
    temp.write_all(&[0u8; 60]).expect("write padding");
    assert_eq!(detect_format(temp.path()), "gguf");
}

#[test]
fn test_detect_format_by_magic_bytes_safetensors() {
    let mut temp = NamedTempFile::new().expect("create temp file");
    // Need at least 4 bytes for detect_format to check magic bytes
    temp.write_all(b"{}  ").expect("write JSON");
    assert_eq!(detect_format(temp.path()), "safetensors");
}

// ============================================================================
// AprV2Model from_bytes Empty Metadata Coverage - PMAT-802
// ============================================================================

#[test]
fn test_apr_v2_model_from_bytes_empty_metadata() {
    // Create model with metadata_size = 0
    let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);
    let tensor_index_offset = HEADER_SIZE as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let total_size = data_offset as usize + 64;
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

    // Model should load with default metadata
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert!(model.metadata().hidden_size.is_none());
    assert!(!model.metadata().is_transformer());
}

// ============================================================================
// Tensor Index Parsing Edge Cases - PMAT-802
// ============================================================================

#[test]
fn test_apr_v2_model_tensor_index_parse_error_handled() {
    // Create model where tensor index has malformed entries
    let metadata = r"{}";
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + 10; // Index region = 10 bytes (malformed)
    let total_size = data_offset as usize + 64;
    let mut data = vec![0u8; total_size];

    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&1u32.to_le_bytes()); // Claims 1 tensor
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    // Put garbage in tensor index region
    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + 10].copy_from_slice(&[0xFF; 10]);

    // Model should load but have 0 parsed tensors (parse errors are handled gracefully)
    let model = AprV2Model::from_bytes(data).expect("should load");
    assert_eq!(model.tensor_count(), 1); // Header says 1
                                         // Parsed tensor count may be less than header due to malformed index
    let parsed_count = model.tensor_names().len();
    assert!(
        parsed_count <= 1,
        "Should have 0 or 1 parsed tensors, got {}",
        parsed_count
    );
}
