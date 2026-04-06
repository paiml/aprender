//! T-COV-95 Coverage Bridge (Part 14 - GgufToAprQ4KConverter::convert full pipeline)
//!
//! Targets uncovered lines in convert/mod.rs:
//!   - GgufToAprQ4KConverter::convert() (lines 525-831) - GGUF->APR Q4K file pipeline
//!   - Metadata extraction, tensor iteration, binary index, alignment, CRC32
//!   - Error paths: missing file, truncated tensor, bounds check

use super::*;
use crate::apr::{HEADER_SIZE, MAGIC};
use crate::gguf::test_factory::build_minimal_llama_gguf;
use std::io::Read;

fn write_gguf_to_tempfile(gguf_data: &[u8]) -> (tempfile::NamedTempFile, std::path::PathBuf) {
    use std::io::Write;
    let mut tmp = tempfile::NamedTempFile::new().expect("create temp GGUF");
    tmp.write_all(gguf_data).expect("write GGUF");
    tmp.flush().expect("flush");
    let path = tmp.path().to_path_buf();
    (tmp, path)
}

// ============================================================================
// Happy path: full GGUF -> APR Q4K conversion
// ============================================================================

#[test]
fn test_q4k_convert_llama_produces_valid_apr() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    let result = GgufToAprQ4KConverter::convert(&gguf_path, &apr_path);
    assert!(result.is_ok(), "convert() failed: {:?}", result.err());

    let stats = result.unwrap();
    assert_eq!(stats.architecture, "llama");
    assert_eq!(stats.num_layers, 1);
    assert_eq!(stats.hidden_size, 64);
    assert!(stats.tensor_count > 0, "should have tensors");
    assert!(stats.q4k_tensor_count > 0, "should have Q4K tensors");
    assert!(stats.total_bytes > 0, "should have written bytes");
}

#[test]
fn test_q4k_convert_output_has_apr_magic() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    let apr_data = std::fs::read(&apr_path).expect("read APR");
    assert!(apr_data.len() >= HEADER_SIZE, "APR file too small");
    assert_eq!(&apr_data[0..4], &MAGIC, "APR magic mismatch");
    assert_eq!(apr_data[4], 2, "version major should be 2");
    assert_eq!(apr_data[5], 0, "version minor should be 0");
}

#[test]
fn test_q4k_convert_output_header_offsets_valid() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    let apr_data = std::fs::read(&apr_path).expect("read APR");

    // Parse header offsets
    let tensor_count = u32::from_le_bytes(apr_data[8..12].try_into().unwrap());
    let metadata_offset = u64::from_le_bytes(apr_data[12..20].try_into().unwrap());
    let metadata_len = u32::from_le_bytes(apr_data[20..24].try_into().unwrap());
    let tensor_index_offset = u64::from_le_bytes(apr_data[24..32].try_into().unwrap());
    let data_offset = u64::from_le_bytes(apr_data[32..40].try_into().unwrap());

    assert!(tensor_count > 0, "should have tensors in header");
    assert_eq!(
        metadata_offset, HEADER_SIZE as u64,
        "metadata starts after header"
    );
    assert!(metadata_len > 0, "metadata should not be empty");
    assert!(
        tensor_index_offset > metadata_offset,
        "tensor index after metadata"
    );
    assert!(data_offset > tensor_index_offset, "data after tensor index");
    assert_eq!(data_offset % 64, 0, "data offset must be 64-byte aligned");
    assert!(
        (apr_data.len() as u64) >= data_offset,
        "file must be at least as large as data offset"
    );
}

#[test]
fn test_q4k_convert_metadata_contains_architecture() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    let apr_data = std::fs::read(&apr_path).expect("read APR");

    // Extract metadata JSON
    let metadata_offset = u64::from_le_bytes(apr_data[12..20].try_into().unwrap()) as usize;
    let metadata_len = u32::from_le_bytes(apr_data[20..24].try_into().unwrap()) as usize;
    let metadata_bytes = &apr_data[metadata_offset..metadata_offset + metadata_len];
    let metadata: serde_json::Value =
        serde_json::from_slice(metadata_bytes).expect("parse metadata JSON");

    assert_eq!(metadata["architecture"], "llama");
    assert_eq!(metadata["hidden_size"], 64);
    assert_eq!(metadata["num_hidden_layers"], 1);
    assert_eq!(metadata["num_attention_heads"], 4);
    assert_eq!(metadata["num_key_value_heads"], 4);
    assert_eq!(metadata["vocab_size"], 32);
    assert_eq!(metadata["intermediate_size"], 128);
    assert_eq!(metadata["model_type"], "transformer_lm_q4k");
    assert!(metadata["quantization"]["quant_type"] == "Q4_K");
}

#[test]
fn test_q4k_convert_tensor_count_matches_stats() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    let stats = GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    let apr_data = std::fs::read(&apr_path).expect("read APR");
    let header_tensor_count = u32::from_le_bytes(apr_data[8..12].try_into().unwrap()) as usize;
    assert_eq!(header_tensor_count, stats.tensor_count);
}

#[test]
fn test_q4k_convert_crc32_checksum_valid() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    let apr_data = std::fs::read(&apr_path).expect("read APR");
    let stored_checksum = u32::from_le_bytes(apr_data[40..44].try_into().unwrap());

    // Recompute checksum over header bytes (excluding checksum field at [40..44])
    let header = &apr_data[0..HEADER_SIZE];
    let recomputed = compute_apr_header_checksum(header);
    assert_eq!(stored_checksum, recomputed, "CRC32 checksum mismatch");
}

// ============================================================================
// GQA model (different num_kv_heads)
// ============================================================================

#[test]
fn test_q4k_convert_gqa_model() {
    // GQA: 8 query heads, 2 KV heads (ratio 4)
    let gguf_data = build_minimal_llama_gguf(32, 128, 256, 8, 2);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    let stats = GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");
    assert_eq!(stats.hidden_size, 128);

    let apr_data = std::fs::read(&apr_path).expect("read APR");
    let metadata_offset = u64::from_le_bytes(apr_data[12..20].try_into().unwrap()) as usize;
    let metadata_len = u32::from_le_bytes(apr_data[20..24].try_into().unwrap()) as usize;
    let metadata: serde_json::Value =
        serde_json::from_slice(&apr_data[metadata_offset..metadata_offset + metadata_len])
            .expect("parse metadata");

    assert_eq!(metadata["num_attention_heads"], 8);
    assert_eq!(metadata["num_key_value_heads"], 2);
}

// ============================================================================
// Error paths
// ============================================================================

#[test]
fn test_q4k_convert_missing_input_file() {
    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let result = GgufToAprQ4KConverter::convert(
        std::path::Path::new("/nonexistent/model.gguf"),
        tmp_out.path(),
    );
    assert!(result.is_err(), "should fail for missing input");
}

#[test]
fn test_q4k_convert_invalid_gguf_data() {
    let bad_data = vec![0xFF; 200];
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&bad_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let result = GgufToAprQ4KConverter::convert(&gguf_path, tmp_out.path());
    assert!(result.is_err(), "should fail for invalid GGUF");
}

#[test]
fn test_q4k_convert_empty_gguf() {
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&[]);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let result = GgufToAprQ4KConverter::convert(&gguf_path, tmp_out.path());
    assert!(result.is_err(), "should fail for empty GGUF");
}

// ============================================================================
// Roundtrip: convert GGUF -> APR Q4K -> load APR back
// ============================================================================

#[test]
fn test_q4k_convert_roundtrip_loadable() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    // Verify APR file can be loaded back as AprTransformer
    let apr_data = std::fs::read(&apr_path).expect("read APR");
    let result = crate::apr_transformer::AprTransformer::from_apr_bytes(&apr_data);
    assert!(
        result.is_ok(),
        "APR file should be loadable: {:?}",
        result.err()
    );

    let transformer = result.unwrap();
    assert_eq!(transformer.config.architecture, "llama");
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 1);
}

// ============================================================================
// Dtype coverage: verify Q4K tensors are correctly typed in output
// ============================================================================

#[test]
fn test_q4k_convert_has_quantized_flag() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    let apr_data = std::fs::read(&apr_path).expect("read APR");
    // Flags at offset [6..8]
    let flags = u16::from_le_bytes(apr_data[6..8].try_into().unwrap());
    assert_eq!(flags & 0x0020, 0x0020, "QUANTIZED flag should be set");
}

#[test]
fn test_q4k_convert_q4k_tensor_count_correct() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    let stats = GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    // build_minimal_llama_gguf has 7 Q4K tensors:
    // attn_q, attn_k, attn_v, attn_output, ffn_up, ffn_down, ffn_gate
    assert_eq!(stats.q4k_tensor_count, 7, "should have 7 Q4K tensors");
}

// ============================================================================
// Rope type inference
// ============================================================================

#[test]
fn test_q4k_convert_rope_type_in_metadata() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let (_tmp_in, gguf_path) = write_gguf_to_tempfile(&gguf_data);

    let tmp_out = tempfile::NamedTempFile::new().expect("create temp APR");
    let apr_path = tmp_out.path().to_path_buf();

    GgufToAprQ4KConverter::convert(&gguf_path, &apr_path).expect("should convert");

    let apr_data = std::fs::read(&apr_path).expect("read APR");
    let metadata_offset = u64::from_le_bytes(apr_data[12..20].try_into().unwrap()) as usize;
    let metadata_len = u32::from_le_bytes(apr_data[20..24].try_into().unwrap()) as usize;
    let metadata: serde_json::Value =
        serde_json::from_slice(&apr_data[metadata_offset..metadata_offset + metadata_len])
            .expect("parse metadata");

    // LLaMA uses NEOX style (type 2) or NORM (type 0) depending on metadata
    assert!(
        metadata.get("rope_type").is_some(),
        "rope_type should be in metadata"
    );
}
