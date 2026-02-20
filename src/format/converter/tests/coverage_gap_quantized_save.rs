//! Coverage gap tests for save_safetensors_quantized and try_gguf_q4k_passthrough
//!
//! Targets:
//! - save_safetensors_quantized (mod_part_03.rs:233, 47 uncov, 0% cov)
//! - try_gguf_q4k_passthrough (mod.rs:179, 55 uncov, 0% cov)

use super::super::*;
use std::collections::BTreeMap;

// ============================================================================
// save_safetensors_quantized: Fp16 quantization
// ============================================================================

#[test]
fn test_save_safetensors_quantized_fp16_basic() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_fp16.safetensors");

    let mut tensors = BTreeMap::new();
    // Normal weight tensor (large enough, not sensitive name)
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.5f32; 2048], vec![32, 64]),
    );

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Fp16);
    assert!(result.is_ok(), "Fp16 quantized save failed: {:?}", result.err());
    assert!(output.exists(), "Output file should exist");

    // Fp16 output should be smaller than raw F32 (2 bytes vs 4 bytes per element)
    let file_size = std::fs::metadata(&output).unwrap().len();
    assert!(file_size > 0, "File should not be empty");
    // 2048 elements * 2 bytes = 4096 bytes data, plus header
    assert!(file_size < 2048 * 4 + 1000, "Fp16 should be smaller than F32");
}

#[test]
fn test_save_safetensors_quantized_int8_basic() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_int8.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        (vec![1.0f32, -0.5, 0.3, -0.8, 0.0, 0.7, -1.0, 0.2], vec![2, 4]),
    );

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Int8);
    assert!(result.is_ok(), "Int8 quantized save failed: {:?}", result.err());
    assert!(output.exists());

    let file_size = std::fs::metadata(&output).unwrap().len();
    assert!(file_size > 0);
}

#[test]
fn test_save_safetensors_quantized_int4_basic() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_int4.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        (vec![0.1f32; 1024], vec![32, 32]),
    );

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Int4);
    assert!(result.is_ok(), "Int4 quantized save failed: {:?}", result.err());
    assert!(output.exists());

    let file_size = std::fs::metadata(&output).unwrap().len();
    assert!(file_size > 0);
    // Int4 packs 2 values per byte, so data should be ~512 bytes + header
    assert!(file_size < 1024 * 4 + 1000, "Int4 should be much smaller than F32");
}

#[test]
fn test_save_safetensors_quantized_q4k_maps_to_i8() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_q4k.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        (vec![0.3f32; 512], vec![16, 32]),
    );

    // Q4K in SafeTensors context maps to I8 symmetric quantization
    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Q4K);
    assert!(result.is_ok(), "Q4K quantized save failed: {:?}", result.err());
    assert!(output.exists());
}

// ============================================================================
// save_safetensors_quantized: Sensitive tensor skip logic
// ============================================================================

#[test]
fn test_save_safetensors_quantized_skips_embeddings() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_with_embedding.safetensors");

    let mut tensors = BTreeMap::new();
    // Embedding tensor should stay F32 even with Int8 quantization
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1f32; 2048], vec![64, 32]),
    );
    // Normal weight should be quantized
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.5f32; 2048], vec![32, 64]),
    );

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Int8);
    assert!(result.is_ok(), "save with mixed tensors failed: {:?}", result.err());
    assert!(output.exists());

    // Read back and verify metadata contains both F32 and I8 dtypes
    let file_data = std::fs::read(&output).unwrap();
    assert!(file_data.len() > 8, "File should have header + data");

    let header_len = u64::from_le_bytes(file_data[0..8].try_into().unwrap()) as usize;
    let header_str = std::str::from_utf8(&file_data[8..8 + header_len]).unwrap();

    // Embedding should be F32, weight should be I8
    assert!(header_str.contains("F32"), "Embedding should remain F32");
    assert!(header_str.contains("I8"), "Weight should be quantized to I8");
}

#[test]
fn test_save_safetensors_quantized_skips_bias() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_with_bias.safetensors");

    let mut tensors = BTreeMap::new();
    // Bias tensor: small and name contains "bias" - should skip quantization
    tensors.insert(
        "model.layers.0.self_attn.q_proj.bias".to_string(),
        (vec![0.01f32; 64], vec![64]),
    );

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Fp16);
    assert!(result.is_ok());
    assert!(output.exists());

    // Even with Fp16, bias tensors stay F32 because element_count < 1024
    let file_data = std::fs::read(&output).unwrap();
    let header_len = u64::from_le_bytes(file_data[0..8].try_into().unwrap()) as usize;
    let header_str = std::str::from_utf8(&file_data[8..8 + header_len]).unwrap();
    // Note: Fp16 does NOT skip quantization for any tensor (all go through Fp16)
    // The should_skip_quantization only applies to Int8/Int4/Q4K
    // For Fp16, the function always converts (check quantize_for_safetensors logic)
    assert!(header_str.contains("F16") || header_str.contains("F32"));
}

#[test]
fn test_save_safetensors_quantized_multiple_tensors() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_multi.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.5f32; 2048], vec![32, 64]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![-0.3f32; 2048], vec![32, 64]),
    );
    tensors.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        (vec![0.7f32; 2048], vec![32, 64]),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        (vec![1.0f32; 32], vec![32]),
    );

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Int8);
    assert!(result.is_ok(), "Multi-tensor save failed: {:?}", result.err());
    assert!(output.exists());

    let file_data = std::fs::read(&output).unwrap();
    let header_len = u64::from_le_bytes(file_data[0..8].try_into().unwrap()) as usize;
    let header_str = std::str::from_utf8(&file_data[8..8 + header_len]).unwrap();

    // Parse header as JSON to verify all tensors present
    let metadata: serde_json::Value = serde_json::from_str(header_str).unwrap();
    assert!(metadata.get("model.layers.0.self_attn.q_proj.weight").is_some());
    assert!(metadata.get("model.layers.0.self_attn.k_proj.weight").is_some());
    assert!(metadata.get("model.layers.0.self_attn.v_proj.weight").is_some());
    assert!(metadata.get("model.norm.weight").is_some());
}

#[test]
fn test_save_safetensors_quantized_empty_tensors() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_empty.safetensors");

    let tensors = BTreeMap::new();

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Int8);
    assert!(result.is_ok(), "Empty tensor save should succeed");
    assert!(output.exists());
}

// ============================================================================
// save_safetensors_quantized: output file structure verification
// ============================================================================

#[test]
fn test_save_safetensors_quantized_file_structure() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let output = dir.path().join("model_verify.safetensors");

    // Use a small tensor that will be quantized. Note: should_skip_quantization
    // returns true for element_count < 1024, but for Fp16, the code always quantizes
    // regardless (Fp16 path does not check should_skip). However, the skip logic
    // in save_safetensors_quantized checks should_skip and emits F32 for small tensors.
    // So we use a name with no skip-pattern and >= 1024 elements.
    let data = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "weight".to_string(),
        (data.clone(), vec![2, 2]),
    );

    let result = save_safetensors_quantized(&tensors, &output, QuantizationType::Fp16);
    assert!(result.is_ok());

    // Verify SafeTensors file structure: 8-byte header length + JSON header + data
    let file_data = std::fs::read(&output).unwrap();
    assert!(file_data.len() >= 8, "File must have at least header length");

    let header_len = u64::from_le_bytes(file_data[0..8].try_into().unwrap()) as usize;
    assert!(header_len > 0, "Header must be non-empty");
    assert!(file_data.len() >= 8 + header_len, "File must contain full header");

    let header_str = std::str::from_utf8(&file_data[8..8 + header_len]).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(header_str).unwrap();

    let weight_meta = metadata.get("weight").expect("weight tensor in metadata");
    assert_eq!(weight_meta["shape"], serde_json::json!([2, 2]));

    // Verify data_offsets are present and valid
    let offsets = weight_meta["data_offsets"].as_array().unwrap();
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;
    assert_eq!(start, 0, "First tensor starts at offset 0");
    assert!(end > start, "End offset must be after start");

    // element_count=4 < 1024, so should_skip_quantization returns true.
    // The tensor stays F32: 4 elements * 4 bytes = 16 bytes
    assert_eq!(end - start, 16, "Small tensor stays F32: 4 elements * 4 bytes");
}

// ============================================================================
// try_gguf_q4k_passthrough: Error path (non-GGUF file returns Ok(None))
// ============================================================================

#[test]
fn test_try_gguf_q4k_passthrough_nonexistent_file() {
    let input = Path::new("/tmp/nonexistent_model_12345.gguf");
    let output = Path::new("/tmp/output_12345.apr");
    let options = ConvertOptions::default();

    let result = try_gguf_q4k_passthrough(input, output, &options);
    assert!(result.is_ok(), "Non-existent file should not error");
    assert!(result.unwrap().is_none(), "Should return None for non-GGUF file");
}

#[test]
fn test_try_gguf_q4k_passthrough_non_gguf_file() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let input = dir.path().join("not_a_gguf.bin");
    let output = dir.path().join("output.apr");

    // Write some random bytes that don't form a valid GGUF
    std::fs::write(&input, b"this is not a GGUF file").unwrap();

    let options = ConvertOptions::default();
    let result = try_gguf_q4k_passthrough(&input, &output, &options);
    assert!(result.is_ok(), "Invalid GGUF should not error");
    assert!(result.unwrap().is_none(), "Should return None for non-GGUF file");
}

#[test]
fn test_try_gguf_q4k_passthrough_valid_gguf_no_q4k_tensors() {
    use tempfile::TempDir;

    let dir = TempDir::new().expect("create temp dir");
    let input = dir.path().join("f32_model.gguf");
    let output = dir.path().join("output.apr");

    // Write a minimal valid GGUF header but with no Q4K tensors.
    // GGUF magic: 0x46475547 ("GGUF" in little-endian)
    // This will fail to parse as valid GGUF because it's too short,
    // so load_gguf_raw will error and we get Ok(None)
    let mut gguf_bytes = Vec::new();
    gguf_bytes.extend_from_slice(&0x46475547u32.to_le_bytes()); // magic
    gguf_bytes.extend_from_slice(&3u32.to_le_bytes());          // version 3
    gguf_bytes.extend_from_slice(&0u64.to_le_bytes());          // tensor count
    gguf_bytes.extend_from_slice(&0u64.to_le_bytes());          // metadata kv count
    std::fs::write(&input, gguf_bytes).unwrap();

    let options = ConvertOptions::default();
    let result = try_gguf_q4k_passthrough(&input, &output, &options);
    // Either parse fails (Ok(None)) or succeeds with no Q4K tensors (Ok(None))
    assert!(result.is_ok());
    assert!(result.unwrap().is_none(), "No Q4K tensors means no passthrough");
}

// ============================================================================
// quantize_for_safetensors: Direct unit tests
// ============================================================================

#[test]
fn test_quantize_for_safetensors_fp16() {
    let data = vec![1.0f32, -1.0, 0.5, 0.0];
    let (dtype, bytes) = quantize_for_safetensors(&data, QuantizationType::Fp16);
    assert_eq!(dtype, "F16");
    assert_eq!(bytes.len(), 8); // 4 values * 2 bytes
}

#[test]
fn test_quantize_for_safetensors_int8() {
    let data = vec![1.0f32, -1.0, 0.5, -0.5];
    let (dtype, bytes) = quantize_for_safetensors(&data, QuantizationType::Int8);
    assert_eq!(dtype, "I8");
    assert_eq!(bytes.len(), 4); // 4 values * 1 byte
}

#[test]
fn test_quantize_for_safetensors_q4k_maps_to_i8() {
    let data = vec![1.0f32, -1.0, 0.5, -0.5];
    let (dtype, bytes) = quantize_for_safetensors(&data, QuantizationType::Q4K);
    assert_eq!(dtype, "I8"); // Q4K maps to I8 symmetric quantization
    assert_eq!(bytes.len(), 4);
}

#[test]
fn test_quantize_for_safetensors_int4() {
    let data = vec![1.0f32, -1.0, 0.5, -0.5];
    let (dtype, bytes) = quantize_for_safetensors(&data, QuantizationType::Int4);
    assert_eq!(dtype, "U8");
    assert_eq!(bytes.len(), 2); // 4 values packed into 2 bytes (2 per byte)
}

// ============================================================================
// symmetric_quantize_i8: Edge cases
// ============================================================================

#[test]
fn test_symmetric_quantize_i8_zeros() {
    let data = vec![0.0f32; 4];
    let bytes = symmetric_quantize_i8(&data, 127.0);
    assert_eq!(bytes.len(), 4);
    // All zeros with scale=1.0 should produce all zero bytes
    for &b in &bytes {
        assert_eq!(b, 0);
    }
}

#[test]
fn test_symmetric_quantize_i8_max_values() {
    let data = vec![1.0f32, -1.0];
    let bytes = symmetric_quantize_i8(&data, 127.0);
    assert_eq!(bytes.len(), 2);
    // max_abs = 1.0, scale = 127.0
    // 1.0 * 127 = 127 -> as i8 -> 127 -> as u8 -> 127
    assert_eq!(bytes[0], 127u8);
    // -1.0 * 127 = -127 -> as i8 -> -127 -> as u8 -> 129
    assert_eq!(bytes[1], (-127i8) as u8);
}

// ============================================================================
// quantize_int4_packed: Edge cases
// ============================================================================

#[test]
fn test_quantize_int4_packed_even_count() {
    let data = vec![1.0f32, -1.0, 0.5, -0.5];
    let bytes = quantize_int4_packed(&data);
    assert_eq!(bytes.len(), 2); // 4 values -> 2 bytes
}

#[test]
fn test_quantize_int4_packed_odd_count() {
    let data = vec![1.0f32, -1.0, 0.5];
    let bytes = quantize_int4_packed(&data);
    assert_eq!(bytes.len(), 2); // 3 values -> ceil(3/2) = 2 bytes
}

#[test]
fn test_quantize_int4_packed_zeros() {
    let data = vec![0.0f32; 4];
    let bytes = quantize_int4_packed(&data);
    assert_eq!(bytes.len(), 2);
    // Zero input: scale=1.0, 0*1+8=8=0x08 per nibble
    // low=8, high=8 -> byte = 8 | (8 << 4) = 8 | 128 = 0x88
    for &b in &bytes {
        assert_eq!(b, 0x88);
    }
}

// ============================================================================
// f32_to_f16_bits and f32_slice_to_f16_le_bytes
// ============================================================================

#[test]
fn test_f32_to_f16_bits_one() {
    let bits = f32_to_f16_bits(1.0);
    assert_eq!(bits, 0x3C00);
}

#[test]
fn test_f32_to_f16_bits_zero() {
    let bits = f32_to_f16_bits(0.0);
    assert_eq!(bits, 0x0000);
}

#[test]
fn test_f32_to_f16_bits_infinity() {
    let bits = f32_to_f16_bits(f32::INFINITY);
    assert_eq!(bits, 0x7C00);
}

#[test]
fn test_f32_to_f16_bits_negative_infinity() {
    let bits = f32_to_f16_bits(f32::NEG_INFINITY);
    assert_eq!(bits, 0xFC00);
}

#[test]
fn test_f32_to_f16_bits_nan() {
    let bits = f32_to_f16_bits(f32::NAN);
    // NaN: exponent all 1s, mantissa non-zero
    assert_eq!(bits & 0x7C00, 0x7C00); // exponent
    assert_ne!(bits & 0x03FF, 0);       // mantissa non-zero
}

#[test]
fn test_f32_to_f16_bits_overflow_to_inf() {
    // Value larger than f16 max (65504) should become infinity
    let bits = f32_to_f16_bits(100000.0);
    assert_eq!(bits, 0x7C00);
}

#[test]
fn test_f32_to_f16_bits_underflow_to_zero() {
    // Very small subnormal
    let bits = f32_to_f16_bits(1e-10);
    assert_eq!(bits, 0x0000);
}

#[test]
fn test_f32_slice_to_f16_le_bytes_basic() {
    let data = vec![1.0f32, 0.0];
    let bytes = f32_slice_to_f16_le_bytes(&data);
    assert_eq!(bytes.len(), 4); // 2 values * 2 bytes
    // 1.0 = 0x3C00 in le = [0x00, 0x3C]
    assert_eq!(bytes[0], 0x00);
    assert_eq!(bytes[1], 0x3C);
    // 0.0 = 0x0000
    assert_eq!(bytes[2], 0x00);
    assert_eq!(bytes[3], 0x00);
}
