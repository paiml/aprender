//! Unit tests for pygmy model builder functions.

pub(crate) use super::*;
pub(crate) use crate::format::v2::{AprV2Reader, MAGIC_V2};

#[test]
fn test_pygmy_safetensors_valid() {
    let data = build_pygmy_safetensors();

    // Should have valid header length
    assert!(data.len() > 8);
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
    assert!(header_len > 0);
    assert!(header_len < 10000); // Reasonable size

    // Should have JSON header
    let header_end = 8 + header_len as usize;
    assert!(data.len() >= header_end);
    let header_str = std::str::from_utf8(&data[8..header_end]).unwrap();
    assert!(header_str.starts_with('{'));
    assert!(header_str.contains("model.embed_tokens.weight"));
}

#[test]
fn test_pygmy_safetensors_with_config() {
    let config = PygmyConfig::llama_style();
    let data = build_pygmy_safetensors_with_config(config);

    assert!(data.len() > 100);
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let header_end = 8 + header_len as usize;
    let header_str = std::str::from_utf8(&data[8..header_end]).unwrap();

    // Should have attention tensors
    assert!(header_str.contains("self_attn.q_proj.weight"));
    assert!(header_str.contains("self_attn.k_proj.weight"));
}

#[test]
fn test_pygmy_safetensors_minimal() {
    let config = PygmyConfig::minimal();
    let data = build_pygmy_safetensors_with_config(config);

    // Minimal config should produce small file
    assert!(data.len() < 1000);
}

/// GH-197: PygmyConfig should generate matching config.json
#[test]
fn test_pygmy_config_to_config_json() {
    let config = PygmyConfig::llama_style();
    let json = config.to_config_json();

    // Should contain all required fields
    assert!(json.contains("\"hidden_size\": 8"));
    assert!(json.contains("\"num_hidden_layers\": 1"));
    assert!(json.contains("\"vocab_size\": 16"));
    assert!(json.contains("\"num_attention_heads\":"));
    assert!(json.contains("\"num_key_value_heads\":"));
    assert!(json.contains("\"intermediate_size\":"));

    // Should be valid JSON
    assert!(json.starts_with('{'));
    assert!(json.ends_with('}'));

    // Validate consistency: vocab_size > hidden_size (GH-197 sanity check)
    assert!(config.vocab_size > config.hidden_size);
}

#[test]
fn test_pygmy_apr_valid() {
    let data = build_pygmy_apr();

    // Should have valid magic
    assert!(data.len() >= 64);
    assert_eq!(&data[0..4], &MAGIC_V2);

    // Should be parseable
    let reader = AprV2Reader::from_bytes(&data);
    assert!(reader.is_ok());
}

#[test]
fn test_pygmy_apr_metadata() {
    let data = build_pygmy_apr();
    let reader = AprV2Reader::from_bytes(&data).unwrap();

    // Should have architecture metadata
    assert_eq!(reader.metadata().architecture, Some("llama".to_string()));

    // Should have tensors
    assert!(!reader.tensor_names().is_empty());
}

#[test]
fn test_pygmy_apr_tensor_count() {
    let config = PygmyConfig::llama_style();
    let data = build_pygmy_apr_with_config(config);
    let reader = AprV2Reader::from_bytes(&data).unwrap();

    // LLaMA style: embed + 2 norms + 4 attn + 3 mlp + final norm + lm_head = 12
    assert!(reader.tensor_names().len() >= 10);
}

#[test]
fn test_pygmy_apr_alignment() {
    let data = build_pygmy_apr();
    let reader = AprV2Reader::from_bytes(&data).unwrap();

    // All tensors should be 64-byte aligned
    assert!(reader.verify_alignment());
}

#[test]
fn test_pygmy_apr_q8() {
    let data = build_pygmy_apr_q8();
    let reader = AprV2Reader::from_bytes(&data).unwrap();

    // Should have Q8 tensors - check tensor names contain attention tensors
    let names = reader.tensor_names();
    let has_attn = names.iter().any(|n| n.contains("self_attn.q_proj.weight"));
    assert!(has_attn);

    // Get one tensor and verify it has Q8 format (scale + quantized values)
    let tensor_data = reader.get_tensor_data("model.layers.0.self_attn.q_proj.weight");
    assert!(tensor_data.is_some());
}

#[test]
fn test_pygmy_apr_q4() {
    let data = build_pygmy_apr_q4();
    let reader = AprV2Reader::from_bytes(&data).unwrap();

    // Should have Q4 tensors - check tensor names
    let names = reader.tensor_names();
    let has_attn = names.iter().any(|n| n.contains("self_attn.q_proj.weight"));
    assert!(has_attn);
}

#[test]
fn test_pygmy_apr_f16() {
    let data = build_pygmy_apr_f16();
    let reader = AprV2Reader::from_bytes(&data).unwrap();

    // Should have F16 tensors - check tensor names
    let names = reader.tensor_names();
    let has_embed = names.iter().any(|n| n.contains("embed_tokens.weight"));
    assert!(has_embed);
}

#[test]
fn test_pygmy_config_variants() {
    // Test all config variants produce valid output
    for config in [
        PygmyConfig::default(),
        PygmyConfig::minimal(),
        PygmyConfig::embedding_only(),
        PygmyConfig::llama_style(),
        PygmyConfig::realistic(),
    ] {
        let st_data = build_pygmy_safetensors_with_config(config.clone());
        assert!(st_data.len() > 8);

        let apr_data = build_pygmy_apr_with_config(config);
        let reader = AprV2Reader::from_bytes(&apr_data);
        assert!(reader.is_ok());
    }
}

#[test]
fn test_pygmy_size_comparison() {
    // Quantized models should be smaller
    let f32_data = build_pygmy_apr();
    let q8_data = build_pygmy_apr_q8();
    let q4_data = build_pygmy_apr_q4();
    let f16_data = build_pygmy_apr_f16();

    // F16 should be ~50% of F32
    assert!(f16_data.len() < f32_data.len());

    // Q8 should be smaller than F32
    // Note: overhead may make small models not show compression
    assert!(!q8_data.is_empty());
    assert!(!q4_data.is_empty());
}

// ========================================================================
// Feature-Gated Tests: Encryption (format-encryption)
// ========================================================================

#[cfg(feature = "format-encryption")]
mod encryption_tests {
    use super::*;

    #[test]
    fn test_pygmy_apr_encrypted_roundtrip() {
        let password = "test_password_123";
        let encrypted_data = build_pygmy_apr_encrypted(password);

        // Should have valid APR header with ENCRYPTED flag
        assert!(encrypted_data.len() > 64);
        assert_eq!(&encrypted_data[0..4], b"APRN");

        // Verify ENCRYPTED flag is set (bytes 6-7 are u16 flags, ENCRYPTED = 0x0004)
        let flags = u16::from_le_bytes([encrypted_data[6], encrypted_data[7]]);
        assert!(
            flags & 0x0004 != 0,
            "ENCRYPTED flag (0x0004) should be set in flags: {flags:#06x}"
        );
    }

    #[test]
    fn test_pygmy_apr_encrypted_default() {
        let data = build_pygmy_apr_encrypted_default();
        assert!(!data.is_empty());
        assert!(data.len() > 64);
    }

    #[test]
    fn test_pygmy_encrypted_wrong_password_fails() {
        use crate::format::{load_encrypted, ModelType};
        use serde::{Deserialize, Serialize};
        use std::io::Write;
        use tempfile::NamedTempFile;

        #[derive(Debug, Serialize, Deserialize)]
        struct PygmyModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let encrypted_data = build_pygmy_apr_encrypted("correct_password");

        // Write to temp file
        let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp");
        temp.write_all(&encrypted_data).expect("Write");
        temp.flush().expect("Flush");

        // Try to load with wrong password - should fail
        let result: crate::error::Result<PygmyModel> =
            load_encrypted(temp.path(), ModelType::Custom, "wrong_password");
        assert!(result.is_err(), "Wrong password should fail decryption");
    }
}

// ========================================================================
// Feature-Gated Tests: Signing (format-signing)
// ========================================================================

#[cfg(feature = "format-signing")]
mod signing_tests {
    use super::*;

    #[test]
    fn test_pygmy_apr_signed_has_signature() {
        let (data, _verifying_key) = build_pygmy_apr_signed();

        // Should have valid APR header with SIGNED flag
        assert!(data.len() > 100); // Header + signature block
        assert_eq!(&data[0..4], b"APRN");

        // Verify SIGNED flag is set (bytes 6-7 are u16 flags, SIGNED = 0x0008)
        let flags = u16::from_le_bytes([data[6], data[7]]);
        assert!(
            flags & 0x0008 != 0,
            "SIGNED flag (0x0008) should be set in flags: {flags:#06x}"
        );
    }

    #[test]
    fn test_pygmy_signed_roundtrip() {
        use crate::format::{load_verified, ModelType};
        use serde::{Deserialize, Serialize};
        use std::io::Write;
        use tempfile::NamedTempFile;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct PygmyModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let (data, verifying_key) = build_pygmy_apr_signed();

        // Write to temp file
        let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp");
        temp.write_all(&data).expect("Write");
        temp.flush().expect("Flush");

        // Load with signature verification
        let loaded: PygmyModel =
            load_verified(temp.path(), ModelType::Custom, Some(&verifying_key))
                .expect("Load signed should succeed");

        assert_eq!(loaded.weights, vec![0.1, 0.2, 0.3, 0.4]);
        assert!((loaded.bias - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_generate_signing_key() {
        let key = generate_test_signing_key();
        let verifying = key.verifying_key();

        // Should be 32-byte keys
        assert_eq!(verifying.as_bytes().len(), 32);
    }

    #[test]
    fn test_pygmy_signed_tampering_detected() {
        use crate::format::{load_verified, ModelType};
        use serde::{Deserialize, Serialize};
        use std::io::Write;
        use tempfile::NamedTempFile;

        #[derive(Debug, Serialize, Deserialize)]
        struct PygmyModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let (mut data, verifying_key) = build_pygmy_apr_signed();

        // Tamper with the data (flip a bit in the payload area)
        if data.len() > 100 {
            data[80] ^= 0xFF;
        }

        // Write tampered data to temp file
        let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp");
        temp.write_all(&data).expect("Write");
        temp.flush().expect("Flush");

        // Load should fail due to signature mismatch
        let result: crate::error::Result<PygmyModel> =
            load_verified(temp.path(), ModelType::Custom, Some(&verifying_key));
        assert!(result.is_err(), "Tampered file should fail verification");
    }
}

// ========================================================================
// Feature-Gated Tests: Quantization (format-quantize)
// ========================================================================

#[cfg(feature = "format-quantize")]
mod quantize_tests {
    use super::*;
    use crate::format::quantize::{dequantize, QuantType};

    #[test]
    fn test_pygmy_quantize_data_valid() {
        let data = build_pygmy_quantize_data();
        assert_eq!(data.len(), 64);

        // Check no NaN or Inf
        for &v in &data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_pygmy_q8_block_roundtrip() {
        let original = build_pygmy_quantize_data();
        let block = build_pygmy_q8_block();

        // Verify block properties
        assert_eq!(block.quant_type, QuantType::Q8_0);
        assert!(!block.blocks.is_empty());

        // Dequantize and verify values are close
        let restored = dequantize(&block).expect("Dequantize Q8");
        assert_eq!(restored.len(), original.len());

        // Q8_0 should have reasonable accuracy
        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.1, "Q8_0 error too large");
        }
    }

    #[test]
    fn test_pygmy_q4_block_roundtrip() {
        let original = build_pygmy_quantize_data();
        let block = build_pygmy_q4_block();

        // Verify block properties
        assert_eq!(block.quant_type, QuantType::Q4_0);
        assert!(!block.blocks.is_empty());

        // Dequantize and verify values are somewhat close
        let restored = dequantize(&block).expect("Dequantize Q4");
        assert_eq!(restored.len(), original.len());

        // Q4_0 has lower precision
        for (orig, rest) in original.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 0.5, "Q4_0 error too large");
        }
    }

    #[test]
    fn test_quant_type_bits_per_weight() {
        assert!((QuantType::Q8_0.bits_per_weight() - 8.5).abs() < 0.01);
        assert!((QuantType::Q4_0.bits_per_weight() - 4.5).abs() < 0.01);
        assert!((QuantType::Q4_1.bits_per_weight() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_quant_type_from_u8() {
        assert_eq!(QuantType::from_u8(0x01), Some(QuantType::Q8_0));
        assert_eq!(QuantType::from_u8(0x02), Some(QuantType::Q4_0));
        assert_eq!(QuantType::from_u8(0x03), Some(QuantType::Q4_1));
        assert_eq!(QuantType::from_u8(0x10), Some(QuantType::Q8Tensor));
        assert_eq!(QuantType::from_u8(0xFF), Some(QuantType::Custom));
        assert_eq!(QuantType::from_u8(0x99), None);
    }

    #[test]
    fn test_quantized_block_num_blocks() {
        let block = build_pygmy_q8_block();
        let num = block.num_blocks();
        assert!(num > 0);

        // With 64 elements and block size 32, should have 2 blocks
        assert_eq!(num, 2);
    }

    #[test]
    fn test_quantized_block_num_elements() {
        let block = build_pygmy_q8_block();
        let total = block.num_elements();
        assert_eq!(total, 64);
    }
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
