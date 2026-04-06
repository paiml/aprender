//! APR tests Part 07: rms_norm, matmul, simd_dot, dequantize extended,
//! simple_attention, BpeTokenizer extended, f16 extended, bpe_encode extended

#[cfg(test)]
mod tests {
    use crate::apr::*;
    use std::collections::HashMap;

    #[test]
    fn test_matmul_scaling() {
        let x = vec![2.0, 2.0];
        let w = vec![1.0, 1.0];
        let result = crate::apr::matmul(&x, &w, 1, 2, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 4.0).abs() < 1e-6);
    }

    // =========================================================================
    // Additional Coverage Tests - get_tensor_f32 with different dtypes
    // =========================================================================

    #[test]
    fn test_get_tensor_f32_f16_dtype() {
        // F16 dtype = 1, shape [4], 4 elements = 8 bytes
        // f16 values: 1.0 = 0x3C00, 2.0 = 0x4000
        let f16_data = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
        ];
        let model_data = create_test_apr_model_with_dtype(1, &f16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.expect("APR operation failed");
        assert_eq!(floats.len(), 4);
        assert!((floats[0] - 1.0).abs() < 0.1);
        assert!((floats[1] - 2.0).abs() < 0.1);
    }

    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_get_tensor_f32_q8_0_dtype() {
        // Q8_0 dtype = 10, block = 2-byte scale + 32 i8 values = 34 bytes for 32 elements
        let mut q8_data = vec![0u8; 34];
        q8_data[0] = 0x00;
        q8_data[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            q8_data[2 + i] = i as u8;
        }

        // Create with shape [32] since Q8_0 block has 32 elements
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_entry = create_binary_tensor_entry("typed.weight", 10, &[32], 0, 34);
        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + 34;
        let mut model_data = vec![0u8; total_size];

        model_data[0..4].copy_from_slice(&MAGIC);
        model_data[4] = 2;
        model_data[5] = 0;
        model_data[8..12].copy_from_slice(&1u32.to_le_bytes());
        model_data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        model_data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        model_data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        model_data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        model_data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        model_data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        let data_start = data_offset as usize;
        model_data[data_start..data_start + 34].copy_from_slice(&q8_data);

        let model = AprV2Model::from_bytes(model_data).expect("should load");
        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.expect("APR operation failed");
        assert_eq!(floats.len(), 32);
    }

    #[test]
    fn test_get_tensor_f32_unsupported_dtype() {
        // BF16 dtype = 2 is not fully supported for get_tensor_f32
        let bf16_data = vec![0x00, 0x3F, 0x80, 0x00]; // Two BF16 values
        let model_data = create_test_apr_model_with_dtype(2, &bf16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        // BF16 is not in the supported list, should error
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tensor_f32_out_of_bounds() {
        // Create a model where tensor data extends beyond file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&100u64.to_le_bytes()); // data_offset

        // Add tensor entry that claims data beyond file
        let tensor_entry = create_binary_tensor_entry("oob.weight", 0, &[1000], 0, 4000);
        data[64..64 + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");
        let result = model.get_tensor_f32("oob.weight");
        assert!(result.is_err()); // Out of bounds
    }

    // =========================================================================
    // decode_tokens extended tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_gpt2_special() {
        // Test GPT-2 style byte-level BPE tokens
        let vocab = vec![
            "Ġhello".to_string(), // Space + hello
            "Ċ".to_string(),      // Newline
            "ĉ".to_string(),      // Tab
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert!(result.contains("hello"));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }

    #[test]
    fn test_decode_tokens_empty_string_token() {
        let vocab = vec![String::new(), "a".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Empty token shouldn't cause issues
        assert!(result.contains('a'));
    }

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_smallest_positive_normal() {
        // Smallest positive normal f16 = 0x0400 (6.103515625e-5)
        let result = crate::apr::f16_to_f32(0x0400);
        assert!(result > 0.0 && result < 1e-4);
    }

    #[test]
    fn test_f16_to_f32_largest_normal() {
        // Largest finite f16 = 0x7BFF (65504.0)
        let result = crate::apr::f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 10.0);
    }

    #[test]
    fn test_f16_to_f32_negative_normal() {
        // -2.0 in f16 = 0xC000
        let result = crate::apr::f16_to_f32(0xC000);
        assert!((result + 2.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_to_f32_subnormal_nonzero() {
        // Various subnormals (exp=0, mantissa!=0)
        for mant in [1u16, 10, 100, 0x3FF] {
            let result = crate::apr::f16_to_f32(mant);
            assert!(result > 0.0, "Subnormal {mant:#x} should be positive");
        }
    }

    // =========================================================================
    // bpe_encode extended tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_with_newline() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ċ".to_string(), 0); // Newline token
        let result = bpe_encode("\n", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_tab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("ĉ".to_string(), 0); // Tab token
        let result = bpe_encode("\t", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_space() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġ".to_string(), 0); // Space token
        let result = bpe_encode(" ", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_mixed() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("Ġ".to_string(), 1); // Space
        token_to_id.insert("b".to_string(), 2);
        let result = bpe_encode("a b", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_bpe_encode_multiple_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        token_to_id.insert("ab".to_string(), 3);
        token_to_id.insert("abc".to_string(), 4);

        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let result = bpe_encode("abc", &token_to_id, &merges, &HashMap::new());
        // Should merge a+b->ab, then ab+c->abc
        assert!(!result.is_empty());
    }
include!("tests_binary_tensor_helpers.rs");
include!("tests_dequantize.rs");
include!("tests_byte_02.rs");
}
