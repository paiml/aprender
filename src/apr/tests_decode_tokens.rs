//! Additional tests for APR module - Part 03
//!
//! This file provides comprehensive tests for uncovered functions in mod.rs and submodules.
//! Focus: dtype utilities, embedded tokenizer metadata, inference helpers, BPE tokenizer.

#[cfg(test)]
mod tests {
    use crate::apr::*;
    use std::collections::HashMap;

    #[test]
    fn test_decode_tokens_with_bpe_special_chars() {
        let vocab = vec!["Ġhello".to_string(), "Ċ".to_string(), "ĉtab".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert_eq!(text, " hello\n\ttab");
    }

    #[test]
    fn test_decode_tokens_out_of_bounds() {
        let vocab = vec!["test".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[0, 100]);
        assert!(text.contains("test"));
        assert!(text.contains("[100]"));
    }

    #[test]
    fn test_decode_tokens_empty() {
        let vocab = vec!["test".to_string()];
        let text = AprV2Model::decode_tokens(&vocab, &[]);
        assert!(text.is_empty());
    }

    // =========================================================================
    // AprV2Model predict tests (linear model inference)
    // =========================================================================

    #[test]
    fn test_predict_empty_model() {
        // Create minimal APR with no tensors
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load model");
        let output = model.predict(&[1.0, 2.0, 3.0]).expect("predict");

        // Empty model sums inputs
        assert_eq!(output.len(), 1);
        assert!((output[0] - 6.0).abs() < 0.001);
    }

    // =========================================================================
    // AprV2Model forward error cases
    // =========================================================================

    #[test]
    fn test_forward_empty_tokens() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load");
        let result = model.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_non_transformer() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load");
        let result = model.forward(&[1, 2, 3]);
        assert!(result.is_err());
    }

    // =========================================================================
    // AprV2Model generate error case
    // =========================================================================

    #[test]
    fn test_generate_empty_input() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        let model = AprV2Model::from_bytes(data).expect("load");
        let result = model.generate(&[], 10, None);
        assert!(result.is_err());
    }

    // =========================================================================
    // simd_dot tests
    // =========================================================================

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!((result - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_empty() {
        let result = simd_dot(&[], &[]);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let result = simd_dot(&a, &b);
        // Only first 2 elements are used
        assert!((result - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_large_vector() {
        // Test AVX2 path with >8 elements
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 32];
        let result = simd_dot(&a, &b);
        // sum of 0..31 = 496
        assert!((result - 496.0).abs() < 0.01);
    }

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_negative_zero() {
        // Negative zero: sign=1, exp=0, mant=0 -> 0x8000
        let result = f16_to_f32(0x8000);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_one() {
        // f16 for 1.0 = 0x3C00
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        // f16 for -1.0 = 0xBC00
        let result = f16_to_f32(0xBC00);
        assert!((result + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_positive_inf() {
        // +Inf: sign=0, exp=31, mant=0 -> 0x7C00
        let result = f16_to_f32(0x7C00);
        assert!(result.is_infinite());
        assert!(result.is_sign_positive());
    }

    #[test]
    fn test_f16_to_f32_negative_inf() {
        // -Inf: sign=1, exp=31, mant=0 -> 0xFC00
        let result = f16_to_f32(0xFC00);
        assert!(result.is_infinite());
        assert!(result.is_sign_negative());
    }
include!("tests_dtype_and_tokenizer.rs");
}
