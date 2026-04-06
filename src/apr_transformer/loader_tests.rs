
// ============================================================================
// Tests for APR Transformer Loaders (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AprQuantizationType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_quant_type_bits_per_weight() {
        assert!((AprQuantizationType::F32.bits_per_weight() - 32.0).abs() < 0.001);
        assert!((AprQuantizationType::Q4_K.bits_per_weight() - 4.5).abs() < 0.001);
        assert!((AprQuantizationType::Q8_0.bits_per_weight() - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_quant_type_bytes_per_block() {
        assert_eq!(AprQuantizationType::F32.bytes_per_block(), 4);
        assert_eq!(AprQuantizationType::Q4_K.bytes_per_block(), 144);
        assert_eq!(AprQuantizationType::Q8_0.bytes_per_block(), 36);
    }

    #[test]
    fn test_quant_type_values_per_block() {
        assert_eq!(AprQuantizationType::F32.values_per_block(), 1);
        assert_eq!(AprQuantizationType::Q4_K.values_per_block(), 256);
        assert_eq!(AprQuantizationType::Q8_0.values_per_block(), 32);
    }

    #[test]
    fn test_quant_type_to_byte() {
        assert_eq!(AprQuantizationType::F32.to_byte(), 0);
        assert_eq!(AprQuantizationType::Q4_K.to_byte(), 1);
        assert_eq!(AprQuantizationType::Q8_0.to_byte(), 2);
    }

    #[test]
    fn test_quant_type_from_byte() {
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
        assert_eq!(AprQuantizationType::from_byte(3), None);
        assert_eq!(AprQuantizationType::from_byte(255), None);
    }

    #[test]
    fn test_quant_type_roundtrip() {
        for qt in [
            AprQuantizationType::F32,
            AprQuantizationType::Q4_K,
            AprQuantizationType::Q8_0,
        ] {
            assert_eq!(AprQuantizationType::from_byte(qt.to_byte()), Some(qt));
        }
    }

    #[test]
    fn test_quant_type_default() {
        assert_eq!(AprQuantizationType::default(), AprQuantizationType::F32);
    }

    // -------------------------------------------------------------------------
    // QuantizedAprTransformer Tests
    // -------------------------------------------------------------------------

    fn make_test_config() -> AprTransformerConfig {
        AprTransformerConfig {
            architecture: "apr".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        }
    }

    #[test]
    fn test_quantized_transformer_new_f32() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        assert_eq!(qt.quantization_type(), AprQuantizationType::F32);
        assert!((qt.bits_per_weight() - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_quantized_transformer_new_q4k() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
        assert_eq!(qt.quantization_type(), AprQuantizationType::Q4_K);
        assert!((qt.bits_per_weight() - 4.5).abs() < 0.001);
    }

    #[test]
    fn test_quantized_transformer_new_q8_0() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q8_0);
        assert_eq!(qt.quantization_type(), AprQuantizationType::Q8_0);
        assert!((qt.bits_per_weight() - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_quantized_transformer_config() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        assert_eq!(qt.config().hidden_dim, 64);
        assert_eq!(qt.config().num_layers, 2);
        assert_eq!(qt.config().vocab_size, 100);
    }

    #[test]
    fn test_quantized_transformer_weight_bytes() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        // Should be non-zero (embeddings + layers + norm + lm_head)
        assert!(qt.weight_bytes() > 0);
    }

    #[test]
    fn test_quantized_transformer_f32_equivalent_bytes() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        // Should match num_parameters * 4
        assert_eq!(qt.f32_equivalent_bytes(), qt.num_parameters() * 4);
    }

    #[test]
    fn test_quantized_transformer_num_parameters() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        // Should be non-zero
        assert!(qt.num_parameters() > 0);
    }

    #[test]
    fn test_quantized_transformer_calculate_quantized_bytes_f32() {
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(100, AprQuantizationType::F32);
        assert_eq!(bytes, 400); // 100 * 4 bytes
    }

    #[test]
    fn test_quantized_transformer_calculate_quantized_bytes_q4k() {
        // 256 values = 1 block = 144 bytes
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(256, AprQuantizationType::Q4_K);
        assert_eq!(bytes, 144);

        // 257 values = 2 blocks = 288 bytes
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(257, AprQuantizationType::Q4_K);
        assert_eq!(bytes, 288);
    }

    #[test]
    fn test_quantized_transformer_calculate_quantized_bytes_q8_0() {
        // 32 values = 1 block = 36 bytes
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(32, AprQuantizationType::Q8_0);
        assert_eq!(bytes, 36);

        // 33 values = 2 blocks = 72 bytes
        let bytes =
            QuantizedAprTransformer::calculate_quantized_bytes(33, AprQuantizationType::Q8_0);
        assert_eq!(bytes, 72);
    }

    #[test]
    fn test_quantized_transformer_forward_empty() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let result = qt.forward(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_transformer_forward_single() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let result = qt.forward(&[1]).unwrap();
        assert_eq!(result.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_forward_multiple() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let result = qt.forward(&[1, 2, 3]).unwrap();
        assert_eq!(result.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_forward_oov_token() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        // Token ID > vocab_size should use zero embedding
        let result = qt.forward(&[999]).unwrap();
        assert_eq!(result.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_forward_with_cache() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let mut cache = AprKVCache::new(&config);
        let result = qt.forward_with_cache(1, &mut cache, 0).unwrap();
        assert_eq!(result.len(), config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_forward_with_cache_multiple() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::F32);
        let mut cache = AprKVCache::new(&config);

        // Forward multiple tokens using cache
        for i in 0..5 {
            let result = qt.forward_with_cache(i, &mut cache, i as usize).unwrap();
            assert_eq!(result.len(), config.vocab_size);
        }
    }

    #[test]
    fn test_quantized_transformer_to_bytes() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let bytes = qt.to_bytes().unwrap();
        assert!(bytes.len() >= APR_TRANSFORMER_HEADER_SIZE);
        // Verify magic
        assert_eq!(&bytes[0..4], MAGIC);
    }

    #[test]
    fn test_quantized_transformer_from_bytes() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
        let bytes = qt.to_bytes().unwrap();

        let qt2 = QuantizedAprTransformer::from_bytes(&bytes).unwrap();
        assert_eq!(qt2.quantization_type(), AprQuantizationType::Q4_K);
        assert_eq!(qt2.config().hidden_dim, config.hidden_dim);
        assert_eq!(qt2.config().vocab_size, config.vocab_size);
    }

    #[test]
    fn test_quantized_transformer_from_bytes_too_small() {
        let bytes = vec![0u8; 10];
        let result = QuantizedAprTransformer::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_transformer_from_bytes_invalid_magic() {
        let mut bytes = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];
        bytes[0..4].copy_from_slice(b"XXXX"); // Invalid magic
        let result = QuantizedAprTransformer::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_transformer_from_bytes_invalid_quant_type() {
        let config = make_test_config();
        let qt = QuantizedAprTransformer::new(config, AprQuantizationType::F32);
        let mut bytes = qt.to_bytes().unwrap();
        bytes[48] = 99; // Invalid quant type
        let result = QuantizedAprTransformer::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_transformer_roundtrip_all_types() {
        for quant_type in [
            AprQuantizationType::F32,
            AprQuantizationType::Q4_K,
            AprQuantizationType::Q8_0,
        ] {
            let config = make_test_config();
            let qt = QuantizedAprTransformer::new(config.clone(), quant_type);
            let bytes = qt.to_bytes().unwrap();
            let qt2 = QuantizedAprTransformer::from_bytes(&bytes).unwrap();
            assert_eq!(qt2.quantization_type(), quant_type);
            assert_eq!(qt2.config().hidden_dim, config.hidden_dim);
        }
    }
}
