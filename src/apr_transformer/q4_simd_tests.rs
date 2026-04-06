
// =============================================================================
// Tests for SIMD Q4 Transformer (PMAT-802: T-COV-95)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // QuantizedAprTensorQ4 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_quantized_tensor_q4_new() {
        let data = vec![0u8; 36]; // 2 blocks * 18 bytes
        let tensor = QuantizedAprTensorQ4::new(data.clone(), 32, 2);
        assert_eq!(tensor.data.len(), 36);
        assert_eq!(tensor.in_dim, 32);
        assert_eq!(tensor.out_dim, 2);
    }

    #[test]
    fn test_quantized_tensor_q4_zeros_small() {
        let tensor = QuantizedAprTensorQ4::zeros(32, 1); // 32 elements = 1 block
        assert_eq!(tensor.in_dim, 32);
        assert_eq!(tensor.out_dim, 1);
        // 1 block = 18 bytes
        assert_eq!(tensor.data.len(), 18);
        assert!(tensor.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_quantized_tensor_q4_zeros_multiple_blocks() {
        let tensor = QuantizedAprTensorQ4::zeros(64, 2); // 128 elements = 4 blocks
        // 4 blocks = 72 bytes
        assert_eq!(tensor.data.len(), 72);
    }

    #[test]
    fn test_quantized_tensor_q4_zeros_partial_block() {
        let tensor = QuantizedAprTensorQ4::zeros(33, 1); // 33 elements = 2 blocks (rounds up)
        // 2 blocks = 36 bytes
        assert_eq!(tensor.data.len(), 36);
    }

    #[test]
    fn test_quantized_tensor_q4_expected_bytes() {
        // 32 elements = 1 block = 18 bytes
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(32), 18);
        // 33 elements = 2 blocks = 36 bytes
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(33), 36);
        // 64 elements = 2 blocks = 36 bytes
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(64), 36);
        // 256 elements = 8 blocks = 144 bytes
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(256), 144);
    }

    #[test]
    fn test_quantized_tensor_q4_expected_bytes_zero() {
        // 0 elements = 0 blocks = 0 bytes
        assert_eq!(QuantizedAprTensorQ4::expected_bytes(0), 0);
    }

    // -------------------------------------------------------------------------
    // AprInferenceScratch Tests
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
    fn test_inference_scratch_from_config() {
        let config = make_test_config();
        let scratch = AprInferenceScratch::from_config(&config);

        // Check buffer sizes match config
        assert_eq!(scratch.hidden.len(), 64); // hidden_dim
        assert_eq!(scratch.normed.len(), 64); // hidden_dim
        assert_eq!(scratch.qkv_out.len(), 192); // hidden_dim * 3
        assert_eq!(scratch.q.len(), 64); // hidden_dim
        assert_eq!(scratch.k.len(), 64); // hidden_dim
        assert_eq!(scratch.v.len(), 64); // hidden_dim
        assert_eq!(scratch.attn_out.len(), 64); // hidden_dim
        assert_eq!(scratch.ffn_input.len(), 64); // hidden_dim
        assert_eq!(scratch.ffn_up.len(), 128); // intermediate_dim
        assert_eq!(scratch.ffn_gate.len(), 128); // intermediate_dim
        assert_eq!(scratch.ffn_out.len(), 64); // hidden_dim
    }

    #[test]
    fn test_inference_scratch_initialized_to_zero() {
        let config = make_test_config();
        let scratch = AprInferenceScratch::from_config(&config);

        // All buffers should be zero-initialized
        assert!(scratch.hidden.iter().all(|&x| x == 0.0));
        assert!(scratch.normed.iter().all(|&x| x == 0.0));
        assert!(scratch.qkv_out.iter().all(|&x| x == 0.0));
        assert!(scratch.attn_out.iter().all(|&x| x == 0.0));
        assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_inference_scratch_clear() {
        let config = make_test_config();
        let mut scratch = AprInferenceScratch::from_config(&config);

        // Modify buffers
        scratch.hidden[0] = 1.0;
        scratch.normed[0] = 2.0;
        scratch.attn_out[0] = 3.0;
        scratch.ffn_up[0] = 4.0;

        // Clear and verify
        scratch.clear();
        assert!(scratch.hidden.iter().all(|&x| x == 0.0));
        assert!(scratch.normed.iter().all(|&x| x == 0.0));
        assert!(scratch.attn_out.iter().all(|&x| x == 0.0));
        assert!(scratch.ffn_up.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_inference_scratch_large_config() {
        let config = AprTransformerConfig {
            architecture: "apr".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        };
        let scratch = AprInferenceScratch::from_config(&config);

        assert_eq!(scratch.hidden.len(), 4096);
        assert_eq!(scratch.ffn_up.len(), 11008);
        assert_eq!(scratch.ffn_gate.len(), 11008);
    }
}
