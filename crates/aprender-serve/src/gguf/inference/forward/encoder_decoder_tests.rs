//! GH-177: Tests for encoder-decoder forward pass

#[cfg(test)]
mod tests {
    use crate::gguf::test_helpers::create_q4k_test_data;
    use crate::gguf::types::GGUF_TYPE_Q4_K;
    use crate::gguf::ArchConstraints;
    use crate::gguf::OwnedQKVWeights;
    use crate::gguf::{GGUFConfig, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor};

    fn create_t5_config(hidden_dim: usize, num_layers: usize) -> GGUFConfig {
        GGUFConfig {
            architecture: "t5".to_string(),
            constraints: ArchConstraints::from_architecture("t5"),
            hidden_dim,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: hidden_dim * 4,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-6,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: Some(0),
            eos_token_id: Some(1),
        }
    }

    fn create_t5_layer(
        hidden_dim: usize,
        intermediate_dim: usize,
        num_heads: usize,
    ) -> OwnedQuantizedLayer {
        let head_dim = hidden_dim / num_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_heads * head_dim;
        let qkv_out = q_dim + 2 * kv_dim;

        OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: Some(vec![0.0f32; hidden_dim]),
            qkv_weight: OwnedQKVWeights::Fused(OwnedQuantizedTensor {
                data: vec![0u8; create_q4k_test_data(hidden_dim, qkv_out).data.len()],
                in_dim: hidden_dim,
                out_dim: qkv_out,
                qtype: GGUF_TYPE_Q4_K,
            }),
            qkv_bias: None,
            attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_test_data(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_test_data(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: None, // T5 uses GELU, no gate
            ffn_gate_bias: None,
            ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
            ffn_norm_bias: Some(vec![0.0f32; hidden_dim]),
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        }
    }

    #[test]
    fn test_encode_rejects_decoder_only() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: ArchConstraints::from_architecture("llama"),
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 64,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1f32; 100 * 32],
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0f32; 32],
            output_norm_bias: None,
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 128],
                in_dim: 32,
                out_dim: 100,
                qtype: GGUF_TYPE_Q4_K,
            },
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let result = model.encode(&[1, 2, 3]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("encoder-decoder"),
            "Expected encoder-decoder error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_decode_rejects_decoder_only() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: ArchConstraints::from_architecture("llama"),
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 64,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1f32; 100 * 32],
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0f32; 32],
            output_norm_bias: None,
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 128],
                in_dim: 32,
                out_dim: 100,
                qtype: GGUF_TYPE_Q4_K,
            },
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let encoder_out = crate::gguf::inference::forward::EncoderOutput {
            hidden_states: vec![0.0f32; 3 * 32],
            seq_len: 3,
            hidden_dim: 32,
        };
        let result = model.decode(&[1, 2], &encoder_out);
        assert!(result.is_err());
    }

    #[test]
    fn test_t5_is_encoder_decoder() {
        let config = create_t5_config(32, 1);
        assert!(config.is_encoder_decoder());
    }

    #[test]
    fn test_encode_returns_correct_shape() {
        let hidden_dim = 32;
        let config = create_t5_config(hidden_dim, 1);
        let layer = create_t5_layer(hidden_dim, hidden_dim * 4, 4);

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1f32; 100 * hidden_dim],
            position_embedding: None,
            layers: vec![layer],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0f32; hidden_dim],
            output_norm_bias: Some(vec![0.0f32; hidden_dim]),
            lm_head_weight: create_q4k_test_data(hidden_dim, 100),
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let input_tokens = vec![2u32, 5, 10];
        let result = model.encode(&input_tokens);
        assert!(result.is_ok(), "encode() failed: {:?}", result.err());

        let enc_out = result.unwrap();
        assert_eq!(enc_out.seq_len, 3);
        assert_eq!(enc_out.hidden_dim, hidden_dim);
        assert_eq!(enc_out.hidden_states.len(), 3 * hidden_dim);

        // Hidden states should be non-zero (not just embedding pass-through)
        let sum: f32 = enc_out.hidden_states.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Encoder output should be non-zero");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let hidden_dim = 32;
        let vocab_size = 100;
        let config = create_t5_config(hidden_dim, 1);
        let layer = create_t5_layer(hidden_dim, hidden_dim * 4, 4);

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1f32; vocab_size * hidden_dim],
            position_embedding: None,
            layers: vec![layer],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0f32; hidden_dim],
            output_norm_bias: Some(vec![0.0f32; hidden_dim]),
            lm_head_weight: create_q4k_test_data(hidden_dim, vocab_size),
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        // Encode
        let enc_out = model.encode(&[2, 5, 10]).expect("encode failed");
        assert_eq!(enc_out.seq_len, 3);

        // Decode
        let logits = model.decode(&[0, 3], &enc_out).expect("decode failed");
        assert_eq!(logits.len(), vocab_size, "Logits should be vocab_size");

        // Logits should be non-zero (real matmul, not placeholder)
        let sum: f32 = logits.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Logits should be non-zero after LM head");
    }

    #[test]
    fn test_encode_token_out_of_range() {
        let hidden_dim = 32;
        let config = create_t5_config(hidden_dim, 1);
        let layer = create_t5_layer(hidden_dim, hidden_dim * 4, 4);

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1f32; 100 * hidden_dim],
            position_embedding: None,
            layers: vec![layer],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0f32; hidden_dim],
            output_norm_bias: Some(vec![0.0f32; hidden_dim]),
            lm_head_weight: create_q4k_test_data(hidden_dim, 100),
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        // Token ID 999 is out of range for vocab_size=100
        let result = model.encode(&[999]);
        assert!(result.is_err());
    }
}
