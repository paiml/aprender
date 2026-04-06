
// ============================================================================
// GpuModelQ4 CPU Logic Tests (Phase 42)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu_model_q4_tests {
    use super::*;
    use crate::apr_transformer::QuantizedAprTransformerQ4;
    use crate::gpu::adapters::apr_q4::{AprQ4ToGpuAdapter, GpuModelQ4, LayerNorms};

    fn create_test_gpu_model_q4() -> GpuModelQ4 {
        GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 64,
                num_layers: 2,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size: 1000,
                intermediate_dim: 128,
                context_length: 256,
                rope_theta: 10000.0,
                eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.1; 1000 * 64],
            output_norm_weight: vec![1.0; 64],
            layer_norms: vec![
                LayerNorms {
                    attn_norm: vec![1.0; 64],
                    ffn_norm: vec![1.0; 64],
                },
                LayerNorms {
                    attn_norm: vec![1.0; 64],
                    ffn_norm: vec![1.0; 64],
                },
            ],
            num_layers: 2,
            has_gate: true,
        }
    }

    #[test]
    fn test_layer_norms_clone() {
        let norms = LayerNorms {
            attn_norm: vec![1.0, 2.0, 3.0],
            ffn_norm: vec![4.0, 5.0, 6.0],
        };

        let cloned = norms.clone();

        assert_eq!(cloned.attn_norm, norms.attn_norm);
        assert_eq!(cloned.ffn_norm, norms.ffn_norm);
    }

    #[test]
    fn test_layer_norms_debug() {
        let norms = LayerNorms {
            attn_norm: vec![1.0],
            ffn_norm: vec![2.0],
        };

        let debug_str = format!("{:?}", norms);
        assert!(debug_str.contains("LayerNorms"));
    }

    #[test]
    fn test_gpu_model_q4_config() {
        let model = create_test_gpu_model_q4();
        assert_eq!(model.num_layers, 2);
        assert!(model.has_gate);
        assert_eq!(model.config.hidden_dim, 64);
    }

    #[test]
    fn test_gpu_model_q4_clone() {
        let model = create_test_gpu_model_q4();
        let cloned = model.clone();

        assert_eq!(cloned.num_layers, model.num_layers);
        assert_eq!(cloned.has_gate, model.has_gate);
        assert_eq!(cloned.token_embedding.len(), model.token_embedding.len());
    }

    #[test]
    fn test_gpu_model_q4_debug() {
        let model = create_test_gpu_model_q4();
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("GpuModelQ4"));
    }

    #[test]
    fn test_create_model_with_gate() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert!(model.has_gate);
        assert_eq!(model.num_layers, 2);
    }

    #[test]
    fn test_create_model_without_gate() {
        let apr = create_test_quantized_apr(false);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert!(!model.has_gate);
    }

    #[test]
    fn test_create_model_layer_norms_extracted() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.layer_norms.len(), 2);
        assert_eq!(model.layer_norms[0].attn_norm.len(), 64);
        assert_eq!(model.layer_norms[0].ffn_norm.len(), 64);
    }

    #[test]
    fn test_create_model_embeddings_copied() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.token_embedding.len(), apr.token_embedding.len());
        assert_eq!(model.output_norm_weight.len(), apr.output_norm_weight.len());
    }

    #[test]
    fn test_create_model_config_preserved() {
        let apr = create_test_quantized_apr(true);
        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.config.hidden_dim, apr.config.hidden_dim);
        assert_eq!(model.config.num_heads, apr.config.num_heads);
        assert_eq!(model.config.num_kv_heads, apr.config.num_kv_heads);
        assert_eq!(model.config.vocab_size, apr.config.vocab_size);
    }

    #[test]
    fn test_create_model_empty_layers() {
        let apr = QuantizedAprTransformerQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 64,
                num_layers: 0,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size: 100,
                intermediate_dim: 128,
                context_length: 256,
                rope_theta: 10000.0,
                eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.1; 100 * 64],
            layers: vec![],
            output_norm_weight: vec![1.0; 64],
            lm_head_weight: QuantizedAprTensorQ4::zeros(64, 100),
        };

        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.num_layers, 0);
        assert!(!model.has_gate);
        assert!(model.layer_norms.is_empty());
    }

    fn create_test_quantized_apr(with_gate: bool) -> QuantizedAprTransformerQ4 {
        let hidden_dim = 64;
        let intermediate_dim = 128;
        let vocab_size = 100;

        QuantizedAprTransformerQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim,
                num_layers: 2,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size,
                intermediate_dim,
                context_length: 256,
                rope_theta: 10000.0,
                eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.1; vocab_size * hidden_dim],
            layers: vec![
                create_test_q4_layer_full(hidden_dim, 4, 4, intermediate_dim, with_gate),
                create_test_q4_layer_full(hidden_dim, 4, 4, intermediate_dim, with_gate),
            ],
            output_norm_weight: vec![1.0; hidden_dim],
            lm_head_weight: QuantizedAprTensorQ4::zeros(hidden_dim, vocab_size),
        }
    }

    fn create_test_q4_layer_full(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        with_gate: bool,
    ) -> QuantizedAprLayerQ4 {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        QuantizedAprLayerQ4 {
            attn_norm_weight: vec![1.0; hidden_dim],
            qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
            attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
            ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
            ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
            ffn_gate_weight: if with_gate {
                Some(QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim))
            } else {
                None
            },
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_q4_layer(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
) -> QuantizedAprLayerQ4 {
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    QuantizedAprLayerQ4 {
        attn_norm_weight: vec![1.0; hidden_dim],
        qkv_weight: QuantizedAprTensorQ4::zeros(hidden_dim, qkv_out_dim),
        attn_output_weight: QuantizedAprTensorQ4::zeros(hidden_dim, hidden_dim),
        ffn_up_weight: QuantizedAprTensorQ4::zeros(hidden_dim, intermediate_dim),
        ffn_down_weight: QuantizedAprTensorQ4::zeros(intermediate_dim, hidden_dim),
        ffn_gate_weight: None,
        ffn_norm_weight: Some(vec![1.0; hidden_dim]),
    }
}
