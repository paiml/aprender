
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_factory::{build_minimal_llama_gguf, build_minimal_phi2_gguf};

    // ============================================================================
    // QuantizedGGUFTransformerLayer tests
    // ============================================================================

    #[test]
    fn test_quantized_layer_struct_fields() {
        // Verify struct field layout and optional fields
        let layer = QuantizedGGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: None,
            qkv_weight: QKVWeights::Fused(QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 192,
                qtype: GGUF_TYPE_Q4_K,
            }),
            qkv_bias: None,
            attn_output_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 50,
                num_elements: 64,
                qtype: GGUF_TYPE_Q4_K,
            },
            attn_output_bias: None,
            ffn_up_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 256,
                qtype: GGUF_TYPE_Q4_K,
            },
            ffn_up_bias: None,
            ffn_down_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 64,
                qtype: GGUF_TYPE_Q4_K,
            },
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        assert_eq!(layer.attn_norm_weight.len(), 64);
        assert!(layer.attn_norm_bias.is_none());
        assert!(layer.ffn_gate_weight.is_none());
    }

    #[test]
    fn test_quantized_layer_with_optional_fields() {
        let layer = QuantizedGGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 32],
            attn_norm_bias: Some(vec![0.0; 32]),
            qkv_weight: QKVWeights::Separate {
                q: QuantizedTensorRef {
                    offset: 0,
                    byte_size: 50,
                    num_elements: 32,
                    qtype: GGUF_TYPE_Q4_0,
                },
                k: QuantizedTensorRef {
                    offset: 50,
                    byte_size: 50,
                    num_elements: 32,
                    qtype: GGUF_TYPE_Q4_0,
                },
                v: QuantizedTensorRef {
                    offset: 100,
                    byte_size: 50,
                    num_elements: 32,
                    qtype: GGUF_TYPE_Q4_0,
                },
            },
            qkv_bias: Some(vec![0.0; 96]),
            attn_output_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 50,
                num_elements: 32,
                qtype: GGUF_TYPE_Q4_0,
            },
            attn_output_bias: Some(vec![0.0; 32]),
            ffn_up_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 128,
                qtype: GGUF_TYPE_Q4_0,
            },
            ffn_up_bias: Some(vec![0.0; 128]),
            ffn_down_weight: QuantizedTensorRef {
                offset: 0,
                byte_size: 50,
                num_elements: 32,
                qtype: GGUF_TYPE_Q4_0,
            },
            ffn_down_bias: Some(vec![0.0; 32]),
            ffn_gate_weight: Some(QuantizedTensorRef {
                offset: 0,
                byte_size: 100,
                num_elements: 128,
                qtype: GGUF_TYPE_Q4_0,
            }),
            ffn_gate_bias: Some(vec![0.0; 128]),
            ffn_norm_weight: Some(vec![1.0; 32]),
            ffn_norm_bias: Some(vec![0.0; 32]),
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        assert!(layer.attn_norm_bias.is_some());
        assert!(layer.qkv_bias.is_some());
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    // ============================================================================
    // QuantizedGGUFTransformer tests using test_factory
    // ============================================================================

    #[test]
    fn test_quantized_transformer_from_llama_gguf() {
        // build_minimal_llama_gguf(vocab_size, hidden_dim, intermediate_dim, num_heads, num_kv_heads)
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data);

        assert!(
            transformer.is_ok(),
            "Failed to load transformer: {:?}",
            transformer.err()
        );
        let transformer = transformer.expect("transformer");

        assert_eq!(transformer.config.hidden_dim, 64);
        assert_eq!(transformer.config.num_layers, 1); // test_factory builds 1 layer
        assert_eq!(transformer.layers.len(), 1);
        assert_eq!(transformer.token_embedding.len(), 100 * 64);
    }

    #[test]
    fn test_quantized_transformer_from_phi2_gguf() {
        // build_minimal_phi2_gguf(vocab_size, hidden_dim, intermediate_dim, num_heads)
        let gguf_data = build_minimal_phi2_gguf(100, 64, 256, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data);

        assert!(
            transformer.is_ok(),
            "Failed to load transformer: {:?}",
            transformer.err()
        );
        let transformer = transformer.expect("transformer");

        assert_eq!(transformer.config.hidden_dim, 64);
        assert_eq!(transformer.layers.len(), 1);
    }

    #[test]
    fn test_quantized_transformer_config() {
        let gguf_data = build_minimal_llama_gguf(200, 128, 512, 8, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).expect("transformer");

        assert_eq!(transformer.config.hidden_dim, 128);
        // intermediate_dim is inferred from ffn_up tensor, may differ from input
        assert!(transformer.config.intermediate_dim > 0);
        assert_eq!(transformer.config.num_heads, 8);
        assert_eq!(transformer.config.num_kv_heads, 4);
        assert_eq!(transformer.config.vocab_size, 200);
    }

    #[test]
    fn test_quantized_transformer_output_norm() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).expect("transformer");

        // Output norm weight should be hidden_dim size
        assert_eq!(transformer.output_norm_weight.len(), 64);
    }

    #[test]
    fn test_quantized_transformer_lm_head() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).expect("transformer");

        // LM head should have elements (may use tied embeddings with F32 type 0)
        assert!(transformer.lm_head_weight.num_elements > 0);
        // qtype can be 0 (F32) for tied embeddings
        assert!(transformer.lm_head_weight.byte_size > 0);
    }

    #[test]
    fn test_quantized_transformer_layer_attn_norm() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).expect("transformer");

        let layer = &transformer.layers[0];
        assert_eq!(layer.attn_norm_weight.len(), 64);
    }

    #[test]
    fn test_quantized_transformer_layer_qkv() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).expect("transformer");

        let layer = &transformer.layers[0];
        // QKV should be either fused or separate
        match &layer.qkv_weight {
            QKVWeights::Fused(ref tensor) => {
                assert!(tensor.num_elements > 0);
            },
            QKVWeights::Separate { q, k, v } => {
                assert!(q.num_elements > 0);
                assert!(k.num_elements > 0);
                assert!(v.num_elements > 0);
            },
        }
    }

    #[test]
    fn test_quantized_transformer_layer_ffn() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).expect("transformer");

        let layer = &transformer.layers[0];
        assert!(layer.ffn_up_weight.num_elements > 0);
        assert!(layer.ffn_down_weight.num_elements > 0);
    }

    #[test]
    fn test_quantized_transformer_has_data_ref() {
        let gguf_data = build_minimal_llama_gguf(100, 64, 256, 4, 4);
        let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to parse GGUF");
        let transformer = QuantizedGGUFTransformer::from_gguf(&model, &gguf_data).expect("transformer");

        // The data reference should point to the original data
        assert!(!transformer.data.is_empty());
        assert_eq!(transformer.data.len(), gguf_data.len());
    }
}
