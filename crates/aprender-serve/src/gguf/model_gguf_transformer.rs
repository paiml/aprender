
    #[test]
    fn test_gguf_transformer_struct() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 512,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let transformer = GGUFTransformer {
            config,
            token_embedding: vec![0.0; 256000], // 1000 vocab * 256 hidden
            position_embedding: None,
            layers: vec![],
            output_norm_weight: vec![1.0; 256],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; 256000],
            lm_head_bias: None,
        };

        assert_eq!(transformer.config.architecture, "llama");
        assert_eq!(transformer.token_embedding.len(), 256000);
        assert!(transformer.layers.is_empty());
    }

    #[test]
    fn test_gguf_transformer_layer_struct() {
        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 256],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 256 * 768], // 3 * hidden_dim
            qkv_bias: None,
            attn_output_weight: vec![0.0; 256 * 256],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 256 * 512]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 256 * 512],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 512 * 256],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 256]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        assert_eq!(layer.attn_norm_weight.len(), 256);
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    #[test]
    fn test_owned_quantized_model_clone() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1; 6400],
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0; 64],
            output_norm_bias: None,
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 128],
                in_dim: 64,
                out_dim: 100,
                qtype: GGUF_TYPE_Q4_K,
            },
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(5),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let cloned = model.clone();
        assert_eq!(cloned.config.architecture, "test");
        assert_eq!(cloned.token_embedding.len(), 6400);

        // CUDA executor is not cloned
        #[cfg(feature = "cuda")]
        {
            assert!(cloned.cuda_executor.is_none());
            assert_eq!(
                cloned
                    .cuda_kernel_count
                    .load(std::sync::atomic::Ordering::Relaxed),
                0
            );
        }
    }

    #[test]
    fn test_owned_quantized_model_debug() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "debug_test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("debug_test"),
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.0; 1600],
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0; 32],
            output_norm_bias: Some(vec![0.0; 32]),
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![],
                in_dim: 32,
                out_dim: 50,
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

        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("token_embedding_len"));
        assert!(debug_str.contains("1600"));
    }

    #[test]
    fn test_gguf_transformer_with_biases() {
        let config = GGUFConfig {
            architecture: "phi2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
            hidden_dim: 128,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 500,
            intermediate_dim: 256,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let transformer = GGUFTransformer {
            config,
            token_embedding: vec![0.0; 64000],
            position_embedding: None,
            layers: vec![],
            output_norm_weight: vec![1.0; 128],
            output_norm_bias: Some(vec![0.0; 128]),
            lm_head_weight: vec![0.0; 64000],
            lm_head_bias: Some(vec![0.0; 500]),
        };

        assert!(transformer.output_norm_bias.is_some());
        assert!(transformer.lm_head_bias.is_some());
        assert_eq!(transformer.lm_head_bias.as_ref().expect("as_ref").len(), 500);
    }

    #[test]
    fn test_gguf_transformer_with_layers() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let layer1 = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 64 * 192],
            qkv_bias: None,
            attn_output_weight: vec![0.0; 64 * 64],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 64 * 128]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 64 * 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 128 * 64],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 64]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        let layer2 = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 64 * 192],
            qkv_bias: None,
            attn_output_weight: vec![0.0; 64 * 64],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 64 * 128]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 64 * 128],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 128 * 64],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 64]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        let transformer = GGUFTransformer {
            config,
            token_embedding: vec![0.0; 6400],
            position_embedding: None,
            layers: vec![layer1, layer2],
            output_norm_weight: vec![1.0; 64],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; 6400],
            lm_head_bias: None,
        };

        assert_eq!(transformer.layers.len(), 2);
        assert_eq!(transformer.layers[0].qkv_weight.len(), 64 * 192);
    }

    #[test]
    fn test_gguf_transformer_layer_without_gate() {
        // Models like phi-2 don't use SwiGLU, so no gate
        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 128],
            attn_norm_bias: Some(vec![0.0; 128]),
            qkv_weight: vec![0.0; 128 * 384],
            qkv_bias: Some(vec![0.0; 384]),
            attn_output_weight: vec![0.0; 128 * 128],
            attn_output_bias: Some(vec![0.0; 128]),
            ffn_gate_weight: None, // No gate for phi-2 style
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 128 * 512],
            ffn_up_bias: Some(vec![0.0; 512]),
            ffn_down_weight: vec![0.0; 512 * 128],
            ffn_down_bias: Some(vec![0.0; 128]),
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        assert!(layer.ffn_gate_weight.is_none());
        assert!(layer.qkv_bias.is_some());
        assert!(layer.attn_output_bias.is_some());
    }

    #[test]
    fn test_gguf_transformer_layer_all_biases() {
        let layer = GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 64],
            attn_norm_bias: Some(vec![0.0; 64]),
            qkv_weight: vec![0.0; 64 * 192],
            qkv_bias: Some(vec![0.0; 192]),
            attn_output_weight: vec![0.0; 64 * 64],
            attn_output_bias: Some(vec![0.0; 64]),
            ffn_gate_weight: Some(vec![0.0; 64 * 128]),
            ffn_gate_bias: Some(vec![0.0; 128]),
            ffn_up_weight: vec![0.0; 64 * 128],
            ffn_up_bias: Some(vec![0.0; 128]),
            ffn_down_weight: vec![0.0; 128 * 64],
            ffn_down_bias: Some(vec![0.0; 64]),
            ffn_norm_weight: Some(vec![1.0; 64]),
            ffn_norm_bias: Some(vec![0.0; 64]),
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        };

        assert!(layer.attn_norm_bias.is_some());
        assert!(layer.qkv_bias.is_some());
        assert!(layer.attn_output_bias.is_some());
        assert!(layer.ffn_gate_bias.is_some());
        assert!(layer.ffn_up_bias.is_some());
        assert!(layer.ffn_down_bias.is_some());
        assert!(layer.ffn_norm_bias.is_some());
    }

    #[test]
    fn test_owned_quantized_model_with_bias() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "phi2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.1; 6400],
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0; 64],
            output_norm_bias: Some(vec![0.0; 64]),
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 128],
                in_dim: 64,
                out_dim: 100,
                qtype: GGUF_TYPE_Q4_K,
            },
            lm_head_bias: Some(vec![0.0; 100]),
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        assert!(model.output_norm_bias.is_some());
        assert!(model.lm_head_bias.is_some());
        assert_eq!(model.lm_head_bias.as_ref().expect("as_ref").len(), 100);
    }

    #[test]
    fn test_owned_quantized_model_clone_preserves_data() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q6_K;

        let config = GGUFConfig {
            architecture: "test_clone".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test_clone"),
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 50,
            intermediate_dim: 64,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-6,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = OwnedQuantizedModel {
            config,
            token_embedding: vec![0.5, 0.6, 0.7, 0.8],
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0, 1.0, 1.0],
            output_norm_bias: Some(vec![0.1, 0.2, 0.3]),
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![1, 2, 3, 4, 5],
                in_dim: 32,
                out_dim: 50,
                qtype: GGUF_TYPE_Q6_K,
            },
            lm_head_bias: Some(vec![0.0; 50]),
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(10),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let cloned = model.clone();

        // Verify all data is cloned
        assert_eq!(cloned.token_embedding, model.token_embedding);
        assert_eq!(cloned.output_norm_weight, model.output_norm_weight);
        assert_eq!(cloned.output_norm_bias, model.output_norm_bias);
        assert_eq!(cloned.lm_head_weight.data, model.lm_head_weight.data);
        assert_eq!(cloned.lm_head_weight.qtype, GGUF_TYPE_Q6_K);
        assert_eq!(cloned.config.architecture, "test_clone");
    }
