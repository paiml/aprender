
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned_quantized_model_debug_shows_all_fields() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q8_0;

        let config = GGUFConfig {
            architecture: "llama_debug".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama_debug"),
            hidden_dim: 128,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 256,
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
            token_embedding: vec![0.0; 128000],
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: vec![1.0; 128],
            output_norm_bias: None,
            lm_head_weight: OwnedQuantizedTensor {
                data: vec![0u8; 1000],
                in_dim: 128,
                out_dim: 1000,
                qtype: GGUF_TYPE_Q8_0,
            },
            lm_head_bias: None,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(42),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("llama_debug"));
        assert!(debug_str.contains("token_embedding_len"));
        assert!(debug_str.contains("128000"));
        assert!(debug_str.contains("layers_count"));
        assert!(debug_str.contains("output_norm_weight_len"));
        assert!(debug_str.contains("has_output_norm_bias"));
        assert!(debug_str.contains("lm_head_weight"));
        assert!(debug_str.contains("has_lm_head_bias"));
        #[cfg(feature = "cuda")]
        assert!(debug_str.contains("cuda_enabled"));
    }

    #[test]
    fn test_gguf_config_fields() {
        let config = GGUFConfig {
            architecture: "mistral".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("mistral"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA style
            vocab_size: 32000,
            intermediate_dim: 14336,
            context_length: 8192,
            rope_theta: 1000000.0,
            eps: 1e-5,
            rope_type: 1,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.architecture, "mistral");
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.intermediate_dim, 14336);
        assert_eq!(config.context_length, 8192);
        assert!((config.rope_theta - 1000000.0).abs() < 0.1);
        assert!((config.eps - 1e-5).abs() < 1e-10);
        assert_eq!(config.rope_type, 1);
    }

    #[test]
    fn test_mapped_gguf_model_file_not_found() {
        let result = MappedGGUFModel::from_path("/nonexistent/path/model.gguf");
        assert!(result.is_err());
    }

    // =========================================================================
    // GH-40 FALSIFICATION: OwnedQuantizedModel API contract stability
    // =========================================================================

    /// GH-40: OwnedQuantizedModel must expose all public fields required by
    /// the inference pipeline. If a field is renamed, removed, or made private,
    /// this test fails — catching API breakage at compile time.
    #[test]
    fn test_falsify_gh40_api_contract_all_fields_accessible() {
        use super::super::quantized::OwnedQuantizedTensor;
        use super::super::types::GGUF_TYPE_Q4_K;

        let config = GGUFConfig {
            architecture: "gh40_test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("gh40_test"),
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
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };

        // GH-40 contract: every field below must be pub and have the expected type.
        // If ANY of these lines fails to compile, the API contract is broken.
        let _arch: &str = &model.config.architecture;
        let _hidden: usize = model.config.hidden_dim;
        let _embed: &[f32] = &model.token_embedding;
        let _layers: &[super::super::quantized::OwnedQuantizedLayer] = &model.layers;
        let _norm: &[f32] = &model.output_norm_weight;
        let _norm_bias: &Option<Vec<f32>> = &model.output_norm_bias;
        let _lm_data: &[u8] = &model.lm_head_weight.data;
        let _lm_in: usize = model.lm_head_weight.in_dim;
        let _lm_out: usize = model.lm_head_weight.out_dim;
        let _lm_qt: u32 = model.lm_head_weight.qtype;
        let _lm_bias: &Option<Vec<f32>> = &model.lm_head_bias;

        // GH-40 contract: config fields required by inference
        assert_eq!(model.config.hidden_dim, 64);
        assert_eq!(model.config.num_heads, 2);
        assert_eq!(model.config.num_kv_heads, 2);
        assert_eq!(model.config.vocab_size, 100);
        assert_eq!(model.config.intermediate_dim, 128);
        assert!(model.config.rope_theta > 0.0);
        assert!(model.config.eps > 0.0);
    }

    /// GH-40: GGUFConfig must have bos_token_id field (added in the fix).
    /// Without this field, tokenizer initialization fails silently.
    #[test]
    fn test_falsify_gh40_config_has_bos_token_id() {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 100,
            intermediate_dim: 128,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
        };
        assert_eq!(
            config.bos_token_id,
            Some(1),
            "GH-40: bos_token_id must be accessible"
        );
    }
include!("model_gguf_transformer.rs");
}
