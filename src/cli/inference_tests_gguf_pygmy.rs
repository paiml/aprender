//! T-COV-95 Active Pygmy CLI Inference Tests (In-Memory)
//!
//! Uses Active Pygmy models for in-memory inference testing.
//! No file I/O - pure setup/teardown in memory.

#[cfg(test)]
mod active_pygmy_inference {
    use crate::apr::test_factory::build_executable_pygmy_apr;
    use crate::apr::AprV2Model;
    use crate::gguf::test_factory::build_executable_pygmy_gguf;
    use crate::gguf::{GGUFModel, QuantizedGGUFTransformer};

    // =========================================================================
    // GGUF Active Pygmy - In-Memory Structure Tests
    // =========================================================================

    #[test]
    fn test_gguf_pygmy_parse_in_memory() {
        let data = build_executable_pygmy_gguf();
        let model = GGUFModel::from_bytes(&data);
        assert!(model.is_ok(), "GGUF parse failed: {:?}", model.err());
    }

    #[test]
    fn test_gguf_pygmy_transformer_in_memory() {
        let data = build_executable_pygmy_gguf();
        let gguf = GGUFModel::from_bytes(&data).expect("gguf");
        let transformer = QuantizedGGUFTransformer::from_gguf(&gguf, &data);
        assert!(
            transformer.is_ok(),
            "Transformer load failed: {:?}",
            transformer.err()
        );

        let t = transformer.expect("t");
        assert_eq!(t.config.hidden_dim, 32);
        assert_eq!(t.config.vocab_size, 32);
    }

    #[test]
    fn test_gguf_pygmy_transformer_config() {
        let data = build_executable_pygmy_gguf();
        let gguf = GGUFModel::from_bytes(&data).expect("gguf");
        let t = QuantizedGGUFTransformer::from_gguf(&gguf, &data).expect("t");

        assert_eq!(t.config.architecture, "llama");
        assert_eq!(t.config.num_layers, 1);
        assert_eq!(t.config.num_heads, 4);
        assert_eq!(t.config.num_kv_heads, 4);
    }

    #[test]
    fn test_gguf_pygmy_transformer_layers() {
        let data = build_executable_pygmy_gguf();
        let gguf = GGUFModel::from_bytes(&data).expect("gguf");
        let t = QuantizedGGUFTransformer::from_gguf(&gguf, &data).expect("t");

        assert_eq!(t.layers.len(), 1);
        assert_eq!(t.layers[0].attn_norm_weight.len(), 32);
    }

    #[test]
    fn test_gguf_pygmy_output_norm() {
        let data = build_executable_pygmy_gguf();
        let gguf = GGUFModel::from_bytes(&data).expect("gguf");
        let t = QuantizedGGUFTransformer::from_gguf(&gguf, &data).expect("t");

        assert_eq!(t.output_norm_weight.len(), 32);
        assert!(t.output_norm_weight.iter().all(|&v| v.is_finite()));
    }

    // =========================================================================
    // APR Active Pygmy - In-Memory Inference
    // =========================================================================

    #[test]
    fn test_apr_pygmy_parse_in_memory() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data);
        assert!(model.is_ok(), "APR parse failed: {:?}", model.err());
    }

    #[test]
    fn test_apr_pygmy_metadata_in_memory() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data).expect("model");

        let meta = model.metadata();
        assert_eq!(meta.hidden_size, Some(8));
        assert_eq!(meta.num_layers, Some(1));
        assert_eq!(meta.vocab_size, Some(10));
    }

    #[test]
    fn test_apr_pygmy_forward_in_memory() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data).expect("model");

        let logits = model.forward(&[1]);
        assert!(logits.is_ok(), "Forward failed: {:?}", logits.err());

        let l = logits.expect("l");
        assert_eq!(l.len(), 10);
        assert!(l.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_apr_pygmy_forward_multi_token() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data).expect("model");

        let logits = model.forward(&[0, 1, 2, 3]);
        assert!(logits.is_ok());
        assert_eq!(logits.expect("logits").len(), 10);
    }

    #[test]
    fn test_apr_pygmy_generate_in_memory() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data).expect("model");

        let tokens = model.generate(&[1], 3, None);
        assert!(tokens.is_ok(), "Generate failed: {:?}", tokens.err());
        assert!(!tokens.expect("tokens").is_empty());
    }

    #[test]
    fn test_apr_pygmy_all_tokens() {
        let data = build_executable_pygmy_apr();
        let model = AprV2Model::from_bytes(data).expect("model");

        // Test all valid tokens (vocab_size = 10)
        for token in 0..10u32 {
            let logits = model.forward(&[token]);
            assert!(logits.is_ok(), "Token {} failed", token);
        }
    }

    // =========================================================================
    // Cross-Format Structural Comparison
    // =========================================================================

    #[test]
    fn test_both_formats_parse_successfully() {
        // GGUF
        let gguf_data = build_executable_pygmy_gguf();
        let gguf = GGUFModel::from_bytes(&gguf_data);
        assert!(gguf.is_ok());

        // APR
        let apr_data = build_executable_pygmy_apr();
        let apr = AprV2Model::from_bytes(apr_data);
        assert!(apr.is_ok());
    }

    #[test]
    fn test_apr_forward_produces_finite_logits() {
        let apr_data = build_executable_pygmy_apr();
        let apr_m = AprV2Model::from_bytes(apr_data).expect("apr_m");
        let apr_logits = apr_m.forward(&[0]).expect("apr_logits");

        assert!(apr_logits.iter().all(|&v| v.is_finite()));
        assert!(!apr_logits.iter().all(|&v| v == 0.0)); // Not all zeros
    }

    #[test]
    fn test_pygmy_models_are_small() {
        let gguf_data = build_executable_pygmy_gguf();
        let apr_data = build_executable_pygmy_apr();

        // Both should be < 20KB (Active Pygmy property)
        assert!(
            gguf_data.len() < 20_000,
            "GGUF too large: {}",
            gguf_data.len()
        );
        assert!(apr_data.len() < 10_000, "APR too large: {}", apr_data.len());
    }
}
