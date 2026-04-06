
// ============================================================================
// Tests for APR Transformer Configuration (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AprKVCache Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apr_kv_cache_new() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), config.context_length);
        assert_eq!(cache.num_kv_heads(), config.num_kv_heads);
        assert_eq!(cache.head_dim(), config.hidden_dim / config.num_heads);
    }

    #[test]
    fn test_apr_kv_cache_append_and_get() {
        let config = AprTransformerConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            hidden_dim: 64,
            context_length: 128,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

        // Append to layer 0
        let k = vec![1.0f32; kv_size];
        let v = vec![2.0f32; kv_size];
        cache.append(0, &k, &v);
        cache.advance(); // F-REGR-231: explicit advance required

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        // Get from layer 0
        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.len(), kv_size);
        assert_eq!(v_out.len(), kv_size);
        assert!((k_out[0] - 1.0).abs() < 0.001);
        assert!((v_out[0] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_apr_kv_cache_multiple_positions() {
        let config = AprTransformerConfig {
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            hidden_dim: 32,
            context_length: 64,
            ..Default::default()
        };
        let mut cache = AprKVCache::new(&config);
        let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

        // Append 3 positions (num_layers=1, so layer 0 is last layer - auto-advances)
        for i in 0..3 {
            let k = vec![(i + 1) as f32; kv_size];
            let v = vec![(i + 10) as f32; kv_size];
            cache.append(0, &k, &v);
            // No advance() needed - append() auto-advances on last layer
        }

        assert_eq!(cache.len(), 3);

        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.len(), 3 * kv_size);
        assert_eq!(v_out.len(), 3 * kv_size);
    }

    #[test]
    fn test_apr_kv_cache_clear() {
        let config = AprTransformerConfig::default();
        let mut cache = AprKVCache::new(&config);
        let kv_size = config.num_kv_heads * (config.hidden_dim / config.num_heads);

        cache.append(0, &vec![1.0; kv_size], &vec![2.0; kv_size]);
        cache.advance(); // F-REGR-231: explicit advance required
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_apr_kv_cache_debug() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("AprKVCache"));
    }

    #[test]
    fn test_apr_kv_cache_clone() {
        let config = AprTransformerConfig::default();
        let cache = AprKVCache::new(&config);
        let cloned = cache.clone();
        assert_eq!(cloned.len(), cache.len());
        assert_eq!(cloned.capacity(), cache.capacity());
    }

    // -------------------------------------------------------------------------
    // GenerateConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_config_default() {
        let config = GenerateConfig::default();
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 1.0).abs() < 0.001);
        assert!((config.top_p - 0.9).abs() < 0.001);
        assert_eq!(config.top_k, 0);
        assert!((config.repetition_penalty - 1.0).abs() < 0.001);
        assert!(!config.trace);
    }

    #[test]
    fn test_generate_config_debug() {
        let config = GenerateConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GenerateConfig"));
    }

    #[test]
    fn test_generate_config_clone() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            repetition_penalty: 1.1,
            trace: true,
            stop_tokens: vec![],
        };
        let cloned = config.clone();
        assert_eq!(cloned.max_tokens, 100);
        assert!((cloned.temperature - 0.8).abs() < 0.001);
        assert!(cloned.trace);
    }

    // -------------------------------------------------------------------------
    // AprTransformerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apr_transformer_config_default() {
        let config = AprTransformerConfig::default();
        assert_eq!(config.architecture, "unknown");
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.intermediate_dim, 2048);
        assert_eq!(config.context_length, 2048);
        assert!((config.rope_theta - 10000.0).abs() < 0.001);
        assert!((config.eps - 1e-5).abs() < 1e-7);
    }

    #[test]
    fn test_apr_transformer_config_debug() {
        let config = AprTransformerConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AprTransformerConfig"));
    }

    #[test]
    fn test_apr_transformer_config_clone() {
        let config = AprTransformerConfig {
            architecture: "llama".to_string(),
            hidden_dim: 4096,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.architecture, "llama");
        assert_eq!(cloned.hidden_dim, 4096);
    }

    #[test]
    fn test_apr_transformer_config_eq() {
        let config1 = AprTransformerConfig::default();
        let config2 = AprTransformerConfig::default();
        assert_eq!(config1, config2);

        let config3 = AprTransformerConfig {
            hidden_dim: 1024,
            ..Default::default()
        };
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_apr_transformer_config_serialization() {
        let config = AprTransformerConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: AprTransformerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, deserialized);
    }

    // -------------------------------------------------------------------------
    // AprTransformerLayer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apr_transformer_layer_empty() {
        let layer = AprTransformerLayer::empty(512, 2048);
        assert_eq!(layer.attn_norm_weight.len(), 512);
        assert_eq!(layer.qkv_weight.len(), 512 * 3 * 512);
        assert_eq!(layer.attn_output_weight.len(), 512 * 512);
        assert_eq!(layer.ffn_up_weight.len(), 512 * 2048);
        assert_eq!(layer.ffn_down_weight.len(), 2048 * 512);
        assert!(layer.attn_norm_bias.is_none());
        assert!(layer.ffn_gate_weight.is_none());
    }

    #[test]
    fn test_apr_transformer_layer_empty_gqa() {
        // GQA: 8 query heads, 2 kv heads, head_dim = 64
        let hidden_dim = 512; // 8 heads * 64 head_dim
        let num_heads = 8;
        let num_kv_heads = 2;
        let intermediate_dim = 2048;

        let layer =
            AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

        let head_dim = hidden_dim / num_heads; // 64
        let kv_dim = num_kv_heads * head_dim; // 2 * 64 = 128
        let qkv_out_dim = hidden_dim + 2 * kv_dim; // 512 + 256 = 768

        assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
        assert_eq!(layer.attn_norm_weight.len(), hidden_dim);
    }

    #[test]
    fn test_apr_transformer_layer_num_parameters() {
        let layer = AprTransformerLayer::empty(64, 128);
        let params = layer.num_parameters();

        // Count expected parameters
        let expected = 64  // attn_norm_weight
            + 64 * 3 * 64  // qkv_weight
            + 64 * 64      // attn_output_weight
            + 64 * 128     // ffn_up_weight
            + 128 * 64; // ffn_down_weight

        assert_eq!(params, expected);
    }

    #[test]
    fn test_apr_transformer_layer_num_parameters_with_bias() {
        let mut layer = AprTransformerLayer::empty(64, 128);
        layer.attn_norm_bias = Some(vec![0.0; 64]);
        layer.qkv_bias = Some(vec![0.0; 3 * 64]);
        layer.ffn_up_bias = Some(vec![0.0; 128]);

        let params_without = AprTransformerLayer::empty(64, 128).num_parameters();
        let params_with = layer.num_parameters();

        assert_eq!(params_with, params_without + 64 + 3 * 64 + 128);
    }

    #[test]
    fn test_apr_transformer_layer_debug() {
        let layer = AprTransformerLayer::empty(64, 128);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("AprTransformerLayer"));
    }

    #[test]
    fn test_apr_transformer_layer_clone() {
        let layer = AprTransformerLayer::empty(64, 128);
        let cloned = layer.clone();
        assert_eq!(cloned.attn_norm_weight.len(), layer.attn_norm_weight.len());
    }

    #[test]
    fn test_apr_transformer_layer_serialization() {
        let layer = AprTransformerLayer::empty(32, 64);
        let json = serde_json::to_string(&layer).expect("serialize");
        let deserialized: AprTransformerLayer = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.attn_norm_weight.len(), 32);
    }

    // -------------------------------------------------------------------------
    // Q4KLayerWeights Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_q4k_layer_weights_default() {
        let weights = Q4KLayerWeights::default();
        assert!(weights.qkv_weight.is_none());
        assert!(weights.attn_q_weight.is_none());
        assert!(weights.attn_k_weight.is_none());
        assert!(weights.attn_v_weight.is_none());
        assert!(weights.ffn_gate_weight.is_none());
        assert!(weights.ffn_up_weight.is_none());
        assert!(weights.ffn_down_weight.is_none());
    }

    #[test]
    fn test_q4k_layer_weights_debug() {
        let weights = Q4KLayerWeights::default();
        let debug_str = format!("{:?}", weights);
        assert!(debug_str.contains("Q4KLayerWeights"));
    }

    #[test]
    fn test_q4k_layer_weights_clone() {
        let mut weights = Q4KLayerWeights::default();
        weights.qkv_weight = Some(vec![1, 2, 3, 4]);
        let cloned = weights.clone();
        assert_eq!(cloned.qkv_weight, Some(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_q4k_layer_weights_serialization() {
        let mut weights = Q4KLayerWeights::default();
        weights.attn_q_weight = Some(vec![0x12, 0x34]);
        let json = serde_json::to_string(&weights).expect("serialize");
        let deserialized: Q4KLayerWeights = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.attn_q_weight, Some(vec![0x12, 0x34]));
    }
}
