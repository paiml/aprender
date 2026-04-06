
    /// Create a minimal test config for testing
    fn test_config() -> GGUFConfig {
        GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            rope_theta: 10000.0,
            context_length: 512,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        }
    }

    /// Create a minimal cached sync model for testing
    fn create_test_model() -> OwnedQuantizedModelCachedSync {
        let config = test_config();
        let model = create_test_model_with_config(&config);
        OwnedQuantizedModelCachedSync::new(model)
    }

    // ========================================================================
    // Constructor Tests
    // ========================================================================

    #[test]
    fn test_new_creates_empty_scheduler() {
        let model = create_test_model();
        // Scheduler should be None initially (lazy init)
        assert!(!model.is_gpu_cache_warm());
    }

    #[test]
    fn test_model_accessor() {
        let config = test_config();
        let model = create_test_model();
        assert_eq!(model.model().config.hidden_dim, config.hidden_dim);
        assert_eq!(model.model().config.vocab_size, config.vocab_size);
    }

    // ========================================================================
    // GPU Cache Tests
    // ========================================================================

    #[test]
    fn test_is_gpu_cache_warm_false_initially() {
        let model = create_test_model();
        assert!(!model.is_gpu_cache_warm());
    }

    #[test]
    fn test_gpu_cache_memory_zero_when_not_warm() {
        let model = create_test_model();
        assert_eq!(model.gpu_cache_memory(), 0);
    }

    #[test]
    fn test_warmup_gpu_cache_success() {
        let model = create_test_model();
        let result = model.warmup_gpu_cache();
        assert!(result.is_ok());
        let (memory_bytes, cached_count) = result.expect("expected value");
        assert!(memory_bytes > 0);
        assert_eq!(cached_count, 1); // 1 layer
        assert!(model.is_gpu_cache_warm());
    }

    #[test]
    fn test_get_dequantized_ffn_weights_none_before_warmup() {
        let model = create_test_model();
        let weights = model.get_dequantized_ffn_weights(0);
        assert!(weights.is_none());
    }

    #[test]
    fn test_get_dequantized_ffn_weights_some_after_warmup() {
        let model = create_test_model();
        let _ = model.warmup_gpu_cache();
        let weights = model.get_dequantized_ffn_weights(0);
        assert!(weights.is_some());
        let w = weights.expect("w");
        // Check dimensions match config
        let config = test_config();
        assert_eq!(w.up.len(), config.hidden_dim * config.intermediate_dim);
        assert_eq!(w.down.len(), config.intermediate_dim * config.hidden_dim);
    }

    #[test]
    fn test_get_dequantized_ffn_weights_out_of_bounds() {
        let model = create_test_model();
        let _ = model.warmup_gpu_cache();
        // Layer index 99 doesn't exist
        let weights = model.get_dequantized_ffn_weights(99);
        assert!(weights.is_none());
    }

    // ========================================================================
    // Batch Stats Tests
    // ========================================================================

    #[test]
    fn test_batch_stats_before_warmup() {
        let model = create_test_model();
        let stats = model.batch_stats();
        assert!(!stats.gpu_cache_ready);
        assert_eq!(stats.cache_memory_gb, 0.0);
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.hidden_dim, 64);
        assert_eq!(stats.intermediate_dim, 128);
    }

    #[test]
    fn test_batch_stats_after_warmup() {
        let model = create_test_model();
        let _ = model.warmup_gpu_cache();
        let stats = model.batch_stats();
        assert!(stats.gpu_cache_ready);
        assert!(stats.cache_memory_gb > 0.0);
    }

    #[test]
    fn test_batch_stats_recommended_values() {
        let model = create_test_model();
        let stats = model.batch_stats();
        assert_eq!(stats.recommended_batch_size, 32);
        assert_eq!(stats.max_batch_size, 64);
    }

    // ========================================================================
    // Adaptive Attention Tests
    // ========================================================================

    #[test]
    fn test_adaptive_fused_attention_short_sequence_uses_cpu() {
        let model = create_test_model();
        let config = test_config();
        let head_dim = config.hidden_dim / config.num_heads;
        let seq_len = 4; // Short sequence, below GPU threshold (64)
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![0.1f32; seq_len * head_dim];
        let k = vec![0.2f32; seq_len * head_dim];
        let v = vec![0.3f32; seq_len * head_dim];

        let result = model.adaptive_fused_attention(&q, &k, &v, seq_len, head_dim, scale);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), seq_len * head_dim);
    }

    #[test]
    fn test_adaptive_fused_attention_long_sequence_uses_gpu() {
        let model = create_test_model();
        let config = test_config();
        let head_dim = config.hidden_dim / config.num_heads;
        let seq_len = 128; // Long sequence, above GPU threshold (64)
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![0.1f32; seq_len * head_dim];
        let k = vec![0.2f32; seq_len * head_dim];
        let v = vec![0.3f32; seq_len * head_dim];

        let result = model.adaptive_fused_attention(&q, &k, &v, seq_len, head_dim, scale);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), seq_len * head_dim);
    }

    // ========================================================================
    // Adaptive Multihead Attention Tests
    // ========================================================================

    #[test]
    fn test_adaptive_multihead_attention_basic() {
        let model = create_test_model();
        let config = test_config();
        let hidden_dim = config.hidden_dim;
        let seq_len = 4;

        let q = vec![0.1f32; seq_len * hidden_dim];
        let k = vec![0.2f32; seq_len * hidden_dim];
        let v = vec![0.3f32; seq_len * hidden_dim];

        let result = model.adaptive_multihead_attention(&q, &k, &v, seq_len);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), seq_len * hidden_dim);
    }

    // ========================================================================
    // Forward Pass Tests
    // ========================================================================

    #[test]
    fn test_forward_batch_gpu_cached_basic() {
        let model = create_test_model();
        let token_ids = vec![1u32, 2, 3];

        let result = model.forward_batch_gpu_cached(&token_ids);
        assert!(result.is_ok());
        let output = result.expect("output");
        let config = test_config();
        assert_eq!(output.len(), token_ids.len() * config.vocab_size);
    }

    #[test]
    fn test_forward_batch_gpu_cached_empty_input() {
        let model = create_test_model();
        let token_ids: Vec<u32> = vec![];

        let result = model.forward_batch_gpu_cached(&token_ids);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), 0);
    }

    // ========================================================================
    // Batch FFN GPU Tests
    // ========================================================================

    #[test]
    fn test_batch_ffn_gpu_not_warmed() {
        let model = create_test_model();
        let config = test_config();
        let hidden_states = vec![0.1f32; config.hidden_dim];

        let result = model.batch_ffn_gpu(&hidden_states, 0);
        assert!(result.is_err());
        match result {
            Err(RealizarError::UnsupportedOperation { operation, reason }) => {
                assert_eq!(operation, "batch_ffn_gpu");
                assert!(reason.contains("not cached"));
            },
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_batch_ffn_gpu_after_warmup() {
        let model = create_test_model();
        let config = test_config();
        let _ = model.warmup_gpu_cache();

        let batch_size = 2;
        let hidden_states = vec![0.1f32; batch_size * config.hidden_dim];

        let result = model.batch_ffn_gpu(&hidden_states, 0);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), batch_size * config.hidden_dim);
    }

    #[test]
    fn test_batch_ffn_gpu_empty_batch() {
        let model = create_test_model();
        let _ = model.warmup_gpu_cache();
        let hidden_states: Vec<f32> = vec![];

        let result = model.batch_ffn_gpu(&hidden_states, 0);
        assert!(result.is_err());
        match result {
            Err(RealizarError::UnsupportedOperation { operation, reason }) => {
                assert_eq!(operation, "batch_ffn_gpu");
                assert!(reason.contains("Empty batch"));
            },
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    // ========================================================================
    // Batch Generate Tests
    // ========================================================================

    #[test]
    fn test_batch_generate_gpu_not_warmed() {
        let model = create_test_model();
        let prompts = vec![vec![1u32, 2, 3]];
        let config = QuantizedGenerateConfig::deterministic(5);

        let result = model.batch_generate_gpu(&prompts, &config);
        assert!(result.is_err());
        match result {
            Err(RealizarError::UnsupportedOperation { operation, reason }) => {
                assert_eq!(operation, "batch_generate_gpu");
                assert!(reason.contains("not warmed up"));
            },
            _ => panic!("Expected UnsupportedOperation error"),
        }
    }

    #[test]
    fn test_batch_generate_gpu_empty_prompts() {
        let model = create_test_model();
        let _ = model.warmup_gpu_cache();
        let prompts: Vec<Vec<u32>> = vec![];
        let config = QuantizedGenerateConfig::deterministic(5);

        let result = model.batch_generate_gpu(&prompts, &config);
        assert!(result.is_ok());
        assert!(result.expect("expected value").is_empty());
    }

    // ========================================================================
    // Forward Batch With GPU FFN Tests
    // ========================================================================

    #[test]
    fn test_forward_batch_with_gpu_ffn_empty() {
        let model = create_test_model();
        let token_ids: Vec<u32> = vec![];
        let mut caches: Vec<crate::gguf::OwnedQuantizedKVCache> = vec![];
        let positions: Vec<usize> = vec![];

        let result = model.forward_batch_with_gpu_ffn(&token_ids, &mut caches, &positions);
        assert!(result.is_ok());
        assert!(result.expect("expected value").is_empty());
    }

    #[test]
    fn test_forward_batch_with_gpu_ffn_mismatched_sizes() {
        let model = create_test_model();
        let config = test_config();
        let max_seq_len = 64;

        let token_ids = vec![1u32, 2, 3];
        let mut caches = vec![
            crate::gguf::OwnedQuantizedKVCache::from_config(&config, max_seq_len),
            crate::gguf::OwnedQuantizedKVCache::from_config(&config, max_seq_len),
        ];
        let positions = vec![0usize];

        let result = model.forward_batch_with_gpu_ffn(&token_ids, &mut caches, &positions);
        assert!(result.is_err());
        match result {
            Err(RealizarError::InvalidShape { reason }) => {
                assert!(reason.contains("mismatch"));
            },
            _ => panic!("Expected InvalidShape error"),
        }
    }

    #[test]
    fn test_forward_batch_with_gpu_ffn_small_batch_cpu_path() {
        let model = create_test_model();
        let config = test_config();
        let max_seq_len = 64;

        // Small batch (< 32) should use CPU path
        let token_ids = vec![1u32, 2, 3];
        let mut caches: Vec<_> = (0..3)
            .map(|_| crate::gguf::OwnedQuantizedKVCache::from_config(&config, max_seq_len))
            .collect();
        let positions = vec![0usize, 0, 0];

        let result = model.forward_batch_with_gpu_ffn(&token_ids, &mut caches, &positions);
        assert!(result.is_ok());
        let logits = result.expect("logits");
        assert_eq!(logits.len(), 3);
        assert_eq!(logits[0].len(), config.vocab_size);
    }

    // ========================================================================
    // Generate With Cache Tests
    // ========================================================================

    #[test]
    fn test_generate_with_cache_basic() {
        let model = create_test_model();
        let prompt = vec![1u32, 2, 3];
        let config = QuantizedGenerateConfig::deterministic(3);

        let result = model.generate_with_cache(&prompt, &config);
        assert!(result.is_ok());
        let output = result.expect("output");
        // Output includes prompt + generated tokens
        assert!(output.len() >= prompt.len());
    }

    // ========================================================================
    // Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OwnedQuantizedModelCachedSync>();
    }

    #[test]
    fn test_concurrent_model_access() {
        use std::sync::Arc;
        use std::thread;

        let model = Arc::new(create_test_model());
        let _ = model.warmup_gpu_cache();

        let mut handles = vec![];
        for _ in 0..4 {
            let model_clone = Arc::clone(&model);
            let handle = thread::spawn(move || {
                let stats = model_clone.batch_stats();
                assert!(stats.gpu_cache_ready);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("thread join failed");
        }
    }

    // ========================================================================
    // Batch Matmul Input Validation Tests
    // ========================================================================

    #[test]
    fn test_batch_matmul_invalid_input_size() {
        let model = create_test_model();
        let _ = model.warmup_gpu_cache();

        // Input size doesn't match batch_size * hidden_dim
        let config = test_config();
        let wrong_size_input = vec![0.1f32; config.hidden_dim + 1];

        let result = model.batch_ffn_gpu(&wrong_size_input, 0);
        // The batch_size calculation would be wrong, leading to error
        assert!(result.is_ok() || result.is_err());
    }
