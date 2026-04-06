
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // KV Cache Initialization Tests
    // ========================================================================

    #[test]
    fn test_init_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let n_layers = 4usize;
        let n_kv_heads = 4usize;
        let head_dim = 64usize;
        let max_seq_len = 1024usize;

        let result =
            exec.init_kv_cache_gpu(n_layers, n_kv_heads, head_dim, max_seq_len, n_kv_heads * 4);
        assert!(result.is_ok());

        // Verify cache is initialized
        assert!(exec.has_kv_cache_gpu());
        assert!(exec.kv_cache_max_len > 0);
    }

    #[test]
    fn test_init_batched_kv_cache_gpu_requires_kv_cache() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Without init_kv_cache_gpu first, should fail
        let result = exec.init_batched_kv_cache_gpu(4, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_init_batched_kv_cache_gpu_after_kv_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // First init regular KV cache
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Then init batched cache
        let result = exec.init_batched_kv_cache_gpu(4, 8);
        assert!(result.is_ok());

        // Verify batched cache is initialized
        assert_eq!(exec.batched_kv_allocated_batch, 8);
    }

    // ========================================================================
    // KV Cache State Tests
    // ========================================================================

    #[test]
    fn test_has_kv_cache_gpu_initial_false() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(!exec.has_kv_cache_gpu());
    }

    #[test]
    fn test_kv_cache_len_uninitialized() {
        let Some(exec) = create_executor() else {
            return;
        };
        // Uninitialized layer should return 0
        assert_eq!(exec.kv_cache_len(0), 0);
        assert_eq!(exec.kv_cache_len(99), 0);
    }

    #[test]
    fn test_kv_cache_len_after_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Initially should be 0 for each layer
        assert_eq!(exec.kv_cache_len(0), 0);
        assert_eq!(exec.kv_cache_len(1), 0);
    }

    // ========================================================================
    // KV Cache Reset Tests
    // ========================================================================

    #[test]
    fn test_reset_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Reset should succeed
        exec.reset_kv_cache_gpu();

        // All lengths should be 0
        assert_eq!(exec.kv_cache_len(0), 0);
    }

    #[test]
    fn test_reset_batched_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Init regular cache first
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();
        exec.init_batched_kv_cache_gpu(4, 8).unwrap();

        exec.reset_batched_kv_cache_gpu();

        // Batched lengths should all be 0
        assert!(exec.batched_kv_lengths.iter().all(|&len| len == 0));
    }

    // ========================================================================
    // RoPE Configuration Tests
    // ========================================================================

    #[test]
    fn test_set_rope_theta() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.set_rope_theta(10000.0);
        assert_eq!(exec.rope_theta, 10000.0);

        exec.set_rope_theta(500000.0); // Longer context
        assert_eq!(exec.rope_theta, 500000.0);
    }

    #[test]
    fn test_set_rope_type() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.set_rope_type(0); // NORM
        assert_eq!(exec.rope_type, 0);

        exec.set_rope_type(2); // NEOX (GPT-NeoX style)
        assert_eq!(exec.rope_type, 2);
    }

    // ========================================================================
    // KV Cache Rollback Tests
    // ========================================================================

    #[test]
    fn test_rollback_kv_cache_gpu() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Rollback to position 5
        exec.rollback_kv_cache_gpu(5);

        // All layers should be rolled back to 5
        for layer in 0..4 {
            assert!(exec.kv_cache_len(layer) <= 5);
        }
    }

    #[test]
    fn test_rollback_to_zero() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.init_kv_cache_gpu(4, 4, 64, 1024, 16).unwrap();

        // Rollback to 0 should be equivalent to reset
        exec.rollback_kv_cache_gpu(0);

        assert_eq!(exec.kv_cache_len(0), 0);
    }

    // ========================================================================
    // Flash Attention Cached Tests
    // ========================================================================

    #[test]
    fn test_flash_attention_cached_requires_kv_cache() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        // Without KV cache initialization
        let q = vec![1.0f32; 256];
        let k = vec![1.0f32; 256];
        let v = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 256];

        // flash_attention_cached takes (layer_idx, q, current_k, current_v, output)
        let result = exec.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_incremental_attention_gpu_requires_kv_cache() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let q = vec![1.0f32; 256];
        let k = vec![1.0f32; 256];
        let v = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 256];

        // incremental_attention_gpu takes (layer_idx, q, current_k, current_v, output)
        let result = exec.incremental_attention_gpu(0, &q, &k, &v, &mut output);
        assert!(result.is_err());
    }

    // ========================================================================
    // KV Cache Memory Calculation Tests
    // ========================================================================

    #[test]
    fn test_kv_cache_memory_calculation() {
        // Test memory calculation for KV cache
        let n_layers = 32usize;
        let n_kv_heads = 8usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        let per_layer_bytes = 2 * max_seq_len * n_kv_heads * head_dim * 4; // K + V, f32
        let total_bytes = n_layers * per_layer_bytes;

        // Verify it's a reasonable size (1-10 GB range for large models)
        assert!(total_bytes > 1_000_000_000); // > 1GB
        assert!(total_bytes < 20_000_000_000); // < 20GB
    }

    #[test]
    fn test_gqa_kv_cache_savings() {
        // Test memory savings from GQA (fewer KV heads)
        let n_layers = 32usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        // MHA: 32 KV heads
        let mha_per_layer = 2 * max_seq_len * 32 * head_dim * 4;
        let mha_total = n_layers * mha_per_layer;

        // GQA: 8 KV heads (4x savings)
        let gqa_per_layer = 2 * max_seq_len * 8 * head_dim * 4;
        let gqa_total = n_layers * gqa_per_layer;

        assert_eq!(mha_total / gqa_total, 4);
    }

    // ========================================================================
    // QWEN-007: Q8 KV Cache Tests
    // ========================================================================

    #[test]
    fn test_q8_kv_cache_init() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Initialize Q8 KV cache
        let result = exec.init_kv_cache_q8_gpu(
            4,   // num_layers
            8,   // num_heads
            4,   // num_kv_heads (GQA)
            128, // head_dim (divisible by 32)
            512, // max_len
        );
        assert!(result.is_ok(), "Q8 KV cache init failed: {:?}", result);
        assert!(exec.is_kv_cache_q8_enabled());
    }

    #[test]
    fn test_q8_kv_cache_invalid_head_dim() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // head_dim not divisible by 32 should fail
        let result = exec.init_kv_cache_q8_gpu(4, 8, 4, 100, 512);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("divisible by 32"));
        }
    }

    #[test]
    fn test_q8_kv_cache_memory_calculation() {
        // Test Q8 memory calculation vs FP32
        let n_layers = 32usize;
        let n_kv_heads = 8usize;
        let head_dim = 128usize;
        let max_seq_len = 4096usize;

        // FP32: 4 bytes per value
        let fp32_bytes = n_layers * 2 * n_kv_heads * max_seq_len * head_dim * 4;

        // Q8: 1 byte per value + 4 bytes per 32 values (scale)
        let q8_values = n_layers * 2 * n_kv_heads * max_seq_len * head_dim * 1;
        let q8_scales = n_layers * 2 * n_kv_heads * max_seq_len * (head_dim / 32) * 4;
        let q8_bytes = q8_values + q8_scales;

        // Q8 should be ~4x smaller (actually 4x / (1 + 1/8) ≈ 3.56x due to scales)
        let reduction = fp32_bytes as f64 / q8_bytes as f64;
        assert!(
            reduction > 3.5,
            "Expected >3.5x reduction, got {:.2}x",
            reduction
        );
        assert!(
            reduction < 4.0,
            "Expected <4x reduction, got {:.2}x",
            reduction
        );
    }

    #[test]
    fn test_q8_kv_cache_memory_methods() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Initialize Q8 KV cache
        exec.init_kv_cache_q8_gpu(4, 8, 4, 128, 512).unwrap();

        let q8_mem = exec.kv_cache_q8_memory_bytes();
        let fp32_equiv = exec.kv_cache_fp32_equivalent_bytes();

        assert!(q8_mem > 0, "Q8 memory should be > 0");
        assert!(fp32_equiv > q8_mem, "FP32 equivalent should be > Q8 memory");

        let reduction = fp32_equiv as f64 / q8_mem as f64;
        assert!(
            reduction > 3.5,
            "Expected >3.5x reduction, got {:.2}x",
            reduction
        );
    }

    #[test]
    fn test_q8_kv_cache_write_read_roundtrip() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_kv_heads = 4;
        let head_dim = 64; // Divisible by 32
        let max_len = 16;

        // Initialize Q8 KV cache
        exec.init_kv_cache_q8_gpu(2, 8, num_kv_heads, head_dim, max_len)
            .unwrap();

        // Create test K/V vectors with known values
        let size = num_kv_heads * head_dim;
        let k: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..size).map(|i| (i as f32) * -0.01).collect();

        // Write to position 0
        exec.write_kv_q8(0, 0, &k, &v).unwrap();

        // Read back
        let (k_out, v_out) = exec.read_kv_q8(0, 0, 1).unwrap();

        // Verify dimensions
        assert_eq!(k_out.len(), size, "K output size mismatch");
        assert_eq!(v_out.len(), size, "V output size mismatch");

        // Verify values are close (Q8 has ~1% quantization error max)
        for i in 0..size {
            let k_err = (k[i] - k_out[i]).abs();
            let v_err = (v[i] - v_out[i]).abs();
            // Allow 1% relative error or 0.01 absolute error
            let k_tol = (k[i].abs() * 0.02).max(0.02);
            let v_tol = (v[i].abs() * 0.02).max(0.02);
            assert!(
                k_err < k_tol,
                "K[{}]: expected {}, got {}, err {} > tol {}",
                i,
                k[i],
                k_out[i],
                k_err,
                k_tol
            );
            assert!(
                v_err < v_tol,
                "V[{}]: expected {}, got {}, err {} > tol {}",
                i,
                v[i],
                v_out[i],
                v_err,
                v_tol
            );
        }
    }

    #[test]
    fn test_q8_kv_cache_multiple_positions() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_kv_heads = 2;
        let head_dim = 32; // Minimal divisible by 32
        let max_len = 8;

        exec.init_kv_cache_q8_gpu(1, 4, num_kv_heads, head_dim, max_len)
            .unwrap();

        let size = num_kv_heads * head_dim;

        // Write to multiple positions
        for pos in 0..4 {
            let k: Vec<f32> = (0..size).map(|i| (pos as f32 + i as f32) * 0.1).collect();
            let v: Vec<f32> = (0..size).map(|i| -(pos as f32 + i as f32) * 0.1).collect();
            exec.write_kv_q8(0, pos, &k, &v).unwrap();
        }

        // Read all positions at once
        let (k_all, v_all) = exec.read_kv_q8(0, 0, 4).unwrap();

        assert_eq!(k_all.len(), 4 * size, "K all size mismatch");
        assert_eq!(v_all.len(), 4 * size, "V all size mismatch");
    }

    #[test]
    fn test_q8_kv_cache_not_enabled_error() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Don't initialize Q8 cache
        let k = vec![1.0f32; 128];
        let v = vec![1.0f32; 128];

        let result = exec.write_kv_q8(0, 0, &k, &v);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enabled"));
    }
