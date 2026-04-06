
#[test]
fn test_streaming_kv_cache_fp16_append_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Append to layer 0
    cache.append(0, &k, &v);
    assert_eq!(cache.len(), 0); // Only advances after last layer

    // Append to layer 1 (last layer)
    cache.append(1, &k, &v);
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_fp16_get_range_f32_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }

    let (keys, values) = cache.get_range_f32(0, 0, 1);
    assert_eq!(keys.len(), kv_dim);
    assert_eq!(values.len(), kv_dim);

    // Check values are approximately correct (FP16 precision)
    for key in &keys {
        assert!((key - 1.0).abs() < 0.01);
    }
    for val in &values {
        assert!((val - 2.0).abs() < 0.01);
    }
}

#[test]
fn test_streaming_kv_cache_fp16_get_range_raw_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }

    let (keys_raw, values_raw) = cache.get_range_raw(0, 0, 1);
    assert_eq!(keys_raw.len(), kv_dim);
    assert_eq!(values_raw.len(), kv_dim);
}

#[test]
fn test_streaming_kv_cache_fp16_get_valid_f32_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    // Add 3 positions
    for _ in 0..3 {
        for layer in 0..2 {
            cache.append(layer, &k, &v);
        }
    }

    let (keys, values) = cache.get_valid_f32(0);
    assert_eq!(keys.len(), 3 * kv_dim);
    assert_eq!(values.len(), 3 * kv_dim);
}

#[test]
fn test_streaming_kv_cache_fp16_clear_ext_cov() {
    let mut cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let kv_dim = 4 * 32;
    let k = vec![1.0f32; kv_dim];
    let v = vec![2.0f32; kv_dim];

    for layer in 0..2 {
        cache.append(layer, &k, &v);
    }
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_streaming_kv_cache_fp16_memory_bytes_ext_cov() {
    let cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let expected_size = 2 * 10 * 4 * 32 * 2 * 2; // layers * pos * heads * dim * kv * fp16
    assert_eq!(cache.memory_bytes(), expected_size);
}

#[test]
fn test_streaming_kv_cache_fp16_memory_mb_ext_cov() {
    let cache = StreamingKVCacheFp16::new(2, 10, 4, 32);
    let bytes = cache.memory_bytes();
    let expected_mb = bytes as f64 / (1024.0 * 1024.0);
    assert!((cache.memory_mb() - expected_mb).abs() < 0.001);
}

// --- GpuModelConfig Tests ---
#[test]
fn test_gpu_model_config_head_dim_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };
    assert_eq!(config.head_dim(), 32); // 256 / 8
}

#[test]
fn test_gpu_model_config_kv_dim_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 4, // GQA
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };
    assert_eq!(config.kv_dim(), 128); // 4 * 32 (num_kv_heads * head_dim)
}

#[test]
fn test_gpu_model_config_qkv_dim_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 4, // GQA
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };
    // qkv_dim = hidden_dim + 2 * kv_dim = 256 + 2 * 128 = 512
    assert_eq!(config.qkv_dim(), 512);
}

#[test]
fn test_gpu_model_config_is_gqa_true_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 4, // Less than num_heads = GQA
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };
    assert!(config.is_gqa());
}

#[test]
fn test_gpu_model_config_is_gqa_false_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8, // Equal = MHA (not GQA)
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };
    assert!(!config.is_gqa());
}

// --- GpuGenerateConfig Tests ---
#[test]
fn test_gpu_generate_config_default_ext_cov() {
    let config = GpuGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_deterministic_ext_cov() {
    let config = GpuGenerateConfig::deterministic(100);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_sampling_ext_cov() {
    let config = GpuGenerateConfig::with_sampling(50, 0.7, 40);
    assert_eq!(config.max_tokens, 50);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_gpu_generate_config_with_stop_tokens_ext_cov() {
    let config = GpuGenerateConfig::deterministic(100).with_stop_tokens(vec![0, 1, 2]);
    assert_eq!(config.stop_tokens, vec![0, 1, 2]);
}

#[test]
fn test_gpu_generate_config_chained_ext_cov() {
    let config = GpuGenerateConfig::with_sampling(100, 0.8, 50).with_stop_tokens(vec![123]);
    assert_eq!(config.max_tokens, 100);
    assert_eq!(config.temperature, 0.8);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.stop_tokens, vec![123]);
}

// --- AttentionBuffers Tests ---
#[test]
fn test_attention_buffers_new_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };
    let buffers = AttentionBuffers::new(&config, 100);

    assert_eq!(buffers.q_buffer.len(), 256);
    assert_eq!(buffers.scores_buffer.len(), 8 * 100); // num_heads * max_seq_len
    assert_eq!(buffers.output_buffer.len(), 256);
    assert_eq!(buffers.kv_proj_buffer.len(), 256);
    assert_eq!(buffers.ffn_buffer.len(), 512);
    assert_eq!(buffers.max_seq_len, 100);
}

#[test]
fn test_attention_buffers_reset_ext_cov() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };
    let mut buffers = AttentionBuffers::new(&config, 100);

    // Fill with non-zero values
    buffers.q_buffer.fill(1.0);
    buffers.scores_buffer.fill(2.0);
    buffers.output_buffer.fill(3.0);
    buffers.kv_proj_buffer.fill(4.0);
    buffers.ffn_buffer.fill(5.0);

    buffers.reset();

    assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.kv_proj_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.ffn_buffer.iter().all(|&x| x == 0.0));
}

// --- WeightType Tests ---
#[test]
fn test_weight_type_variants_ext_cov() {
    let qkv = WeightType::Qkv;
    let output = WeightType::Output;
    let ffn_fc1 = WeightType::FfnFc1;
    let ffn_fc2 = WeightType::FfnFc2;
    let lm_head = WeightType::LmHead;

    // Just ensure they are distinct and can be debug-printed
    let debug_qkv = format!("{:?}", qkv);
    let debug_output = format!("{:?}", output);
    let debug_fc1 = format!("{:?}", ffn_fc1);
    let debug_fc2 = format!("{:?}", ffn_fc2);
    let debug_lm_head = format!("{:?}", lm_head);

    assert!(debug_qkv.contains("Qkv"));
    assert!(debug_output.contains("Output"));
    assert!(debug_fc1.contains("FfnFc1"));
    assert!(debug_fc2.contains("FfnFc2"));
    assert!(debug_lm_head.contains("LmHead"));
}

#[test]
fn test_weight_type_clone_ext_cov() {
    let original = WeightType::Qkv;
    let cloned = original;
    assert!(matches!(cloned, WeightType::Qkv));
}

// --- ComputeBackend Tests ---
#[test]
fn test_compute_backend_default_ext_cov() {
    let backend = ComputeBackend::default();
    assert!(matches!(backend, ComputeBackend::Auto));
}

#[test]
fn test_compute_backend_variants_ext_cov() {
    let gpu = ComputeBackend::Gpu;
    let cpu = ComputeBackend::Cpu;
    let auto = ComputeBackend::Auto;

    assert!(matches!(gpu, ComputeBackend::Gpu));
    assert!(matches!(cpu, ComputeBackend::Cpu));
    assert!(matches!(auto, ComputeBackend::Auto));
}

#[test]
fn test_compute_backend_equality_ext_cov() {
    assert_eq!(ComputeBackend::Gpu, ComputeBackend::Gpu);
    assert_eq!(ComputeBackend::Cpu, ComputeBackend::Cpu);
    assert_eq!(ComputeBackend::Auto, ComputeBackend::Auto);
    assert_ne!(ComputeBackend::Gpu, ComputeBackend::Cpu);
}

#[test]
fn test_compute_backend_clone_ext_cov() {
    let original = ComputeBackend::Gpu;
    let cloned = original;
    assert_eq!(cloned, ComputeBackend::Gpu);
}

// ============================================================================
// Deep Coverage Tests (_deep_gcov_ prefix)
// Testing: error handling paths, fallback paths, scheduler initialization,
//          memory allocation failures, batch processing edge cases
// ============================================================================

// --- Scheduler Fallback and Error Handling Tests ---
#[test]
fn test_scheduler_fallback_when_cuda_unavailable_deep_gcov() {
    // Test that HybridScheduler gracefully falls back to CPU when GPU isn't available
    // or when workload is too small
    let mut scheduler = HybridScheduler::with_threshold(1_000_000).expect("test");

    // With very high threshold, should use CPU for small operations
    assert!(!scheduler.should_use_gpu(2, 2, 2)); // 8 elements << 1_000_000

    // Matmul should still work via CPU fallback
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = scheduler.matmul(&a, &b, 2, 2, 2);
    assert!(result.is_ok());
    let c = result.expect("GPU operation failed");
    assert!((c[0] - 19.0).abs() < 1e-5);
}

#[test]
fn test_batch_processing_empty_input_deep_gcov() {
    // Test batch processing with empty batches
    let mut scheduler = HybridScheduler::new().expect("test");

    let empty_ops: Vec<MatmulOp> = vec![];
    let result = scheduler.matmul_batch(&empty_ops);
    assert!(result.is_ok());
    assert!(result.expect("GPU operation failed").is_empty());
}

#[test]
fn test_batch_processing_single_op_deep_gcov() {
    // Test batch processing with single operation
    let mut scheduler = HybridScheduler::new().expect("test");

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let ops = vec![(a, b, 2, 2, 2)];

    let result = scheduler.matmul_batch(&ops);
    assert!(result.is_ok());
    let results = result.expect("GPU operation failed");
    assert_eq!(results.len(), 1);
    assert!((results[0][0] - 19.0).abs() < 1e-5);
}

#[test]
fn test_batch_processing_multiple_ops_deep_gcov() {
    // Test batch processing with multiple operations
    let mut scheduler = HybridScheduler::new().expect("test");

    let ops = vec![
        (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0], 2, 2, 2),
        (vec![1.0, 0.0, 0.0, 1.0], vec![2.0, 0.0, 0.0, 2.0], 2, 2, 2), // Identity-like
    ];

    let result = scheduler.matmul_batch(&ops);
    assert!(result.is_ok());
    let results = result.expect("GPU operation failed");
    assert_eq!(results.len(), 2);
}

#[test]
fn test_gpu_compute_matmul_zero_dimensions_deep_gcov() {
    // Test matmul with zero-size inputs (edge case)
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");

    // Empty matrices
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    // Should handle 0x0 @ 0x0 gracefully (returns empty result)
    let result = compute.matmul(&a, &b, 0, 0, 0);
    assert!(result.is_ok());
    assert!(result.expect("GPU operation failed").is_empty());
}

#[test]
fn test_gpu_compute_matmul_large_k_dimension_deep_gcov() {
    // Test matmul with large inner dimension
    let mut compute = GpuCompute::new(ComputeBackend::Cpu).expect("test");

    let k = 128;
    let a: Vec<f32> = vec![0.1; k]; // 1 x k
    let b: Vec<f32> = vec![0.1; k]; // k x 1

    let result = compute.matmul(&a, &b, 1, k, 1);
    assert!(result.is_ok());
    let c = result.expect("GPU operation failed");
    assert_eq!(c.len(), 1);
    // 1x128 @ 128x1 = 1x1, each element contributes 0.01
    assert!((c[0] - (k as f32 * 0.01)).abs() < 1e-4);
}
