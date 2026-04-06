
impl Default for DispatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GGUFConfig {
        GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 2,
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

    // ============================================================================
    // InferenceScratchBuffer tests
    // ============================================================================

    #[test]
    fn test_inference_scratch_buffer_from_config() {
        let config = test_config();
        let buf = InferenceScratchBuffer::from_config(&config);

        assert_eq!(buf.hidden.len(), 64);
        assert_eq!(buf.normed.len(), 64);
        assert_eq!(buf.qkv.len(), 64 * 3); // hidden_dim * 3
        assert_eq!(buf.logits.len(), 100); // vocab_size
        assert_eq!(buf.ffn_up.len(), 128); // intermediate_dim
    }

    #[test]
    fn test_inference_scratch_buffer_reset() {
        let config = test_config();
        let mut buf = InferenceScratchBuffer::from_config(&config);

        // Set some values
        buf.hidden[0] = 1.0;
        buf.normed[0] = 2.0;

        buf.reset();

        assert!((buf.hidden[0] - 0.0).abs() < f32::EPSILON);
        assert!((buf.normed[0] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_scratch_buffer_q8k_buffers() {
        let config = test_config();
        let buf = InferenceScratchBuffer::from_config(&config);

        // Q8K uses 256-element super-blocks
        // hidden_dim=64 -> ceil(64/256)=1 super-block -> 1 scale
        assert!(!buf.q8k_hidden_scales.is_empty());
        assert!(buf.q8k_hidden_quants.len() >= 64);
    }

    #[test]
    fn test_inference_scratch_buffer_debug() {
        let config = test_config();
        let buf = InferenceScratchBuffer::from_config(&config);
        let debug = format!("{:?}", buf);
        assert!(debug.contains("InferenceScratchBuffer"));
    }

    // ============================================================================
    // OwnedInferenceScratchBuffer tests
    // ============================================================================

    #[test]
    fn test_owned_scratch_buffer_from_config() {
        let config = test_config();
        let buf = OwnedInferenceScratchBuffer::from_config(&config);

        // qkv = hidden_dim + 2 * kv_dim
        // kv_dim = num_kv_heads * head_dim = 4 * 16 = 64
        // qkv = 64 + 2 * 64 = 192
        let head_dim = 64 / 4; // 16
        let kv_dim = 4 * head_dim;
        let expected_qkv = 64 + 2 * kv_dim;
        assert_eq!(buf.qkv.len(), expected_qkv);
        assert_eq!(buf.attn_out.len(), 64);
        assert_eq!(buf.logits.len(), 100);
    }

    #[test]
    fn test_owned_scratch_buffer_reset() {
        let config = test_config();
        let mut buf = OwnedInferenceScratchBuffer::from_config(&config);

        // Add some data
        buf.qkv.push(1.0);
        buf.attn_out.push(2.0);

        buf.reset();

        // All vectors should be cleared
        assert!(buf.qkv.is_empty());
        assert!(buf.attn_out.is_empty());
        assert!(buf.ffn_up.is_empty());
        assert!(buf.logits.is_empty());
    }

    #[test]
    fn test_owned_scratch_buffer_debug() {
        let config = test_config();
        let buf = OwnedInferenceScratchBuffer::from_config(&config);
        let debug = format!("{:?}", buf);
        assert!(debug.contains("OwnedInferenceScratchBuffer"));
    }

    // ============================================================================
    // ContiguousKVCache tests
    // ============================================================================

    #[test]
    fn test_contiguous_kv_cache_new() {
        let cache = ContiguousKVCache::new(2, 64, 512);
        assert!(cache.is_contiguous());
        assert!(cache.is_cache_aligned());
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 512);
    }

    #[test]
    fn test_contiguous_kv_cache_from_config() {
        let config = test_config();
        let cache = ContiguousKVCache::from_config(&config, 256);
        assert!(cache.is_contiguous());
        assert_eq!(cache.max_len(), 256);
    }

    #[test]
    fn test_contiguous_kv_cache_append_and_advance() {
        let mut cache = ContiguousKVCache::new(2, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        let cached_k = cache.get_k(0);
        assert_eq!(cached_k.len(), 4);
        assert!((cached_k[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_contiguous_kv_cache_get_k_v() {
        let mut cache = ContiguousKVCache::new(2, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        let cached_k = cache.get_k(0);
        let cached_v = cache.get_v(0);
        assert_eq!(cached_k.len(), 4);
        assert_eq!(cached_v.len(), 4);
        assert!((cached_v[0] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_contiguous_kv_cache_get_k_v_mut() {
        let mut cache = ContiguousKVCache::new(1, 4, 10);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        let k_mut = cache.get_k_mut(0);
        k_mut[0] = 99.0;

        let cached_k = cache.get_k(0);
        assert!((cached_k[0] - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_contiguous_kv_cache_reset() {
        let mut cache = ContiguousKVCache::new(1, 4, 10);
        cache.append(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_contiguous_kv_cache_reset_and_zero() {
        let mut cache = ContiguousKVCache::new(1, 4, 10);
        cache.append(0, &[1.0; 4], &[2.0; 4]);
        cache.advance();

        cache.reset_and_zero();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_contiguous_kv_cache_memory_bytes() {
        let cache = ContiguousKVCache::new(2, 64, 128);
        let mem = cache.memory_bytes();
        // At least 2 layers * 128 * 64 * 4 bytes * 2 (k+v) = 131072
        assert!(mem >= 131072);
    }

    #[test]
    fn test_contiguous_kv_cache_layer_stride() {
        let cache = ContiguousKVCache::new(2, 64, 128);
        let stride = cache.layer_stride();
        // Should be cache-line aligned
        assert!(stride.is_multiple_of(FLOATS_PER_CACHE_LINE));
    }

    #[test]
    fn test_contiguous_kv_cache_invalid_layer() {
        let cache = ContiguousKVCache::new(2, 4, 10);
        // Invalid layer returns empty slice
        assert!(cache.get_k(99).is_empty());
        assert!(cache.get_v(99).is_empty());
    }

    #[test]
    fn test_contiguous_kv_cache_prefetch() {
        let cache = ContiguousKVCache::new(2, 64, 128);
        // Prefetch should not panic
        cache.prefetch_k(0);
        cache.prefetch_v(0);
        cache.prefetch_k(99); // Invalid layer should be safe
    }

    // ============================================================================
    // DispatchMetrics tests
    // ============================================================================

    #[test]
    fn test_dispatch_metrics_new() {
        let metrics = DispatchMetrics::new();
        assert_eq!(metrics.cpu_dispatches(), 0);
        assert_eq!(metrics.gpu_dispatches(), 0);
        assert_eq!(metrics.total_dispatches(), 0);
    }

    #[test]
    fn test_dispatch_metrics_default() {
        let metrics = DispatchMetrics::default();
        assert_eq!(metrics.cpu_dispatches(), 0);
    }

    #[test]
    fn test_dispatch_metrics_record_cpu() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_dispatch();
        metrics.record_cpu_dispatch();
        assert_eq!(metrics.cpu_dispatches(), 2);
        assert_eq!(metrics.gpu_dispatches(), 0);
    }

    #[test]
    fn test_dispatch_metrics_record_gpu() {
        let metrics = DispatchMetrics::new();
        metrics.record_gpu_dispatch();
        assert_eq!(metrics.gpu_dispatches(), 1);
    }

    #[test]
    fn test_dispatch_metrics_gpu_ratio() {
        let metrics = DispatchMetrics::new();
        assert!((metrics.gpu_ratio() - 0.0).abs() < f64::EPSILON);

        metrics.record_cpu_dispatch();
        metrics.record_gpu_dispatch();
        assert!((metrics.gpu_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dispatch_metrics_cpu_latency() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        metrics.record_cpu_latency(std::time::Duration::from_micros(200));

        assert_eq!(metrics.cpu_latency_count(), 2);
        assert!((metrics.cpu_latency_mean_us() - 150.0).abs() < 1.0);
        assert_eq!(metrics.cpu_latency_min_us(), 100);
        assert_eq!(metrics.cpu_latency_max_us(), 200);
    }

    #[test]
    fn test_dispatch_metrics_gpu_latency() {
        let metrics = DispatchMetrics::new();
        metrics.record_gpu_latency(std::time::Duration::from_micros(50));

        assert_eq!(metrics.gpu_latency_count(), 1);
        assert!((metrics.gpu_latency_mean_us() - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_dispatch_metrics_buckets() {
        let metrics = DispatchMetrics::new();
        // <100us -> bucket 0
        metrics.record_cpu_latency(std::time::Duration::from_micros(50));
        // 100-500us -> bucket 1
        metrics.record_cpu_latency(std::time::Duration::from_micros(200));
        // 500-1000us -> bucket 2
        metrics.record_cpu_latency(std::time::Duration::from_micros(700));

        let buckets = metrics.cpu_latency_buckets();
        assert_eq!(buckets[0], 1);
        assert_eq!(buckets[1], 1);
        assert_eq!(buckets[2], 1);
    }

    #[test]
    fn test_dispatch_metrics_variance_and_stddev() {
        let metrics = DispatchMetrics::new();
        // With only 1 sample, variance should be 0
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        assert!((metrics.cpu_latency_variance_us() - 0.0).abs() < 0.001);

        // With 2 identical samples, variance should be 0
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));
        assert!(metrics.cpu_latency_variance_us().is_finite());
    }

    #[test]
    fn test_dispatch_metrics_percentiles() {
        let metrics = DispatchMetrics::new();
        // All in first bucket
        for _ in 0..100 {
            metrics.record_cpu_latency(std::time::Duration::from_micros(50));
        }

        let p50 = metrics.cpu_latency_p50_us();
        let p95 = metrics.cpu_latency_p95_us();
        let p99 = metrics.cpu_latency_p99_us();
        assert!(p50 >= 0.0);
        assert!(p95 >= p50);
        assert!(p99 >= p95);
    }

    #[test]
    fn test_dispatch_metrics_bucket_boundaries() {
        let metrics = DispatchMetrics::new();
        let boundaries = metrics.bucket_boundaries_us();
        assert_eq!(boundaries.len(), 5);
        assert!(boundaries[0].contains("0-"));
    }

    #[test]
    fn test_dispatch_metrics_reset() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_dispatch();
        metrics.record_gpu_dispatch();
        metrics.record_cpu_latency(std::time::Duration::from_micros(100));

        metrics.reset();

        assert_eq!(metrics.cpu_dispatches(), 0);
        assert_eq!(metrics.gpu_dispatches(), 0);
        assert_eq!(metrics.cpu_latency_count(), 0);
    }

    #[test]
    fn test_dispatch_metrics_speedup() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_latency(std::time::Duration::from_micros(1000));
        metrics.record_gpu_latency(std::time::Duration::from_micros(100));

        let speedup = metrics.cpu_gpu_speedup();
        assert!((speedup - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_dispatch_metrics_cv() {
        let metrics = DispatchMetrics::new();
        // No samples -> CV = 0
        assert!((metrics.cpu_latency_cv() - 0.0).abs() < 0.001);
        assert!((metrics.gpu_latency_cv() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dispatch_metrics_elapsed_and_throughput() {
        let metrics = DispatchMetrics::new();
        metrics.record_cpu_dispatch();

        // Elapsed should be very small
        let elapsed = metrics.elapsed_seconds();
        assert!(elapsed >= 0.0);

        // Throughput calculation
        let _throughput = metrics.throughput_rps();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_dispatch_metrics_debug() {
        let metrics = DispatchMetrics::new();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("DispatchMetrics"));
    }
}
