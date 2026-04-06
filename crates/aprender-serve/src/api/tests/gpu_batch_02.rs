
#[test]
fn test_gpu_batch_stats_gpu_used_flag() {
    let stats_gpu = GpuBatchStats {
        batch_size: 32,
        gpu_used: true,
        total_tokens: 320,
        processing_time_ms: 10.0,
        throughput_tps: 32000.0,
    };
    assert!(stats_gpu.gpu_used);

    let stats_cpu = GpuBatchStats {
        batch_size: 4,
        gpu_used: false,
        total_tokens: 40,
        processing_time_ms: 100.0,
        throughput_tps: 400.0,
    };
    assert!(!stats_cpu.gpu_used);
}

// ============================================================================
// GpuStatusResponse Chaos Tests
// ============================================================================

#[test]
fn test_gpu_status_response_not_ready() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    assert!(!response.cache_ready);
    assert_eq!(response.cache_memory_bytes, 0);
}

#[test]
fn test_gpu_status_response_ready_with_memory() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 24_000_000_000, // 24GB
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    assert!(response.cache_ready);
    assert_eq!(response.cache_memory_bytes, 24_000_000_000);
}

#[test]
fn test_gpu_status_response_zero_thresholds() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1000,
        batch_threshold: 0,
        recommended_min_batch: 0,
    };

    assert_eq!(response.batch_threshold, 0);
    assert_eq!(response.recommended_min_batch, 0);
}

// ============================================================================
// BatchConfig Boundary Tests
// ============================================================================

#[test]
fn test_batch_config_should_process_boundary() {
    let config = BatchConfig::default();

    // Test exactly at optimal_batch
    assert!(config.should_process(config.optimal_batch));

    // Test one below
    assert!(!config.should_process(config.optimal_batch - 1));

    // Test one above
    assert!(config.should_process(config.optimal_batch + 1));
}

#[test]
fn test_batch_config_meets_minimum_boundary() {
    let config = BatchConfig::default();

    // Test exactly at min_batch
    assert!(config.meets_minimum(config.min_batch));

    // Test one below
    assert!(!config.meets_minimum(config.min_batch - 1));

    // Test zero
    assert!(!config.meets_minimum(0));
}

// ============================================================================
// Clone and Debug Coverage
// ============================================================================

#[test]
fn test_batch_config_clone_eq() {
    let config1 = BatchConfig::default();
    let config2 = config1.clone();

    assert_eq!(config1.window_ms, config2.window_ms);
    assert_eq!(config1.min_batch, config2.min_batch);
    assert_eq!(config1.optimal_batch, config2.optimal_batch);
    assert_eq!(config1.max_batch, config2.max_batch);
    assert_eq!(config1.queue_size, config2.queue_size);
    assert_eq!(config1.gpu_threshold, config2.gpu_threshold);
}

#[test]
fn test_batch_config_debug() {
    let config = BatchConfig::default();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("BatchConfig"));
    assert!(debug_str.contains("window_ms"));
    assert!(debug_str.contains("min_batch"));
}

#[test]
fn test_gpu_batch_request_clone() {
    let request = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 10,
        temperature: 0.5,
        top_k: 20,
        stop: vec!["STOP".to_string()],
    };
    let cloned = request.clone();

    assert_eq!(request.prompts, cloned.prompts);
    assert_eq!(request.temperature, cloned.temperature);
    assert_eq!(request.stop, cloned.stop);
}

#[test]
fn test_gpu_batch_response_clone() {
    let response = GpuBatchResponse {
        results: vec![],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 1,
            processing_time_ms: 1.0,
            throughput_tps: 1.0,
        },
    };
    let cloned = response.clone();

    assert_eq!(response.stats.batch_size, cloned.stats.batch_size);
}

#[test]
fn test_gpu_status_response_clone() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1000,
        batch_threshold: 32,
        recommended_min_batch: 16,
    };
    let cloned = response.clone();

    assert_eq!(response.cache_ready, cloned.cache_ready);
    assert_eq!(response.cache_memory_bytes, cloned.cache_memory_bytes);
}

// ============================================================================
// Serialization Chaos Tests
// ============================================================================

#[test]
fn test_gpu_batch_request_json_roundtrip() {
    let request = GpuBatchRequest {
        prompts: vec!["hello".to_string(), "world".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec![],
    };

    let json = serde_json::to_string(&request).expect("JSON serialization failed");
    let decoded: GpuBatchRequest = serde_json::from_str(&json).expect("JSON deserialization failed");

    assert_eq!(decoded.prompts, request.prompts);
    assert_eq!(decoded.max_tokens, request.max_tokens);
}

#[test]
fn test_gpu_batch_response_json_roundtrip() {
    let response = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "test".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 3,
            processing_time_ms: 10.0,
            throughput_tps: 300.0,
        },
    };

    let json = serde_json::to_string(&response).expect("JSON serialization failed");
    let decoded: GpuBatchResponse = serde_json::from_str(&json).expect("JSON deserialization failed");

    assert_eq!(decoded.results.len(), response.results.len());
}

#[test]
fn test_gpu_status_response_json_roundtrip() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 24_000_000_000,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let json = serde_json::to_string(&response).expect("JSON serialization failed");
    let decoded: GpuStatusResponse = serde_json::from_str(&json).expect("JSON deserialization failed");

    assert_eq!(decoded.cache_ready, response.cache_ready);
    assert_eq!(decoded.cache_memory_bytes, response.cache_memory_bytes);
}

// ============================================================================
// Memory Pressure Simulation
// ============================================================================

#[test]
fn test_large_batch_result_allocation() {
    // Simulate large batch result
    let token_ids: Vec<u32> = (0..100_000).collect();

    let result = GpuBatchResult {
        index: 0,
        token_ids,
        text: "x".repeat(100_000),
        num_generated: 100_000,
    };

    assert_eq!(result.token_ids.len(), 100_000);
    assert_eq!(result.text.len(), 100_000);
}

#[test]
fn test_large_batch_response_allocation() {
    // Many results
    let mut results = Vec::with_capacity(1000);
    for i in 0..1000 {
        results.push(GpuBatchResult {
            index: i,
            token_ids: vec![i as u32],
            text: format!("result_{}", i),
            num_generated: 1,
        });
    }

    let response = GpuBatchResponse {
        results,
        stats: GpuBatchStats {
            batch_size: 1000,
            gpu_used: true,
            total_tokens: 1000,
            processing_time_ms: 1.0,
            throughput_tps: 1000000.0,
        },
    };

    assert_eq!(response.results.len(), 1000);
}

// ============================================================================
// Config Preset Tests
// ============================================================================

#[test]
fn test_batch_config_presets() {
    let default_config = BatchConfig::default();
    assert_eq!(default_config.window_ms, 50);
    assert_eq!(default_config.min_batch, 4);
    assert_eq!(default_config.optimal_batch, 32);
    assert_eq!(default_config.max_batch, 64);

    let low_latency = BatchConfig::low_latency();
    assert!(low_latency.window_ms < default_config.window_ms);
    assert!(low_latency.max_batch < default_config.max_batch);

    let high_throughput = BatchConfig::high_throughput();
    assert!(high_throughput.window_ms >= default_config.window_ms);
    assert!(high_throughput.max_batch > default_config.max_batch);
}
