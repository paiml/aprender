use super::*;

// =========================================================================
// GpuBatchRequest tests
// =========================================================================

#[test]
fn test_gpu_batch_request_basic() {
    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec![],
    };
    assert_eq!(request.prompts.len(), 2);
    assert_eq!(request.max_tokens, 50);
    assert!((request.temperature - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_gpu_batch_request_serialization() {
    let request = GpuBatchRequest {
        prompts: vec!["test".to_string()],
        max_tokens: 100,
        temperature: 0.5,
        top_k: 10,
        stop: vec!["END".to_string()],
    };
    let json = serde_json::to_string(&request).expect("serialize");
    assert!(json.contains("test"));
    assert!(json.contains("100"));
    assert!(json.contains("END"));
}

#[test]
fn test_gpu_batch_request_deserialization() {
    let json = r#"{"prompts": ["hello", "world"], "max_tokens": 50}"#;
    let request: GpuBatchRequest = serde_json::from_str(json).expect("deserialize");
    assert_eq!(request.prompts.len(), 2);
    assert_eq!(request.max_tokens, 50);
}

#[test]
fn test_gpu_batch_request_clone() {
    let request = GpuBatchRequest {
        prompts: vec!["a".to_string()],
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop: vec![],
    };
    let cloned = request.clone();
    assert_eq!(request.prompts, cloned.prompts);
}

#[test]
fn test_gpu_batch_request_debug() {
    let request = GpuBatchRequest {
        prompts: vec![],
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop: vec![],
    };
    let debug = format!("{:?}", request);
    assert!(debug.contains("GpuBatchRequest"));
}

// =========================================================================
// GpuBatchResponse tests
// =========================================================================

#[test]
fn test_gpu_batch_response_basic() {
    let response = GpuBatchResponse {
        results: vec![GpuBatchResult {
            index: 0,
            token_ids: vec![1, 2, 3],
            text: "hello".to_string(),
            num_generated: 3,
        }],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: true,
            total_tokens: 3,
            processing_time_ms: 10.0,
            throughput_tps: 300.0,
        },
    };
    assert_eq!(response.results.len(), 1);
    assert!(response.stats.gpu_used);
}

#[test]
fn test_gpu_batch_response_serialization() {
    let response = GpuBatchResponse {
        results: vec![],
        stats: GpuBatchStats {
            batch_size: 4,
            gpu_used: false,
            total_tokens: 100,
            processing_time_ms: 50.0,
            throughput_tps: 2000.0,
        },
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("batch_size"));
    assert!(json.contains("gpu_used"));
}

#[test]
fn test_gpu_batch_response_clone() {
    let response = GpuBatchResponse {
        results: vec![],
        stats: GpuBatchStats {
            batch_size: 1,
            gpu_used: false,
            total_tokens: 10,
            processing_time_ms: 5.0,
            throughput_tps: 2000.0,
        },
    };
    let cloned = response.clone();
    assert_eq!(response.stats.batch_size, cloned.stats.batch_size);
}

// =========================================================================
// GpuBatchResult tests
// =========================================================================

#[test]
fn test_gpu_batch_result_basic() {
    let result = GpuBatchResult {
        index: 5,
        token_ids: vec![10, 20, 30],
        text: "generated text".to_string(),
        num_generated: 3,
    };
    assert_eq!(result.index, 5);
    assert_eq!(result.token_ids.len(), 3);
    assert_eq!(result.num_generated, 3);
}

#[test]
fn test_gpu_batch_result_serialization() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![1, 2],
        text: "hi".to_string(),
        num_generated: 2,
    };
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("index"));
    assert!(json.contains("token_ids"));
    assert!(json.contains("hi"));
}

#[test]
fn test_gpu_batch_result_clone() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![1],
        text: "test".to_string(),
        num_generated: 1,
    };
    let cloned = result.clone();
    assert_eq!(result.text, cloned.text);
}

#[test]
fn test_gpu_batch_result_debug() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![],
        text: String::new(),
        num_generated: 0,
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("GpuBatchResult"));
}

// =========================================================================
// GpuBatchStats tests
// =========================================================================

#[test]
fn test_gpu_batch_stats_basic() {
    let stats = GpuBatchStats {
        batch_size: 8,
        gpu_used: true,
        total_tokens: 256,
        processing_time_ms: 100.0,
        throughput_tps: 2560.0,
    };
    assert_eq!(stats.batch_size, 8);
    assert!(stats.gpu_used);
    assert_eq!(stats.total_tokens, 256);
}

#[test]
fn test_gpu_batch_stats_serialization() {
    let stats = GpuBatchStats {
        batch_size: 4,
        gpu_used: false,
        total_tokens: 100,
        processing_time_ms: 50.0,
        throughput_tps: 2000.0,
    };
    let json = serde_json::to_string(&stats).expect("serialize");
    assert!(json.contains("batch_size"));
    assert!(json.contains("throughput_tps"));
}

#[test]
fn test_gpu_batch_stats_clone() {
    let stats = GpuBatchStats {
        batch_size: 1,
        gpu_used: false,
        total_tokens: 10,
        processing_time_ms: 5.0,
        throughput_tps: 2000.0,
    };
    let cloned = stats.clone();
    assert_eq!(stats.total_tokens, cloned.total_tokens);
}

#[test]
fn test_gpu_batch_stats_debug() {
    let stats = GpuBatchStats {
        batch_size: 1,
        gpu_used: false,
        total_tokens: 0,
        processing_time_ms: 0.0,
        throughput_tps: 0.0,
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("GpuBatchStats"));
}

// =========================================================================
// GpuWarmupResponse tests
// =========================================================================

#[test]
fn test_gpu_warmup_response_success() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 1_000_000,
        num_layers: 12,
        message: "Warmup complete".to_string(),
    };
    assert!(response.success);
    assert_eq!(response.num_layers, 12);
}

#[test]
fn test_gpu_warmup_response_failure() {
    let response = GpuWarmupResponse {
        success: false,
        memory_bytes: 0,
        num_layers: 0,
        message: "GPU not available".to_string(),
    };
    assert!(!response.success);
}

#[test]
fn test_gpu_warmup_response_serialization() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 500_000,
        num_layers: 6,
        message: "OK".to_string(),
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("success"));
    assert!(json.contains("memory_bytes"));
}

#[test]
fn test_gpu_warmup_response_clone() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 100,
        num_layers: 1,
        message: "test".to_string(),
    };
    let cloned = response.clone();
    assert_eq!(response.message, cloned.message);
}

#[test]
fn test_gpu_warmup_response_debug() {
    let response = GpuWarmupResponse {
        success: false,
        memory_bytes: 0,
        num_layers: 0,
        message: String::new(),
    };
    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuWarmupResponse"));
}

// =========================================================================
// GpuStatusResponse tests
// =========================================================================

#[test]
fn test_gpu_status_response_ready() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 2_000_000,
        batch_threshold: 32,
        recommended_min_batch: 4,
    };
    assert!(response.cache_ready);
    assert_eq!(response.batch_threshold, 32);
}

#[test]
fn test_gpu_status_response_not_ready() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 4,
    };
    assert!(!response.cache_ready);
}

#[test]
fn test_gpu_status_response_serialization() {
    let response = GpuStatusResponse {
        cache_ready: true,
        cache_memory_bytes: 1000,
        batch_threshold: 16,
        recommended_min_batch: 2,
    };
    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("cache_ready"));
    assert!(json.contains("batch_threshold"));
}

#[test]
fn test_gpu_status_response_clone() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 8,
        recommended_min_batch: 1,
    };
    let cloned = response.clone();
    assert_eq!(response.batch_threshold, cloned.batch_threshold);
}

#[test]
fn test_gpu_status_response_debug() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 4,
    };
    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuStatusResponse"));
}

// =========================================================================
// BatchConfig tests (feature = "gpu")
// =========================================================================

#[cfg(feature = "gpu")]
mod batch_config_tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.window_ms, 50);
        assert_eq!(config.min_batch, 4);
        assert_eq!(config.optimal_batch, 32);
        assert_eq!(config.max_batch, 64);
        assert_eq!(config.queue_size, 1024);
        assert_eq!(config.gpu_threshold, 32);
    }

    #[test]
    fn test_batch_config_low_latency() {
        let config = BatchConfig::low_latency();
        assert_eq!(config.window_ms, 5);
        assert_eq!(config.min_batch, 2);
        assert_eq!(config.optimal_batch, 8);
        assert_eq!(config.max_batch, 16);
    }

    #[test]
    fn test_batch_config_high_throughput() {
        let config = BatchConfig::high_throughput();
        assert_eq!(config.window_ms, 100);
        assert_eq!(config.min_batch, 8);
        assert_eq!(config.optimal_batch, 32);
        assert_eq!(config.max_batch, 128);
    }

    #[test]
    fn test_batch_config_should_process_true() {
        let config = BatchConfig::default();
        assert!(config.should_process(32));
        assert!(config.should_process(64));
    }

    #[test]
    fn test_batch_config_should_process_false() {
        let config = BatchConfig::default();
        assert!(!config.should_process(31));
        assert!(!config.should_process(1));
    }

    #[test]
    fn test_batch_config_meets_minimum_true() {
        let config = BatchConfig::default();
        assert!(config.meets_minimum(4));
        assert!(config.meets_minimum(100));
    }

    #[test]
    fn test_batch_config_meets_minimum_false() {
        let config = BatchConfig::default();
        assert!(!config.meets_minimum(3));
        assert!(!config.meets_minimum(0));
    }

    #[test]
    fn test_batch_config_clone() {
        let config = BatchConfig::high_throughput();
        let cloned = config.clone();
        assert_eq!(config.max_batch, cloned.max_batch);
    }

    #[test]
    fn test_batch_config_debug() {
        let config = BatchConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("BatchConfig"));
    }
}

include!("gpu_handlers_tests_continuous_batch.rs");
