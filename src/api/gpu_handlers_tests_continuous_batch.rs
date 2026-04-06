
// =========================================================================
// ContinuousBatchResponse tests (feature = "gpu")
// =========================================================================

#[cfg(feature = "gpu")]
mod continuous_batch_response_tests {
    use super::*;

    #[test]
    fn test_continuous_batch_response_single() {
        let response = ContinuousBatchResponse::single(vec![1, 2, 3], 1, 5.0);
        assert_eq!(response.token_ids, vec![1, 2, 3]);
        assert_eq!(response.prompt_len, 1);
        assert!(!response.batched);
        assert_eq!(response.batch_size, 1);
        assert!((response.latency_ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_continuous_batch_response_batched() {
        let response = ContinuousBatchResponse::batched(vec![4, 5, 6], 2, 8, 10.0);
        assert_eq!(response.token_ids, vec![4, 5, 6]);
        assert_eq!(response.prompt_len, 2);
        assert!(response.batched);
        assert_eq!(response.batch_size, 8);
    }

    #[test]
    fn test_continuous_batch_response_clone() {
        let response = ContinuousBatchResponse::single(vec![1], 1, 1.0);
        let cloned = response.clone();
        assert_eq!(response.batched, cloned.batched);
    }

    #[test]
    fn test_continuous_batch_response_debug() {
        let response = ContinuousBatchResponse::single(vec![], 0, 0.0);
        let debug = format!("{:?}", response);
        assert!(debug.contains("ContinuousBatchResponse"));
    }
}

// =========================================================================
// Edge case tests
// =========================================================================

#[test]
fn test_gpu_batch_request_empty_prompts() {
    let request = GpuBatchRequest {
        prompts: vec![],
        max_tokens: 10,
        temperature: 0.0,
        top_k: 1,
        stop: vec![],
    };
    assert!(request.prompts.is_empty());
}

#[test]
fn test_gpu_batch_result_empty_tokens() {
    let result = GpuBatchResult {
        index: 0,
        token_ids: vec![],
        text: String::new(),
        num_generated: 0,
    };
    assert!(result.token_ids.is_empty());
    assert_eq!(result.num_generated, 0);
}

#[test]
fn test_gpu_batch_stats_zero_throughput() {
    let stats = GpuBatchStats {
        batch_size: 1,
        gpu_used: false,
        total_tokens: 0,
        processing_time_ms: 0.0,
        throughput_tps: 0.0,
    };
    assert!((stats.throughput_tps - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_gpu_warmup_response_large_memory() {
    let response = GpuWarmupResponse {
        success: true,
        memory_bytes: 8_000_000_000, // 8GB
        num_layers: 96,
        message: "Large model".to_string(),
    };
    assert_eq!(response.memory_bytes, 8_000_000_000);
}
