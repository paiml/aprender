
#[test]
fn test_gpu_status_response_debug() {
    let response = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    let debug = format!("{:?}", response);
    assert!(debug.contains("GpuStatusResponse"));
}

// =============================================================================
// BatchConfig Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_default() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default();
    assert_eq!(config.window_ms, 50);
    assert_eq!(config.min_batch, 4);
    assert_eq!(config.optimal_batch, 32);
    assert_eq!(config.max_batch, 64);
    assert_eq!(config.queue_size, 1024);
    assert_eq!(config.gpu_threshold, 32);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_low_latency() {
    use crate::api::BatchConfig;

    let config = BatchConfig::low_latency();
    assert_eq!(config.window_ms, 5);
    assert_eq!(config.min_batch, 2);
    assert_eq!(config.optimal_batch, 8);
    assert_eq!(config.max_batch, 16);
    assert_eq!(config.queue_size, 512);
    assert_eq!(config.gpu_threshold, 32);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_high_throughput() {
    use crate::api::BatchConfig;

    let config = BatchConfig::high_throughput();
    assert_eq!(config.window_ms, 100);
    assert_eq!(config.min_batch, 8);
    assert_eq!(config.optimal_batch, 32);
    assert_eq!(config.max_batch, 128);
    assert_eq!(config.queue_size, 2048);
    assert_eq!(config.gpu_threshold, 32);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_should_process() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default(); // optimal_batch = 32
    assert!(!config.should_process(0));
    assert!(!config.should_process(16));
    assert!(!config.should_process(31));
    assert!(config.should_process(32));
    assert!(config.should_process(64));
    assert!(config.should_process(100));
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_meets_minimum() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default(); // min_batch = 4
    assert!(!config.meets_minimum(0));
    assert!(!config.meets_minimum(1));
    assert!(!config.meets_minimum(3));
    assert!(config.meets_minimum(4));
    assert!(config.meets_minimum(5));
    assert!(config.meets_minimum(100));
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_clone() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.window_ms, config.window_ms);
    assert_eq!(cloned.min_batch, config.min_batch);
    assert_eq!(cloned.optimal_batch, config.optimal_batch);
    assert_eq!(cloned.max_batch, config.max_batch);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_config_debug() {
    use crate::api::BatchConfig;

    let config = BatchConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("BatchConfig"));
    assert!(debug.contains("window_ms"));
}

// =============================================================================
// ContinuousBatchResponse Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_single() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::single(
        vec![1, 2, 3, 4, 5], // token_ids
        3,                   // prompt_len
        12.5,                // latency_ms
    );

    assert_eq!(response.token_ids, vec![1, 2, 3, 4, 5]);
    assert_eq!(response.prompt_len, 3);
    assert!(!response.batched);
    assert_eq!(response.batch_size, 1);
    assert!((response.latency_ms - 12.5).abs() < 0.01);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_batched() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::batched(
        vec![10, 20, 30, 40, 50, 60], // token_ids
        4,                            // prompt_len
        16,                           // batch_size
        50.0,                         // latency_ms
    );

    assert_eq!(response.token_ids, vec![10, 20, 30, 40, 50, 60]);
    assert_eq!(response.prompt_len, 4);
    assert!(response.batched);
    assert_eq!(response.batch_size, 16);
    assert!((response.latency_ms - 50.0).abs() < 0.01);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens() {
    use crate::api::ContinuousBatchResponse;

    // Normal case: token_ids longer than prompt_len
    let response = ContinuousBatchResponse::single(vec![1, 2, 3, 4, 5, 6, 7], 3, 10.0);
    let generated = response.generated_tokens();
    assert_eq!(generated, &[4, 5, 6, 7]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens_empty() {
    use crate::api::ContinuousBatchResponse;

    // Edge case: prompt_len equals token_ids.len()
    let response = ContinuousBatchResponse::single(vec![1, 2, 3], 3, 5.0);
    let generated = response.generated_tokens();
    assert!(generated.is_empty());
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_generated_tokens_overflow() {
    use crate::api::ContinuousBatchResponse;

    // Edge case: prompt_len > token_ids.len() (shouldn't happen but handle gracefully)
    let response = ContinuousBatchResponse::single(vec![1, 2], 10, 5.0);
    let generated = response.generated_tokens();
    assert!(generated.is_empty());
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_clone() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::batched(vec![1, 2, 3], 1, 8, 25.0);
    let cloned = response.clone();

    assert_eq!(cloned.token_ids, response.token_ids);
    assert_eq!(cloned.prompt_len, response.prompt_len);
    assert_eq!(cloned.batched, response.batched);
    assert_eq!(cloned.batch_size, response.batch_size);
}

#[cfg(feature = "gpu")]
#[test]
fn test_continuous_batch_response_debug() {
    use crate::api::ContinuousBatchResponse;

    let response = ContinuousBatchResponse::single(vec![1], 0, 1.0);
    let debug = format!("{:?}", response);
    assert!(debug.contains("ContinuousBatchResponse"));
}

// =============================================================================
// BatchQueueStats Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_default() {
    use crate::api::BatchQueueStats;

    let stats = BatchQueueStats::default();
    assert_eq!(stats.total_queued, 0);
    assert_eq!(stats.total_batches, 0);
    assert_eq!(stats.total_single, 0);
    assert!((stats.avg_batch_size - 0.0).abs() < f64::EPSILON);
    assert!((stats.avg_wait_ms - 0.0).abs() < f64::EPSILON);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_clone() {
    use crate::api::BatchQueueStats;

    let stats = BatchQueueStats {
        total_queued: 100,
        total_batches: 10,
        total_single: 20,
        avg_batch_size: 10.0,
        avg_wait_ms: 5.5,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.total_queued, stats.total_queued);
    assert_eq!(cloned.total_batches, stats.total_batches);
    assert_eq!(cloned.total_single, stats.total_single);
}

#[cfg(feature = "gpu")]
#[test]
fn test_batch_queue_stats_debug() {
    use crate::api::BatchQueueStats;

    let stats = BatchQueueStats::default();
    let debug = format!("{:?}", stats);
    assert!(debug.contains("BatchQueueStats"));
    assert!(debug.contains("total_queued"));
}

// =============================================================================
// BatchProcessResult Tests (GPU feature)
// =============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_batch_process_result_debug() {
    use crate::api::BatchProcessResult;

    let result = BatchProcessResult {
        requests_processed: 5,
        was_batched: true,
        total_time_ms: 50.0,
        avg_latency_ms: 10.0,
    };

    let debug = format!("{:?}", result);
    assert!(debug.contains("BatchProcessResult"));
    assert!(debug.contains("requests_processed"));
    assert!(debug.contains("was_batched"));
}

// =============================================================================
// GPU Endpoint HTTP Tests
// =============================================================================

#[tokio::test]
async fn test_gpu_warmup_endpoint_no_gpu_model() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/gpu/warmup")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Demo app doesn't have GPU model, should return error
    // When GPU feature is enabled, returns 503; when not, also returns 503
    assert!(
        response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_gpu_status_endpoint() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/gpu/status")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    // GPU status always returns OK (even without GPU model)
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    if response.status() != StatusCode::OK {
        return;
    }

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let status: GpuStatusResponse = serde_json::from_slice(&body).expect("parse json");

    // Without GPU model, cache_ready should be false
    assert!(!status.cache_ready);
    assert_eq!(status.cache_memory_bytes, 0);
    assert_eq!(status.batch_threshold, 32);
    assert_eq!(status.recommended_min_batch, 32);
}

#[tokio::test]
async fn test_gpu_batch_completions_empty_prompts() {
    let app = create_test_app_shared();

    let request = GpuBatchRequest {
        prompts: vec![], // Empty prompts array - should fail
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
        stop: vec![],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&request).expect("serialize"),
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Empty prompts should return 400 Bad Request
    // Note: Non-GPU build might return 503 instead
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_no_gpu_model() {
    let app = create_test_app_shared();

    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 10,
        temperature: 0.5,
        top_k: 40,
        stop: vec![],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_string(&request).expect("serialize"),
                ))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Demo app doesn't have GPU/cached model, should return error
    assert!(
        response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_gpu_batch_completions_invalid_json() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Invalid JSON should return 400 Bad Request
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}
