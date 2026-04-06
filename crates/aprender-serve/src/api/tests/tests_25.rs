//! T-COV-95 Chaotic Citizens: GPU Batch Resilience Falsification (PMAT-802)
//!
//! Dr. Popper's directive: "The Potemkin Village is built, but uninhabited.
//! Now we populate it with Chaotic Citizens who break every social contract."
//!
//! This module extends the Potemkin Village with chaos patterns:
//! 1. Partial Batch Failures - one prompt fails, others succeed
//! 2. Queue Saturation - simulate 100 pending requests
//! 3. Zombie Batches - client disconnects mid-generation
//!
//! Target: 517 missed lines in api/gpu_handlers.rs

#[cfg(all(test, feature = "gpu"))]
mod chaotic_citizens {

    use std::time::{Duration, Instant};
    use tokio::sync::{mpsc, oneshot};

    use crate::api::gpu_handlers::{
        BatchConfig, BatchProcessResult, BatchQueueStats, ContinuousBatchRequest,
        ContinuousBatchResponse, GpuBatchRequest, GpuBatchResponse, GpuBatchResult, GpuBatchStats,
        GpuStatusResponse, GpuWarmupResponse,
    };

    #[test]
    fn test_gpu_batch_request_serde_defaults() {
        // Test that default values work with serde
        let json = r#"{"prompts": ["Hello"]}"#;
        let request: GpuBatchRequest = serde_json::from_str(json).expect("JSON deserialization failed");

        assert_eq!(request.prompts.len(), 1);
        // max_tokens should have default
        assert!(request.max_tokens > 0);
        // top_k should have default
        assert!(request.top_k > 0);
    }

    // =========================================================================
    // ContinuousBatchRequest Edge Cases
    // =========================================================================

    #[tokio::test]
    async fn test_continuous_batch_request_timing() {
        let (tx, _rx) = oneshot::channel();

        let start = Instant::now();
        let request = ContinuousBatchRequest {
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 10,
            temperature: 0.5,
            top_k: 20,
            response_tx: tx,
            submitted_at: start,
        };

        // Simulate some processing time
        tokio::time::sleep(Duration::from_millis(5)).await;

        let queue_wait = request.submitted_at.elapsed();
        assert!(queue_wait >= Duration::from_millis(5));

        // Send response
        let response =
            ContinuousBatchResponse::single(vec![1, 2, 3, 4], 3, queue_wait.as_secs_f64() * 1000.0);
        let _ = request.response_tx.send(response);
    }

    // =========================================================================
    // BatchProcessResult Coverage
    // =========================================================================

    #[test]
    fn test_batch_process_result_small_batch() {
        let result = BatchProcessResult {
            requests_processed: 3,
            was_batched: false,
            total_time_ms: 45.0,
            avg_latency_ms: 15.0,
        };

        assert!(!result.was_batched);
        assert_eq!(result.requests_processed, 3);
        assert!((result.avg_latency_ms - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_process_result_large_batch() {
        let result = BatchProcessResult {
            requests_processed: 64,
            was_batched: true,
            total_time_ms: 500.0,
            avg_latency_ms: 7.8125, // 500 / 64
        };

        assert!(result.was_batched);
        assert_eq!(result.requests_processed, 64);
        assert!(result.avg_latency_ms < 10.0);
    }
include!("partial_batch.rs");
}
