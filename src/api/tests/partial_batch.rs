
    // =========================================================================
    // Chaos Pattern 1: Partial Batch Failures
    // =========================================================================

    #[test]
    fn test_partial_batch_one_success_one_failure_response() {
        // Simulate a batch where one request succeeds, another fails
        let success_response = ContinuousBatchResponse::batched(vec![1, 2, 3, 4, 5], 2, 2, 15.5);
        let failure_response = ContinuousBatchResponse::single(
            vec![1, 2], // Just prompt tokens returned on failure
            2,
            0.1,
        );

        // Success should have generated tokens
        assert_eq!(success_response.generated_tokens().len(), 3);
        assert!(success_response.batched);
        assert_eq!(success_response.batch_size, 2);

        // Failure returns only prompt
        assert_eq!(failure_response.generated_tokens().len(), 0);
        assert!(!failure_response.batched);
    }

    #[test]
    fn test_partial_batch_mixed_latencies() {
        // Some requests fast, some slow - simulates partial failure
        let responses: Vec<ContinuousBatchResponse> = vec![
            ContinuousBatchResponse::batched(vec![1, 2, 3], 1, 4, 5.0), // Fast
            ContinuousBatchResponse::batched(vec![4, 5, 6], 1, 4, 500.0), // Slow (timeout-like)
            ContinuousBatchResponse::batched(vec![7, 8, 9], 1, 4, 10.0), // Normal
            ContinuousBatchResponse::single(vec![10], 1, 1000.0),       // Degraded to single
        ];

        let total_latency: f64 = responses.iter().map(|r| r.latency_ms).sum();
        let avg_latency = total_latency / responses.len() as f64;

        // Verify mixed batch/single processing
        assert_eq!(responses.iter().filter(|r| r.batched).count(), 3);
        assert_eq!(responses.iter().filter(|r| !r.batched).count(), 1);
        assert!(avg_latency > 100.0); // High due to slow request
    }

    #[test]
    fn test_batch_result_with_empty_generation() {
        // Prompt so long that max_tokens=0 effectively
        let response = ContinuousBatchResponse::batched(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // All prompt
            10,                                  // prompt_len = total
            8,
            25.0,
        );

        // No tokens generated
        assert!(response.generated_tokens().is_empty());
        assert_eq!(response.prompt_len, 10);
    }

    // =========================================================================
    // Chaos Pattern 2: Queue Saturation
    // =========================================================================

    #[tokio::test]
    async fn test_queue_saturation_100_requests() {
        // Create config with small queue to force saturation behavior
        let config = BatchConfig {
            window_ms: 10,
            min_batch: 4,
            optimal_batch: 32,
            max_batch: 64,
            queue_size: 128, // Allows 100 but will stress the system
            gpu_threshold: 32,
        };

        // Create channel
        let (tx, mut rx) = mpsc::channel::<ContinuousBatchRequest>(config.queue_size);

        // Simulate 100 pending requests
        let mut response_channels = Vec::with_capacity(100);

        for i in 0..100 {
            let (resp_tx, resp_rx) = oneshot::channel();
            response_channels.push(resp_rx);

            let request = ContinuousBatchRequest {
                prompt_tokens: vec![1, 2, 3, (i % 255) as u32],
                max_tokens: 10,
                temperature: 0.7,
                top_k: 40,
                response_tx: resp_tx,
                submitted_at: Instant::now(),
            };

            // Send should succeed (queue size 128 > 100)
            tx.send(request).await.expect("Queue should not be full");
        }

        // Verify all 100 are queued
        let mut count = 0;
        while let Ok(Some(_)) = tokio::time::timeout(Duration::from_millis(1), rx.recv()).await {
            count += 1;
            if count >= 100 {
                break;
            }
        }
        assert_eq!(count, 100, "Should have 100 requests queued");
    }

    #[tokio::test]
    async fn test_queue_saturation_overflow_behavior() {
        // Tiny queue to force overflow
        let config = BatchConfig {
            window_ms: 10,
            min_batch: 2,
            optimal_batch: 4,
            max_batch: 8,
            queue_size: 5, // Very small
            gpu_threshold: 4,
        };

        let (tx, _rx) = mpsc::channel::<ContinuousBatchRequest>(config.queue_size);

        // Fill the queue
        let mut success_count = 0;
        for i in 0..10 {
            let (resp_tx, _) = oneshot::channel();
            let request = ContinuousBatchRequest {
                prompt_tokens: vec![i as u32],
                max_tokens: 5,
                temperature: 0.0,
                top_k: 1,
                response_tx: resp_tx,
                submitted_at: Instant::now(),
            };

            match tx.try_send(request) {
                Ok(()) => success_count += 1,
                Err(_) => break, // Queue full
            }
        }

        // Should succeed for queue_size requests, then fail
        assert!(
            success_count >= 4 && success_count <= 6,
            "Expected 4-6 successes before overflow, got {}",
            success_count
        );
    }

    #[test]
    fn test_batch_queue_stats_under_load() {
        let mut stats = BatchQueueStats::default();

        // Simulate 100 requests processed in batches
        stats.total_queued = 100;
        stats.total_batches = 10; // 10 batches of ~10 each
        stats.total_single = 5; // 5 fell back to single
        stats.avg_batch_size = 9.5;
        stats.avg_wait_ms = 25.0;

        assert_eq!(stats.total_queued, 100);
        assert!(stats.avg_batch_size > 9.0);
        assert!(stats.avg_wait_ms < 50.0);
    }

    // =========================================================================
    // Chaos Pattern 3: Zombie Batches (Client Disconnect)
    // =========================================================================

    #[tokio::test]
    async fn test_zombie_batch_client_disconnect() {
        // Simulate client disconnecting by dropping the receiver
        let (resp_tx, resp_rx) = oneshot::channel::<ContinuousBatchResponse>();

        // Client "disconnects" - drops the receiver
        drop(resp_rx);

        // Server tries to send response - should not panic
        let response = ContinuousBatchResponse::batched(vec![1, 2, 3, 4, 5], 2, 1, 10.0);

        // Send should fail gracefully (returns Err but doesn't panic)
        let result = resp_tx.send(response);
        assert!(result.is_err(), "Send should fail when receiver dropped");
    }

    #[tokio::test]
    async fn test_zombie_batch_multiple_disconnects() {
        // Simulate batch where 3/5 clients disconnect
        let mut channels: Vec<(
            oneshot::Sender<ContinuousBatchResponse>,
            Option<oneshot::Receiver<ContinuousBatchResponse>>,
        )> = Vec::new();

        for i in 0..5 {
            let (tx, rx) = oneshot::channel();
            if i < 3 {
                // Clients 0, 1, 2 disconnect (drop receiver)
                channels.push((tx, None));
            } else {
                // Clients 3, 4 stay connected
                channels.push((tx, Some(rx)));
            }
        }

        let mut send_failures = 0;
        let mut send_successes = 0;

        for (tx, _) in channels {
            let response = ContinuousBatchResponse::batched(vec![1, 2, 3], 1, 5, 10.0);

            match tx.send(response) {
                Ok(()) => send_successes += 1,
                Err(_) => send_failures += 1,
            }
        }

        // 3 failures (disconnected), 2 successes
        assert_eq!(send_failures, 3);
        assert_eq!(send_successes, 2);
    }

    #[tokio::test]
    async fn test_zombie_batch_timeout_then_disconnect() {
        // Client waits, then times out and disconnects
        let (resp_tx, resp_rx) = oneshot::channel::<ContinuousBatchResponse>();

        // Spawn a task that times out waiting
        let handle = tokio::spawn(async move {
            match tokio::time::timeout(Duration::from_millis(5), resp_rx).await {
                Ok(Ok(_)) => "received",
                Ok(Err(_)) => "channel_closed",
                Err(_) => "timeout",
            }
        });

        // Simulate slow processing (wide gap vs 5ms timeout for CI robustness)
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Client has already timed out, receiver dropped
        let response = ContinuousBatchResponse::single(vec![1], 1, 100.0);
        let _send_result = resp_tx.send(response);

        let client_result = handle.await.expect("test value should be present");

        // Client timed out
        assert_eq!(client_result, "timeout");
        // Send may or may not fail depending on timing
        // (receiver might not be dropped yet if timeout just happened)
    }

    // =========================================================================
    // BatchConfig Edge Cases
    // =========================================================================

    #[test]
    fn test_batch_config_edge_should_process() {
        let config = BatchConfig::default();

        // Below optimal - should not process
        assert!(!config.should_process(31));
        // At optimal - should process
        assert!(config.should_process(32));
        // Above optimal - should process
        assert!(config.should_process(100));
    }

    #[test]
    fn test_batch_config_edge_meets_minimum() {
        let config = BatchConfig::default();

        // Below min - doesn't meet
        assert!(!config.meets_minimum(3));
        // At min - meets
        assert!(config.meets_minimum(4));
        // Above min - meets
        assert!(config.meets_minimum(10));
    }

    #[test]
    fn test_batch_config_low_latency() {
        let config = BatchConfig::low_latency();

        assert_eq!(config.window_ms, 5);
        assert_eq!(config.min_batch, 2);
        assert_eq!(config.optimal_batch, 8);
        assert_eq!(config.max_batch, 16);
        // GPU threshold > max_batch means GPU batch effectively disabled
        assert!(config.gpu_threshold > config.max_batch);
    }

    #[test]
    fn test_batch_config_high_throughput() {
        let config = BatchConfig::high_throughput();

        assert_eq!(config.window_ms, 100);
        assert_eq!(config.min_batch, 8);
        assert_eq!(config.optimal_batch, 32);
        assert_eq!(config.max_batch, 128);
        assert_eq!(config.gpu_threshold, 32);
    }

    // =========================================================================
    // Response Type Edge Cases
    // =========================================================================

    #[test]
    fn test_gpu_batch_response_empty_results() {
        let response = GpuBatchResponse {
            results: vec![],
            stats: GpuBatchStats {
                batch_size: 0,
                gpu_used: false,
                total_tokens: 0,
                processing_time_ms: 0.0,
                throughput_tps: 0.0,
            },
        };

        assert!(response.results.is_empty());
        assert_eq!(response.stats.batch_size, 0);
    }

    #[test]
    fn test_gpu_batch_result_serialization() {
        let result = GpuBatchResult {
            index: 42,
            token_ids: vec![1, 2, 3, 4, 5],
            text: "Hello world".to_string(),
            num_generated: 5,
        };

        let json = serde_json::to_string(&result).expect("JSON serialization failed");
        assert!(json.contains("42"));
        assert!(json.contains("Hello world"));

        let parsed: GpuBatchResult = serde_json::from_str(&json).expect("JSON deserialization failed");
        assert_eq!(parsed.index, 42);
        assert_eq!(parsed.num_generated, 5);
    }

    #[test]
    fn test_gpu_batch_stats_high_throughput() {
        let stats = GpuBatchStats {
            batch_size: 64,
            gpu_used: true,
            total_tokens: 6400, // 100 tokens * 64 requests
            processing_time_ms: 1000.0,
            throughput_tps: 6400.0, // 6400 tokens / 1 second
        };

        assert!(stats.gpu_used);
        assert!(stats.throughput_tps > 1000.0);
    }

    #[test]
    fn test_gpu_warmup_response_success() {
        let response = GpuWarmupResponse {
            success: true,
            memory_bytes: 1024 * 1024 * 512, // 512 MB
            num_layers: 32,
            message: "GPU cache warmed up: 32 layers, 0.54 GB".to_string(),
        };

        assert!(response.success);
        assert_eq!(response.num_layers, 32);
        assert!(response.message.contains("32 layers"));
    }

    #[test]
    fn test_gpu_warmup_response_failure() {
        let response = GpuWarmupResponse {
            success: false,
            memory_bytes: 0,
            num_layers: 0,
            message: "CUDA error: out of memory".to_string(),
        };

        assert!(!response.success);
        assert_eq!(response.memory_bytes, 0);
        assert!(response.message.contains("error"));
    }

    #[test]
    fn test_gpu_status_response_warm() {
        let response = GpuStatusResponse {
            cache_ready: true,
            cache_memory_bytes: 1024 * 1024 * 256,
            batch_threshold: 32,
            recommended_min_batch: 8,
        };

        assert!(response.cache_ready);
        assert_eq!(response.batch_threshold, 32);
    }

    #[test]
    fn test_gpu_status_response_cold() {
        let response = GpuStatusResponse {
            cache_ready: false,
            cache_memory_bytes: 0,
            batch_threshold: 32,
            recommended_min_batch: 4,
        };

        assert!(!response.cache_ready);
        assert_eq!(response.cache_memory_bytes, 0);
    }

    // =========================================================================
    // GpuBatchRequest Edge Cases
    // =========================================================================

    #[test]
    fn test_gpu_batch_request_empty_prompts() {
        let request = GpuBatchRequest {
            prompts: vec![],
            max_tokens: 100,
            temperature: 0.7,
            top_k: 40,
            stop: vec![],
        };

        assert!(request.prompts.is_empty());
    }

    #[test]
    fn test_gpu_batch_request_many_prompts() {
        let request = GpuBatchRequest {
            prompts: (0..100).map(|i| format!("Prompt {}", i)).collect(),
            max_tokens: 50,
            temperature: 0.0, // Greedy
            top_k: 1,
            stop: vec!["<|endoftext|>".to_string()],
        };

        assert_eq!(request.prompts.len(), 100);
        assert_eq!(request.temperature, 0.0);
        assert_eq!(request.stop.len(), 1);
    }
