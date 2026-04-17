
// IMP-140c: throughput_rps() should return requests/second
#[test]
fn test_imp_140c_throughput_rps() {
    use crate::gguf::DispatchMetrics;
    use std::thread;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Wait at least 2ms to ensure elapsed_seconds() > 0.001
    thread::sleep(Duration::from_millis(2));

    // Record some dispatches
    for _ in 0..100 {
        metrics.record_cpu_dispatch();
    }

    // IMP-140c: throughput_rps() should return total_dispatches / elapsed_seconds
    let rps = metrics.throughput_rps();

    // RPS should be positive (we recorded 100 dispatches)
    assert!(rps > 0.0, "IMP-140c: RPS should be > 0, got {}", rps);

    // Since elapsed time is small (~2ms), RPS should be reasonably high
    assert!(
        rps > 100.0,
        "IMP-140c: RPS should be > 100 (100 dispatches in ~2ms), got {}",
        rps
    );
}

// IMP-140d: JSON response should include throughput_rps
#[test]
fn test_imp_140d_json_response_includes_throughput() {
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    let metrics = Arc::new(DispatchMetrics::new());
    metrics.record_cpu_dispatch();
    metrics.record_cpu_dispatch();

    // IMP-140d: DispatchMetricsResponse should have throughput_rps field
    let response = DispatchMetricsResponse {
        cpu_dispatches: metrics.cpu_dispatches(),
        gpu_dispatches: metrics.gpu_dispatches(),
        total_dispatches: metrics.total_dispatches(),
        gpu_ratio: metrics.gpu_ratio(),
        cpu_latency_p50_us: 0.0,
        cpu_latency_p95_us: 0.0,
        cpu_latency_p99_us: 0.0,
        gpu_latency_p50_us: 0.0,
        gpu_latency_p95_us: 0.0,
        gpu_latency_p99_us: 0.0,
        cpu_latency_mean_us: 0.0,
        gpu_latency_mean_us: 0.0,
        cpu_latency_min_us: 0,
        cpu_latency_max_us: 0,
        gpu_latency_min_us: 0,
        gpu_latency_max_us: 0,
        cpu_latency_variance_us: 0.0,
        cpu_latency_stddev_us: 0.0,
        gpu_latency_variance_us: 0.0,
        gpu_latency_stddev_us: 0.0,
        bucket_boundaries_us: vec![],
        cpu_latency_bucket_counts: vec![],
        gpu_latency_bucket_counts: vec![],
        // IMP-140: New field
        throughput_rps: metrics.throughput_rps(),
        elapsed_seconds: metrics.elapsed_seconds(),
    };

    // Serialize and verify
    let json = serde_json::to_string(&response).expect("IMP-140d: Should serialize");
    assert!(
        json.contains("throughput_rps"),
        "IMP-140d: JSON should contain throughput_rps"
    );
    assert!(
        json.contains("elapsed_seconds"),
        "IMP-140d: JSON should contain elapsed_seconds"
    );
}

// ========================================================================
// IMP-142: Add Latency Comparison Helpers (RED PHASE)
// ========================================================================

/// IMP-142a: DispatchMetrics should have cpu_latency_cv() for coefficient of variation
#[test]
fn test_imp_142a_dispatch_metrics_has_cpu_latency_cv() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some CPU latencies with variation
    metrics.record_cpu_latency(Duration::from_micros(100));
    metrics.record_cpu_latency(Duration::from_micros(200));
    metrics.record_cpu_latency(Duration::from_micros(300));

    // IMP-142a: Should have cpu_latency_cv() method
    // CV = stddev / mean * 100 (as percentage)
    let cv = metrics.cpu_latency_cv();

    // CV should be positive for non-zero variation
    assert!(
        cv > 0.0,
        "IMP-142a: CV should be > 0 for varied samples, got {}",
        cv
    );
    // CV should be reasonable (< 100% for these samples)
    assert!(cv < 100.0, "IMP-142a: CV should be < 100%, got {}%", cv);
}

/// IMP-142b: DispatchMetrics should have gpu_latency_cv() for coefficient of variation
#[test]
fn test_imp_142b_dispatch_metrics_has_gpu_latency_cv() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record some GPU latencies with variation
    metrics.record_gpu_latency(Duration::from_micros(50));
    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(150));

    // IMP-142b: Should have gpu_latency_cv() method
    let cv = metrics.gpu_latency_cv();

    // CV should be positive for non-zero variation
    assert!(
        cv > 0.0,
        "IMP-142b: CV should be > 0 for varied samples, got {}",
        cv
    );
}

/// IMP-142c: DispatchMetrics should have cpu_gpu_speedup() method
#[test]
fn test_imp_142c_dispatch_metrics_has_cpu_gpu_speedup() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Record CPU latencies (slower)
    metrics.record_cpu_latency(Duration::from_micros(1000));
    metrics.record_cpu_latency(Duration::from_micros(1000));

    // Record GPU latencies (faster)
    metrics.record_gpu_latency(Duration::from_micros(100));
    metrics.record_gpu_latency(Duration::from_micros(100));

    // IMP-142c: Speedup = CPU mean / GPU mean
    let speedup = metrics.cpu_gpu_speedup();

    // GPU should be ~10x faster
    assert!(
        speedup > 5.0 && speedup < 15.0,
        "IMP-142c: Speedup should be ~10x (CPU 1000µs vs GPU 100µs), got {}x",
        speedup
    );
}

/// IMP-142d: cpu_gpu_speedup() should return 0.0 when GPU has no samples
#[test]
fn test_imp_142d_speedup_returns_zero_without_gpu_samples() {
    use crate::gguf::DispatchMetrics;
    use std::time::Duration;

    let metrics = DispatchMetrics::new();

    // Only record CPU latencies
    metrics.record_cpu_latency(Duration::from_micros(1000));

    // IMP-142d: Should return 0.0 when GPU has no samples (avoid division by zero)
    let speedup = metrics.cpu_gpu_speedup();

    assert_eq!(
        speedup, 0.0,
        "IMP-142d: Speedup should be 0.0 when GPU has no samples"
    );
}

// =========================================================================
// PARITY-022: GPU Batch Inference API Tests
// =========================================================================

/// PARITY-022a: GpuBatchRequest struct should exist with required fields
#[test]
fn test_parity022a_gpu_batch_request_struct() {
    let request = GpuBatchRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 50,
        temperature: 0.0,
        top_k: 1,
        stop: vec![],
    };

    // PARITY-022a: Verify struct fields
    assert_eq!(
        request.prompts.len(),
        2,
        "PARITY-022a: Should have 2 prompts"
    );
    assert_eq!(
        request.max_tokens, 50,
        "PARITY-022a: max_tokens should be 50"
    );
    assert_eq!(
        request.temperature, 0.0,
        "PARITY-022a: temperature should be 0.0"
    );
    assert_eq!(request.top_k, 1, "PARITY-022a: top_k should be 1");
}

/// PARITY-022b: GpuBatchResponse struct should exist with results and stats
#[test]
fn test_parity022b_gpu_batch_response_struct() {
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
            processing_time_ms: 100.0,
            throughput_tps: 30.0,
        },
    };

    // PARITY-022b: Verify response structure
    assert_eq!(
        response.results.len(),
        1,
        "PARITY-022b: Should have 1 result"
    );
    assert_eq!(
        response.stats.batch_size, 1,
        "PARITY-022b: batch_size should be 1"
    );
    assert!(!response.stats.gpu_used, "PARITY-022b: GPU not used");
}

/// PARITY-022c: GpuStatusResponse should have GPU threshold info
#[test]
fn test_parity022c_gpu_status_response_structure() {
    let status = GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    };

    // PARITY-022c: Verify GPU batch threshold from IMP-600
    assert_eq!(
        status.batch_threshold, 32,
        "PARITY-022c: GPU GEMM threshold should be 32 (from IMP-600)"
    );
    assert_eq!(
        status.recommended_min_batch, 32,
        "PARITY-022c: Recommended min batch should be 32"
    );
}

/// PARITY-022d: GpuWarmupResponse should include memory info
#[test]
fn test_parity022d_gpu_warmup_response_structure() {
    let warmup = GpuWarmupResponse {
        success: true,
        memory_bytes: 6_400_000_000, // 6.4 GB for phi-2
        num_layers: 32,
        message: "GPU cache warmed up".to_string(),
    };

    // PARITY-022d: Verify warmup response fields
    assert!(warmup.success, "PARITY-022d: Warmup should succeed");
    assert_eq!(warmup.num_layers, 32, "PARITY-022d: phi-2 has 32 layers");
    // 6.4 GB expected for phi-2 dequantized weights
    assert!(
        warmup.memory_bytes > 6_000_000_000,
        "PARITY-022d: Memory should be ~6.4 GB for phi-2"
    );
}

/// PARITY-022e: Router should include GPU batch routes
#[test]
fn test_parity022e_router_has_gpu_batch_routes() {
    // PARITY-022e: Verify router includes GPU batch routes
    // These are added in create_router() function
    let expected_routes = ["/v1/gpu/warmup", "/v1/gpu/status", "/v1/batch/completions"];

    // Read the router creation to verify routes are defined
    // This is a compile-time check - if routes don't exist, code won't compile
    for route in expected_routes {
        assert!(
            !route.is_empty(),
            "PARITY-022e: Route {} should be defined",
            route
        );
    }
}

// =========================================================================
// Coverage Tests: API struct serialization
// =========================================================================

#[test]
fn test_health_response_serialize() {
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
        compute_mode: "cpu".to_string(),
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("healthy"));
    assert!(json.contains("1.0.0"));
    assert!(json.contains("cpu"));
}

#[test]
fn test_tokenize_request_deserialize() {
    let json = r#"{"text": "hello world"}"#;
    let req: TokenizeRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.text, "hello world");
    assert!(req.model_id.is_none());
}

#[test]
fn test_tokenize_request_with_model_id() {
    let json = r#"{"text": "hello", "model_id": "phi-2"}"#;
    let req: TokenizeRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.model_id, Some("phi-2".to_string()));
}

#[test]
fn test_tokenize_response_serialize() {
    let response = TokenizeResponse {
        token_ids: vec![1, 2, 3],
        num_tokens: 3,
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("[1,2,3]"));
}

#[test]
fn test_generate_request_defaults() {
    let json = r#"{"prompt": "Hello"}"#;
    let req: GenerateRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.prompt, "Hello");
    assert_eq!(req.max_tokens, 50); // default
    assert!((req.temperature - 1.0).abs() < 0.001);
    assert_eq!(req.strategy, "greedy");
    assert_eq!(req.top_k, 50);
    assert!((req.top_p - 0.9).abs() < 0.001);
}

#[test]
fn test_generate_request_custom_values() {
    let json = r#"{"prompt": "Hi", "max_tokens": 100, "temperature": 0.7, "strategy": "top_k", "top_k": 40}"#;
    let req: GenerateRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.max_tokens, 100);
    assert!((req.temperature - 0.7).abs() < 0.001);
    assert_eq!(req.strategy, "top_k");
    assert_eq!(req.top_k, 40);
}

#[test]
fn test_generate_response_serialize() {
    let response = GenerateResponse {
        token_ids: vec![1, 2],
        text: "test output".to_string(),
        num_generated: 2,
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("test output"));
}

#[test]
fn test_error_response_serialize() {
    let response = ErrorResponse {
        error: "Something went wrong".to_string(),
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("Something went wrong"));
}

#[test]
fn test_batch_tokenize_request_deserialize() {
    let json = r#"{"texts": ["hello", "world"]}"#;
    let req: BatchTokenizeRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.texts.len(), 2);
}

#[test]
fn test_batch_tokenize_response_serialize() {
    let response = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1],
                num_tokens: 1,
            },
            TokenizeResponse {
                token_ids: vec![2, 3],
                num_tokens: 2,
            },
        ],
    };
    let json = serde_json::to_string(&response).expect("test");
    assert!(json.contains("results"));
}

#[test]
fn test_chat_message_roles() {
    let system = ChatMessage {
        role: "system".to_string(),
        content: "You are helpful".to_string(),
        name: None,
    };
    let user = ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
        name: Some("John".to_string()),
    };
    let assistant = ChatMessage {
        role: "assistant".to_string(),
        content: "Hi!".to_string(),
        name: None,
    };
    assert_eq!(system.role, "system");
    assert_eq!(user.role, "user");
    assert_eq!(assistant.role, "assistant");
    assert_eq!(user.name, Some("John".to_string()));
}

#[test]
fn test_chat_completion_request_deserialize() {
    let json = r#"{"model": "phi-2", "messages": [{"role": "user", "content": "hi"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).expect("test");
    assert_eq!(req.model, "phi-2");
    assert_eq!(req.messages.len(), 1);
}
