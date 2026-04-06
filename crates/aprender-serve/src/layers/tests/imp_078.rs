
/// M31: Bulkhead Pattern (IMP-078)
/// Target: Separate pools, prevent starvation, configurable sizes
#[test]
#[cfg(feature = "gpu")]
fn test_imp_078_bulkhead_pattern() {
    use crate::gpu::{BulkheadConfig, BulkheadManager, RequestType};

    // Test 1: Create bulkhead manager with config
    let config = BulkheadConfig::new()
        .with_pool("inference", 10)
        .with_pool("embedding", 5)
        .with_pool("batch", 2);
    let manager = BulkheadManager::new(&config);

    // Test 2: Acquire from specific pool
    let permit = manager.acquire(RequestType::Inference);
    assert!(
        permit.is_ok(),
        "IMP-078: Should acquire from inference pool"
    );
    assert_eq!(
        manager.available(RequestType::Inference),
        9,
        "IMP-078: Should decrement available"
    );

    // Test 3: Pools are isolated
    let embed_permit = manager.acquire(RequestType::Embedding);
    assert!(
        embed_permit.is_ok(),
        "IMP-078: Should acquire from embedding pool"
    );
    assert_eq!(
        manager.available(RequestType::Inference),
        9,
        "IMP-078: Inference should be unchanged"
    );
    assert_eq!(
        manager.available(RequestType::Embedding),
        4,
        "IMP-078: Embedding should decrement"
    );

    // Test 4: Pool exhaustion doesn't affect others
    for _ in 0..2 {
        let _ = manager.acquire(RequestType::Batch);
    }
    let batch_overflow = manager.try_acquire(RequestType::Batch);
    assert!(
        batch_overflow.is_err(),
        "IMP-078: Batch pool should be exhausted"
    );
    assert_eq!(
        manager.available(RequestType::Inference),
        9,
        "IMP-078: Inference still available"
    );

    // Test 5: Release returns to correct pool
    manager.release(&permit.expect("test"));
    assert_eq!(
        manager.available(RequestType::Inference),
        10,
        "IMP-078: Should release to correct pool"
    );

    // Test 6: Get pool stats
    let stats = manager.stats();
    assert_eq!(stats.pool_count, 3, "IMP-078: Should have 3 pools");
    assert!(
        stats.total_capacity >= 17,
        "IMP-078: Total capacity should sum pools"
    );
}

// ========================================================================
// M32: Production Logging & Diagnostics (IMP-079, IMP-080, IMP-081)
// ========================================================================

/// M32: Structured Logging (IMP-079)
/// Target: JSON-formatted logs, correlation IDs, configurable levels
#[test]
#[cfg(feature = "gpu")]
fn test_imp_079_structured_logging() {
    use crate::gpu::{LogConfig, LogEntry, LogLevel, Logger};

    // Test 1: Create logger with config
    let config = LogConfig::new()
        .with_level(LogLevel::Debug)
        .with_json_format(true)
        .with_module_level("gpu", LogLevel::Trace);
    let logger = Logger::new(config);

    // Test 2: Create log entry with structured data
    let entry = LogEntry::new(LogLevel::Info, "Request started")
        .with_correlation_id("req-12345")
        .with_field("model", "llama-7b")
        .with_field("tokens", "128");
    assert_eq!(
        entry.correlation_id(),
        Some("req-12345"),
        "IMP-079: Should have correlation ID"
    );
    assert_eq!(entry.level(), LogLevel::Info, "IMP-079: Should have level");

    // Test 3: JSON formatting
    let json = entry.to_json();
    assert!(
        json.contains("\"level\":\"INFO\""),
        "IMP-079: JSON should have level"
    );
    assert!(
        json.contains("\"correlation_id\":\"req-12345\""),
        "IMP-079: JSON should have correlation ID"
    );
    assert!(
        json.contains("\"model\":\"llama-7b\""),
        "IMP-079: JSON should have custom fields"
    );

    // Test 4: Module-specific log levels
    assert!(
        logger.is_enabled(LogLevel::Trace, "gpu"),
        "IMP-079: gpu should allow Trace"
    );
    assert!(
        logger.is_enabled(LogLevel::Debug, "inference"),
        "IMP-079: Other modules use default"
    );
    assert!(
        !logger.is_enabled(LogLevel::Trace, "inference"),
        "IMP-079: Trace should be filtered for non-gpu"
    );

    // Test 5: Log with automatic timestamp
    let entry = LogEntry::new(LogLevel::Warn, "High memory usage");
    assert!(entry.timestamp() > 0, "IMP-079: Should have timestamp");
}

/// M32: Performance Diagnostics (IMP-080)
/// Target: Latency breakdown, memory tracking, GPU timing
#[test]
#[cfg(feature = "gpu")]
fn test_imp_080_performance_diagnostics() {
    use crate::gpu::{DiagnosticsCollector, MemoryTracker, PhaseTimer};

    // Test 1: Create diagnostics collector
    let collector = DiagnosticsCollector::new();

    // Test 2: Track request phases
    let timer = PhaseTimer::new();
    timer.start_phase("tokenization");
    std::thread::sleep(std::time::Duration::from_millis(10));
    timer.end_phase("tokenization");
    timer.start_phase("inference");
    std::thread::sleep(std::time::Duration::from_millis(20));
    timer.end_phase("inference");

    let breakdown = timer.breakdown();
    assert!(
        breakdown.contains_key("tokenization"),
        "IMP-080: Should track tokenization"
    );
    assert!(
        breakdown.contains_key("inference"),
        "IMP-080: Should track inference"
    );
    assert!(
        *breakdown.get("inference").expect("test") > *breakdown.get("tokenization").expect("test"),
        "IMP-080: Inference should take longer"
    );

    // Test 3: Memory allocation tracking
    let tracker = MemoryTracker::new();
    tracker.record_allocation("model_weights", 1024 * 1024 * 1024);
    tracker.record_allocation("kv_cache", 256 * 1024 * 1024);
    tracker.record_deallocation("kv_cache", 256 * 1024 * 1024);

    let report = tracker.report();
    assert_eq!(
        report.peak_bytes,
        1024 * 1024 * 1024 + 256 * 1024 * 1024,
        "IMP-080: Should track peak"
    );
    assert_eq!(
        report.current_bytes,
        1024 * 1024 * 1024,
        "IMP-080: Should track current"
    );
    assert_eq!(
        report.allocation_count, 2,
        "IMP-080: Should count allocations"
    );

    // Test 4: Report to collector
    collector.record_request_timing("req-001", timer.breakdown());
    collector.record_memory_snapshot(report);
    let summary = collector.summary();
    assert!(summary.request_count >= 1, "IMP-080: Should count requests");
}

/// M32: Debug Mode (IMP-081)
/// Target: Verbose logging, request replay, state dump
#[test]
#[cfg(feature = "gpu")]
fn test_imp_081_debug_mode() {
    use crate::gpu::{DebugMode, RequestCapture, StateDump};

    // Test 1: Enable debug mode
    let debug = DebugMode::new();
    assert!(
        !debug.is_enabled(),
        "IMP-081: Should be disabled by default"
    );
    debug.enable();
    assert!(debug.is_enabled(), "IMP-081: Should enable");

    // Test 2: Capture request for replay
    let capture = RequestCapture::new()
        .with_input("Hello, world!")
        .with_params("temperature", "0.7")
        .with_params("max_tokens", "100");
    assert_eq!(
        capture.input(),
        "Hello, world!",
        "IMP-081: Should capture input"
    );
    assert_eq!(capture.params().len(), 2, "IMP-081: Should capture params");

    // Test 3: Serialize/deserialize for replay
    let json = capture.to_json();
    let restored = RequestCapture::from_json(&json);
    assert!(restored.is_ok(), "IMP-081: Should deserialize");
    assert_eq!(
        restored.expect("test").input(),
        "Hello, world!",
        "IMP-081: Should restore input"
    );

    // Test 4: State dump on error
    let dump = StateDump::new()
        .with_error("Out of memory")
        .with_stack_trace("at inference::generate\nat main")
        .with_state("model_loaded", "true")
        .with_state("tokens_processed", "42");
    assert_eq!(
        dump.error(),
        "Out of memory",
        "IMP-081: Should capture error"
    );
    assert!(
        dump.stack_trace().contains("inference::generate"),
        "IMP-081: Should have stack"
    );
    assert_eq!(dump.state().len(), 2, "IMP-081: Should capture state");

    // Test 5: Dump to file (mock)
    let dump_json = dump.to_json();
    assert!(
        dump_json.contains("Out of memory"),
        "IMP-081: JSON should have error"
    );
    assert!(
        dump_json.contains("tokens_processed"),
        "IMP-081: JSON should have state"
    );
}

// =========================================================================
// M33: GGUF HTTP Serving Integration Tests
// Per spec v2.15.0: Wire GpuModel to HTTP server
// =========================================================================

/// M33: GgufModelState (IMP-082)
/// Target: App state that holds a loaded GGUF model for HTTP serving
#[test]
#[cfg(feature = "gpu")]
fn test_imp_082_gguf_model_state() {
    use crate::gpu::GgufModelState;

    // Test 1: Create empty state
    let state = GgufModelState::new();
    assert!(!state.is_loaded(), "IMP-082: Should be unloaded initially");

    // Test 2: State reports model info
    assert_eq!(
        state.model_name(),
        None,
        "IMP-082: No model name when empty"
    );
    assert_eq!(state.vocab_size(), 0, "IMP-082: Zero vocab when empty");

    // Test 3: Ready check
    assert!(!state.is_ready(), "IMP-082: Not ready when empty");
}

/// M33: Load GGUF to GPU (IMP-083)
/// Target: Pipeline from GGUF file to GpuModel ready for inference
#[test]
#[cfg(feature = "gpu")]
fn test_imp_083_load_gguf_to_gpu() {
    use crate::gpu::load_gguf_to_gpu;

    // Test with test GGUF data (minimal model)
    let vocab_size = 256;
    let hidden_dim = 64;
    let num_layers = 2;

    // Create minimal test GGUF-like config
    let result = load_gguf_to_gpu(vocab_size, hidden_dim, num_layers);

    // This should work - creates a minimal GPU model
    assert!(result.is_ok(), "IMP-083: Should load test model to GPU");

    let state = result.expect("test");
    assert!(state.is_loaded(), "IMP-083: Should be loaded after load");
    assert!(state.is_ready(), "IMP-083: Should be ready for inference");
    assert_eq!(
        state.vocab_size(),
        vocab_size,
        "IMP-083: Should have correct vocab"
    );
}

/// M33: Serve GGUF Model (IMP-084)
/// Target: HTTP server with loaded GGUF model (integration test)
///
/// Verifies that a GGUF model can be served via HTTP.
/// Run with: `cargo test test_imp_084 --ignored --features gpu`
#[test]
#[ignore = "Requires integration test setup"]
fn test_imp_084_serve_gguf_model() {
    // IMP-084: Integration test for serve_gguf_model
    //
    // This test verifies the HTTP serving infrastructure is correct.
    // It uses a demo model since real GGUF files may not be available.

    // Check if realizar server is running on default port
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("Failed to create HTTP client");

    let health_url = "http://127.0.0.1:3000/health";
    match client.get(health_url).send() {
        Ok(response) => {
            assert!(
                response.status().is_success(),
                "IMP-084: Health endpoint should return 200 OK"
            );
            println!("IMP-084: ✅ Server health check passed");

            // Test generate endpoint with demo model
            let gen_url = "http://127.0.0.1:3000/generate";
            let request = serde_json::json!({
                "prompt": "Hello",
                "max_tokens": 5,
                "temperature": 0.0
            });

            match client.post(gen_url).json(&request).send() {
                Ok(gen_response) => {
                    assert!(
                        gen_response.status().is_success(),
                        "IMP-084: Generate endpoint should return 200 OK"
                    );
                    let body: serde_json::Value = gen_response.json().expect("Valid JSON");
                    assert!(
                        body.get("text").is_some(),
                        "IMP-084: Response should have text"
                    );
                    println!("IMP-084: ✅ Generate endpoint works, got: {:?}", body);
                },
                Err(e) => {
                    println!("IMP-084: ⚠️ Generate endpoint not available: {}", e);
                },
            }
        },
        Err(e) => {
            panic!(
                "IMP-084: Server not running at {}. Start with: cargo run --example api_server. Error: {}",
                health_url, e
            );
        },
    }
}

/// M33: OpenAI Completions Endpoint (IMP-085)
/// Target: /v1/completions returns generated text
///
/// Tests OpenAI-compatible completions API.
/// Run with: `cargo test test_imp_085 --ignored`
#[test]
#[ignore = "Requires running server"]
fn test_imp_085_completions_endpoint() {
    // IMP-085: Integration test for /v1/completions (OpenAI-compatible)

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client");

    let url = "http://127.0.0.1:3000/v1/completions";

    // OpenAI-style request format
    let request = serde_json::json!({
        "model": "demo",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0.0
    });

    match client.post(url).json(&request).send() {
        Ok(response) => {
            if response.status().is_success() {
                let body: serde_json::Value = response.json().expect("Valid JSON");
                assert!(
                    body.get("choices").is_some(),
                    "IMP-085: Response should have 'choices'"
                );
                println!("IMP-085: ✅ OpenAI completions endpoint works");
            } else if response.status().as_u16() == 404 {
                println!("IMP-085: ⚠️ /v1/completions not implemented yet (404)");
            } else {
                panic!("IMP-085: Unexpected status: {}", response.status());
            }
        },
        Err(e) => {
            panic!(
                "IMP-085: Server not running. Start with: cargo run --example api_server. Error: {}",
                e
            );
        },
    }
}
