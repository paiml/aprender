use super::*;

// ============================================================================
// BatchGenerationStats tests
// ============================================================================

#[test]
fn test_batch_generation_stats_basic() {
    let stats = BatchGenerationStats {
        gpu_cache_ready: true,
        cache_memory_gb: 2.5,
        num_layers: 32,
        hidden_dim: 4096,
        intermediate_dim: 11008,
        recommended_batch_size: 8,
        max_batch_size: 16,
    };

    assert!(stats.gpu_cache_ready);
    assert!((stats.cache_memory_gb - 2.5).abs() < 0.01);
    assert_eq!(stats.num_layers, 32);
}

#[test]
fn test_batch_generation_stats_clone() {
    let stats = BatchGenerationStats {
        gpu_cache_ready: false,
        cache_memory_gb: 1.0,
        num_layers: 12,
        hidden_dim: 768,
        intermediate_dim: 3072,
        recommended_batch_size: 4,
        max_batch_size: 8,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.num_layers, stats.num_layers);
    assert_eq!(cloned.hidden_dim, stats.hidden_dim);
}

#[test]
fn test_batch_generation_stats_debug() {
    let stats = BatchGenerationStats {
        gpu_cache_ready: true,
        cache_memory_gb: 3.0,
        num_layers: 24,
        hidden_dim: 2048,
        intermediate_dim: 8192,
        recommended_batch_size: 16,
        max_batch_size: 32,
    };

    let debug = format!("{:?}", stats);
    assert!(debug.contains("BatchGenerationStats"));
    assert!(debug.contains("gpu_cache_ready"));
}

// ============================================================================
// PendingRequest tests
// ============================================================================

#[test]
fn test_pending_request_new() {
    let req = PendingRequest::new(1, vec![100, 200, 300], 50, 0.7, 40);

    assert_eq!(req.id, 1);
    assert_eq!(req.prompt, vec![100, 200, 300]);
    assert_eq!(req.max_tokens, 50);
    assert!((req.temperature - 0.7).abs() < 0.01);
    assert_eq!(req.top_k, 40);
}

#[test]
fn test_pending_request_wait_time() {
    let req = PendingRequest::new(0, vec![1], 10, 1.0, 50);
    std::thread::sleep(std::time::Duration::from_millis(10));
    let wait = req.wait_time();
    assert!(wait >= std::time::Duration::from_millis(5));
}

#[test]
fn test_pending_request_clone() {
    let req = PendingRequest::new(5, vec![1, 2, 3], 100, 0.5, 20);
    let cloned = req.clone();
    assert_eq!(cloned.id, req.id);
    assert_eq!(cloned.prompt, req.prompt);
}

// ============================================================================
// RequestBatch tests
// ============================================================================

#[test]
fn test_request_batch_new() {
    let reqs = vec![
        PendingRequest::new(0, vec![1, 2], 10, 1.0, 50),
        PendingRequest::new(1, vec![3, 4, 5], 20, 0.8, 40),
    ];
    let batch = RequestBatch::new(reqs);

    assert_eq!(batch.size(), 2);
}

#[test]
fn test_request_batch_prompts() {
    let reqs = vec![
        PendingRequest::new(0, vec![100], 10, 1.0, 50),
        PendingRequest::new(1, vec![200, 201], 20, 0.8, 40),
    ];
    let batch = RequestBatch::new(reqs);

    let prompts = batch.prompts();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0], vec![100]);
    assert_eq!(prompts[1], vec![200, 201]);
}

#[test]
fn test_request_batch_avg_wait_time() {
    let batch = RequestBatch::new(Vec::new());
    assert_eq!(batch.avg_wait_time(), std::time::Duration::ZERO);

    let reqs = vec![PendingRequest::new(0, vec![1], 10, 1.0, 50)];
    let batch2 = RequestBatch::new(reqs);
    // avg_wait_time should be accessible
    let _ = batch2.avg_wait_time();
}

// ============================================================================
// BatchRequestCollector tests
// ============================================================================

#[test]
fn test_batch_request_collector_new() {
    let collector = BatchRequestCollector::new();
    assert_eq!(collector.batch_threshold, 32);
    assert_eq!(collector.timeout_ms, 50);
    assert_eq!(collector.max_batch_size, 64);
}

#[test]
fn test_batch_request_collector_with_thresholds() {
    let collector = BatchRequestCollector::with_thresholds(16, 100, 32);
    assert_eq!(collector.batch_threshold, 16);
    assert_eq!(collector.timeout_ms, 100);
    assert_eq!(collector.max_batch_size, 32);
}

#[test]
fn test_batch_request_collector_submit() {
    let collector = BatchRequestCollector::new();
    let id1 = collector.submit(vec![1, 2], 10, 1.0, 50);
    let id2 = collector.submit(vec![3, 4], 20, 0.8, 40);

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
}

#[test]
fn test_batch_request_collector_default() {
    let collector = BatchRequestCollector::default();
    assert_eq!(collector.batch_threshold, 32);
}

// ============================================================================
// BatchingConfig tests
// ============================================================================

#[test]
fn test_batching_config_default() {
    let config = BatchingConfig::default();
    assert!(config.max_batch_size > 0);
    assert!(config.timeout_ms > 0);
}

#[test]
fn test_batching_config_latency_optimized() {
    let config = BatchingConfig::latency_optimized();
    assert!(config.batch_threshold < 32); // Lower than default
    assert!(config.timeout_ms < 50); // Shorter than default
}

// ============================================================================
// SlotState tests
// ============================================================================

#[test]
fn test_slot_state_empty() {
    let state = SlotState::Empty;
    assert!(state.is_empty());
    assert!(!state.is_active());
}

#[test]
fn test_slot_state_active() {
    let state = SlotState::Active {
        request_id: 1,
        prompt_tokens: vec![100, 200],
        generated_tokens: vec![300],
        max_tokens: 50,
        temperature: 0.7,
        top_k: 40,
    };
    assert!(!state.is_empty());
    assert!(state.is_active());
}

#[test]
fn test_slot_state_completed() {
    let state = SlotState::Completed {
        request_id: 1,
        generated_tokens: vec![100, 200, 300],
    };
    assert!(!state.is_empty());
    assert!(!state.is_active());
    assert!(state.is_completed());
}

// ============================================================================
// SpeculativeConfig tests
// ============================================================================

#[test]
fn test_speculative_config_default() {
    let config = SpeculativeConfig::default();
    assert!(config.speculation_length > 0);
    assert!(config.draft_temperature >= 0.0);
}

// ============================================================================
// SpeculativeDecoder tests
// ============================================================================

#[test]
fn test_speculative_decoder_default() {
    let decoder = SpeculativeDecoder::default();
    assert!(decoder.config.speculation_length > 0);
}

// ============================================================================
// ChunkedPrefillStats tests
// ============================================================================

#[test]
fn test_chunked_prefill_stats_basic() {
    let stats = ChunkedPrefillStats {
        total_chunks: 4,
        chunk_size: 128,
        total_tokens: 512,
        total_time_ms: 100.0,
        avg_chunk_time_ms: 25.0,
        ttft_ms: 50.0,
        tokens_per_second: 5120.0,
    };

    assert_eq!(stats.total_chunks, 4);
    assert_eq!(stats.chunk_size, 128);
    assert!((stats.total_time_ms - 100.0).abs() < 0.1);
}

#[test]
fn test_chunked_prefill_stats_clone() {
    let stats = ChunkedPrefillStats {
        total_chunks: 2,
        chunk_size: 64,
        total_tokens: 128,
        total_time_ms: 50.0,
        avg_chunk_time_ms: 25.0,
        ttft_ms: 30.0,
        tokens_per_second: 2560.0,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.total_chunks, stats.total_chunks);
}

#[test]
fn test_chunked_prefill_stats_debug() {
    let stats = ChunkedPrefillStats {
        total_chunks: 1,
        chunk_size: 256,
        total_tokens: 256,
        total_time_ms: 25.0,
        avg_chunk_time_ms: 25.0,
        ttft_ms: 25.0,
        tokens_per_second: 10240.0,
    };

    let debug = format!("{:?}", stats);
    assert!(debug.contains("ChunkedPrefillStats"));
}
