#![allow(clippy::many_single_char_names)]

use crate::gpu::*;
#[test]
fn test_fused_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let output = fused_layernorm(&input, &gamma, &beta, eps);

    assert_eq!(output.len(), 4);
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_layernorm_fused_matches_standard() {
    let input = vec![0.5, 1.5, 2.5, 3.5];
    let gamma = vec![1.0, 2.0, 1.0, 2.0];
    let beta = vec![0.1, 0.2, 0.3, 0.4];
    let eps = 1e-5;

    let standard = standard_layernorm(&input, &gamma, &beta, eps);
    let fused = fused_layernorm(&input, &gamma, &beta, eps);

    for (s, f) in standard.iter().zip(fused.iter()) {
        assert!((s - f).abs() < 1e-4, "Fused should match standard");
    }
}

// ============================================================================
// Coverage Tests - Quantized Operations
// ============================================================================

#[test]
fn test_quantized_dot_q4_basic() {
    // Q4 block: 18 bytes (2 scales + 16 nibbles)
    let block_a = vec![0u8; 18];
    let block_b = vec![0u8; 18];

    let result = quantized_dot_q4(&block_a, &block_b);
    // Zero blocks should give zero result
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_quantized_dot_q8_basic() {
    // Q8 block: 34 bytes (2 bytes scale + 32 bytes quants)
    let block_a = vec![0u8; 34];
    let block_b = vec![0u8; 34];

    let result = quantized_dot_q8(&block_a, &block_b);
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn test_prefetch_read_no_panic() {
    let data = vec![1.0f32; 100];
    // Should not panic
    prefetch_read(&data, 0, 10);
    prefetch_read(&data, 50, 20);
}

#[test]
fn test_sequential_sum_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sequential_sum(&data);
    assert!((result - 15.0).abs() < 1e-5);
}

#[test]
fn test_sum_with_prefetch_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sum_with_prefetch(&data, 2);
    assert!((result - 15.0).abs() < 1e-5);
}

#[test]
fn test_sum_with_prefetch_matches_sequential() {
    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let seq = sequential_sum(&data);
    let prefetch = sum_with_prefetch(&data, 8);
    assert!((seq - prefetch).abs() < 1e-3);
}

// ============================================================================
// Coverage Tests - Matmul Operations
// ============================================================================

#[test]
fn test_naive_matmul_2x2() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let c = naive_matmul(&a, &b, 2, 2, 2);

    // C = [[19, 22], [43, 50]]
    assert!((c[0] - 19.0).abs() < 1e-5);
    assert!((c[1] - 22.0).abs() < 1e-5);
    assert!((c[2] - 43.0).abs() < 1e-5);
    assert!((c[3] - 50.0).abs() < 1e-5);
}

#[test]
fn test_blocked_matmul_2x2() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let c = blocked_matmul(&a, &b, 2, 2, 2, 2);

    assert!((c[0] - 19.0).abs() < 1e-5);
    assert!((c[3] - 50.0).abs() < 1e-5);
}

#[test]
fn test_blocked_matmul_matches_naive() {
    let a: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..16).map(|i| i as f32 * 0.2).collect();

    let c_naive = naive_matmul(&a, &b, 4, 4, 4);
    let c_blocked = blocked_matmul(&a, &b, 4, 4, 4, 2);

    for (n, bl) in c_naive.iter().zip(c_blocked.iter()) {
        assert!((*n - *bl).abs() < 1e-4, "Blocked should match naive");
    }
}

// =========================================================================
// Coverage Tests: Basic types (PMAT-802)
// =========================================================================

#[test]
fn test_contiguous_attention_buffer_debug_cov() {
    let buf = ContiguousAttentionBuffer::new(8, 4, 64);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("ContiguousAttentionBuffer"));
    assert!(buf.is_contiguous());
}

#[test]
fn test_contiguous_attention_buffer_views_cov() {
    let mut buf = ContiguousAttentionBuffer::new(2, 2, 4);
    let (q, k, v, o) = buf.get_views();
    assert_eq!(q.len(), 2 * 2 * 4);
    assert_eq!(k.len(), 2 * 2 * 4);
    assert_eq!(v.len(), 2 * 2 * 4);
    assert_eq!(o.len(), 2 * 2 * 4);

    let (qm, km, vm, om) = buf.get_views_mut();
    qm[0] = 1.0;
    km[0] = 2.0;
    vm[0] = 3.0;
    om[0] = 4.0;
    assert_eq!(qm[0], 1.0);
}

#[test]
fn test_cache_aligned_buffer_debug_cov() {
    let buf = CacheAlignedBuffer::new(128);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("CacheAlignedBuffer"));
}

#[test]
fn test_tensor_pool_debug_cov() {
    let pool = TensorPool::new(1024);
    let debug = format!("{:?}", pool);
    assert!(debug.contains("TensorPool"));
}

#[test]
fn test_forward_arena_debug_cov() {
    let arena = ForwardArena::new(1024);
    let debug = format!("{:?}", arena);
    assert!(debug.contains("ForwardArena"));
}

#[test]
fn test_scratch_buffer_debug_cov() {
    let buf = ScratchBuffer::new(4, 256);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("ScratchBuffer"));
}

#[test]
fn test_quantized_accumulator_debug_clone_cov() {
    let acc = QuantizedAccumulator::new();
    let debug = format!("{:?}", acc);
    assert!(debug.contains("QuantizedAccumulator"));
    let cloned = acc.clone();
    assert!(format!("{:?}", cloned).contains("QuantizedAccumulator"));
}

#[test]
fn test_double_buffer_debug_cov() {
    let db: DoubleBuffer<Vec<f32>> = DoubleBuffer::new(64);
    let debug = format!("{:?}", db);
    assert!(debug.contains("DoubleBuffer"));
}

#[test]
fn test_chunked_processor_debug_clone_cov() {
    let proc = ChunkedProcessor::new(64);
    let debug = format!("{:?}", proc);
    assert!(debug.contains("ChunkedProcessor"));
    let cloned = proc.clone();
    assert!(format!("{:?}", cloned).contains("ChunkedProcessor"));
}

#[test]
fn test_gpu_pipeline_stage_debug_clone_copy_cov() {
    let embed = GpuPipelineStage::Embed;
    let debug = format!("{:?}", embed);
    assert!(debug.contains("Embed"));
    let cloned = embed;
    assert_eq!(cloned, GpuPipelineStage::Embed);

    let attn = GpuPipelineStage::Attention;
    assert!(format!("{:?}", attn).contains("Attention"));

    let ffn = GpuPipelineStage::FFN;
    assert!(format!("{:?}", ffn).contains("FFN"));

    let output = GpuPipelineStage::Output;
    assert!(format!("{:?}", output).contains("Output"));
}

#[test]
fn test_inference_pipeline_debug_cov() {
    let pipeline = InferencePipeline::new(4);
    let debug = format!("{:?}", pipeline);
    assert!(debug.contains("InferencePipeline"));
}

#[test]
fn test_token_batch_debug_cov() {
    let batch = TokenBatch::new(16);
    let debug = format!("{:?}", batch);
    assert!(debug.contains("TokenBatch"));
}

#[test]
fn test_speculative_buffer_debug_cov() {
    let buf = SpeculativeBuffer::new(4);
    let debug = format!("{:?}", buf);
    assert!(debug.contains("SpeculativeBuffer"));
}

#[test]
fn test_inference_batch_scheduler_debug_cov() {
    let scheduler = InferenceBatchScheduler::new();
    let debug = format!("{:?}", scheduler);
    assert!(debug.contains("InferenceBatchScheduler"));
}

#[test]
fn test_async_request_queue_debug_cov() {
    let queue: AsyncRequestQueue<u32> = AsyncRequestQueue::new(10);
    let debug = format!("{:?}", queue);
    assert!(debug.contains("AsyncRequestQueue"));
}

#[test]
fn test_inference_event_notifier_debug_cov() {
    let notifier = InferenceEventNotifier::new();
    let debug = format!("{:?}", notifier);
    assert!(debug.contains("InferenceEventNotifier"));
}

#[test]
fn test_timeout_manager_debug_cov() {
    let mgr = TimeoutManager::new();
    let debug = format!("{:?}", mgr);
    assert!(debug.contains("TimeoutManager"));
}

#[test]
fn test_priority_request_debug_clone_cov() {
    let req = PriorityRequest::new(5, 42u32);
    let debug = format!("{:?}", req);
    assert!(debug.contains("PriorityRequest"));
    let cloned = req.clone();
    assert!(format!("{:?}", cloned).contains("PriorityRequest"));
}

#[test]
fn test_priority_request_queue_debug_cov() {
    let queue: PriorityRequestQueue<u32> = PriorityRequestQueue::new();
    let debug = format!("{:?}", queue);
    assert!(debug.contains("PriorityRequestQueue"));
}

#[test]
fn test_token_rate_limiter_debug_cov() {
    let limiter = TokenRateLimiter::new(100.0, 10);
    let debug = format!("{:?}", limiter);
    assert!(debug.contains("TokenRateLimiter"));
}

#[test]
fn test_resource_tracker_debug_cov() {
    let tracker = ResourceTracker::new(1024 * 1024 * 1024, 16);
    let debug = format!("{:?}", tracker);
    assert!(debug.contains("ResourceTracker"));
}

#[test]
fn test_inference_metrics_debug_cov() {
    let metrics = InferenceMetrics::new();
    let debug = format!("{:?}", metrics);
    assert!(debug.contains("InferenceMetrics"));
}

#[test]
fn test_health_checker_debug_cov() {
    let checker = HealthChecker::new();
    let debug = format!("{:?}", checker);
    assert!(debug.contains("HealthChecker"));
}

#[test]
fn test_shutdown_coordinator_debug_cov() {
    let coord = ShutdownCoordinator::new();
    let debug = format!("{:?}", coord);
    assert!(debug.contains("ShutdownCoordinator"));
}

#[test]
fn test_compute_backend_debug_clone_copy_cov() {
    let cpu = ComputeBackend::Cpu;
    let debug = format!("{:?}", cpu);
    assert!(debug.contains("Cpu"));
    let copied = cpu;
    assert_eq!(copied, ComputeBackend::Cpu);

    let gpu = ComputeBackend::Gpu;
    assert!(format!("{:?}", gpu).contains("Gpu"));

    let auto = ComputeBackend::Auto;
    assert!(format!("{:?}", auto).contains("Auto"));
}

#[test]
fn test_log_level_debug_clone_copy_eq_ord_cov() {
    let debug_level = LogLevel::Debug;
    let debug = format!("{:?}", debug_level);
    assert!(debug.contains("Debug"));
    assert_eq!(debug_level, LogLevel::Debug);

    let info = LogLevel::Info;
    assert!(format!("{:?}", info).contains("Info"));

    let warn = LogLevel::Warn;
    assert!(format!("{:?}", warn).contains("Warn"));

    let error = LogLevel::Error;
    assert!(format!("{:?}", error).contains("Error"));

    // Test ordering
    assert!(LogLevel::Debug < LogLevel::Info);
    assert!(LogLevel::Info < LogLevel::Warn);
    assert!(LogLevel::Warn < LogLevel::Error);
}

#[test]
fn test_log_entry_debug_clone_cov() {
    let entry = LogEntry::new(LogLevel::Info, "test message");
    let debug = format!("{:?}", entry);
    assert!(debug.contains("LogEntry"));
    let cloned = entry.clone();
    assert!(format!("{:?}", cloned).contains("LogEntry"));
}

// Note: MemoryTracker, DiagnosticsCollector, DebugMode don't implement Debug
// Test their basic construction instead
#[test]
fn test_memory_tracker_new_cov() {
    let tracker = MemoryTracker::new();
    // Basic construction succeeds - verify initial state via allocation
    tracker.record_allocation("test", 100);
    tracker.record_deallocation("test", 100);
}

#[test]
fn test_diagnostics_collector_new_cov() {
    let collector = DiagnosticsCollector::new();
    // Basic construction succeeds - verify request_count is zero
    assert_eq!(
        collector
            .request_count
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn test_debug_mode_new_cov() {
    let mode = DebugMode::new();
    // Basic construction succeeds - verify debug mode is off by default
    assert!(!mode.is_enabled());
}

#[test]
fn test_request_capture_debug_cov() {
    let capture = RequestCapture::new();
    let debug = format!("{:?}", capture);
    assert!(debug.contains("RequestCapture"));
}

#[test]
fn test_state_dump_debug_cov() {
    let dump = StateDump::new();
    let debug = format!("{:?}", dump);
    assert!(debug.contains("StateDump"));
}

#[test]
fn test_gguf_model_state_debug_cov() {
    let state = GgufModelState::new();
    let debug = format!("{:?}", state);
    assert!(debug.contains("GgufModelState"));
}

// =========================================================================
// Coverage Tests: Edge cases and function variants
// =========================================================================

#[test]
fn test_exceeds_gpu_buffer_limit_cov() {
    // Test below limit
    let small = 1000;
    assert!(!exceeds_gpu_buffer_limit(small));

    // Test at limit
    let at_limit = 256 * 1024 * 1024 / 4;
    assert!(!exceeds_gpu_buffer_limit(at_limit));

    // Test above limit
    let above_limit = 256 * 1024 * 1024 / 4 + 1;
    assert!(exceeds_gpu_buffer_limit(above_limit));
}

#[test]
fn test_scalar_softmax_empty_cov() {
    let result = scalar_softmax(&[]);
    assert!(result.is_empty());
}

include!("simd_softmax_scalar.rs");
include!("sequential_ffn_parallel.rs");
include!("double_buffer_chunked.rs");
