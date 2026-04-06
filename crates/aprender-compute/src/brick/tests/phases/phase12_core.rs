use super::super::super::*;
use std::time::Duration;

// ========================================================================
// Phase 12 Falsification Tests (F156-F175)
// ========================================================================

/// F156: PerfMetrics accuracy - wall clock drift < 1%
#[test]
fn test_f156_perf_metrics_accuracy() {
    let mut metrics = PerfMetrics::new();

    // Record known values
    metrics.record_load(1000);
    metrics.record_prefill(200, 100);
    metrics.record_decode(50);
    metrics.record_decode(50);

    // Verify calculations
    assert_eq!(metrics.total_ms(), 1300); // 1000 + 200 + 100
    assert_eq!(metrics.time_to_first_token_ms(), 1200); // 1000 + 200
    assert_eq!(metrics.n_eval, 2);

    // Tokens per second: 2 tokens / 100ms = 20 tok/s
    let tps = metrics.tokens_per_second();
    assert!((tps - 20.0).abs() < 0.1, "Expected ~20 tok/s, got {}", tps);

    // Prefill: 100 tokens / 200ms = 500 tok/s
    let prefill_tps = metrics.prefill_tokens_per_second();
    assert!((prefill_tps - 500.0).abs() < 1.0, "Expected ~500 tok/s, got {}", prefill_tps);
}

/// F157: Direct I/O alignment - 4KB aligned
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_f157_direct_io_alignment() {
    let buf = AlignedBuffer::new(8192).expect("allocation should succeed");

    // Verify 4KB alignment
    assert!(is_direct_io_aligned(buf.as_ptr()), "Buffer should be 4KB aligned");
    assert_eq!(buf.as_ptr() as usize % DIRECT_IO_ALIGNMENT, 0);
    assert_eq!(buf.len(), 8192);
    assert!(!buf.is_empty());
}

/// F159: PerfMetrics summary format
#[test]
fn test_f159_perf_metrics_summary() {
    let mut metrics = PerfMetrics::new();
    metrics.record_load(1500);
    metrics.record_prefill(300, 512);
    metrics.record_decode_batch(1000, 20);

    let summary = metrics.summary();
    assert!(summary.contains("load: 1500ms"));
    assert!(summary.contains("prefill: 300ms"));
    assert!(summary.contains("512 tokens"));
    assert!(summary.contains("20 tokens"));
}

/// F160: Balance211 evenness - max-min <= 1
#[test]
fn test_f160_balance211_evenness() {
    // Test various distributions
    for (n, t) in [(10, 3), (100, 7), (17, 4), (1000, 16)] {
        let ranges = balance211(n, t);

        let counts: Vec<usize> = ranges.iter().map(|(_, c)| *c).collect();
        let min_count = *counts.iter().min().unwrap();
        let max_count = *counts.iter().max().unwrap();

        assert!(
            max_count - min_count <= 1,
            "Balance211({}, {}): max-min should be <= 1, got {} - {} = {}",
            n,
            t,
            max_count,
            min_count,
            max_count - min_count
        );

        // Verify total elements sum to n
        let total: usize = counts.iter().sum();
        assert_eq!(total, n, "Total elements should equal n");
    }
}

/// F161: Cache line alignment effective
#[test]
fn test_f161_cache_alignment() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let aligned: CacheAligned<AtomicU64> = CacheAligned::new(AtomicU64::new(42));

    // Verify alignment
    assert_eq!(std::mem::align_of_val(&aligned), 64, "Should be 64-byte aligned");

    // Verify size is at least 64 bytes
    assert!(std::mem::size_of_val(&aligned) >= 64, "Should be at least 64 bytes");

    // Verify value is correct
    assert_eq!(aligned.get().load(Ordering::Relaxed), 42);
}

/// F163: Buffer watermark triggers correctly
#[test]
fn test_f163_watermark_triggers() {
    let wm = BufferWatermarks::new(1024, 8192);

    // Below low watermark - can write
    assert!(wm.can_write(500));
    assert!(!wm.should_backpressure(500));

    // Between watermarks
    assert!(!wm.can_write(2000));
    assert!(!wm.should_backpressure(2000));

    // At high watermark - backpressure
    assert!(!wm.can_write(8192));
    assert!(wm.should_backpressure(8192));

    // Above high watermark
    assert!(wm.should_backpressure(10000));
}

/// F164: Resource pool permit limiting
#[test]
fn test_f164_pool_permit_limiting() {
    let pool: ResourcePool<Vec<u8>> = ResourcePool::new(3, || Vec::with_capacity(1024));

    assert_eq!(pool.available(), 3);

    // Acquire all permits
    let r1 = pool.try_acquire().expect("Should acquire 1");
    assert_eq!(pool.available(), 2);

    let r2 = pool.try_acquire().expect("Should acquire 2");
    assert_eq!(pool.available(), 1);

    let r3 = pool.try_acquire().expect("Should acquire 3");
    assert_eq!(pool.available(), 0);

    // Pool exhausted
    assert!(pool.try_acquire().is_none(), "Pool should be exhausted");

    // Release one
    drop(r1);
    assert_eq!(pool.available(), 1);

    // Can acquire again
    let _r4 = pool.try_acquire().expect("Should acquire after release");
    assert_eq!(pool.available(), 0);

    drop(r2);
    drop(r3);
}

/// F165: Graceful shutdown completes cleanly
#[test]
fn test_f165_shutdown_clean() {
    let shutdown = GracefulShutdown::new(Duration::from_millis(100));

    // No active operations - should complete immediately
    let result = shutdown.shutdown();
    assert_eq!(result, ShutdownResult::Clean);
}

/// F166: Graceful shutdown timeout works
#[test]
fn test_f166_shutdown_timeout() {
    use std::sync::Arc;
    use std::thread;

    let shutdown = Arc::new(GracefulShutdown::new(Duration::from_millis(50)));

    // Register an operation that won't complete
    let guard = shutdown.register().expect("Should register");

    // Start shutdown in another thread
    let shutdown_clone = Arc::clone(&shutdown);
    let handle = thread::spawn(move || shutdown_clone.shutdown());

    // Wait for shutdown to timeout
    let result = handle.join().expect("Thread should complete");

    // Should timeout with 1 remaining operation
    match result {
        ShutdownResult::Timeout { remaining } => {
            assert_eq!(remaining, 1, "Should have 1 remaining operation");
        }
        ShutdownResult::Clean => {
            panic!("Should have timed out");
        }
    }

    // Clean up
    drop(guard);
}

/// F167: DoS limits enforced - rejects oversized
#[test]
fn test_f167_dos_limits_enforced() {
    let limits = ServeLimits::default();

    // Valid request
    assert!(limits.validate_request(50, 1024).is_ok());

    // Too many headers
    let err = limits.validate_request(200, 1024).unwrap_err();
    assert!(matches!(err, LimitError::TooManyHeaders { .. }));

    // Body too large
    let err = limits.validate_request(50, 10 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, LimitError::BodyTooLarge { .. }));
}

/// F168: Connection limit works
#[test]
fn test_f168_connection_limit() {
    let limits = ServeLimits::default().with_max_connections(100);

    // Below limit
    assert!(limits.validate_connections(50).is_ok());
    assert!(limits.validate_connections(99).is_ok());

    // At limit
    let err = limits.validate_connections(100).unwrap_err();
    assert!(matches!(err, LimitError::ConnectionLimitReached { .. }));

    // Above limit
    let err = limits.validate_connections(150).unwrap_err();
    assert!(matches!(err, LimitError::ConnectionLimitReached { .. }));
}

/// F169: Buffer watermark pressure level
#[test]
fn test_f169_watermark_pressure_level() {
    let wm = BufferWatermarks::new(1000, 10000);

    // 0% at empty
    assert!((wm.pressure_level(0) - 0.0).abs() < 0.01);

    // 50% at half
    assert!((wm.pressure_level(5000) - 0.5).abs() < 0.01);

    // 100% at high watermark
    assert!((wm.pressure_level(10000) - 1.0).abs() < 0.01);

    // Capped at 100%
    assert!((wm.pressure_level(20000) - 1.0).abs() < 0.01);
}

/// F170: WatermarkedBuffer flow control
#[test]
fn test_f170_watermarked_buffer_flow() {
    let mut buf = WatermarkedBuffer::new(BufferWatermarks::new(100, 1000));

    // Initially can write
    assert!(buf.can_write());
    assert!(!buf.should_backpressure());

    // Write some data
    buf.write(&[0u8; 500]);
    assert!(!buf.can_write()); // Above low watermark
    assert!(!buf.should_backpressure()); // Below high watermark

    // Write more to trigger backpressure
    buf.write(&[0u8; 600]);
    assert!(buf.should_backpressure()); // At/above high watermark

    // Drain everything to resume writing
    buf.clear();
    assert!(buf.can_write());
    assert!(buf.is_empty());
}

/// F171: Balance211 iterator
#[test]
fn test_f171_balance211_iterator() {
    let mut iter = Balance211Iter::new(10, 3);

    assert_eq!(iter.len(), 3);

    let r1 = iter.next().unwrap();
    assert_eq!(r1, 0..4); // First thread gets 4 items

    let r2 = iter.next().unwrap();
    assert_eq!(r2, 4..7); // Second thread gets 3 items

    let r3 = iter.next().unwrap();
    assert_eq!(r3, 7..10); // Third thread gets 3 items

    assert!(iter.next().is_none());
}

/// F172: InferencePhase enum
#[test]
fn test_f172_inference_phase() {
    let phase = InferencePhase::default();
    assert_eq!(phase, InferencePhase::Prefill);

    let decode = InferencePhase::Decode;
    assert_ne!(decode, InferencePhase::Prefill);
}

/// F173: PerfMetrics reset
#[test]
fn test_f173_perf_metrics_reset() {
    let mut metrics = PerfMetrics::new();
    metrics.record_load(1000);
    metrics.record_prefill(200, 50);
    metrics.record_decode(100);

    assert_ne!(metrics.total_ms(), 0);

    metrics.reset();

    assert_eq!(metrics.t_load_ms, 0);
    assert_eq!(metrics.t_p_eval_ms, 0);
    assert_eq!(metrics.t_eval_ms, 0);
    assert_eq!(metrics.n_p_eval, 0);
    assert_eq!(metrics.n_eval, 0);
    assert_eq!(metrics.total_ms(), 0);
}

/// F174: ServeLimits builder pattern
#[test]
fn test_f174_serve_limits_builder() {
    let limits = ServeLimits::new()
        .with_max_request_size(1024 * 1024)
        .with_max_headers(50)
        .with_max_connections(500);

    assert_eq!(limits.max_request_size, 1024 * 1024);
    assert_eq!(limits.max_headers, 50);
    assert_eq!(limits.max_connections, 500);
}

/// F175: LimitError display
#[test]
fn test_f175_limit_error_display() {
    let err = LimitError::TooManyHeaders { count: 150, max: 100 };
    let msg = format!("{}", err);
    assert!(msg.contains("150"));
    assert!(msg.contains("100"));

    let err = LimitError::BodyTooLarge { size: 5_000_000, max: 2_000_000 };
    let msg = format!("{}", err);
    assert!(msg.contains("5000000"));
    assert!(msg.contains("2000000"));
}

/// F158: Prefetch slice doesn't panic
#[test]
fn test_f158_prefetch_slice() {
    let data: Vec<f32> = vec![1.0; 1024];

    // Should not panic on any locality level
    prefetch_slice(&data, PrefetchLocality::None);
    prefetch_slice(&data, PrefetchLocality::Low);
    prefetch_slice(&data, PrefetchLocality::Moderate);
    prefetch_slice(&data, PrefetchLocality::High);

    // Empty slice should not panic
    let empty: Vec<f32> = vec![];
    prefetch_slice(&empty, PrefetchLocality::High);
}

/// F162: Memory advice enum
#[test]
fn test_f162_memory_advice() {
    // Just verify the enum variants exist and are distinct
    let seq = MemoryAdvice::Sequential;
    let rand = MemoryAdvice::Random;
    let need = MemoryAdvice::WillNeed;
    let dont = MemoryAdvice::DontNeed;

    assert_ne!(seq, rand);
    assert_ne!(need, dont);
    assert_eq!(seq, MemoryAdvice::Sequential);
}

/// F176: Cache line constants
#[test]
fn test_f176_cache_line_constants() {
    assert_eq!(CACHE_LINE_SIZE, 64);
    assert_eq!(CACHE_LINE_SIZE_F32, 16); // 64 / 4 = 16 floats
    assert_eq!(DIRECT_IO_ALIGNMENT, 4096);
}
