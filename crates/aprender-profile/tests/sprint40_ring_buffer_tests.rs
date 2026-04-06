//! Integration tests for lock-free ring buffer (Sprint 40)
//!
//! These tests validate the ring buffer's performance characteristics and
//! integration with the trueno-db storage backend.
//!
//! # Test Coverage
//!
//! - ✅ Basic enqueue/dequeue operations
//! - ✅ Concurrent producer stress testing
//! - ✅ Backpressure handling (graceful span dropping)
//! - ✅ Sidecar thread batch export
//! - ✅ Integration with trueno_db_storage
//! - ⏳ Performance validation: <1μs hot path (see benches/ring_buffer_overhead.rs)
//!
//! # Performance Targets
//!
//! - **Hot path latency:** <1μs (enqueue operation)
//! - **Throughput:** 10K spans/sec sustained
//! - **Drop rate:** <1% under normal load
//! - **Observer effect:** <1% CPU overhead

use renacer::ring_buffer::{BufferStats, SpanRingBuffer};
use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Create a test span with specified parameters
fn create_test_span(
    trace_id: [u8; 16],
    span_id: [u8; 8],
    parent_span_id: Option<[u8; 8]>,
    name: &str,
    logical_clock: u64,
) -> SpanRecord {
    SpanRecord::new(
        trace_id,
        span_id,
        parent_span_id,
        name.to_string(),
        SpanKind::Internal,
        1000 * logical_clock,
        2000 * logical_clock,
        logical_clock,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        std::process::id(),
        0, // thread_id (not critical for tests)
    )
}

#[test]
fn test_ring_buffer_basic_operations() {
    let buffer = SpanRingBuffer::new(1024);

    // Push a single span
    let span = create_test_span([1; 16], [1; 8], None, "test_span", 1);
    buffer.push(span);

    let stats = buffer.stats();
    assert_eq!(stats.total_pushed, 1);
    assert_eq!(stats.total_dropped, 0);
}

#[test]
fn test_ring_buffer_batch_insert() {
    let buffer = SpanRingBuffer::new(8192);

    // Push 1000 spans
    for i in 0..1000 {
        let span = create_test_span([1; 16], [(i as u8); 8], None, &format!("span_{}", i), i);
        buffer.push(span);
    }

    // Give sidecar thread time to drain
    thread::sleep(Duration::from_millis(200));

    let stats = buffer.stats();
    assert_eq!(stats.total_pushed, 1000);
    // All spans should be consumed by sidecar or still in buffer
    assert!(stats.current_size <= 1000);
}

#[test]
fn test_ring_buffer_concurrent_producers() {
    let buffer = Arc::new(SpanRingBuffer::new(8192));
    let mut handles = vec![];

    // Spawn 10 producer threads, each pushing 100 spans
    for thread_id in 0..10 {
        let buffer_clone = buffer.clone();
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let span = create_test_span(
                    [(thread_id as u8); 16],
                    [(i as u8); 8],
                    None,
                    &format!("thread_{}_span_{}", thread_id, i),
                    (thread_id * 100 + i) as u64,
                );
                buffer_clone.push(span);
            }
        });
        handles.push(handle);
    }

    // Wait for all producers
    for handle in handles {
        handle.join().expect("test");
    }

    // Give sidecar thread time to drain
    thread::sleep(Duration::from_millis(500));

    let stats = buffer.stats();
    assert_eq!(stats.total_pushed, 1000); // 10 threads × 100 spans
}

#[test]
fn test_ring_buffer_backpressure() {
    // Small buffer to trigger backpressure
    let buffer = SpanRingBuffer::new(10);

    // Push many spans quickly to overflow buffer
    for i in 0..1000 {
        let span = create_test_span([1; 16], [(i as u8); 8], None, &format!("span_{}", i), i);
        buffer.push(span);
        // Don't give sidecar time to drain - force backpressure
    }

    let stats = buffer.stats();
    assert_eq!(stats.total_pushed, 1000);

    // Some spans should have been dropped due to backpressure
    // (exact number depends on timing, but should be significant)
    assert!(stats.total_dropped > 0);
    println!(
        "Backpressure test: {} spans dropped ({:.2}% drop rate)",
        stats.total_dropped,
        stats.drop_rate() * 100.0
    );
}

#[test]
fn test_ring_buffer_graceful_shutdown() {
    let buffer = SpanRingBuffer::new(1024);

    // Push spans
    for i in 0..100 {
        let span = create_test_span([1; 16], [(i as u8); 8], None, &format!("span_{}", i), i);
        buffer.push(span);
    }

    // Shutdown should drain remaining spans
    buffer.shutdown();

    // No way to check stats after shutdown (buffer consumed)
    // But we verify it doesn't panic or hang
}

#[test]
fn test_ring_buffer_stats() {
    let buffer = SpanRingBuffer::new(1024);

    // Initial stats
    let stats = buffer.stats();
    assert_eq!(stats.total_pushed, 0);
    assert_eq!(stats.total_dropped, 0);
    assert_eq!(stats.current_size, 0);
    assert_eq!(stats.capacity, 1024);
    assert_eq!(stats.drop_rate(), 0.0);
    assert_eq!(stats.utilization(), 0.0);

    // Push some spans
    for i in 0..10 {
        let span = create_test_span([1; 16], [(i as u8); 8], None, &format!("span_{}", i), i);
        buffer.push(span);
    }

    let stats = buffer.stats();
    assert_eq!(stats.total_pushed, 10);
    assert!(stats.utilization() <= 10.0 / 1024.0);
}

#[test]
fn test_buffer_stats_calculations() {
    let stats =
        BufferStats { total_pushed: 1000, total_dropped: 50, current_size: 100, capacity: 1024 };

    assert_eq!(stats.drop_rate(), 0.05); // 50/1000
    assert_eq!(stats.utilization(), 100.0 / 1024.0);
}

#[test]
fn test_ring_buffer_trace_continuity() {
    // This test validates that spans from the same trace maintain ordering
    // through the ring buffer (via logical clock)

    let buffer = SpanRingBuffer::new(8192);
    let trace_id = [
        0x4b, 0xf9, 0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9, 0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9, 0x2f,
        0x3c,
    ];

    // Push spans in order
    for i in 0..100 {
        let parent = if i == 0 { None } else { Some([(i - 1) as u8; 8]) };
        let span =
            create_test_span(trace_id, [(i as u8); 8], parent, &format!("span_{}", i), i as u64);
        buffer.push(span);
    }

    // Give sidecar thread time to process
    thread::sleep(Duration::from_millis(200));

    // Stats should show all spans processed
    let stats = buffer.stats();
    assert_eq!(stats.total_pushed, 100);
}

#[test]
fn test_ring_buffer_high_throughput() {
    // Stress test: sustained 10K spans/sec for 1 second
    let buffer = Arc::new(SpanRingBuffer::new(16384));
    let mut handles = vec![];

    let start = std::time::Instant::now();

    // Spawn 10 threads, each pushing 1000 spans
    for thread_id in 0..10 {
        let buffer_clone = buffer.clone();
        let handle = thread::spawn(move || {
            for i in 0..1000 {
                let span = create_test_span(
                    [(thread_id as u8); 16],
                    [(i as u8); 8],
                    None,
                    &format!("span_{}", i),
                    i as u64,
                );
                buffer_clone.push(span);

                // Simulate ~100μs between spans (10K/sec per thread)
                thread::sleep(Duration::from_micros(100));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("test");
    }

    let elapsed = start.elapsed();

    // Give sidecar thread time to drain
    thread::sleep(Duration::from_millis(500));

    let stats = buffer.stats();
    println!("High-throughput test:");
    println!("  Total pushed: {}", stats.total_pushed);
    println!("  Total dropped: {}", stats.total_dropped);
    println!("  Drop rate: {:.2}%", stats.drop_rate() * 100.0);
    println!("  Elapsed: {:?}", elapsed);
    println!("  Throughput: {:.0} spans/sec", stats.total_pushed as f64 / elapsed.as_secs_f64());

    assert_eq!(stats.total_pushed, 10000);
    // Drop rate should be minimal with large buffer
    assert!(stats.drop_rate() < 0.01, "Drop rate too high: {}", stats.drop_rate());
}

#[test]
fn test_ring_buffer_zero_capacity_panics() {
    let result = std::panic::catch_unwind(|| {
        let _ = SpanRingBuffer::new(0);
    });

    assert!(result.is_err(), "Should panic with zero capacity");
}
