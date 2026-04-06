//! Integration tests for Sprint 41: RLE Compression
//!
//! This tests the run-length encoding module with realistic tight loop scenarios.

use renacer::rle_compression::{compress_spans, decompress_segment};
use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use std::collections::HashMap;

fn create_span(
    span_id: u8,
    logical_clock: u64,
    syscall_name: &str,
    duration_nanos: u64,
) -> SpanRecord {
    SpanRecord::new(
        [1; 16],
        [span_id; 8],
        None,
        syscall_name.to_string(),
        SpanKind::Internal,
        logical_clock * 1000,
        logical_clock * 1000 + duration_nanos,
        logical_clock,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        1234,
        5678,
    )
}

#[test]
fn test_realistic_tight_loop_compression() {
    // Scenario: Real-world tight loop - 10,000 repeated read() calls
    let mut spans = vec![];
    for i in 0..10_000 {
        spans.push(create_span(
            (i % 256) as u8,
            i,
            "read",
            100, // 100ns per read
        ));
    }

    let compressed = compress_spans(&spans, 100).expect("test");

    // Should compress into a single segment
    assert_eq!(compressed.segments.len(), 1);
    assert_eq!(compressed.uncompressed.len(), 0);

    let segment = &compressed.segments[0];
    assert_eq!(segment.syscall_name, "read");
    assert_eq!(segment.count, 10_000);
    assert_eq!(segment.total_duration, 1_000_000); // 10K * 100ns
    assert_eq!(segment.avg_duration, 100);

    // Verify compression ratio
    let ratio = compressed.compression_ratio();
    assert_eq!(ratio, 10_000.0);

    // Verify storage savings
    let savings = compressed.storage_savings_percent();
    assert!((savings - 99.99).abs() < 0.01); // ~99.99% savings
}

#[test]
fn test_262k_compression_target() {
    // Scenario: Achieve 262,144× compression (2^18)
    let mut spans = vec![];
    for i in 0..262_144 {
        spans.push(create_span(
            (i % 256) as u8,
            i,
            "poll",
            50, // 50ns per poll
        ));
    }

    let compressed = compress_spans(&spans, 1000).expect("test");

    // Should compress into a single segment
    assert_eq!(compressed.segments.len(), 1);

    let segment = &compressed.segments[0];
    assert_eq!(segment.count, 262_144);
    assert_eq!(segment.syscall_name, "poll");

    // Verify compression ratio meets target
    let ratio = compressed.compression_ratio();
    assert!(ratio >= 262_144.0, "Expected ratio ≥262,144, got {}", ratio);

    println!("Achieved {}× compression (target: 262,144×)", ratio);
}

#[test]
fn test_mixed_workload_selective_compression() {
    // Scenario: Mixed workload - compress tight loops, keep varied calls
    let mut spans = vec![];

    // Phase 1: 200 repeated reads (compressed)
    for i in 0..200 {
        spans.push(create_span(i as u8, i as u64, "read", 100));
    }

    // Phase 2: 50 varied syscalls (not compressed)
    let syscalls = ["open", "stat", "close", "write", "fsync"];
    for i in 200..250 {
        let syscall = syscalls[(i - 200) % syscalls.len()];
        spans.push(create_span(i as u8, i as u64, syscall, 150));
    }

    // Phase 3: 300 repeated writes (compressed)
    for i in 250..550 {
        spans.push(create_span((i % 256) as u8, i as u64, "write", 80));
    }

    let compressed = compress_spans(&spans, 100).expect("test");

    // Should have 2 compressed segments + 50 uncompressed
    assert_eq!(compressed.segments.len(), 2);
    assert_eq!(compressed.uncompressed.len(), 50);

    // Verify first segment (reads)
    assert_eq!(compressed.segments[0].syscall_name, "read");
    assert_eq!(compressed.segments[0].count, 200);

    // Verify second segment (writes)
    assert_eq!(compressed.segments[1].syscall_name, "write");
    assert_eq!(compressed.segments[1].count, 300);

    // Verify compression ratio
    let ratio = compressed.compression_ratio();
    assert!(ratio > 5.0, "Expected ratio >5, got {}", ratio);
}

#[test]
fn test_decompression_accuracy() {
    // Scenario: Compress and decompress - verify lossless
    let mut spans = vec![];
    for i in 0..1000 {
        spans.push(create_span((i % 256) as u8, i, "recv", 120));
    }

    let compressed = compress_spans(&spans, 500).expect("test");
    assert_eq!(compressed.segments.len(), 1);

    let segment = &compressed.segments[0];
    let decompressed = decompress_segment(segment);

    // Should reconstruct 1000 spans
    assert_eq!(decompressed.len(), 1000);

    // Verify logical clock ordering
    for (i, span) in decompressed.iter().enumerate() {
        assert_eq!(span.logical_clock, i as u64);
        assert_eq!(span.span_name, "recv");
        assert_eq!(span.duration_nanos, 120);
    }
}

#[test]
fn test_varying_durations_statistics() {
    // Scenario: Tight loop with varying durations
    let mut spans = vec![];
    for i in 0..500 {
        let duration = 100 + (i % 50) * 10; // 100-590ns range
        spans.push(create_span((i % 256) as u8, i, "read", duration));
    }

    let compressed = compress_spans(&spans, 100).expect("test");
    assert_eq!(compressed.segments.len(), 1);

    let segment = &compressed.segments[0];

    // Verify duration statistics
    assert_eq!(segment.min_duration, 100);
    assert_eq!(segment.max_duration, 590);
    assert!(segment.avg_duration > 100 && segment.avg_duration < 590);
    assert_eq!(segment.duration_variance(), 490); // max - min
}

#[test]
fn test_poll_loop_compression() {
    // Scenario: Realistic poll() loop in event-driven system
    // 100,000 poll() calls waiting for events
    let mut spans = vec![];
    for i in 0..100_000 {
        spans.push(create_span(
            (i % 256) as u8,
            i,
            "poll",
            1000, // 1μs per poll
        ));
    }

    let compressed = compress_spans(&spans, 10_000).expect("test");

    assert_eq!(compressed.segments.len(), 1);
    let segment = &compressed.segments[0];

    assert_eq!(segment.syscall_name, "poll");
    assert_eq!(segment.count, 100_000);
    assert_eq!(segment.total_duration, 100_000_000); // 100ms total

    // Verify compression ratio
    let ratio = compressed.compression_ratio();
    assert!(ratio >= 100_000.0);

    println!("Poll loop: {}× compression", ratio);
}

#[test]
fn test_network_recv_loop_compression() {
    // Scenario: Network receive loop - 50,000 recv() calls
    let mut spans = vec![];
    for i in 0..50_000 {
        spans.push(create_span(
            (i % 256) as u8,
            i,
            "recvfrom",
            500, // 500ns per recv
        ));
    }

    let compressed = compress_spans(&spans, 5_000).expect("test");

    assert_eq!(compressed.segments.len(), 1);
    let segment = &compressed.segments[0];

    assert_eq!(segment.syscall_name, "recvfrom");
    assert_eq!(segment.count, 50_000);

    // Verify compression
    let ratio = compressed.compression_ratio();
    assert!(ratio >= 50_000.0);

    // Verify decompression
    let decompressed = decompress_segment(segment);
    assert_eq!(decompressed.len(), 50_000);
    assert_eq!(decompressed[0].span_name, "recvfrom");
}

#[test]
fn test_min_run_length_threshold() {
    // Scenario: Test various min_run_length thresholds
    let mut spans = vec![];
    for i in 0..500 {
        spans.push(create_span((i % 256) as u8, i, "read", 100));
    }

    // Test with min_run_length = 1000 (should NOT compress)
    let compressed_high = compress_spans(&spans, 1000).expect("test");
    assert_eq!(compressed_high.segments.len(), 0);
    assert_eq!(compressed_high.uncompressed.len(), 500);

    // Test with min_run_length = 100 (SHOULD compress)
    let compressed_low = compress_spans(&spans, 100).expect("test");
    assert_eq!(compressed_low.segments.len(), 1);
    assert_eq!(compressed_low.uncompressed.len(), 0);
}

#[test]
fn test_multiple_tight_loops() {
    // Scenario: Multiple distinct tight loops
    let mut spans = vec![];

    // Loop 1: 200 reads
    for i in 0..200 {
        spans.push(create_span(i as u8, i as u64, "read", 100));
    }

    // Break: 10 different syscalls
    for i in 200..210 {
        spans.push(create_span(i as u8, i as u64, "stat", 50));
    }

    // Loop 2: 300 writes
    for i in 210..510 {
        spans.push(create_span((i % 256) as u8, i as u64, "write", 80));
    }

    // Break: 5 different syscalls
    for i in 510..515 {
        spans.push(create_span((i % 256) as u8, i as u64, "close", 30));
    }

    // Loop 3: 150 polls
    for i in 515..665 {
        spans.push(create_span((i % 256) as u8, i as u64, "poll", 200));
    }

    let compressed = compress_spans(&spans, 100).expect("test");

    // Should have 3 compressed segments
    assert_eq!(compressed.segments.len(), 3);

    // Verify segments
    assert_eq!(compressed.segments[0].syscall_name, "read");
    assert_eq!(compressed.segments[0].count, 200);

    assert_eq!(compressed.segments[1].syscall_name, "write");
    assert_eq!(compressed.segments[1].count, 300);

    assert_eq!(compressed.segments[2].syscall_name, "poll");
    assert_eq!(compressed.segments[2].count, 150);

    // Verify uncompressed (breaks between loops)
    assert_eq!(compressed.uncompressed.len(), 15); // 10 + 5
}

#[test]
fn test_compression_preserves_metadata() {
    // Scenario: Verify metadata (process_id, thread_id, trace_id) is preserved
    let mut spans = vec![];
    for i in 0..1000 {
        spans.push(create_span((i % 256) as u8, i, "read", 100));
    }

    let compressed = compress_spans(&spans, 500).expect("test");
    assert_eq!(compressed.segments.len(), 1);

    let segment = &compressed.segments[0];

    // Verify metadata
    assert_eq!(segment.process_id, 1234);
    assert_eq!(segment.thread_id, 5678);
    assert_eq!(segment.trace_id, [1; 16]);
}

#[test]
fn test_empty_trace_compression() {
    // Scenario: Empty trace
    let spans: Vec<SpanRecord> = vec![];
    let compressed = compress_spans(&spans, 100).expect("test");

    assert_eq!(compressed.segments.len(), 0);
    assert_eq!(compressed.uncompressed.len(), 0);
    assert_eq!(compressed.original_count, 0);
    assert_eq!(compressed.compression_ratio(), 1.0); // No compression
}

#[test]
fn test_no_compression_needed() {
    // Scenario: All unique syscalls - no compression possible
    let syscalls = ["open", "read", "write", "close", "stat", "fstat", "lseek", "mmap"];
    let mut spans = vec![];
    for (i, &syscall) in syscalls.iter().enumerate() {
        spans.push(create_span(i as u8, i as u64, syscall, 100));
    }

    let compressed = compress_spans(&spans, 5).expect("test");

    // No compression should occur
    assert_eq!(compressed.segments.len(), 0);
    assert_eq!(compressed.uncompressed.len(), 8);
    assert_eq!(compressed.compression_ratio(), 1.0);
}

#[test]
fn test_storage_savings_calculation() {
    // Scenario: Verify storage savings percentage
    let mut spans = vec![];
    for i in 0..10_000 {
        spans.push(create_span((i % 256) as u8, i, "read", 100));
    }

    let compressed = compress_spans(&spans, 100).expect("test");

    // 10,000 spans compressed to 1 segment
    let savings = compressed.storage_savings_percent();

    // Should be ~99.99% savings (10,000 → 1)
    assert!(savings > 99.9, "Expected >99.9% savings, got {}", savings);
    println!("Storage savings: {:.2}%", savings);
}

#[test]
fn test_total_span_count() {
    // Scenario: Verify total span count is correct
    let mut spans = vec![];

    // 500 compressed
    for i in 0..500 {
        spans.push(create_span((i % 256) as u8, i, "read", 100));
    }

    // 50 uncompressed
    let syscalls = ["open", "stat", "close"];
    for i in 500..550 {
        let syscall = syscalls[(i - 500) % syscalls.len()];
        spans.push(create_span((i % 256) as u8, i as u64, syscall, 150));
    }

    let compressed = compress_spans(&spans, 100).expect("test");

    // Verify total count
    let total = compressed.total_span_count();
    assert_eq!(total, 550); // 500 + 50

    // Verify breakdown
    assert_eq!(compressed.segments[0].count, 500);
    assert_eq!(compressed.uncompressed.len(), 50);
}
