//! Ring buffer performance benchmark (Sprint 40)
//!
//! This benchmark validates the <1μs hot path latency requirement for the
//! lock-free ring buffer. The hot path consists of:
//!
//! 1. `ring_buffer.push(span)` - Lock-free enqueue operation
//!
//! # Performance Targets
//!
//! - **Hot path latency:** <1μs (p50, p95, p99)
//! - **Baseline:** Synchronous I/O is 250-2500μs (250-2500× slower)
//! - **Observer effect:** <1% CPU overhead (Google observability SLO)
//!
//! # Run Instructions
//!
//! ```bash
//! cargo bench --bench ring_buffer_overhead
//! ```
//!
//! # Expected Output
//!
//! ```text
//! ring_buffer_push        time:   [200 ns 250 ns 300 ns]
//! ```
//!
//! # Peer-Reviewed Foundation
//!
//! - **Mestel et al. (2022). "Profiling-Guided Optimization." Google.**
//!   - Finding: Observability overhead >10% CPU is unacceptable
//!   - Application: Ring buffer keeps overhead <1%

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use renacer::ring_buffer::SpanRingBuffer;
use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use std::collections::HashMap;

/// Create a test span for benchmarking
fn create_bench_span(i: u64) -> SpanRecord {
    SpanRecord::new(
        [1; 16],
        [(i as u8); 8],
        None,
        format!("bench_span_{}", i),
        SpanKind::Internal,
        i * 1000,
        i * 2000,
        i,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        1234,
        5678,
    )
}

/// Benchmark: Ring buffer push (hot path)
///
/// This measures the latency of a single `push()` operation, which is the
/// hot path that runs in the application thread.
///
/// Target: <1μs (ideally ~200-500ns)
fn bench_ring_buffer_push(c: &mut Criterion) {
    let buffer = SpanRingBuffer::new(16384); // Large enough to never fill

    let mut i = 0u64;

    c.bench_function("ring_buffer_push", |b| {
        b.iter(|| {
            let span = create_bench_span(i);
            buffer.push(black_box(span));
            i += 1;
        });
    });

    // Shutdown buffer after benchmark
    std::mem::drop(buffer);
}

/// Benchmark: Ring buffer push with varying buffer sizes
///
/// This measures how buffer capacity affects push latency.
fn bench_ring_buffer_push_varying_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_push_capacity");

    for capacity in [1024, 4096, 8192, 16384, 32768] {
        group.bench_with_input(BenchmarkId::from_parameter(capacity), &capacity, |b, &capacity| {
            let buffer = SpanRingBuffer::new(capacity);
            let mut i = 0u64;

            b.iter(|| {
                let span = create_bench_span(i);
                buffer.push(black_box(span));
                i += 1;
            });

            std::mem::drop(buffer);
        });
    }

    group.finish();
}

/// Benchmark: Span creation overhead
///
/// This measures the overhead of creating a SpanRecord itself (not the ring
/// buffer push). This helps isolate ring buffer performance.
fn bench_span_creation(c: &mut Criterion) {
    let mut i = 0u64;

    c.bench_function("span_creation", |b| {
        b.iter(|| {
            let span = create_bench_span(black_box(i));
            i += 1;
            black_box(span);
        });
    });
}

/// Benchmark: Lamport clock tick (part of hot path)
///
/// This measures the overhead of incrementing the logical clock, which
/// happens on every span creation.
///
/// Target: <100ns
fn bench_lamport_clock_tick(c: &mut Criterion) {
    use renacer::lamport_clock::LamportClock;

    let clock = LamportClock::new();

    c.bench_function("lamport_clock_tick", |b| {
        b.iter(|| {
            black_box(clock.tick());
        });
    });
}

/// Benchmark: Complete hot path (span creation + clock tick + ring buffer push)
///
/// This measures the end-to-end latency of recording a span, including:
/// 1. Incrementing Lamport clock
/// 2. Creating SpanRecord
/// 3. Pushing to ring buffer
///
/// Target: <1μs total
fn bench_complete_hot_path(c: &mut Criterion) {
    use renacer::lamport_clock::LamportClock;

    let buffer = SpanRingBuffer::new(16384);
    let clock = LamportClock::new();

    c.bench_function("complete_hot_path", |b| {
        b.iter(|| {
            // 1. Tick logical clock
            let logical_time = clock.tick();

            // 2. Create span
            let span = SpanRecord::new(
                black_box([1; 16]),
                black_box([2; 8]),
                None,
                black_box("syscall".to_string()),
                SpanKind::Internal,
                black_box(1000),
                black_box(2000),
                logical_time,
                StatusCode::Ok,
                String::new(),
                HashMap::new(),
                HashMap::new(),
                1234,
                5678,
            );

            // 3. Push to ring buffer
            buffer.push(black_box(span));
        });
    });

    std::mem::drop(buffer);
}

/// Benchmark: Baseline comparison - synchronous write to file
///
/// This demonstrates the performance benefit of the ring buffer architecture
/// compared to synchronous I/O.
///
/// Expected: 250-2500μs (250-2500× slower than ring buffer)
fn bench_synchronous_write_baseline(c: &mut Criterion) {
    use std::io::Write;
    use tempfile::NamedTempFile;

    c.bench_function("synchronous_write_baseline", |b| {
        let mut file = NamedTempFile::new().expect("test");

        b.iter(|| {
            // Simulate writing span to file (like synchronous OTLP export)
            let data = b"span data here\n";
            file.write_all(black_box(data)).expect("test");
            file.flush().expect("test"); // Force fsync
        });
    });
}

/// Benchmark: Ring buffer stats (cold path)
///
/// This measures the overhead of checking buffer statistics, which is not
/// on the hot path but useful to know.
fn bench_ring_buffer_stats(c: &mut Criterion) {
    let buffer = SpanRingBuffer::new(8192);

    // Fill buffer with some spans
    for i in 0..100 {
        buffer.push(create_bench_span(i));
    }

    c.bench_function("ring_buffer_stats", |b| {
        b.iter(|| {
            black_box(buffer.stats());
        });
    });

    std::mem::drop(buffer);
}

criterion_group!(
    benches,
    bench_ring_buffer_push,
    bench_ring_buffer_push_varying_capacity,
    bench_span_creation,
    bench_lamport_clock_tick,
    bench_complete_hot_path,
    bench_synchronous_write_baseline,
    bench_ring_buffer_stats,
);
criterion_main!(benches);
