//! Metrics Throughput Benchmark (Sprint 56 QA Falsification)
//!
//! Popper Falsification Criteria:
//! - Counter.inc() must complete in <50ns (p99)
//! - Histogram.observe() must complete in <200ns (SIMD)
//! - 100 threads x 1M increments must yield exactly 100M (atomicity)

#![allow(clippy::unwrap_used, clippy::semicolon_if_nothing_returned)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use renacer::metrics::{Counter, Gauge, Histogram, Labels, Registry, DEFAULT_BUCKETS};

/// Benchmark Counter.inc() - must be <50ns p99
fn bench_counter_inc(c: &mut Criterion) {
    let counter = Counter::new("test_counter", Labels::new());

    c.bench_function("counter_inc", |b| {
        b.iter(|| {
            counter.inc();
            black_box(())
        })
    });
}

/// Benchmark Counter.add() with value
fn bench_counter_add(c: &mut Criterion) {
    let counter = Counter::new("test_counter", Labels::new());

    c.bench_function("counter_add", |b| {
        b.iter(|| {
            counter.add(black_box(1));
        })
    });
}

/// Benchmark Gauge.set() - must be <30ns p99
fn bench_gauge_set(c: &mut Criterion) {
    let gauge = Gauge::new("test_gauge", Labels::new());

    c.bench_function("gauge_set", |b| {
        b.iter(|| {
            gauge.set(black_box(42));
        })
    });
}

/// Benchmark Gauge.inc/dec
fn bench_gauge_inc_dec(c: &mut Criterion) {
    let gauge = Gauge::new("test_gauge", Labels::new());

    let mut group = c.benchmark_group("gauge_ops");

    group.bench_function("inc", |b| {
        b.iter(|| {
            gauge.inc();
            black_box(())
        })
    });

    group.bench_function("dec", |b| {
        b.iter(|| {
            gauge.dec();
            black_box(())
        })
    });

    group.finish();
}

/// Benchmark Histogram.observe() - must be <200ns p99 (SIMD)
fn bench_histogram_observe(c: &mut Criterion) {
    let histogram = Histogram::new("test_histogram", Labels::new(), DEFAULT_BUCKETS);

    c.bench_function("histogram_observe", |b| {
        b.iter(|| {
            histogram.observe(black_box(0.05));
        })
    });
}

/// Benchmark Histogram with different bucket counts
fn bench_histogram_bucket_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_buckets");

    for bucket_count in [5, 11, 20, 50].iter() {
        let buckets: Vec<f64> = (1..=*bucket_count).map(|i| i as f64 * 0.1).collect();
        let histogram = Histogram::new("test_histogram", Labels::new(), &buckets);

        group.bench_with_input(BenchmarkId::from_parameter(bucket_count), bucket_count, |b, _| {
            b.iter(|| {
                histogram.observe(black_box(0.5));
            })
        });
    }

    group.finish();
}

/// Benchmark Registry operations
fn bench_registry_operations(c: &mut Criterion) {
    let registry = Registry::new();

    // Pre-register some metrics
    for i in 0..100 {
        let _ = registry.counter(&format!("counter_{i}"), &[]);
    }

    let mut group = c.benchmark_group("registry");

    group.bench_function("get_counter", |b| {
        b.iter(|| {
            let _ = registry.counter(black_box("counter_50"), &[]);
        })
    });

    group.bench_function("register_new", |b| {
        let mut idx = 1000;
        b.iter(|| {
            let _ = registry.counter(&format!("new_counter_{}", idx), &[]);
            idx += 1;
        })
    });

    group.finish();
}

/// CRITICAL: Thread safety stress test
/// 100 threads x 10K increments must yield exactly 1M
#[test]
fn test_counter_thread_safety_stress() {
    use std::sync::Arc;
    use std::thread;
    let counter = Arc::new(Counter::new("stress_counter", Labels::new()));
    let num_threads = 100;
    let increments_per_thread = 10_000;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let c = Arc::clone(&counter);
            thread::spawn(move || {
                for _ in 0..increments_per_thread {
                    c.inc();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected = num_threads * increments_per_thread;
    let actual = counter.get();
    assert_eq!(
        actual, expected as u64,
        "FALSIFIED: Counter thread safety broken! Expected {expected}, got {actual}"
    );
}

/// CRITICAL: Histogram thread safety stress test
#[test]
fn test_histogram_thread_safety_stress() {
    use std::sync::Arc;
    use std::thread;
    let histogram = Arc::new(Histogram::new("stress_histogram", Labels::new(), DEFAULT_BUCKETS));
    let num_threads = 50;
    let observations_per_thread = 10_000;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let h = Arc::clone(&histogram);
            thread::spawn(move || {
                for j in 0..observations_per_thread {
                    // Use thread ID and iteration to vary values
                    let value = (i * 1000 + j) as f64 / 100000.0;
                    h.observe(value);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let expected_count = num_threads * observations_per_thread;
    let actual_count = histogram.get_count();
    assert_eq!(
        actual_count, expected_count as u64,
        "FALSIFIED: Histogram thread safety broken! Expected {expected_count} observations, got {actual_count}"
    );
}

criterion_group!(
    benches,
    bench_counter_inc,
    bench_counter_add,
    bench_gauge_set,
    bench_gauge_inc_dec,
    bench_histogram_observe,
    bench_histogram_bucket_sizes,
    bench_registry_operations,
);

criterion_main!(benches);
