/// Sprint 36: Memory Pool Benchmarks
///
/// Measures the performance of object pooling for span data.
/// Compares allocation patterns: pool vs. heap allocation.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use std::borrow::Cow;

/// Simulate span data structure (Sprint 36: with zero-copy Cow)
#[derive(Clone)]
#[allow(dead_code)] // Benchmark mock data
struct SpanData {
    name: Cow<'static, str>,
    attributes: Vec<(Cow<'static, str>, String)>,
    timestamp: u64,
    duration: u64,
}

#[allow(dead_code)] // Benchmark mock helpers
impl SpanData {
    fn new(name: &'static str) -> Self {
        SpanData {
            name: Cow::Borrowed(name),
            attributes: vec![
                (Cow::Borrowed("syscall.name"), name.to_string()),
                (Cow::Borrowed("syscall.result"), "0".to_string()),
            ],
            timestamp: 1234567890,
            duration: 1000,
        }
    }

    fn new_owned(name: String) -> Self {
        SpanData {
            name: Cow::Owned(name.clone()),
            attributes: vec![
                (Cow::Owned("syscall.name".to_string()), name),
                (Cow::Owned("syscall.result".to_string()), "0".to_string()),
            ],
            timestamp: 1234567890,
            duration: 1000,
        }
    }
}

/// Simple object pool implementation for benchmarking
struct SimplePool {
    pool: Vec<SpanData>,
}

impl SimplePool {
    fn new(capacity: usize) -> Self {
        let mut pool = Vec::with_capacity(capacity);
        for i in 0..capacity {
            pool.push(SpanData::new_owned(format!("syscall_{}", i)));
        }
        SimplePool { pool }
    }

    fn acquire(&mut self) -> Option<SpanData> {
        self.pool.pop()
    }

    fn release(&mut self, span: SpanData) {
        self.pool.push(span);
    }
}

/// Benchmark: Direct heap allocation (baseline)
fn bench_heap_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("heap_allocation");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1000));

    group.bench_function("alloc_1000_spans", |b| {
        b.iter(|| {
            let mut spans = Vec::new();
            for i in 0..1000 {
                spans.push(SpanData::new_owned(format!("syscall_{}", i)));
            }
            black_box(spans);
        });
    });

    group.finish();
}

/// Benchmark: Pool allocation
fn bench_pool_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_allocation");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1000));

    group.bench_function("acquire_1000_spans", |b| {
        let mut pool = SimplePool::new(1024);
        b.iter(|| {
            let mut spans = Vec::new();
            for _ in 0..1000 {
                if let Some(span) = pool.acquire() {
                    spans.push(span);
                }
            }
            // Return to pool
            for span in spans {
                pool.release(span);
            }
        });
    });

    group.finish();
}

/// Benchmark: Pool acquire/release cycle
fn bench_pool_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_cycle");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    group.bench_function("acquire_release", |b| {
        let mut pool = SimplePool::new(1024);
        b.iter(|| {
            let span = pool.acquire().expect("test");
            black_box(&span);
            pool.release(span);
        });
    });

    group.finish();
}

/// Benchmark: Pool with different sizes
fn bench_pool_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_sizes");
    group.measurement_time(Duration::from_secs(5));

    for size in [128, 256, 512, 1024, 2048, 4096].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut pool = SimplePool::new(size);
            b.iter(|| {
                // Acquire half the pool
                let mut spans = Vec::new();
                for _ in 0..size / 2 {
                    if let Some(span) = pool.acquire() {
                        spans.push(span);
                    }
                }
                // Return all
                for span in spans {
                    pool.release(span);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark: Allocation patterns under pressure
fn bench_allocation_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_pressure");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);

    // Simulate realistic workload: acquire many, release many
    group.bench_function("realistic_workload", |b| {
        let mut pool = SimplePool::new(1024);
        b.iter(|| {
            let mut active_spans = Vec::new();

            // Burst of 500 allocations
            for _ in 0..500 {
                if let Some(span) = pool.acquire() {
                    active_spans.push(span);
                }
            }

            // Release 250 (simulate partial completion)
            for _ in 0..250 {
                if let Some(span) = active_spans.pop() {
                    pool.release(span);
                }
            }

            // Another burst of 300
            for _ in 0..300 {
                if let Some(span) = pool.acquire() {
                    active_spans.push(span);
                }
            }

            // Release all
            for span in active_spans {
                pool.release(span);
            }
        });
    });

    group.finish();
}

/// Benchmark: Memory footprint (measure via allocation count)
fn bench_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("heap_10k_spans", |b| {
        b.iter(|| {
            let mut spans = Vec::new();
            for i in 0..10000 {
                spans.push(SpanData::new_owned(format!("syscall_{}", i)));
            }
            black_box(spans);
        });
    });

    group.bench_function("pool_10k_spans", |b| {
        let mut pool = SimplePool::new(10000);
        b.iter(|| {
            let mut spans = Vec::new();
            for _ in 0..10000 {
                if let Some(span) = pool.acquire() {
                    spans.push(span);
                }
            }
            for span in spans {
                pool.release(span);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_heap_allocation,
    bench_pool_allocation,
    bench_pool_cycle,
    bench_pool_sizes,
    bench_allocation_pressure,
    bench_memory_footprint
);

criterion_main!(benches);
