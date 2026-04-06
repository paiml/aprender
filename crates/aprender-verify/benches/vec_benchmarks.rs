//! Performance benchmarks for `TruenoVec`
//!
//! These benchmarks are part of Tier 3 (ON-MERGE/NIGHTLY) quality gates.
//! They validate that `TruenoVec` maintains competitive performance characteristics
//! compared to `std::vec::Vec`.
//!
//! Run with: `cargo bench`

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

use certeza::TruenoVec;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ============================================================================
// Push Benchmarks
// ============================================================================

fn bench_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("push");

    for size in [10, 100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("TruenoVec", size), &size, |b, &size| {
            b.iter(|| {
                let mut vec = TruenoVec::new();
                for i in 0..size {
                    vec.push(black_box(i));
                }
                vec
            });
        });

        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &size| {
            b.iter(|| {
                let mut vec = Vec::new();
                for i in 0..size {
                    vec.push(black_box(i));
                }
                vec
            });
        });
    }

    group.finish();
}

fn bench_push_with_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_with_capacity");

    for size in [10, 100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("TruenoVec", size), &size, |b, &size| {
            b.iter(|| {
                let mut vec = TruenoVec::with_capacity(size);
                for i in 0..size {
                    vec.push(black_box(i));
                }
                vec
            });
        });

        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &size| {
            b.iter(|| {
                let mut vec = Vec::with_capacity(size);
                for i in 0..size {
                    vec.push(black_box(i));
                }
                vec
            });
        });
    }

    group.finish();
}

// ============================================================================
// Pop Benchmarks
// ============================================================================

fn bench_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop");

    for size in [10, 100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("TruenoVec", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = TruenoVec::new();
                    for i in 0..size {
                        vec.push(i);
                    }
                    vec
                },
                |mut vec| {
                    while vec.pop().is_some() {}
                    vec
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = Vec::new();
                    for i in 0..size {
                        vec.push(i);
                    }
                    vec
                },
                |mut vec| {
                    while vec.pop().is_some() {}
                    vec
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Get/Index Benchmarks
// ============================================================================

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");

    let size = 10000;
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("TruenoVec", |b| {
        let mut vec = TruenoVec::new();
        for i in 0..size {
            vec.push(i);
        }

        b.iter(|| {
            let mut sum = 0;
            for i in 0..size {
                if let Some(&val) = vec.get(i) {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("std::Vec", |b| {
        let mut vec = Vec::new();
        for i in 0..size {
            vec.push(i);
        }

        b.iter(|| {
            let mut sum = 0;
            for i in 0..size {
                if let Some(&val) = vec.get(i) {
                    sum += val;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

// ============================================================================
// Iterator Benchmarks
// ============================================================================

fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("TruenoVec", size), &size, |b, &size| {
            let mut vec = TruenoVec::new();
            for i in 0..size {
                vec.push(i);
            }

            b.iter(|| {
                let sum: i32 = vec.iter().sum();
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &size| {
            let mut vec = Vec::new();
            for i in 0..size {
                vec.push(i);
            }

            b.iter(|| {
                let sum: i32 = vec.iter().sum();
                black_box(sum)
            });
        });
    }

    group.finish();
}

fn bench_into_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("into_iter");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("TruenoVec", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = TruenoVec::new();
                    for i in 0..size {
                        vec.push(i);
                    }
                    vec
                },
                |vec| {
                    let sum: i32 = vec.into_iter().sum();
                    black_box(sum)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = Vec::new();
                    for i in 0..size {
                        vec.push(i);
                    }
                    vec
                },
                |vec| {
                    let sum: i32 = vec.into_iter().sum();
                    black_box(sum)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Insert/Remove Benchmarks
// ============================================================================

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("TruenoVec_middle", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = TruenoVec::new();
                    for i in 0..size {
                        vec.push(i);
                    }
                    vec
                },
                |mut vec| {
                    for _ in 0..size {
                        vec.insert(size / 2, black_box(42));
                    }
                    vec
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("std::Vec_middle", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = Vec::new();
                    for i in 0..size {
                        vec.push(i);
                    }
                    vec
                },
                |mut vec| {
                    for _ in 0..size {
                        vec.insert(size / 2, black_box(42));
                    }
                    vec
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");

    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("TruenoVec_middle", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = TruenoVec::new();
                    for i in 0..size * 2 {
                        vec.push(i);
                    }
                    vec
                },
                |mut vec| {
                    for _ in 0..size {
                        vec.remove(vec.len() / 2);
                    }
                    vec
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("std::Vec_middle", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut vec = Vec::new();
                    for i in 0..size * 2 {
                        vec.push(i);
                    }
                    vec
                },
                |mut vec| {
                    for _ in 0..size {
                        vec.remove(vec.len() / 2);
                    }
                    vec
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_push,
    bench_push_with_capacity,
    bench_pop,
    bench_get,
    bench_iter,
    bench_into_iter,
    bench_insert,
    bench_remove
);
criterion_main!(benches);
