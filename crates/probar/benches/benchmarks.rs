//! Criterion benchmarks for probar testing framework.
//!
//! Benchmarks assertion evaluation and soft assertion collection
//! which are hot paths during test execution.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jugar_probar::{AssertionResult, SoftAssertions};

fn bench_assertion_result_pass(c: &mut Criterion) {
    c.bench_function("assertion_result_pass", |b| {
        b.iter(|| {
            black_box(AssertionResult::pass());
        });
    });
}

fn bench_assertion_result_fail(c: &mut Criterion) {
    c.bench_function("assertion_result_fail", |b| {
        b.iter(|| {
            black_box(AssertionResult::fail(black_box("expected 42, got 0")));
        });
    });
}

fn bench_soft_assertions_collect(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_assertions");
    for count in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("collect_failures", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut soft = SoftAssertions::new();
                    for i in 0..count {
                        soft.assert_eq(&i, &(i + 1), "values should match");
                    }
                    black_box(soft.failure_count());
                });
            },
        );
    }
    group.finish();
}

fn bench_soft_assertions_all_pass(c: &mut Criterion) {
    c.bench_function("soft_assertions_100_pass", |b| {
        b.iter(|| {
            let mut soft = SoftAssertions::new();
            for i in 0..100 {
                soft.assert_eq(&i, &i, "values match");
            }
            black_box(soft.failure_count());
        });
    });
}

fn bench_soft_assertions_verify(c: &mut Criterion) {
    c.bench_function("soft_assertions_verify_50_failures", |b| {
        b.iter(|| {
            let mut soft = SoftAssertions::new();
            for i in 0..50 {
                soft.assert_eq(&i, &(i + 1), "mismatch");
            }
            let _ = black_box(soft.verify());
        });
    });
}

criterion_group!(
    benches,
    bench_assertion_result_pass,
    bench_assertion_result_fail,
    bench_soft_assertions_collect,
    bench_soft_assertions_all_pass,
    bench_soft_assertions_verify,
);
criterion_main!(benches);
