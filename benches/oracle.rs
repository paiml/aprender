//! Oracle benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use verificar::oracle::{ExecutionResult, IoOracle, Oracle};

fn benchmark_comparison(c: &mut Criterion) {
    let oracle = IoOracle::new();

    let source = ExecutionResult {
        stdout: "hello world\n".repeat(100),
        stderr: String::new(),
        exit_code: 0,
        duration_ms: 10,
    };

    let target = ExecutionResult {
        stdout: "hello world\n".repeat(100),
        stderr: String::new(),
        exit_code: 0,
        duration_ms: 5,
    };

    c.bench_function("oracle_comparison_pass", |b| {
        b.iter(|| oracle.compare(&source, &target));
    });
}

fn benchmark_comparison_mismatch(c: &mut Criterion) {
    let oracle = IoOracle::new();

    let source = ExecutionResult {
        stdout: "hello world\n".repeat(100),
        stderr: String::new(),
        exit_code: 0,
        duration_ms: 10,
    };

    let target = ExecutionResult {
        stdout: "goodbye world\n".repeat(100),
        stderr: String::new(),
        exit_code: 0,
        duration_ms: 5,
    };

    c.bench_function("oracle_comparison_mismatch", |b| {
        b.iter(|| oracle.compare(&source, &target));
    });
}

criterion_group!(benches, benchmark_comparison, benchmark_comparison_mismatch);
criterion_main!(benches);
