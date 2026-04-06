/// Sprint 36: Syscall Tracing Overhead Benchmarks
///
/// Measures the performance overhead of Renacer tracing compared to native execution.
/// These benchmarks help detect performance regressions and validate optimizations.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::process::Command;
use std::time::Duration;

const FIXTURE_PATH: &str = "./benches/fixtures/syscall_heavy";
const RENACER_BIN: &str = "./target/release/renacer";

/// Baseline: Run fixture without any tracing
fn bench_native_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("native");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("syscall_heavy_native", |b| {
        b.iter(|| {
            let output = Command::new(FIXTURE_PATH).output().expect("Failed to run fixture");
            assert!(output.status.success());
            black_box(output);
        });
    });

    group.finish();
}

/// Basic tracing: Renacer without OTLP export
fn bench_basic_tracing(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_tracing");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("syscall_heavy_basic", |b| {
        b.iter(|| {
            let output = Command::new(RENACER_BIN)
                .arg("--")
                .arg(FIXTURE_PATH)
                .output()
                .expect("Failed to run renacer");
            assert!(output.status.success());
            black_box(output);
        });
    });

    group.finish();
}

/// With statistics: Renacer with -c flag
fn bench_with_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("with_statistics");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("syscall_heavy_stats", |b| {
        b.iter(|| {
            let output = Command::new(RENACER_BIN)
                .arg("-c")
                .arg("--")
                .arg(FIXTURE_PATH)
                .output()
                .expect("Failed to run renacer");
            assert!(output.status.success());
            black_box(output);
        });
    });

    group.finish();
}

/// With timing: Renacer with -T flag
fn bench_with_timing(c: &mut Criterion) {
    let mut group = c.benchmark_group("with_timing");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("syscall_heavy_timing", |b| {
        b.iter(|| {
            let output = Command::new(RENACER_BIN)
                .arg("-T")
                .arg("--")
                .arg(FIXTURE_PATH)
                .output()
                .expect("Failed to run renacer");
            assert!(output.status.success());
            black_box(output);
        });
    });

    group.finish();
}

/// Full stack: All features enabled (except OTLP which requires backend)
fn bench_full_stack_no_otlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_stack");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30); // Fewer samples due to longer runtime

    group.bench_function("syscall_heavy_full", |b| {
        b.iter(|| {
            let output = Command::new(RENACER_BIN)
                .arg("-T")
                .arg("-c")
                .arg("--stats-extended")
                .arg("--source")
                .arg("--")
                .arg(FIXTURE_PATH)
                .output()
                .expect("Failed to run renacer");
            assert!(output.status.success());
            black_box(output);
        });
    });

    group.finish();
}

/// Overhead comparison: Compare all configurations side-by-side
fn bench_overhead_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("overhead_comparison");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    let configs = vec![
        ("native", vec![]),
        ("basic", vec!["--"]),
        ("timing", vec!["-T", "--"]),
        ("stats", vec!["-c", "--"]),
        ("full", vec!["-T", "-c", "--stats-extended", "--"]),
    ];

    for (name, args) in configs {
        group.bench_with_input(BenchmarkId::from_parameter(name), &args, |b, args| {
            b.iter(|| {
                let mut cmd = if name == "native" {
                    Command::new(FIXTURE_PATH)
                } else {
                    let mut c = Command::new(RENACER_BIN);
                    for arg in args {
                        c.arg(arg);
                    }
                    c.arg(FIXTURE_PATH);
                    c
                };

                let output = cmd.output().expect("Failed to run command");
                assert!(output.status.success());
                black_box(output);
            });
        });
    }

    group.finish();
}

/// Throughput benchmark: Measure syscalls/sec
fn bench_syscall_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Estimate: syscall_heavy fixture generates ~300 syscalls (100 files * 3 ops each)
    group.throughput(Throughput::Elements(300));

    group.bench_function("syscalls_per_second", |b| {
        b.iter(|| {
            let output = Command::new(RENACER_BIN)
                .arg("--")
                .arg(FIXTURE_PATH)
                .output()
                .expect("Failed to run renacer");
            assert!(output.status.success());
            black_box(output);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_native_baseline,
    bench_basic_tracing,
    bench_with_statistics,
    bench_with_timing,
    bench_full_stack_no_otlp,
    bench_overhead_comparison,
    bench_syscall_throughput
);

criterion_main!(benches);
