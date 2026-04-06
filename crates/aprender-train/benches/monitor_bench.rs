//! Performance benchmarks for the monitor module.
//!
//! Validates that metrics collection overhead stays <1% of training time.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use entrenar::monitor::{
    DriftDetector, HanseiAnalyzer, InMemoryStore, Metric, MetricsCollector, MetricsStore,
    SlidingWindowBaseline,
};

/// Benchmark MetricsCollector::record throughput
fn bench_metrics_record(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetricsCollector");

    for size in &[100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("record", size), size, |b, &size| {
            b.iter(|| {
                let mut collector = MetricsCollector::new();
                for i in 0..size {
                    collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
                }
                black_box(collector)
            });
        });
    }
    group.finish();
}

/// Benchmark summary calculation (Welford's algorithm)
fn bench_metrics_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetricsSummary");

    for size in &[1_000, 10_000, 100_000] {
        // Pre-populate collector
        let mut collector = MetricsCollector::new();
        for i in 0..*size {
            collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
            collector.record(Metric::Accuracy, 0.5 + 0.0001 * f64::from(i));
        }

        group.bench_with_input(BenchmarkId::new("summary", size), size, |b, _| {
            b.iter(|| black_box(collector.summary()));
        });
    }
    group.finish();
}

/// Benchmark InMemoryStore operations
fn bench_in_memory_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryStore");

    group.bench_function("write_batch_1000", |b| {
        b.iter(|| {
            let mut store = InMemoryStore::new();
            let mut collector = MetricsCollector::new();
            for i in 0..1000 {
                collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
            }
            store.write_batch(&collector.to_records()).expect("write_batch must succeed");
            black_box(store)
        });
    });

    // Pre-populate store for query benchmark
    let mut store = InMemoryStore::new();
    let mut collector = MetricsCollector::new();
    for i in 0..10_000 {
        collector.record(Metric::Loss, 1.0 / (f64::from(i) + 1.0));
        collector.record(Metric::Accuracy, 0.5 + 0.0001 * f64::from(i));
    }
    store.write_batch(&collector.to_records()).expect("write_batch must succeed");

    group.bench_function("query_all_10k", |b| {
        b.iter(|| black_box(store.query_all(&Metric::Loss)));
    });
    group.finish();
}

/// Benchmark SlidingWindowBaseline anomaly detection
fn bench_drift_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("DriftDetection");

    group.bench_function("baseline_1000_samples", |b| {
        b.iter(|| {
            let mut baseline = SlidingWindowBaseline::new(1000);
            for i in 0..1000 {
                baseline.update(0.5 + 0.001 * f64::from(i % 10));
            }
            // Check for anomaly
            black_box(baseline.detect_anomaly(10.0, 3.0))
        });
    });

    // Pre-trained baseline for anomaly check
    let mut baseline = SlidingWindowBaseline::new(1000);
    for i in 0..1000 {
        baseline.update(0.5 + 0.001 * f64::from(i % 10));
    }

    group.bench_function("anomaly_check", |b| {
        b.iter(|| black_box(baseline.detect_anomaly(10.0, 3.0)));
    });

    group.bench_function("drift_detector_100_metrics", |b| {
        b.iter(|| {
            let mut detector = DriftDetector::new(100);
            for i in 0..100 {
                detector.check(0.5 + 0.001 * f64::from(i % 10));
            }
            black_box(detector)
        });
    });
    group.finish();
}

/// Benchmark HanseiAnalyzer report generation
fn bench_hansei_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("HanseiReport");

    for size in &[100, 1_000, 10_000] {
        let mut collector = MetricsCollector::new();
        for i in 0..*size {
            collector.record(Metric::Loss, 1.0 - (f64::from(i) / f64::from(*size)));
            collector.record(Metric::Accuracy, 0.5 + 0.5 * (f64::from(i) / f64::from(*size)));
            collector.record(Metric::GradientNorm, 1.0 + 0.01 * f64::from(i % 100));
        }

        group.bench_with_input(BenchmarkId::new("analyze", size), size, |b, _| {
            let analyzer = HanseiAnalyzer::new();
            b.iter(|| black_box(analyzer.analyze("bench-run", &collector, 100.0)));
        });
    }
    group.finish();
}

/// Benchmark batch recording (simulates epoch boundary)
fn bench_batch_record(c: &mut Criterion) {
    let mut group = c.benchmark_group("BatchRecord");

    let metrics: Vec<(Metric, f64)> = vec![
        (Metric::Loss, 0.5),
        (Metric::Accuracy, 0.85),
        (Metric::LearningRate, 0.001),
        (Metric::GradientNorm, 1.5),
        (Metric::Epoch, 1.0),
        (Metric::Batch, 100.0),
    ];

    group.bench_function("record_batch_6_metrics", |b| {
        b.iter(|| {
            let mut collector = MetricsCollector::new();
            collector.record_batch(&metrics);
            black_box(collector)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_metrics_record,
    bench_metrics_summary,
    bench_in_memory_store,
    bench_drift_detection,
    bench_hansei_report,
    bench_batch_record
);
criterion_main!(benches);
