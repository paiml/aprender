/// Sprint 36: OTLP Export Performance Benchmarks
///
/// Measures the performance of OTLP span export operations.
/// Focuses on batch processing, serialization, and network overhead.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Simulate span data for export
#[derive(Clone)]
#[allow(dead_code)] // Benchmark mock data
struct ExportSpan {
    trace_id: String,
    span_id: String,
    name: String,
    attributes: Vec<(String, String)>,
    timestamp_nanos: u64,
    duration_nanos: u64,
}

impl ExportSpan {
    fn new(id: usize) -> Self {
        ExportSpan {
            trace_id: format!("trace_{:032x}", id),
            span_id: format!("span_{:016x}", id),
            name: format!("syscall:open_{}", id),
            attributes: vec![
                ("syscall.name".to_string(), "open".to_string()),
                ("syscall.result".to_string(), "3".to_string()),
                ("code.filepath".to_string(), "/src/main.rs".to_string()),
                ("code.lineno".to_string(), "42".to_string()),
            ],
            timestamp_nanos: 1234567890000000 + (id as u64 * 1000),
            duration_nanos: 1000,
        }
    }
}

/// Simulate batch exporter
struct BatchExporter {
    buffer: Vec<ExportSpan>,
    max_batch_size: usize,
}

impl BatchExporter {
    fn new(max_batch_size: usize) -> Self {
        BatchExporter { buffer: Vec::with_capacity(max_batch_size), max_batch_size }
    }

    fn add(&mut self, span: ExportSpan) -> bool {
        self.buffer.push(span);
        self.buffer.len() >= self.max_batch_size
    }

    fn flush(&mut self) -> Vec<ExportSpan> {
        std::mem::take(&mut self.buffer)
    }
}

/// Benchmark: Individual span export (baseline - inefficient)
fn bench_individual_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("individual_export");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(100));

    group.bench_function("export_100_spans", |b| {
        b.iter(|| {
            for i in 0..100 {
                let span = ExportSpan::new(i);
                // Simulate export (just drop for benchmark)
                black_box(span);
            }
        });
    });

    group.finish();
}

/// Benchmark: Batch export with different batch sizes
fn bench_batch_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_export");
    group.measurement_time(Duration::from_secs(5));

    for batch_size in [16, 32, 64, 128, 256, 512, 1024, 2048].iter() {
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut exporter = BatchExporter::new(batch_size);

                    for i in 0..1000 {
                        let span = ExportSpan::new(i);
                        if exporter.add(span) {
                            let batch = exporter.flush();
                            black_box(batch);
                        }
                    }

                    // Flush remaining
                    let remaining = exporter.flush();
                    black_box(remaining);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Span creation cost
fn bench_span_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_creation");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1000));

    group.bench_function("create_1000_spans", |b| {
        b.iter(|| {
            let mut spans = Vec::new();
            for i in 0..1000 {
                spans.push(ExportSpan::new(i));
            }
            black_box(spans);
        });
    });

    group.finish();
}

/// Benchmark: Span serialization (simulate protobuf encoding)
fn bench_span_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_serialization");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(100));

    group.bench_function("serialize_100_spans", |b| {
        let spans: Vec<_> = (0..100).map(ExportSpan::new).collect();
        b.iter(|| {
            for span in &spans {
                // Simulate serialization by converting to JSON (similar cost)
                let serialized = format!(
                    "{{\"trace_id\":\"{}\",\"span_id\":\"{}\",\"name\":\"{}\"}}",
                    span.trace_id, span.span_id, span.name
                );
                black_box(serialized);
            }
        });
    });

    group.finish();
}

/// Benchmark: Buffer management overhead
fn bench_buffer_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_management");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("vec_push_pop", |b| {
        b.iter(|| {
            let mut buffer = Vec::with_capacity(512);
            for i in 0..512 {
                buffer.push(ExportSpan::new(i));
            }
            black_box(&buffer);
            buffer.clear();
        });
    });

    group.bench_function("vec_mem_take", |b| {
        let mut buffer = Vec::with_capacity(512);
        b.iter(|| {
            for i in 0..512 {
                buffer.push(ExportSpan::new(i));
            }
            let taken = std::mem::take(&mut buffer);
            black_box(taken);
            buffer = Vec::with_capacity(512);
        });
    });

    group.finish();
}

/// Benchmark: Realistic workload simulation
fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workload");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);
    group.throughput(Throughput::Elements(10000));

    group.bench_function("10k_spans_batch_512", |b| {
        b.iter(|| {
            let mut exporter = BatchExporter::new(512);
            let mut export_count = 0;

            for i in 0..10000 {
                let span = ExportSpan::new(i);
                if exporter.add(span) {
                    let batch = exporter.flush();
                    export_count += batch.len();
                    black_box(batch);
                }
            }

            // Flush remaining
            let remaining = exporter.flush();
            export_count += remaining.len();
            black_box((remaining, export_count));
        });
    });

    group.finish();
}

/// Benchmark: Queue saturation behavior
fn bench_queue_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_saturation");
    group.measurement_time(Duration::from_secs(5));

    // Simulate queue filling up faster than it can drain
    group.bench_function("high_pressure", |b| {
        b.iter(|| {
            let mut exporter = BatchExporter::new(256);
            let mut export_count = 0;

            // Rapid span generation
            for i in 0..5000 {
                let span = ExportSpan::new(i);
                if exporter.add(span) {
                    let batch = exporter.flush();
                    export_count += batch.len();
                    black_box(batch);
                }
            }

            black_box(export_count);
        });
    });

    group.finish();
}

/// Benchmark: Batch vs. individual comparison
fn bench_batch_vs_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_individual");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1000));

    group.bench_function("individual_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let span = ExportSpan::new(i);
                black_box(span);
            }
        });
    });

    group.bench_function("batch_512_1000", |b| {
        b.iter(|| {
            let mut exporter = BatchExporter::new(512);
            for i in 0..1000 {
                let span = ExportSpan::new(i);
                if exporter.add(span) {
                    let batch = exporter.flush();
                    black_box(batch);
                }
            }
            let remaining = exporter.flush();
            black_box(remaining);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_individual_export,
    bench_batch_export,
    bench_span_creation,
    bench_span_serialization,
    bench_buffer_management,
    bench_realistic_workload,
    bench_queue_saturation,
    bench_batch_vs_individual
);

criterion_main!(benches);
