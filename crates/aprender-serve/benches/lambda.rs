//! Lambda Handler Benchmarks
//!
//! Per `docs/specifications/serve-deploy-apr.md` Section 8:
//! - Cold start target: <50ms
//! - Warm inference target: <10ms (p50)
//!
//! These benchmarks validate performance against spec targets.

#![cfg(feature = "lambda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use realizar::lambda::{BatchLambdaRequest, LambdaHandler, LambdaMetrics, LambdaRequest};

/// Valid .apr model bytes for benchmarking
static MODEL_BYTES: &[u8] = b"APR\0\x01\x00\x00\x00benchmark_model_data_padding_for_realistic_size";

// =============================================================================
// Benchmark: Cold Start
// =============================================================================

/// Benchmark cold start latency (first invocation)
///
/// Per spec §8.1: Target <50ms cold start
fn bench_cold_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_cold_start");

    // Measure handler creation + first inference
    group.bench_function("handler_creation", |b| {
        b.iter(|| {
            let handler = LambdaHandler::from_bytes(black_box(MODEL_BYTES)).expect("test");
            black_box(handler)
        });
    });

    group.bench_function("first_inference", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;

            for _ in 0..iters {
                // Create fresh handler for each iteration (cold start)
                let handler = LambdaHandler::from_bytes(MODEL_BYTES).expect("test");
                let request = LambdaRequest {
                    features: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                    model_id: None,
                };

                let start = std::time::Instant::now();
                let _ = handler.handle(black_box(&request));
                total += start.elapsed();
            }

            total
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark: Warm Inference
// =============================================================================

/// Benchmark warm inference latency (subsequent invocations)
///
/// Per spec §8.1: Target <10ms p50 latency
fn bench_warm_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_warm_inference");

    // Pre-create handler (warm)
    let handler = LambdaHandler::from_bytes(MODEL_BYTES).expect("test");

    // Warm up with first invocation
    let warmup_request = LambdaRequest {
        features: vec![1.0],
        model_id: None,
    };
    let _ = handler.handle(&warmup_request);

    // Benchmark different feature sizes
    for size in [1, 10, 100, 1000].iter() {
        let features: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let request = LambdaRequest {
            features,
            model_id: None,
        };

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("features", size), &request, |b, req| {
            b.iter(|| handler.handle(black_box(req)))
        });
    }

    group.finish();
}

// =============================================================================
// Benchmark: Batch Inference
// =============================================================================

/// Benchmark batch inference throughput
///
/// Per spec §5.3: Batch inference for throughput optimization
fn bench_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_batch_inference");

    let handler = LambdaHandler::from_bytes(MODEL_BYTES).expect("test");

    // Warm up
    let _ = handler.handle(&LambdaRequest {
        features: vec![1.0],
        model_id: None,
    });

    // Benchmark different batch sizes
    for batch_size in [1, 10, 50, 100].iter() {
        let instances: Vec<LambdaRequest> = (0..*batch_size)
            .map(|i| LambdaRequest {
                features: vec![i as f32; 10],
                model_id: None,
            })
            .collect();

        let batch = BatchLambdaRequest {
            instances,
            max_parallelism: None,
        };

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch,
            |b, batch| b.iter(|| handler.handle_batch(black_box(batch))),
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: Metrics Overhead
// =============================================================================

/// Benchmark metrics collection overhead
fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_metrics");

    group.bench_function("record_success", |b| {
        let mut metrics = LambdaMetrics::new();
        b.iter(|| {
            metrics.record_success(black_box(1.5), black_box(false));
        });
    });

    group.bench_function("record_batch", |b| {
        let mut metrics = LambdaMetrics::new();
        b.iter(|| {
            metrics.record_batch(black_box(10), black_box(2), black_box(5.0));
        });
    });

    group.bench_function("to_prometheus", |b| {
        let mut metrics = LambdaMetrics::new();
        for _ in 0..100 {
            metrics.record_success(1.0, false);
        }

        b.iter(|| metrics.to_prometheus());
    });

    group.bench_function("avg_latency", |b| {
        let mut metrics = LambdaMetrics::new();
        for i in 0..1000 {
            metrics.record_success(i as f64 * 0.1, false);
        }

        b.iter(|| metrics.avg_latency_ms());
    });

    group.finish();
}

// =============================================================================
// Benchmark: End-to-End Pipeline
// =============================================================================

/// Benchmark full inference pipeline (realistic workload)
fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambda_pipeline");

    let handler = LambdaHandler::from_bytes(MODEL_BYTES).expect("test");
    let mut metrics = LambdaMetrics::new();

    // Warm up
    let _ = handler.handle(&LambdaRequest {
        features: vec![1.0],
        model_id: None,
    });

    // Realistic workload: single inference + metrics
    group.bench_function("single_with_metrics", |b| {
        let request = LambdaRequest {
            features: vec![1.0; 50],
            model_id: None,
        };

        b.iter(|| {
            let response = handler.handle(black_box(&request)).expect("test");
            metrics.record_success(response.latency_ms, response.cold_start);
            black_box(response)
        });
    });

    // Realistic workload: batch inference + metrics
    group.bench_function("batch_with_metrics", |b| {
        let batch = BatchLambdaRequest {
            instances: (0..20)
                .map(|i| LambdaRequest {
                    features: vec![i as f32; 50],
                    model_id: None,
                })
                .collect(),
            max_parallelism: None,
        };

        b.iter(|| {
            let response = handler.handle_batch(black_box(&batch)).expect("test");
            metrics.record_batch(
                response.success_count,
                response.error_count,
                response.total_latency_ms,
            );
            black_box(response)
        });
    });

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_cold_start,
    bench_warm_inference,
    bench_batch_inference,
    bench_metrics,
    bench_pipeline,
);

criterion_main!(benches);
