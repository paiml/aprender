//! Benchmarks for Sprint 43: Trueno-DB Query Performance
//!
//! Performance Targets (Sprint 43 Acceptance Criteria):
//! - 1M spans: <20ms p95 query latency
//! - 10M spans: <200ms p95 query latency
//! - Regression detection: <10% latency increase
//!
//! Run with: cargo bench --bench trueno_db_queries

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use renacer::trueno_db_storage::{StorageConfig, TruenoDbStorage};
use std::collections::HashMap;
use tempfile::TempDir;

fn create_test_span(trace_id: [u8; 16], span_idx: u64, start_time_nanos: u64) -> SpanRecord {
    SpanRecord::new(
        trace_id,
        [(span_idx as u8); 8],
        None,
        format!("span_{}", span_idx),
        SpanKind::Internal,
        start_time_nanos,
        start_time_nanos + 1_000_000, // 1ms duration
        span_idx,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        1234,
        5678,
    )
}

fn setup_storage(num_spans: usize, config: StorageConfig) -> (TempDir, TruenoDbStorage, [u8; 16]) {
    let tmp_dir = TempDir::new().expect("test");
    let path = tmp_dir.path().join("bench.parquet");

    let storage = TruenoDbStorage::with_config(&path, config).expect("test");

    // Insert spans in batches
    let trace_id = [0x4b; 16];
    let batch_size = 1000;

    for batch_start in (0..num_spans).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_spans);
        let spans: Vec<_> = (batch_start..batch_end)
            .map(|i| create_test_span(trace_id, i as u64, (i as u64) * 1_000_000))
            .collect();
        storage.insert_batch(&spans).expect("test");
    }

    (tmp_dir, storage, trace_id)
}

fn bench_query_by_trace_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_trace_id");

    // Benchmark different dataset sizes
    for num_spans in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K_spans", num_spans / 1000)),
            num_spans,
            |b, &num_spans| {
                let (_tmp_dir, storage, trace_id) =
                    setup_storage(num_spans, StorageConfig::default());

                b.iter(|| {
                    let results = storage.query_by_trace_id(black_box(&trace_id)).expect("test");
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_query_by_trace_id_and_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_trace_id_and_time");

    for num_spans in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K_spans", num_spans / 1000)),
            num_spans,
            |b, &num_spans| {
                let (_tmp_dir, storage, trace_id) =
                    setup_storage(num_spans, StorageConfig::default());

                // Query middle 10% of time range
                let start_time = (num_spans as u64 * 1_000_000 * 45) / 100; // 45%
                let end_time = (num_spans as u64 * 1_000_000 * 55) / 100; // 55%

                b.iter(|| {
                    let results = storage
                        .query_by_trace_id_and_time(
                            black_box(&trace_id),
                            black_box(start_time),
                            black_box(end_time),
                        )
                        .expect("test");
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_query_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_optimized");

    for num_spans in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}K_spans", num_spans / 1000)),
            num_spans,
            |b, &num_spans| {
                let (_tmp_dir, storage, trace_id) =
                    setup_storage(num_spans, StorageConfig::default());

                let start_time_min = (num_spans as u64 * 1_000_000 * 40) / 100;
                let start_time_max = (num_spans as u64 * 1_000_000 * 60) / 100;
                let process_id = Some(1234u32);

                b.iter(|| {
                    let results = storage
                        .query_optimized(
                            black_box(Some(&trace_id)),
                            black_box(Some(start_time_min)),
                            black_box(Some(start_time_max)),
                            black_box(process_id),
                        )
                        .expect("test");
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn bench_predicate_pushdown_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_pushdown_comparison");

    let num_spans = 10_000;

    // Benchmark WITH predicate pushdown (default)
    group.bench_function("with_pushdown", |b| {
        let config = StorageConfig { predicate_pushdown: true, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        b.iter(|| {
            let results = storage
                .query_optimized(black_box(Some(&trace_id)), None, None, None)
                .expect("test");
            black_box(results);
        });
    });

    // Benchmark WITHOUT predicate pushdown
    group.bench_function("without_pushdown", |b| {
        let config = StorageConfig { predicate_pushdown: false, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        b.iter(|| {
            let results = storage
                .query_optimized(black_box(Some(&trace_id)), None, None, None)
                .expect("test");
            black_box(results);
        });
    });

    group.finish();
}

fn bench_bloom_filter_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_comparison");

    let num_spans = 10_000;

    // Benchmark WITH Bloom filter (default)
    group.bench_function("with_bloom_filter", |b| {
        let config = StorageConfig { bloom_filter_trace_id: true, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        b.iter(|| {
            let results = storage.query_by_trace_id(black_box(&trace_id)).expect("test");
            black_box(results);
        });
    });

    // Benchmark WITHOUT Bloom filter
    group.bench_function("without_bloom_filter", |b| {
        let config = StorageConfig { bloom_filter_trace_id: false, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        b.iter(|| {
            let results = storage.query_by_trace_id(black_box(&trace_id)).expect("test");
            black_box(results);
        });
    });

    group.finish();
}

fn bench_row_group_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_group_size_comparison");

    let num_spans = 20_000;

    // Small row groups (1K)
    group.bench_function("row_group_1K", |b| {
        let config = StorageConfig { row_group_size: 1_000, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        b.iter(|| {
            let results = storage.query_by_trace_id(black_box(&trace_id)).expect("test");
            black_box(results);
        });
    });

    // Medium row groups (10K - default)
    group.bench_function("row_group_10K", |b| {
        let config = StorageConfig { row_group_size: 10_000, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        b.iter(|| {
            let results = storage.query_by_trace_id(black_box(&trace_id)).expect("test");
            black_box(results);
        });
    });

    // Large row groups (100K)
    group.bench_function("row_group_100K", |b| {
        let config = StorageConfig { row_group_size: 100_000, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        b.iter(|| {
            let results = storage.query_by_trace_id(black_box(&trace_id)).expect("test");
            black_box(results);
        });
    });

    group.finish();
}

fn bench_composite_index_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("composite_index_comparison");

    let num_spans = 10_000;

    // Benchmark WITH composite index (default)
    group.bench_function("with_composite_index", |b| {
        let config = StorageConfig { composite_index_trace_time: true, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        let start_time = (num_spans as u64 * 1_000_000 * 40) / 100;
        let end_time = (num_spans as u64 * 1_000_000 * 60) / 100;

        b.iter(|| {
            let results = storage
                .query_by_trace_id_and_time(
                    black_box(&trace_id),
                    black_box(start_time),
                    black_box(end_time),
                )
                .expect("test");
            black_box(results);
        });
    });

    // Benchmark WITHOUT composite index
    group.bench_function("without_composite_index", |b| {
        let config = StorageConfig { composite_index_trace_time: false, ..Default::default() };
        let (_tmp_dir, storage, trace_id) = setup_storage(num_spans, config);

        let start_time = (num_spans as u64 * 1_000_000 * 40) / 100;
        let end_time = (num_spans as u64 * 1_000_000 * 60) / 100;

        b.iter(|| {
            let results = storage
                .query_by_trace_id_and_time(
                    black_box(&trace_id),
                    black_box(start_time),
                    black_box(end_time),
                )
                .expect("test");
            black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_query_by_trace_id,
    bench_query_by_trace_id_and_time,
    bench_query_optimized,
    bench_predicate_pushdown_comparison,
    bench_bloom_filter_comparison,
    bench_row_group_size_comparison,
    bench_composite_index_comparison,
);

criterion_main!(benches);
