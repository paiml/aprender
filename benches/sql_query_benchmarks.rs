//! SQL Query Execution Benchmarks
//!
//! Validates performance targets from GitHub Issue #3:
//! - SIMD aggregations: 2.78x faster than scalar
//! - Top-K (1M rows): 5-28x faster than heap-based
//!
//! Run with: cargo bench --bench `sql_query_benchmarks`

use arrow::array::{Float64Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use trueno_db::query::{QueryEngine, QueryExecutor};
use trueno_db::storage::StorageEngine;

const SMALL_SIZE: usize = 1_000; // 1K rows
const MEDIUM_SIZE: usize = 100_000; // 100K rows
const LARGE_SIZE: usize = 1_000_000; // 1M rows

/// Create test data for benchmarks
fn create_benchmark_data(num_rows: usize) -> StorageEngine {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]));

    let ids: Vec<i32> = (0..num_rows as i32).collect();
    let values: Vec<f64> = (0..num_rows).map(|i| i as f64 * 1.5).collect();
    let quantities: Vec<i32> = (0..num_rows as i32).map(|i| i * 10).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Float64Array::from(values)),
            Arc::new(Int32Array::from(quantities)),
        ],
    )
    .unwrap();

    let mut storage = StorageEngine::new(vec![]);
    storage.append_batch(batch).unwrap();
    storage
}

/// Benchmark SUM aggregation via SQL
fn bench_sql_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_sum_aggregation");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let storage = create_benchmark_data(size);
        let engine = QueryEngine::new();
        let executor = QueryExecutor::new();

        group.bench_with_input(BenchmarkId::new("sql_sum", size), &size, |b, _| {
            b.iter(|| {
                let plan = engine.parse("SELECT SUM(value) FROM table1").unwrap();
                black_box(executor.execute(&plan, &storage).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark AVG aggregation via SQL
fn bench_sql_avg(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_avg_aggregation");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let storage = create_benchmark_data(size);
        let engine = QueryEngine::new();
        let executor = QueryExecutor::new();

        group.bench_with_input(BenchmarkId::new("sql_avg", size), &size, |b, _| {
            b.iter(|| {
                let plan = engine.parse("SELECT AVG(value) FROM table1").unwrap();
                black_box(executor.execute(&plan, &storage).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark COUNT aggregation via SQL
fn bench_sql_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_count_aggregation");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let storage = create_benchmark_data(size);
        let engine = QueryEngine::new();
        let executor = QueryExecutor::new();

        group.bench_with_input(BenchmarkId::new("sql_count", size), &size, |b, _| {
            b.iter(|| {
                let plan = engine.parse("SELECT COUNT(*) FROM table1").unwrap();
                black_box(executor.execute(&plan, &storage).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark MIN/MAX aggregations via SQL
fn bench_sql_min_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_min_max_aggregation");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let storage = create_benchmark_data(size);
        let engine = QueryEngine::new();
        let executor = QueryExecutor::new();

        group.bench_with_input(BenchmarkId::new("sql_min_max", size), &size, |b, _| {
            b.iter(|| {
                let plan = engine.parse("SELECT MIN(value), MAX(value) FROM table1").unwrap();
                black_box(executor.execute(&plan, &storage).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark ORDER BY + LIMIT (Top-K optimization)
fn bench_sql_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_top_k_order_by_limit");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let storage = create_benchmark_data(size);
        let engine = QueryEngine::new();
        let executor = QueryExecutor::new();

        // Top-10 query
        group.bench_with_input(BenchmarkId::new("top_10", size), &size, |b, _| {
            b.iter(|| {
                let plan = engine
                    .parse("SELECT id, value FROM table1 ORDER BY value DESC LIMIT 10")
                    .unwrap();
                black_box(executor.execute(&plan, &storage).unwrap());
            });
        });

        // Top-100 query
        group.bench_with_input(BenchmarkId::new("top_100", size), &size, |b, _| {
            b.iter(|| {
                let plan = engine
                    .parse("SELECT id, value FROM table1 ORDER BY value DESC LIMIT 100")
                    .unwrap();
                black_box(executor.execute(&plan, &storage).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark WHERE filter + aggregation
fn bench_sql_filter_aggregate(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_filter_aggregate");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let storage = create_benchmark_data(size);
        let engine = QueryEngine::new();
        let executor = QueryExecutor::new();

        group.bench_with_input(BenchmarkId::new("filter_sum", size), &size, |b, _| {
            b.iter(|| {
                let plan =
                    engine.parse("SELECT SUM(value) FROM table1 WHERE value > 1000.0").unwrap();
                black_box(executor.execute(&plan, &storage).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark complete query pipeline (parse + execute)
fn bench_sql_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("sql_full_pipeline");

    let storage = create_benchmark_data(MEDIUM_SIZE);
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    group.bench_function("parse_and_execute", |b| {
        b.iter(|| {
            let plan = engine
                .parse(black_box("SELECT SUM(value), AVG(value) FROM table1 WHERE value > 500.0"))
                .unwrap();
            black_box(executor.execute(&plan, &storage).unwrap());
        });
    });

    group.finish();
}

/// Benchmark scalar baseline for comparison (Target: 2.78x slower than SIMD)
fn bench_scalar_baseline_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_baseline_sum");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let values: Vec<f64> = (0..size).map(|i| i as f64 * 1.5).collect();

        group.bench_with_input(BenchmarkId::new("scalar_sum", size), &size, |b, _| {
            b.iter(|| {
                let sum: f64 = black_box(&values).iter().sum();
                black_box(sum);
            });
        });
    }

    group.finish();
}

/// Benchmark heap-based Top-K for comparison (Target: 5-28x slower)
fn bench_heap_based_top_k_baseline(c: &mut Criterion) {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Debug)]
    struct MinHeapItem {
        value: f64,
        #[allow(dead_code)]
        index: usize,
    }

    impl PartialEq for MinHeapItem {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Eq for MinHeapItem {}

    impl Ord for MinHeapItem {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse for min-heap
            other.value.partial_cmp(&self.value).unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for MinHeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut group = c.benchmark_group("heap_based_top_k_baseline");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE] {
        let values: Vec<f64> = (0..size).map(|i| i as f64 * 1.5).collect();

        group.bench_with_input(BenchmarkId::new("heap_top_10", size), &size, |b, _| {
            b.iter(|| {
                let mut heap: BinaryHeap<MinHeapItem> = BinaryHeap::with_capacity(10);
                for (index, &value) in black_box(&values).iter().enumerate() {
                    if heap.len() < 10 {
                        heap.push(MinHeapItem { value, index });
                    } else if let Some(top) = heap.peek() {
                        if value > top.value {
                            heap.pop();
                            heap.push(MinHeapItem { value, index });
                        }
                    }
                }
                black_box(heap);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sql_sum,
    bench_sql_avg,
    bench_sql_count,
    bench_sql_min_max,
    bench_sql_top_k,
    bench_sql_filter_aggregate,
    bench_sql_full_pipeline,
    bench_scalar_baseline_sum,
    bench_heap_based_top_k_baseline
);
criterion_main!(benches);
