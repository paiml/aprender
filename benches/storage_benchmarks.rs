//! Storage backend benchmarks
//!
//! Benchmarks for Arrow storage backend performance:
//! - Parquet loading
//! - Morsel iteration
//! - `RecordBatch` slicing
//!
//! Toyota Way: Measure before optimizing (Genchi Genbutsu)

use arrow::array::{Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::Arc;
use trueno_db::storage::StorageEngine;

/// Create a test `RecordBatch` with specified number of rows
#[allow(clippy::cast_precision_loss)]
fn create_test_batch(num_rows: i32) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float32, false),
        Field::new("category", DataType::Utf8, false),
    ]);

    let id_array = Int32Array::from_iter_values(0..num_rows);
    let value_array = Float32Array::from_iter_values((0..num_rows).map(|i| (i as f32) * 1.5));
    let category_array =
        StringArray::from_iter_values((0..num_rows).map(|i| format!("category_{}", i % 10)));

    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(id_array), Arc::new(value_array), Arc::new(category_array)],
    )
    .unwrap()
}

/// Create a test Parquet file with specified number of rows
fn create_test_parquet(path: &str, num_rows: i32) {
    let batch = create_test_batch(num_rows);
    let schema = batch.schema();

    let file = File::create(path).unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_size(usize::try_from(num_rows).unwrap() / 2) // 2 row groups
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Benchmark Parquet file loading
fn bench_parquet_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("parquet_loading");

    for size in &[1_000, 10_000, 100_000] {
        let path = format!("/tmp/trueno_bench_{size}.parquet");
        create_test_parquet(&path, *size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let storage = StorageEngine::load_parquet(&path).unwrap();
                black_box(storage);
            });
        });

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    group.finish();
}

/// Benchmark morsel iteration
fn bench_morsel_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("morsel_iteration");

    for size in &[10_000, 100_000, 1_000_000] {
        let batch = create_test_batch(*size);
        let storage = StorageEngine::new(vec![batch]);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let morsels: Vec<_> = storage.morsels().collect();
                black_box(morsels);
            });
        });
    }

    group.finish();
}

/// Benchmark `RecordBatch` memory usage calculation
fn bench_batch_memory_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_memory_size");

    for size in &[1_000, 10_000, 100_000] {
        let batch = create_test_batch(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let size = batch.get_array_memory_size();
                black_box(size);
            });
        });
    }

    group.finish();
}

/// Benchmark `RecordBatch` slicing (used by morsel iterator)
fn bench_batch_slicing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_slicing");

    let batch = create_test_batch(100_000);

    for chunk_size in &[1_000, 10_000, 50_000] {
        group.bench_with_input(BenchmarkId::from_parameter(chunk_size), chunk_size, |b, size| {
            b.iter(|| {
                let sliced = batch.slice(0, *size);
                black_box(sliced);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parquet_loading,
    bench_morsel_iteration,
    bench_batch_memory_size,
    bench_batch_slicing
);
criterion_main!(benches);
