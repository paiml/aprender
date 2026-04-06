//! Benchmarks for dataset operations.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::explicit_iter_loop,
    missing_docs
)]

use std::sync::Arc;

use alimentar::{
    parallel::ParallelDataLoader, ArrowDataset, DataLoader, Dataset, Select, Shuffle, Transform,
    WeightedDataLoader,
};
use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn create_dataset(rows: usize) -> ArrowDataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ]));

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let ids: Vec<i32> = (0..rows as i32).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("item_{i}")).collect();
    #[allow(clippy::cast_lossless)]
    let scores: Vec<f64> = ids.iter().map(|i| *i as f64 * 1.5).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Float64Array::from(scores)),
        ],
    )
    .expect("Failed to create batch");

    ArrowDataset::from_batch(batch).expect("Failed to create dataset")
}

fn bench_dataset_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_creation");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| create_dataset(black_box(size)));
        });
    }

    group.finish();
}

fn bench_dataset_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_iteration");

    for size in [1_000, 10_000, 100_000].iter() {
        let dataset = create_dataset(*size);
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &dataset, |b, dataset| {
            b.iter(|| {
                let mut count = 0;
                for _batch in dataset.iter() {
                    count += 1;
                }
                black_box(count)
            });
        });
    }

    group.finish();
}

fn bench_dataloader(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataloader");

    for size in [1_000, 10_000].iter() {
        let dataset = create_dataset(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_32", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let loader = DataLoader::new(dataset.clone()).batch_size(32);
                    let mut count = 0;
                    for _batch in loader {
                        count += 1;
                    }
                    black_box(count)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_128", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let loader = DataLoader::new(dataset.clone()).batch_size(128);
                    let mut count = 0;
                    for _batch in loader {
                        count += 1;
                    }
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

fn bench_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("transforms");

    let dataset = create_dataset(10_000);
    let batch = dataset.get(0).expect("Should have batch");

    group.bench_function("select_2_columns", |b| {
        let select = Select::new(vec!["id", "score"]);
        b.iter(|| {
            let result = select.apply(black_box(batch.clone()));
            black_box(result)
        });
    });

    group.bench_function("shuffle", |b| {
        let shuffle = Shuffle::with_seed(42);
        b.iter(|| {
            let result = shuffle.apply(black_box(batch.clone()));
            black_box(result)
        });
    });

    group.finish();
}

fn bench_row_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_access");

    for size in [1_000, 10_000, 100_000].iter() {
        let dataset = create_dataset(*size);
        group.throughput(Throughput::Elements(100));

        group.bench_with_input(BenchmarkId::from_parameter(size), &dataset, |b, dataset| {
            b.iter(|| {
                // Access 100 random rows
                for i in (0..*size).step_by(size / 100) {
                    let _ = black_box(dataset.get(i));
                }
            });
        });
    }

    group.finish();
}

fn bench_parallel_dataloader(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_dataloader");

    for size in [1_000, 10_000].iter() {
        let dataset = create_dataset(*size);
        group.throughput(Throughput::Elements(*size as u64));

        // Single-threaded (num_workers=0)
        group.bench_with_input(
            BenchmarkId::new("workers_0", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let loader = ParallelDataLoader::new(dataset.clone())
                        .batch_size(32)
                        .num_workers(0);
                    let mut count = 0;
                    for _batch in loader {
                        count += 1;
                    }
                    black_box(count)
                });
            },
        );

        // Two workers
        group.bench_with_input(
            BenchmarkId::new("workers_2", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let loader = ParallelDataLoader::new(dataset.clone())
                        .batch_size(32)
                        .num_workers(2);
                    let mut count = 0;
                    for _batch in loader {
                        count += 1;
                    }
                    black_box(count)
                });
            },
        );

        // Four workers
        group.bench_with_input(
            BenchmarkId::new("workers_4", size),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let loader = ParallelDataLoader::new(dataset.clone())
                        .batch_size(32)
                        .num_workers(4);
                    let mut count = 0;
                    for _batch in loader {
                        count += 1;
                    }
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn bench_weighted_dataloader(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_dataloader");

    for size in [1_000, 10_000].iter() {
        let dataset = create_dataset(*size);
        let weights: Vec<f32> = (0..*size).map(|i| 1.0 + (i as f32 * 0.001)).collect();
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &dataset, |b, dataset| {
            b.iter(|| {
                let loader = WeightedDataLoader::new(dataset.clone(), weights.clone())
                    .expect("Should create loader")
                    .batch_size(32)
                    .num_samples(*size)
                    .seed(42);
                let mut count = 0;
                for _batch in loader {
                    count += 1;
                }
                black_box(count)
            });
        });
    }

    group.finish();
}

fn bench_prefetch_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_comparison");

    let dataset = create_dataset(10_000);
    group.throughput(Throughput::Elements(10_000));

    // No prefetch (prefetch=1)
    group.bench_function("prefetch_1", |b| {
        let dataset = dataset.clone();
        b.iter(|| {
            let loader = ParallelDataLoader::new(dataset.clone())
                .batch_size(32)
                .num_workers(2)
                .prefetch(1);
            let mut count = 0;
            for _batch in loader {
                count += 1;
            }
            black_box(count)
        });
    });

    // Medium prefetch
    group.bench_function("prefetch_4", |b| {
        let dataset = dataset.clone();
        b.iter(|| {
            let loader = ParallelDataLoader::new(dataset.clone())
                .batch_size(32)
                .num_workers(2)
                .prefetch(4);
            let mut count = 0;
            for _batch in loader {
                count += 1;
            }
            black_box(count)
        });
    });

    // High prefetch
    group.bench_function("prefetch_8", |b| {
        let dataset = dataset.clone();
        b.iter(|| {
            let loader = ParallelDataLoader::new(dataset.clone())
                .batch_size(32)
                .num_workers(2)
                .prefetch(8);
            let mut count = 0;
            for _batch in loader {
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dataset_creation,
    bench_dataset_iteration,
    bench_dataloader,
    bench_transforms,
    bench_row_access,
    bench_parallel_dataloader,
    bench_weighted_dataloader,
    bench_prefetch_comparison,
);
criterion_main!(benches);
