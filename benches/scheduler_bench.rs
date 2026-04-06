#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::uninlined_format_args,
    clippy::explicit_iter_loop,
    clippy::doc_markdown
)]
//! Scheduler benchmarks (Iron Lotus Framework: Performance validation)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use repartir::scheduler::{Scheduler, WorkerId};
use repartir::task::{Backend, Task};

fn bench_scheduler_submit(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("scheduler_submit", |b| {
        b.iter(|| {
            rt.block_on(async {
                let scheduler = Scheduler::new();
                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg("test")
                    .backend(Backend::Cpu)
                    .build()
                    .unwrap();

                scheduler.submit(black_box(task)).await.unwrap();
            });
        });
    });
}

/// Benchmark data location tracking overhead (v2.0 Phase 1)
fn bench_data_location_tracking(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("data_location_tracking");

    for num_items in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_items),
            num_items,
            |b, &num_items| {
                b.iter(|| {
                    rt.block_on(async {
                        let scheduler = Scheduler::with_capacity(1000);
                        let tracker = scheduler.data_tracker();
                        let worker_id = WorkerId::new();

                        for i in 0..num_items {
                            let data_key = format!("dataset_{}", i);
                            tracker.track_data(black_box(data_key), worker_id).await;
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark data location batch queries (v2.0 Phase 1)
fn bench_data_location_batch_query(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("data_location_batch_query");

    for num_datasets in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_datasets),
            num_datasets,
            |b, &num_datasets| {
                // Setup: Pre-populate data locations
                let scheduler = rt.block_on(async {
                    let scheduler = Scheduler::with_capacity(1000);
                    let tracker = scheduler.data_tracker();

                    // Create 3 workers
                    let workers: Vec<WorkerId> = (0..3).map(|_| WorkerId::new()).collect();

                    // Track data across workers
                    for i in 0..num_datasets {
                        let data_key = format!("dataset_{}", i);
                        let worker_id = workers[i % workers.len()];
                        tracker.track_data(data_key, worker_id).await;
                    }

                    scheduler
                });

                b.iter(|| {
                    rt.block_on(async {
                        let data_keys: Vec<String> =
                            (0..num_datasets).map(|i| format!("dataset_{}", i)).collect();

                        let _locations =
                            scheduler.data_tracker().locate_data_batch(black_box(&data_keys)).await;
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark affinity calculation overhead (v2.0 Phase 2)
fn bench_affinity_calculation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("affinity_calculation");

    for num_dependencies in [5, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_dependencies),
            num_dependencies,
            |b, &num_dependencies| {
                // Setup: Pre-populate data locations
                let scheduler = rt.block_on(async {
                    let scheduler = Scheduler::with_capacity(1000);
                    let tracker = scheduler.data_tracker();

                    // Create 5 workers with different data distributions
                    let workers: Vec<WorkerId> = (0..5).map(|_| WorkerId::new()).collect();

                    for i in 0..num_dependencies {
                        let data_key = format!("dataset_{}", i);
                        // Worker 0 has most data, others have less
                        if i < num_dependencies * 3 / 4 {
                            tracker.track_data(data_key.clone(), workers[0]).await;
                        }
                        if i < num_dependencies / 2 {
                            tracker.track_data(data_key.clone(), workers[1]).await;
                        }
                        if i < num_dependencies / 4 {
                            tracker.track_data(data_key, workers[2]).await;
                        }
                    }

                    scheduler
                });

                b.iter(|| {
                    rt.block_on(async {
                        let task = Task::builder()
                            .binary("/bin/process")
                            .backend(Backend::Cpu)
                            .build()
                            .unwrap();

                        let data_keys: Vec<String> =
                            (0..num_dependencies).map(|i| format!("dataset_{}", i)).collect();

                        // This internally calculates affinity
                        scheduler
                            .submit_with_data_locality(black_box(task), black_box(&data_keys))
                            .await
                            .unwrap();

                        // Dequeue to prevent queue overflow
                        let _ = scheduler.next_task().await;
                    });
                });
            },
        );
    }

    group.finish();
}

/// Compare submit vs submit_with_affinity performance (v2.0 Phase 2)
fn bench_submit_comparison(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("submit_comparison");

    // Standard submit (no locality)
    group.bench_function("submit_standard", |b| {
        b.iter(|| {
            rt.block_on(async {
                let scheduler = Scheduler::with_capacity(1000);
                let task =
                    Task::builder().binary("/bin/process").backend(Backend::Cpu).build().unwrap();

                scheduler.submit(black_box(task)).await.unwrap();
            });
        });
    });

    // Submit with affinity (locality-aware)
    group.bench_function("submit_with_affinity", |b| {
        b.iter(|| {
            rt.block_on(async {
                let scheduler = Scheduler::with_capacity(1000);
                let tracker = scheduler.data_tracker();

                // Pre-populate some data locations
                let worker_id = WorkerId::new();
                for i in 0..10 {
                    tracker.track_data(format!("dataset_{}", i), worker_id).await;
                }

                let task =
                    Task::builder().binary("/bin/process").backend(Backend::Cpu).build().unwrap();

                let data_keys: Vec<String> = (0..10).map(|i| format!("dataset_{}", i)).collect();

                scheduler
                    .submit_with_data_locality(black_box(task), black_box(&data_keys))
                    .await
                    .unwrap();

                // Dequeue to prevent queue overflow
                let _ = scheduler.next_task().await;
            });
        });
    });

    group.finish();
}

/// Benchmark locality metrics tracking overhead (v2.0 Phase 2)
fn bench_locality_metrics(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("locality_metrics_access", |b| {
        let scheduler = rt.block_on(async {
            let scheduler = Scheduler::with_capacity(1000);

            // Submit some tasks to populate metrics
            for _ in 0..100 {
                let task =
                    Task::builder().binary("/bin/echo").backend(Backend::Cpu).build().unwrap();
                scheduler.submit(task).await.unwrap();
            }

            scheduler
        });

        b.iter(|| {
            rt.block_on(async {
                let metrics = scheduler.locality_metrics().await;
                black_box(metrics.hit_rate());
            });
        });
    });
}

criterion_group!(
    benches,
    bench_scheduler_submit,
    bench_data_location_tracking,
    bench_data_location_batch_query,
    bench_affinity_calculation,
    bench_submit_comparison,
    bench_locality_metrics
);
criterion_main!(benches);
