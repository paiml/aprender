#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::explicit_iter_loop,
    clippy::cast_sign_loss,
    clippy::semicolon_if_nothing_returned
)]
//! Pool benchmarks comparing repartir to Ray/Dask patterns

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use repartir::task::{Backend, Task};
use repartir::Pool;

/// Benchmark task submission throughput
fn bench_pool_submit_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_submit_throughput");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for num_tasks in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*num_tasks as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_tasks),
            num_tasks,
            |b, &num_tasks| {
                b.iter(|| {
                    rt.block_on(async {
                        let pool =
                            Pool::builder().cpu_workers(4).max_queue_size(10_000).build().unwrap();

                        let mut tasks = Vec::new();
                        for i in 0..num_tasks {
                            let task = Task::builder()
                                .binary("/bin/echo")
                                .arg(format!("task_{i}"))
                                .backend(Backend::Cpu)
                                .build()
                                .unwrap();

                            tasks.push(pool.submit(black_box(task)));
                        }

                        // Wait for all tasks
                        for task in tasks {
                            let _ = task.await;
                        }

                        pool.shutdown().await;
                    })
                });
            },
        );
    }
    group.finish();
}

/// Benchmark parallel speedup vs sequential execution
fn bench_parallel_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_speedup");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Sequential baseline
    group.bench_function("sequential_10_tasks", |b| {
        b.iter(|| {
            for i in 0..10 {
                std::process::Command::new("/bin/sh")
                    .arg("-c")
                    .arg(format!("sleep 0.01 && echo {i}"))
                    .output()
                    .unwrap();
            }
        });
    });

    // Parallel with pool
    group.bench_function("parallel_10_tasks_4_workers", |b| {
        b.iter(|| {
            rt.block_on(async {
                let pool = Pool::builder().cpu_workers(4).build().unwrap();

                let mut tasks = Vec::new();
                for i in 0..10 {
                    let task = Task::builder()
                        .binary("/bin/sh")
                        .arg("-c")
                        .arg(format!("sleep 0.01 && echo {i}"))
                        .backend(Backend::Cpu)
                        .build()
                        .unwrap();

                    tasks.push(pool.submit(task));
                }

                for task in tasks {
                    let _ = task.await;
                }

                pool.shutdown().await;
            })
        });
    });

    group.finish();
}

/// Benchmark task execution latency distribution
fn bench_task_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_latency");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("single_task_echo", |b| {
        b.iter(|| {
            rt.block_on(async {
                let pool = Pool::builder().cpu_workers(1).build().unwrap();

                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg("benchmark")
                    .backend(Backend::Cpu)
                    .build()
                    .unwrap();

                let _ = pool.submit(black_box(task)).await;
                pool.shutdown().await;
            })
        });
    });

    group.finish();
}

/// Benchmark CPU-intensive workload distribution
fn bench_cpu_intensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_intensive");
    group.sample_size(10); // Reduce sample size for long-running benchmarks
    let rt = tokio::runtime::Runtime::new().unwrap();

    for num_workers in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_workers}_workers")),
            num_workers,
            |b, &num_workers| {
                b.iter(|| {
                    rt.block_on(async {
                        let pool = Pool::builder().cpu_workers(num_workers).build().unwrap();

                        // Submit CPU-bound tasks (calculate primes)
                        let mut tasks = Vec::new();
                        for _ in 0..16 {
                            let task = Task::builder()
                                .binary("/bin/sh")
                                .arg("-c")
                                .arg("seq 1 1000 | wc -l")
                                .backend(Backend::Cpu)
                                .build()
                                .unwrap();

                            tasks.push(pool.submit(task));
                        }

                        for task in tasks {
                            let _ = task.await;
                        }

                        pool.shutdown().await;
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed priority workload
fn bench_priority_scheduling(c: &mut Criterion) {
    use repartir::task::Priority;

    let mut group = c.benchmark_group("priority_scheduling");
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("mixed_priority_20_tasks", |b| {
        b.iter(|| {
            rt.block_on(async {
                let pool = Pool::builder().cpu_workers(2).build().unwrap();

                let mut tasks = Vec::new();

                // Mix of priorities
                for i in 0..20 {
                    let priority = match i % 3 {
                        0 => Priority::High,
                        1 => Priority::Normal,
                        _ => Priority::Low,
                    };

                    let task = Task::builder()
                        .binary("/bin/echo")
                        .arg(format!("task_{i}"))
                        .backend(Backend::Cpu)
                        .priority(priority)
                        .build()
                        .unwrap();

                    tasks.push(pool.submit(task));
                }

                for task in tasks {
                    let _ = task.await;
                }

                pool.shutdown().await;
            })
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pool_submit_throughput,
    bench_parallel_speedup,
    bench_task_latency,
    bench_cpu_intensive,
    bench_priority_scheduling
);
criterion_main!(benches);
