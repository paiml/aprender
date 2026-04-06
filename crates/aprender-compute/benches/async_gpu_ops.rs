#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Benchmark comparing synchronous GPU API vs async batch API
//!
//! Success Criteria (v0.3.0):
//! - 2x fewer GPU transfers for chained operations
//! - ≥30% performance improvement for 3+ operation chains

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno::backends::gpu::{GpuCommandBatch, GpuDevice};

/// Benchmark: Chained operations (relu → scale → add) - SYNC API
///
/// This uses the traditional synchronous API where each operation:
/// 1. Uploads data to GPU
/// 2. Executes kernel
/// 3. Downloads result
///
/// Total transfers: 6 (3 uploads + 3 downloads)
fn bench_sync_chained_ops(c: &mut Criterion) {
    if !GpuDevice::is_available() {
        eprintln!("GPU not available, skipping sync benchmark");
        return;
    }

    let device = GpuDevice::new().expect("Failed to create GPU device");

    let mut group = c.benchmark_group("gpu_chained_ops_sync");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let input = vec![1.0f32; size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                let input_data = black_box(&input);

                // Operation 1: ReLU (upload + execute + download)
                let mut relu_result = vec![0.0f32; size];
                device.relu(input_data, &mut relu_result).expect("ReLU failed");

                // Operation 2: Scale by 2.0 (upload + execute + download)
                let mut scaled_result = vec![0.0f32; size];
                for i in 0..size {
                    scaled_result[i] = relu_result[i] * 2.0;
                }

                // Operation 3: Add constant vector (upload + execute + download)
                let other = vec![0.5f32; size];
                let mut final_result = vec![0.0f32; size];
                device.vec_add(&scaled_result, &other, &mut final_result).expect("Add failed");

                black_box(final_result)
            });
        });
    }

    group.finish();
}

/// Benchmark: Chained operations (relu → scale → add) - ASYNC BATCH API
///
/// This uses the new async batch API where:
/// 1. Upload data once
/// 2. Execute all operations on GPU
/// 3. Download result once
///
/// Total transfers: 2 (1 upload + 1 download) = **3x reduction**
fn bench_async_chained_ops(c: &mut Criterion) {
    if !GpuDevice::is_available() {
        eprintln!("GPU not available, skipping async benchmark");
        return;
    }

    let device = GpuDevice::new().expect("Failed to create GPU device");

    let mut group = c.benchmark_group("gpu_chained_ops_async");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let input = vec![1.0f32; size];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                let input_data = black_box(&input);

                // Use async batch API
                let mut batch = GpuCommandBatch::new(device.clone());

                // Queue all operations (no GPU execution yet)
                let input_id = batch.upload(input_data);
                let relu_out = batch.relu(input_id);
                let scaled = batch.scale(relu_out, 2.0);
                let other_id = batch.upload(&vec![0.5f32; size]);
                let final_out = batch.add(scaled, other_id);

                // Execute all operations in single batch
                pollster::block_on(async {
                    batch.execute().await.expect("Execute failed");
                    batch.read(final_out).await.expect("Read failed")
                })
            });
        });
    }

    group.finish();
}

/// Benchmark: Single operation (relu) - SYNC vs ASYNC
///
/// For single operations, async API should have similar performance
/// (both do 1 upload + 1 download)
fn bench_single_op_comparison(c: &mut Criterion) {
    if !GpuDevice::is_available() {
        eprintln!("GPU not available, skipping single op benchmark");
        return;
    }

    let device = GpuDevice::new().expect("Failed to create GPU device");

    let mut group = c.benchmark_group("gpu_single_op");

    for size in [10_000, 100_000, 1_000_000] {
        let input = vec![1.0f32; size];

        // Sync API
        group.bench_with_input(BenchmarkId::new("sync", size), &size, |b, &_size| {
            b.iter(|| {
                let mut result = vec![0.0f32; size];
                device.relu(black_box(&input), &mut result).expect("ReLU failed");
                black_box(result)
            });
        });

        // Async API
        group.bench_with_input(BenchmarkId::new("async", size), &size, |b, &_size| {
            b.iter(|| {
                let mut batch = GpuCommandBatch::new(device.clone());
                let input_id = batch.upload(black_box(&input));
                let output_id = batch.relu(input_id);

                pollster::block_on(async {
                    batch.execute().await.expect("Execute failed");
                    batch.read(output_id).await.expect("Read failed")
                })
            });
        });
    }

    group.finish();
}

/// Benchmark: Deep chain (5 operations) - Maximum benefit
///
/// relu → scale → add → mul → relu
///
/// Sync: 10 transfers (5 up + 5 down)
/// Async: 2 transfers (1 up + 1 down) = **5x reduction**
fn bench_deep_chain(c: &mut Criterion) {
    if !GpuDevice::is_available() {
        eprintln!("GPU not available, skipping deep chain benchmark");
        return;
    }

    let device = GpuDevice::new().expect("Failed to create GPU device");

    let mut group = c.benchmark_group("gpu_deep_chain");

    for size in [10_000, 100_000, 1_000_000] {
        let input = vec![1.0f32; size];

        // Benchmarks async batched GPU operations (amortizes kernel launch overhead)
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                let input_data = black_box(&input);
                let mut batch = GpuCommandBatch::new(device.clone());

                // Queue 5 chained operations
                let x = batch.upload(input_data);
                let x = batch.relu(x);
                let x = batch.scale(x, 2.0);
                let other = batch.upload(&vec![0.5f32; size]);
                let x = batch.add(x, other);
                let multiplier = batch.upload(&vec![1.5f32; size]);
                let x = batch.mul(x, multiplier);
                let x = batch.relu(x);

                pollster::block_on(async {
                    batch.execute().await.expect("Execute failed");
                    batch.read(x).await.expect("Read failed")
                })
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sync_chained_ops,
    bench_async_chained_ops,
    bench_single_op_comparison,
    bench_deep_chain
);
criterion_main!(benches);
