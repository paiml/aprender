//! `PCIe` Transfer Benchmarks and 5x Rule Validation (CORE-008)
//!
//! Toyota Way: Genchi Genbutsu (go and see, measure don't guess)
//!
//! This benchmark empirically validates the 5x rule from CORE-002:
//! GPU is only worthwhile when `compute_time` > 5 * `transfer_time`
//!
//! Measurements:
//! 1. `PCIe` transfer time (CPU → GPU VRAM)
//! 2. GPU compute time for SUM aggregation
//! 3. Transfer overhead vs compute time ratio
//!
//! Expected findings:
//! - Small datasets (<1M rows): Transfer dominates, SIMD faster
//! - Large datasets (>10M rows): Compute dominates, GPU faster
//! - Crossover point validates cost model from `BackendDispatcher`
//!
//! References:
//! - Gregg & Hazelwood (2011): `PCIe` bus bottleneck analysis
//! - CORE-002: Cost-based backend dispatcher with 5x rule
//!
//! Run with: cargo bench --bench `pcie_analysis` --features gpu

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "gpu")]
use criterion::{black_box, BenchmarkId};
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
use arrow::array::Int32Array;
#[cfg(feature = "gpu")]
use trueno_db::gpu::GpuEngine;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

// Dataset sizes for testing
#[allow(dead_code)]
const SMALL: usize = 1_000; // 1K rows = 4KB
#[allow(dead_code)]
const MEDIUM: usize = 100_000; // 100K rows = 400KB
#[allow(dead_code)]
const LARGE: usize = 1_000_000; // 1M rows = 4MB
#[allow(dead_code)]
const XLARGE: usize = 10_000_000; // 10M rows = 40MB

#[cfg(feature = "gpu")]
/// Benchmark PCIe transfer time (CPU → GPU VRAM)
fn bench_pcie_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcie_transfer");
    group.sample_size(20); // Fewer samples for GPU benchmarks

    // Try to initialize GPU
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = runtime.block_on(async { GpuEngine::new().await });

    if engine.is_err() {
        eprintln!("⚠️  GPU not available, skipping PCIe benchmarks");
        eprintln!("   This is expected on CI or machines without GPU");
        return;
    }

    let engine = engine.unwrap();
    let device = &engine.device;

    // Benchmark different data sizes
    for size in [SMALL, MEDIUM, LARGE, XLARGE] {
        let data: Vec<i32> = (0..size as i32).collect();
        let size_mb = (size * 4) as f64 / (1024.0 * 1024.0);

        group.bench_with_input(
            BenchmarkId::new("cpu_to_gpu", format!("{}MB", size_mb as usize)),
            &data,
            |b, data| {
                b.iter(|| {
                    // Measure only the buffer creation (PCIe transfer)
                    let _buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("PCIe Transfer Benchmark"),
                        contents: bytemuck::cast_slice(black_box(data)),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "gpu")]
/// Benchmark GPU compute time for SUM operation
fn bench_gpu_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_compute_sum");
    group.sample_size(20);

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = runtime.block_on(async { GpuEngine::new().await });

    if engine.is_err() {
        eprintln!("⚠️  GPU not available, skipping GPU compute benchmarks");
        return;
    }

    let engine = engine.unwrap();

    for size in [SMALL, MEDIUM, LARGE, XLARGE] {
        let data: Vec<i32> = (0..size as i32).collect();
        let array = Int32Array::from(data);
        let size_mb = (size * 4) as f64 / (1024.0 * 1024.0);

        group.bench_with_input(
            BenchmarkId::new("sum_i32", format!("{}MB", size_mb as usize)),
            &array,
            |b, array| {
                b.to_async(&runtime)
                    .iter(|| async { engine.sum_i32(black_box(array)).await.unwrap() });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "gpu")]
/// Validate the 5x rule: measure transfer vs compute time
fn bench_5x_rule_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("5x_rule_validation");
    group.sample_size(10); // Fewer samples for this analysis

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = runtime.block_on(async { GpuEngine::new().await });

    if engine.is_err() {
        eprintln!("⚠️  GPU not available, skipping 5x rule validation");
        return;
    }

    let engine = engine.unwrap();
    let device = &engine.device;
    let queue = &engine.queue;

    println!("\n📊 5x Rule Validation (CORE-008)");
    println!("   Measuring transfer vs compute time for different dataset sizes\n");

    for size in [SMALL, MEDIUM, LARGE, XLARGE] {
        let data: Vec<i32> = (0..size as i32).collect();
        let size_mb = (size * 4) as f64 / (1024.0 * 1024.0);

        // Measure transfer time
        let start = Instant::now();
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transfer Time Measurement"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        device.poll(wgpu::Maintain::Wait);
        let transfer_time = start.elapsed();

        // Measure compute time (simplified - just dispatch)
        let array = Int32Array::from(data);
        let start = Instant::now();
        let _result = runtime.block_on(async { engine.sum_i32(&array).await.unwrap() });
        let total_time = start.elapsed();

        // Approximate compute time (total - transfer)
        let compute_time = total_time.saturating_sub(transfer_time);

        let ratio = if transfer_time.as_secs_f64() > 0.0 {
            compute_time.as_secs_f64() / transfer_time.as_secs_f64()
        } else {
            0.0
        };

        println!("   Size: {:.1}MB", size_mb);
        println!("     Transfer: {:?}", transfer_time);
        println!("     Compute:  {:?}", compute_time);
        println!("     Ratio:    {:.2}x", ratio);
        println!("     GPU worthwhile: {}", if ratio > 5.0 { "✅ YES" } else { "❌ NO" });
        println!();

        // Prevent unused variable warning
        let _ = input_buffer;
        let _ = queue;
    }

    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(pcie_benches, bench_pcie_transfer, bench_gpu_compute, bench_5x_rule_validation);

#[cfg(not(feature = "gpu"))]
fn bench_gpu_not_available(_c: &mut Criterion) {
    eprintln!("GPU benchmarks require --features gpu");
}

#[cfg(not(feature = "gpu"))]
criterion_group!(pcie_benches, bench_gpu_not_available);

criterion_main!(pcie_benches);
