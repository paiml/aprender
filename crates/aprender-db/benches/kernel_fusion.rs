//! Kernel Fusion Benchmarks (CORE-003)
//!
//! Toyota Way: Muda elimination (waste of intermediate memory writes)
//!
//! This benchmark proves that fused filter+sum is faster than separate operations:
//! - Unfused: Filter → intermediate buffer → SUM (2 GPU passes, 1 memory write)
//! - Fused: Filter+SUM in single pass (1 GPU pass, 0 intermediate writes)
//!
//! Expected results:
//! - Fused should be 1.5-2x faster than unfused
//! - Larger datasets show bigger fusion benefit (amortize overhead)
//!
//! References:
//! - Wu et al. (2012): Kernel fusion execution model
//! - Neumann (2011): JIT compilation for queries
//! - CORE-003: JIT WGSL compiler for kernel fusion
//!
//! Run with: cargo bench --bench `kernel_fusion` --features gpu

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "gpu")]
use criterion::{black_box, BenchmarkId};

#[cfg(feature = "gpu")]
use arrow::array::Int32Array;
#[cfg(feature = "gpu")]
use trueno_db::gpu::GpuEngine;

// Dataset sizes for fusion analysis
#[allow(dead_code)]
const SMALL: usize = 1_000; // 1K rows
#[allow(dead_code)]
const MEDIUM: usize = 100_000; // 100K rows
#[allow(dead_code)]
const LARGE: usize = 1_000_000; // 1M rows
#[allow(dead_code)]
const XLARGE: usize = 10_000_000; // 10M rows

#[cfg(feature = "gpu")]
/// Benchmark fused filter+sum vs unfused (separate operations)
fn bench_fusion_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_fusion");
    group.sample_size(20); // Fewer samples for GPU benchmarks

    // Try to initialize GPU
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = runtime.block_on(async { GpuEngine::new().await });

    if engine.is_err() {
        eprintln!("⚠️  GPU not available, skipping kernel fusion benchmarks");
        eprintln!("   This is expected on CI or machines without GPU");
        return;
    }

    let engine = engine.unwrap();

    // Test different dataset sizes
    for size in [MEDIUM, LARGE] {
        let data: Vec<i32> = (0..size as i32).collect();
        let array = Int32Array::from(data);

        // Benchmark: Fused filter+sum (JIT-compiled single-pass kernel)
        group.bench_with_input(
            BenchmarkId::new("fused_filter_sum", format!("{}K", size / 1000)),
            &array,
            |b, array| {
                b.to_async(&runtime).iter(|| async {
                    // Single pass: Filter WHERE value > 500 AND SUM
                    engine.fused_filter_sum(black_box(array), 500, "gt").await.unwrap()
                });
            },
        );

        // Benchmark: Unfused (separate filter + sum operations)
        // Note: This is a simplified simulation - full unfused would require
        // implementing a separate filter kernel that writes intermediate buffer
        group.bench_with_input(
            BenchmarkId::new("unfused_filter_sum", format!("{}K", size / 1000)),
            &array,
            |b, array| {
                b.to_async(&runtime).iter(|| async {
                    // Simulate unfused: CPU filter → intermediate buffer → GPU sum
                    // In practice, this would be 2 GPU kernels with intermediate write
                    let filtered: Vec<i32> =
                        array.values().iter().copied().filter(|&x| x > 500).collect();
                    let filtered_array = Int32Array::from(filtered);
                    engine.sum_i32(black_box(&filtered_array)).await.unwrap()
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "gpu")]
/// Benchmark different filter operators with JIT compilation
fn bench_jit_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_operators");
    group.sample_size(20);

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = runtime.block_on(async { GpuEngine::new().await });

    if engine.is_err() {
        eprintln!("⚠️  GPU not available, skipping JIT operator benchmarks");
        return;
    }

    let engine = engine.unwrap();

    // 1M element dataset
    let data: Vec<i32> = (0..LARGE as i32).collect();
    let array = Int32Array::from(data);

    // Test different operators (should all compile and cache separately)
    for op in ["gt", "lt", "eq", "gte", "lte"] {
        group.bench_with_input(BenchmarkId::new("fused_filter_sum", op), &array, |b, array| {
            b.to_async(&runtime).iter(|| async {
                engine.fused_filter_sum(black_box(array), 500_000, op).await.unwrap()
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
/// Benchmark shader cache effectiveness
fn bench_cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("shader_cache");
    group.sample_size(50); // More samples since cache should make this fast

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = runtime.block_on(async { GpuEngine::new().await });

    if engine.is_err() {
        eprintln!("⚠️  GPU not available, skipping cache benchmarks");
        return;
    }

    let engine = engine.unwrap();

    let data: Vec<i32> = (0..MEDIUM as i32).collect();
    let array = Int32Array::from(data);

    // First call: Cold cache (includes JIT compilation)
    group.bench_function(BenchmarkId::new("first_call", "cold_cache"), |b| {
        b.to_async(&runtime).iter(|| async {
            // Each iteration uses a different threshold to avoid caching
            // (this simulates first-time compilation cost)
            static mut COUNTER: i32 = 0;
            let threshold = unsafe {
                COUNTER += 1;
                COUNTER
            };
            engine.fused_filter_sum(black_box(&array), threshold, "gt").await.unwrap()
        });
    });

    // Subsequent calls: Warm cache (cached shader, no compilation)
    group.bench_function(BenchmarkId::new("cached_call", "warm_cache"), |b| {
        b.to_async(&runtime).iter(|| async {
            // Same threshold every time → cached shader
            engine.fused_filter_sum(black_box(&array), 1000, "gt").await.unwrap()
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(
    fusion_benches,
    bench_fusion_comparison,
    bench_jit_operators,
    bench_cache_effectiveness
);

#[cfg(not(feature = "gpu"))]
fn bench_gpu_not_available(_c: &mut Criterion) {
    eprintln!("GPU benchmarks require --features gpu");
}

#[cfg(not(feature = "gpu"))]
criterion_group!(fusion_benches, bench_gpu_not_available);

criterion_main!(fusion_benches);
