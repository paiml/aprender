//! CUDA Executor Benchmarks
//!
//! Benchmarks comparing CPU vs GPU execution via trueno-gpu runtime.
//! Run with: cargo bench --bench cuda_executor --features cuda
//!
//! Requires NVIDIA GPU with CUDA driver installed.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "cuda")]
use realizar::cuda::{CudaExecutor, CudaKernels, KernelType};

// ============================================================================
// PTX Generation Benchmarks (no GPU required)
// ============================================================================

#[cfg(feature = "cuda")]
fn bench_ptx_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ptx_generation");
    let kernels = CudaKernels::new();

    // GEMM kernel generation
    for size in [256, 512, 1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::new("gemm_tiled", size), size, |b, &size| {
            let kernel_type = KernelType::GemmTiled {
                m: size,
                n: size,
                k: size,
                tile_size: 32,
            };
            b.iter(|| black_box(kernels.generate_ptx(&kernel_type)))
        });
    }

    // Softmax kernel generation
    for dim in [256, 1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::new("softmax", dim), dim, |b, &dim| {
            let kernel_type = KernelType::Softmax { dim };
            b.iter(|| black_box(kernels.generate_ptx(&kernel_type)))
        });
    }

    // Q4_K kernel generation
    for k in [1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::new("q4k_gemm", k), k, |b, &k| {
            let kernel_type = KernelType::QuantizedGemm { m: 1, n: 4096, k };
            b.iter(|| black_box(kernels.generate_ptx(&kernel_type)))
        });
    }

    // FlashAttention kernel generation
    for seq_len in [512, 1024, 2048].iter() {
        group.bench_with_input(
            BenchmarkId::new("attention", seq_len),
            seq_len,
            |b, &seq_len| {
                let kernel_type = KernelType::Attention {
                    seq_len,
                    head_dim: 64,
                    causal: true,
                };
                b.iter(|| black_box(kernels.generate_ptx(&kernel_type)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// CPU vs GPU GEMM Benchmarks (requires GPU)
// ============================================================================

#[cfg(feature = "cuda")]
fn bench_gemm_gpu(c: &mut Criterion) {
    // Skip if no CUDA available
    if !CudaExecutor::is_available() {
        println!("CUDA not available, skipping GPU benchmarks");
        return;
    }

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            println!("Failed to create CUDA executor: {}", e);
            return;
        },
    };

    // Print GPU info
    if let Ok(name) = executor.device_name() {
        println!("GPU: {}", name);
    }
    if let Ok((free, total)) = executor.memory_info() {
        println!(
            "GPU Memory: {:.1} GB free / {:.1} GB total",
            free as f64 / 1e9,
            total as f64 / 1e9
        );
    }

    let mut group = c.benchmark_group("gemm_gpu");

    for size in [128, 256, 512, 1024].iter() {
        let m = *size;
        let n = *size;
        let _k = *size;
        let elements = (m * n) as u64;

        group.throughput(Throughput::Elements(elements));

        // GPU GEMM
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |b, &size| {
            let a = vec![1.0f32; (size * size) as usize];
            let b_mat = vec![1.0f32; (size * size) as usize];
            let mut c = vec![0.0f32; (size * size) as usize];

            b.iter(|| {
                executor
                    .gemm(&a, &b_mat, &mut c, size, size, size)
                    .expect("test");
                black_box(&c);
            })
        });

        // CPU GEMM (naive) for comparison
        group.bench_with_input(BenchmarkId::new("cpu_naive", size), size, |b, &size| {
            let a = vec![1.0f32; (size * size) as usize];
            let b_mat = vec![1.0f32; (size * size) as usize];
            let mut c = vec![0.0f32; (size * size) as usize];
            let size = size as usize;

            b.iter(|| {
                for i in 0..size {
                    for j in 0..size {
                        let mut sum = 0.0f32;
                        for kk in 0..size {
                            sum += a[i * size + kk] * b_mat[kk * size + j];
                        }
                        c[i * size + j] = sum;
                    }
                }
                black_box(&c);
            })
        });
    }

    group.finish();
}

// ============================================================================
// Softmax Benchmarks (requires GPU)
// ============================================================================

#[cfg(feature = "cuda")]
fn bench_softmax_gpu(c: &mut Criterion) {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("softmax_gpu");

    for dim in [256, 1024, 4096].iter() {
        let dim = *dim as usize;
        group.throughput(Throughput::Elements(dim as u64));

        // GPU Softmax
        group.bench_with_input(BenchmarkId::new("gpu", dim), &dim, |b, &dim| {
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();

            b.iter(|| {
                executor.softmax(&mut data).expect("test");
                black_box(&data);
            })
        });

        // CPU Softmax for comparison
        group.bench_with_input(BenchmarkId::new("cpu", dim), &dim, |b, &dim| {
            let mut data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();

            b.iter(|| {
                // Numerically stable softmax
                let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = data.iter().map(|x| (x - max).exp()).sum();
                for x in &mut data {
                    *x = (*x - max).exp() / sum;
                }
                black_box(&data);
            })
        });
    }

    group.finish();
}

// ============================================================================
// CUDA Availability Check
// ============================================================================

#[cfg(feature = "cuda")]
fn bench_cuda_availability(c: &mut Criterion) {
    c.bench_function("cuda_is_available", |b| {
        b.iter(|| black_box(CudaExecutor::is_available()))
    });

    c.bench_function("cuda_device_count", |b| {
        b.iter(|| black_box(CudaExecutor::num_devices()))
    });
}

// ============================================================================
// Criterion Groups
// ============================================================================

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_ptx_generation,
    bench_cuda_availability,
    bench_gemm_gpu,
    bench_softmax_gpu,
);

#[cfg(feature = "cuda")]
criterion_main!(benches);

// Fallback when cuda feature not enabled
#[cfg(not(feature = "cuda"))]
fn main() {
    println!(
        "CUDA feature not enabled. Run with: cargo bench --bench cuda_executor --features cuda"
    );
}
