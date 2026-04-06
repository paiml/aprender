//! PTX Generation and Driver Type Benchmarks
//!
//! Run with: cargo bench

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use trueno_gpu::driver::{DevicePtr, LaunchConfig};
use trueno_gpu::ptx::{PtxKernel, PtxModule, PtxType};

// ============================================================================
// PTX Generation Benchmarks
// ============================================================================

fn bench_module_emit(c: &mut Criterion) {
    let kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .param(PtxType::U32, "n");

    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .address_size(64)
        .add_kernel(kernel);

    c.bench_function("ptx_module_emit", |b| b.iter(|| black_box(module.emit())));
}

fn bench_kernel_builder(c: &mut Criterion) {
    c.bench_function("ptx_kernel_build", |b| {
        b.iter(|| {
            black_box(
                PtxKernel::new("test_kernel")
                    .param(PtxType::U64, "ptr1")
                    .param(PtxType::U64, "ptr2")
                    .param(PtxType::F32, "scale")
                    .param(PtxType::U32, "n"),
            )
        })
    });
}

fn bench_module_builder(c: &mut Criterion) {
    c.bench_function("ptx_module_build", |b| {
        b.iter(|| {
            black_box(
                PtxModule::new()
                    .version(8, 0)
                    .target("sm_70")
                    .address_size(64),
            )
        })
    });
}

// ============================================================================
// Driver Type Benchmarks
// ============================================================================

fn bench_launch_config(c: &mut Criterion) {
    let mut group = c.benchmark_group("launch_config");

    for size in [1024, 65536, 1_000_000, 16_777_216].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("linear", size), size, |b, &size| {
            b.iter(|| black_box(LaunchConfig::linear(size as u32, 256)))
        });
    }

    group.finish();
}

fn bench_device_ptr_offset(c: &mut Criterion) {
    let ptr: DevicePtr<f32> = unsafe { DevicePtr::from_raw(0x1000_0000) };

    c.bench_function("device_ptr_offset", |b| {
        b.iter(|| {
            let mut p = ptr;
            for i in 0..1000 {
                p = p.byte_offset(i * 4);
            }
            black_box(p)
        })
    });
}

fn bench_launch_config_total_threads(c: &mut Criterion) {
    let configs = [
        LaunchConfig::linear(1024, 256),
        LaunchConfig::grid_2d(16, 16, 16, 16),
        LaunchConfig {
            grid: (32, 32, 32),
            block: (8, 8, 8),
            shared_mem: 0,
        },
    ];

    c.bench_function("launch_config_total_threads", |b| {
        b.iter(|| {
            let mut total = 0u64;
            for config in &configs {
                total += config.total_threads();
            }
            black_box(total)
        })
    });
}

criterion_group!(
    benches,
    bench_module_emit,
    bench_kernel_builder,
    bench_module_builder,
    bench_launch_config,
    bench_device_ptr_offset,
    bench_launch_config_total_threads,
);
criterion_main!(benches);
