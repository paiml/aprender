#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Benchmarks for Tiling Compute Blocks
//!
//! Profiles:
//! - TiledQ4KMatvec::execute_scalar hot path
//! - f16_to_f32 conversion overhead
//! - Index calculation overhead

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::tiling::{
    pack_a_index, pack_b_index, swizzle_index, TcbGeometry, TcbIndexCalculator, TiledQ4KMatvec,
    TilingConfig, Q4K_SUPERBLOCK_BYTES,
};

/// Benchmark Q4K MatVec scalar execution
fn bench_q4k_matvec_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4k_matvec_scalar");

    for size in [256, 1024, 4096] {
        let matvec = TiledQ4KMatvec::new(size, size);
        let weights = vec![0u8; matvec.total_superblocks() * Q4K_SUPERBLOCK_BYTES];
        let input = vec![1.0f32; size];
        let mut output = vec![0.0f32; size];

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                matvec.execute_scalar(
                    black_box(&weights),
                    black_box(&input),
                    black_box(&mut output),
                );
            });
        });

        // Benchmark parallel version
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, _| {
            b.iter(|| {
                matvec.execute_parallel(
                    black_box(&weights),
                    black_box(&input),
                    black_box(&mut output),
                );
            });
        });
    }
    group.finish();
}

/// Benchmark arithmetic intensity calculation
fn bench_arithmetic_intensity(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcb_geometry");

    let geometries = [
        ("micro_4x8x256", TcbGeometry::new(4, 8, 256)),
        ("midi_64x64x256", TcbGeometry::new(64, 64, 256)),
        ("macro_512x512x512", TcbGeometry::new(512, 512, 512)),
    ];

    for (name, geom) in geometries {
        group.bench_function(BenchmarkId::new("arithmetic_intensity", name), |b| {
            b.iter(|| black_box(geom).arithmetic_intensity());
        });

        group.bench_function(BenchmarkId::new("total_flops", name), |b| {
            b.iter(|| black_box(geom).total_flops());
        });

        group.bench_function(BenchmarkId::new("fits_in_cache", name), |b| {
            b.iter(|| black_box(geom).fits_in_cache(256 * 1024));
        });
    }
    group.finish();
}

/// Benchmark index calculator operations
fn bench_index_calculator(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_calculator");

    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 4096, 4096, 4096);

    group.bench_function("macro_tile_offset", |b| {
        b.iter(|| {
            for i in 0..256 {
                black_box(calc.macro_tile_offset(black_box(i)));
            }
        });
    });

    group.bench_function("is_boundary_tile", |b| {
        b.iter(|| {
            for i in 0..256 {
                black_box(calc.is_boundary_tile(black_box(i)));
            }
        });
    });

    group.bench_function("block_to_linear_offset", |b| {
        b.iter(|| {
            for i in 0..256 {
                black_box(calc.block_to_linear_offset(black_box(i), 4096));
            }
        });
    });

    group.bench_function("a_offset", |b| {
        b.iter(|| {
            for i in 0..16 {
                for j in 0..16 {
                    black_box(calc.a_offset(black_box(i), black_box(j)));
                }
            }
        });
    });

    group.finish();
}

/// Benchmark memory layout helpers
fn bench_memory_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout");

    group.bench_function("pack_a_index_1k", |b| {
        b.iter(|| {
            for row in 0..32 {
                for col in 0..32 {
                    black_box(pack_a_index(row, col, 4, 256, 64));
                }
            }
        });
    });

    group.bench_function("pack_b_index_1k", |b| {
        b.iter(|| {
            for row in 0..32 {
                for col in 0..32 {
                    black_box(pack_b_index(row, col, 8, 64, 64));
                }
            }
        });
    });

    group.bench_function("swizzle_index_1k", |b| {
        b.iter(|| {
            for i in 0..1024 {
                black_box(swizzle_index(i));
            }
        });
    });

    group.finish();
}

/// Benchmark TilingConfig validation
fn bench_config_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiling_config");

    let configs = [
        ("gpu_q4k_matvec", TilingConfig::gpu_q4k_matvec()),
        ("cpu_avx512_matmul", TilingConfig::cpu_avx512_matmul()),
        ("cpu_avx2_matmul", TilingConfig::cpu_avx2_matmul()),
    ];

    for (name, config) in configs {
        group.bench_function(BenchmarkId::new("validate", name), |b| {
            b.iter(|| black_box(&config).validate());
        });

        group.bench_function(BenchmarkId::new("num_macro_tiles", name), |b| {
            b.iter(|| black_box(&config).num_macro_tiles(4096, 4096));
        });

        group.bench_function(BenchmarkId::new("midi_tiles_per_macro", name), |b| {
            b.iter(|| black_box(&config).midi_tiles_per_macro());
        });
    }

    group.finish();
}

/// Benchmark Q4K statistics calculation
fn bench_q4k_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4k_stats");

    for size in [1024, 4096, 8192] {
        let matvec = TiledQ4KMatvec::new(size, size);

        group.bench_function(BenchmarkId::new("stats", size), |b| {
            b.iter(|| black_box(&matvec).stats());
        });

        group.bench_function(BenchmarkId::new("optimal_parallel_rows", size), |b| {
            b.iter(|| black_box(&matvec).optimal_parallel_rows(256 * 1024));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_q4k_matvec_scalar,
    bench_arithmetic_intensity,
    bench_index_calculator,
    bench_memory_layout,
    bench_config_validation,
    bench_q4k_stats,
);

criterion_main!(benches);
