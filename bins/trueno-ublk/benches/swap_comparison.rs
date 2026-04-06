//! BENCH-001 v2.1.0: Swap Technology Comparison Benchmarks
//!
//! Scientific falsification benchmarks comparing compression technologies.
//! Uses criterion for statistical rigor with regression detection.
//!
//! References:
//! - Yi et al. (2005) "Improving Computer Architecture Simulation Methodology"
//! - Gregg, B. (2016) "The Flame Graph" - CACM 59(6):48-57
//!
//! Run with: cargo bench --bench swap_comparison

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::time::Duration;
use trueno_zram_core::{Algorithm, CompressorBuilder};

const PAGE_SIZE: usize = 4096;

/// Data patterns for benchmarking (per BENCH-001 spec)
#[derive(Clone, Copy)]
enum Workload {
    /// W1: All zeros - 100% compressible
    Zeros,
    /// W2: Text-like - ~70% compressible
    Text,
    /// W3: Mixed - ~35% compressible
    Mixed,
    /// W4: Random - 0% compressible
    Random,
}

impl Workload {
    fn name(&self) -> &'static str {
        match self {
            Workload::Zeros => "W1-ZEROS",
            Workload::Text => "W2-TEXT",
            Workload::Mixed => "W3-MIXED",
            Workload::Random => "W4-RANDOM",
        }
    }

    fn generate_page(&self) -> [u8; PAGE_SIZE] {
        let mut data = [0u8; PAGE_SIZE];
        match self {
            Workload::Zeros => {
                // Already all zeros
            }
            Workload::Text => {
                // Simulated text: repetitive ASCII patterns
                for i in 0..PAGE_SIZE {
                    data[i] = (32 + (i % 95)) as u8; // Printable ASCII
                }
            }
            Workload::Mixed => {
                // 50% compressible, 50% random
                let half = PAGE_SIZE / 2;
                for i in 0..half {
                    data[i] = (i % 256) as u8;
                }
                let mut seed: u64 = 0xCAFEBABE;
                for i in half..PAGE_SIZE {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    data[i] = (seed >> 33) as u8;
                }
            }
            Workload::Random => {
                // Incompressible random data
                let mut seed: u64 = 0xDEADBEEF;
                for i in 0..PAGE_SIZE {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    data[i] = (seed >> 33) as u8;
                }
            }
        }
        data
    }
}

/// Benchmark LZ4 compression across workloads
fn bench_lz4_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_compress");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .expect("Failed to create compressor");

    for workload in [Workload::Zeros, Workload::Text, Workload::Mixed, Workload::Random] {
        let page = workload.generate_page();

        group.throughput(Throughput::Bytes(PAGE_SIZE as u64));
        group.bench_with_input(BenchmarkId::new(workload.name(), PAGE_SIZE), &page, |b, page| {
            b.iter(|| {
                let result = compressor.compress(black_box(page));
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark LZ4 decompression across workloads
fn bench_lz4_decompression(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz4_decompress");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .expect("Failed to create compressor");

    for workload in [Workload::Zeros, Workload::Text, Workload::Mixed, Workload::Random] {
        let page = workload.generate_page();
        let compressed = compressor.compress(&page).expect("Compression failed");

        group.throughput(Throughput::Bytes(PAGE_SIZE as u64));
        group.bench_with_input(
            BenchmarkId::new(workload.name(), PAGE_SIZE),
            &compressed,
            |b, compressed_page| {
                b.iter(|| {
                    let result = compressor.decompress(black_box(compressed_page));
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark ZSTD compression across workloads
fn bench_zstd_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_compress");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Zstd { level: 1 })
        .build()
        .expect("Failed to create compressor");

    for workload in [Workload::Zeros, Workload::Text, Workload::Mixed, Workload::Random] {
        let page = workload.generate_page();

        group.throughput(Throughput::Bytes(PAGE_SIZE as u64));
        group.bench_with_input(BenchmarkId::new(workload.name(), PAGE_SIZE), &page, |b, page| {
            b.iter(|| {
                let result = compressor.compress(black_box(page));
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark batch compression throughput (P6-BATCH pattern)
fn bench_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(15));

    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .expect("Failed to create compressor");

    // Batch sizes corresponding to P6-BATCH (iodepth=128)
    for batch_size in [32, 64, 128, 256] {
        let pages: Vec<[u8; PAGE_SIZE]> =
            (0..batch_size).map(|_| Workload::Text.generate_page()).collect();

        group.throughput(Throughput::Bytes((batch_size * PAGE_SIZE) as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_compress", batch_size),
            &pages,
            |b, pages| {
                b.iter(|| {
                    for page in pages {
                        let _ = black_box(compressor.compress(black_box(page)));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark compression ratio (not timing, but ratio measurement)
fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");
    group.sample_size(10); // Ratio is deterministic, few samples needed

    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .expect("Failed to create compressor");

    for workload in [Workload::Zeros, Workload::Text, Workload::Mixed, Workload::Random] {
        let page = workload.generate_page();
        let compressed = compressor.compress(&page).expect("Compression failed");
        let ratio = PAGE_SIZE as f64 / compressed.data.len() as f64;

        // Log ratio for analysis (criterion will capture this)
        println!(
            "[RATIO] {}: {:.2}x ({} -> {} bytes)",
            workload.name(),
            ratio,
            PAGE_SIZE,
            compressed.data.len()
        );

        group.bench_with_input(BenchmarkId::new(workload.name(), "ratio"), &page, |b, page| {
            b.iter(|| {
                let compressed = compressor.compress(black_box(page)).expect("fail");
                black_box(compressed.data.len())
            })
        });
    }

    group.finish();
}

criterion_group!(
    name = swap_benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets =
        bench_lz4_compression,
        bench_lz4_decompression,
        bench_zstd_compression,
        bench_batch_throughput,
        bench_compression_ratio
);

criterion_main!(swap_benches);
