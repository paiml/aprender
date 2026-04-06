//! IOPS Benchmarks for trueno-ublk
//!
//! Measures I/O operations per second for the compression daemon.
//! Target: 800K+ IOPS (Ming Lei LPC 2022 reference: ublk can achieve 1.2M IOPS)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::Duration;
use trueno_ublk::{BatchConfig, BatchedPageStore};
use trueno_zram_core::Algorithm;

const PAGE_SIZE: usize = 4096;
const SECTORS_PER_PAGE: u64 = 8;

/// Create a test page store for benchmarking
fn create_test_store() -> Arc<BatchedPageStore> {
    Arc::new(BatchedPageStore::new(Algorithm::Lz4))
}

/// Create a test page store with custom batch threshold
fn create_test_store_with_threshold(batch_threshold: usize) -> Arc<BatchedPageStore> {
    let config = BatchConfig {
        batch_threshold,
        flush_timeout: Duration::from_millis(10),
        gpu_batch_size: 4000,
    };
    Arc::new(BatchedPageStore::with_config(Algorithm::Lz4, config))
}

/// Generate test data with specified compressibility
fn generate_test_data(compressibility: f64) -> [u8; PAGE_SIZE] {
    let mut data = [0u8; PAGE_SIZE];
    let compressible_bytes = (PAGE_SIZE as f64 * compressibility) as usize;

    // Compressible portion: repetitive pattern
    for i in 0..compressible_bytes {
        data[i] = (i % 256) as u8;
    }

    // Incompressible portion: pseudo-random
    let mut seed: u64 = 0xDEADBEEF;
    for i in compressible_bytes..PAGE_SIZE {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        data[i] = (seed >> 33) as u8;
    }

    data
}

/// Benchmark sequential writes
fn bench_sequential_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_writes");

    for pages in [100, 1000, 10000].iter() {
        let store = create_test_store();
        let data = generate_test_data(0.7); // 70% compressible

        group.throughput(Throughput::Elements(*pages as u64));
        group.bench_with_input(BenchmarkId::new("pages", pages), pages, |b, &pages| {
            b.iter(|| {
                for i in 0..pages {
                    let sector = (i as u64) * SECTORS_PER_PAGE;
                    let _ = black_box(store.store(sector, &data));
                }
                let _ = store.flush_batch();
            });
        });
    }

    group.finish();
}

/// Benchmark random writes
fn bench_random_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_writes");

    for pages in [100, 1000].iter() {
        let store = create_test_store();
        let data = generate_test_data(0.7);

        // Pre-generate random sectors
        let mut sectors: Vec<u64> = (0..*pages).map(|i| (i * SECTORS_PER_PAGE) as u64).collect();
        // Simple shuffle using deterministic pattern
        for i in (1..sectors.len()).rev() {
            let j = (i * 31337) % (i + 1);
            sectors.swap(i, j);
        }

        group.throughput(Throughput::Elements(*pages as u64));
        group.bench_with_input(BenchmarkId::new("pages", pages), &sectors, |b, sectors| {
            b.iter(|| {
                for &sector in sectors {
                    let _ = black_box(store.store(sector, &data));
                }
                let _ = store.flush_batch();
            });
        });
    }

    group.finish();
}

/// Benchmark sequential reads
fn bench_sequential_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_reads");

    for pages in [100, 1000, 10000].iter() {
        let store = create_test_store();
        let data = generate_test_data(0.7);

        // Pre-populate store
        for i in 0..*pages {
            let sector = (i as u64) * SECTORS_PER_PAGE;
            let _ = store.store(sector, &data);
        }
        let _ = store.flush_batch();

        group.throughput(Throughput::Elements(*pages as u64));
        group.bench_with_input(BenchmarkId::new("pages", pages), pages, |b, &pages| {
            let mut buffer = [0u8; PAGE_SIZE];
            b.iter(|| {
                for i in 0..pages {
                    let sector = (i as u64) * SECTORS_PER_PAGE;
                    let _ = black_box(store.load(sector, &mut buffer));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark different compression ratios
fn bench_compression_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");

    for ratio in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0].iter() {
        let store = create_test_store();
        let data = generate_test_data(*ratio);
        let pages = 1000;

        group.throughput(Throughput::Elements(pages));
        group.bench_with_input(
            BenchmarkId::new("ratio", format!("{:.0}%", ratio * 100.0)),
            &pages,
            |b, &pages| {
                b.iter(|| {
                    for i in 0..pages {
                        let sector = (i as u64) * SECTORS_PER_PAGE;
                        let _ = black_box(store.store(sector, &data));
                    }
                    let _ = store.flush_batch();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch threshold sizes
fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_threshold");

    for batch_threshold in [100, 500, 1000, 2000, 4000].iter() {
        let store = create_test_store_with_threshold(*batch_threshold);
        let data = generate_test_data(0.7);
        let pages = 1000u64;

        group.throughput(Throughput::Elements(pages));
        group.bench_with_input(
            BenchmarkId::new("threshold", batch_threshold),
            &pages,
            |b, &pages| {
                b.iter(|| {
                    for i in 0..pages {
                        let sector = (i as u64) * SECTORS_PER_PAGE;
                        let _ = black_box(store.store(sector, &data));
                    }
                    let _ = store.flush_batch();
                });
            },
        );
    }

    group.finish();
}

/// Configure Criterion for meaningful measurements
fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
}

criterion_group! {
    name = iops_benches;
    config = criterion_config();
    targets =
        bench_sequential_writes,
        bench_random_writes,
        bench_sequential_reads,
        bench_compression_ratios,
        bench_batch_sizes,
}

criterion_main!(iops_benches);
