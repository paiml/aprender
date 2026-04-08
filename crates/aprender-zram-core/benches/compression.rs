//! Compression benchmarks using Criterion.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use trueno_zram_core::{lz4, zstd, PAGE_SIZE};

fn generate_test_data() -> Vec<[u8; PAGE_SIZE]> {
    let mut pages = Vec::with_capacity(1000);

    // Zero pages (highly compressible)
    for _ in 0..250 {
        pages.push([0u8; PAGE_SIZE]);
    }

    // Repeating pattern
    for i in 0..250 {
        let mut page = [0u8; PAGE_SIZE];
        let pattern = [(i % 256) as u8, ((i + 1) % 256) as u8];
        for (j, byte) in page.iter_mut().enumerate() {
            *byte = pattern[j % 2];
        }
        pages.push(page);
    }

    // Sequential
    for _ in 0..250 {
        let mut page = [0u8; PAGE_SIZE];
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        pages.push(page);
    }

    // Pseudo-random (hard to compress)
    let mut state = 12345u64;
    for _ in 0..250 {
        let mut page = [0u8; PAGE_SIZE];
        for byte in &mut page {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }
        pages.push(page);
    }

    pages
}

fn benchmark_lz4_compress(c: &mut Criterion) {
    let pages = generate_test_data();
    let total_bytes = pages.len() * PAGE_SIZE;

    let mut group = c.benchmark_group("LZ4");
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("compress", |b| {
        b.iter(|| {
            for page in &pages {
                black_box(lz4::compress(page).unwrap());
            }
        });
    });

    // Pre-compress for decompression benchmark
    let compressed: Vec<_> = pages.iter().map(|p| lz4::compress(p).unwrap()).collect();

    group.bench_function("decompress_scalar", |b| {
        b.iter(|| {
            let mut output = [0u8; PAGE_SIZE];
            for data in &compressed {
                black_box(lz4::decompress(data, &mut output).unwrap());
            }
        });
    });

    group.bench_function("decompress_simd", |b| {
        b.iter(|| {
            let mut output = [0u8; PAGE_SIZE];
            for data in &compressed {
                black_box(lz4::decompress_simd(data, &mut output).unwrap());
            }
        });
    });

    group.finish();
}

fn benchmark_zstd_compress(c: &mut Criterion) {
    let pages = generate_test_data();
    let total_bytes = pages.len() * PAGE_SIZE;

    let mut group = c.benchmark_group("ZSTD");
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("compress_level1", |b| {
        b.iter(|| {
            for page in &pages {
                black_box(zstd::compress(page, 1).unwrap());
            }
        });
    });

    group.bench_function("compress_level3", |b| {
        b.iter(|| {
            for page in &pages {
                black_box(zstd::compress(page, 3).unwrap());
            }
        });
    });

    // Pre-compress for decompression benchmark
    let compressed: Vec<_> = pages
        .iter()
        .map(|p| zstd::compress(p, 3).unwrap())
        .collect();

    group.bench_function("decompress", |b| {
        b.iter(|| {
            let mut output = [0u8; PAGE_SIZE];
            for data in &compressed {
                black_box(zstd::decompress(data, &mut output).unwrap());
            }
        });
    });

    group.finish();
}

fn benchmark_compression_ratio(c: &mut Criterion) {
    let pages = generate_test_data();

    let mut group = c.benchmark_group("Ratio");

    // Just measure, don't benchmark speed
    group.bench_function("lz4_ratio_calculation", |b| {
        b.iter(|| {
            let mut total_in = 0usize;
            let mut total_out = 0usize;
            for page in &pages {
                total_in += PAGE_SIZE;
                total_out += lz4::compress(page).unwrap().len();
            }
            black_box(total_in as f64 / total_out as f64)
        });
    });

    group.finish();
}

fn benchmark_lz4_compressible(c: &mut Criterion) {
    // Test with only highly compressible data (best-case performance)
    let mut pages = Vec::with_capacity(1000);

    // Zero pages
    for _ in 0..500 {
        pages.push([0u8; PAGE_SIZE]);
    }

    // Repeating pattern
    for i in 0..500 {
        let mut page = [0u8; PAGE_SIZE];
        let pattern = [(i % 256) as u8, ((i + 1) % 256) as u8];
        for (j, byte) in page.iter_mut().enumerate() {
            *byte = pattern[j % 2];
        }
        pages.push(page);
    }

    let total_bytes = pages.len() * PAGE_SIZE;
    let compressed: Vec<_> = pages.iter().map(|p| lz4::compress(p).unwrap()).collect();

    let mut group = c.benchmark_group("LZ4_Compressible");
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("decompress_scalar", |b| {
        b.iter(|| {
            let mut output = [0u8; PAGE_SIZE];
            for data in &compressed {
                black_box(lz4::decompress(data, &mut output).unwrap());
            }
        });
    });

    group.bench_function("decompress_simd", |b| {
        b.iter(|| {
            let mut output = [0u8; PAGE_SIZE];
            for data in &compressed {
                black_box(lz4::decompress_simd(data, &mut output).unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_lz4_compress,
    benchmark_zstd_compress,
    benchmark_compression_ratio,
    benchmark_lz4_compressible
);
criterion_main!(benches);
