//! Benchmarks for content-addressed storage.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use pacha::storage::ContentAddress;

fn bench_content_address_from_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_address");

    for size in [64, 256, 1024, 4096, 16384, 65536].iter() {
        let data: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("from_bytes", size), &data, |b, data| {
            b.iter(|| ContentAddress::from_bytes(black_box(data)))
        });
    }

    group.finish();
}

fn bench_content_address_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");

    for size in [1024, 16384, 65536].iter() {
        let data: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();
        let addr = ContentAddress::from_bytes(&data);

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("verify", size), &data, |b, data| {
            b.iter(|| addr.verify(black_box(data)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_content_address_from_bytes, bench_content_address_verify);
criterion_main!(benches);
