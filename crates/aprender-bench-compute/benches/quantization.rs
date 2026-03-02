//! Quantization benchmarks: aprender Q4_0/Q8_0 quantize + dequantize.
//!
//! Quantization is the key to fitting large models in memory. The dequant
//! path runs on every matmul during inference — it must be fast.
//!
//! ## Reference
//!
//! No ndarray equivalent exists for quantized formats. Instead we benchmark:
//! - **Quantize throughput**: How fast can we compress f32 → Q4_0/Q8_0?
//! - **Dequantize throughput**: How fast can we decompress Q4_0/Q8_0 → f32?
//! - **Round-trip**: quantize + dequantize (measures total overhead)
//!
//! The "reference" is the throughput ceiling: memcpy of the same f32 data.
//! This shows how close quantization is to pure memory bandwidth.
//!
//! ## Throughput metric
//!
//! Reports elements/s (f32 values processed).

use aprender::format::quantize::{dequantize, quantize, QuantType};
use aprender_bench_compute::sizes::QUANT_SIZES;
use aprender_bench_compute::deterministic_f32;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_quantize_q8_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_Q8_0");

    for &size in QUANT_SIZES {
        let data = deterministic_f32(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| black_box(quantize(&data, &[size], QuantType::Q8_0)));
        });
    }

    group.finish();
}

fn bench_quantize_q4_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_Q4_0");

    for &size in QUANT_SIZES {
        let data = deterministic_f32(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| black_box(quantize(&data, &[size], QuantType::Q4_0)));
        });
    }

    group.finish();
}

fn bench_dequantize_q8_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize_Q8_0");

    for &size in QUANT_SIZES {
        let data = deterministic_f32(size);
        let q = quantize(&data, &[size], QuantType::Q8_0).expect("quantize Q8_0");
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| black_box(dequantize(&q)));
        });
    }

    group.finish();
}

fn bench_dequantize_q4_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequantize_Q4_0");

    for &size in QUANT_SIZES {
        let data = deterministic_f32(size);
        let q = quantize(&data, &[size], QuantType::Q4_0).expect("quantize Q4_0");
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| black_box(dequantize(&q)));
        });
    }

    group.finish();
}

fn bench_memcpy_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("memcpy_baseline");

    for &size in QUANT_SIZES {
        let data = deterministic_f32(size);
        group.throughput(Throughput::Bytes((size * 4) as u64));

        group.bench_with_input(BenchmarkId::new("f32_clone", size), &size, |bench, _| {
            bench.iter(|| black_box(data.clone()));
        });
    }

    group.finish();
}

fn bench_roundtrip_q8_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_Q8_0");

    for &size in QUANT_SIZES {
        let data = deterministic_f32(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| {
                let q = quantize(&data, &[size], QuantType::Q8_0).unwrap();
                black_box(dequantize(&q))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quantize_q8_0,
    bench_quantize_q4_0,
    bench_dequantize_q8_0,
    bench_dequantize_q4_0,
    bench_memcpy_baseline,
    bench_roundtrip_q8_0,
);
criterion_main!(benches);
