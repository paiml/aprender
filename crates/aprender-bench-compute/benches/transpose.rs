//! Transpose benchmarks: aprender Tensor::transpose vs ndarray.
//!
//! Transpose of K^T runs once per attention layer per token. For 7B (32 layers):
//! 32 transposes per token. At 100 tok/s → 3200 transposes/s.
//!
//! aprender uses a naive double loop. ndarray uses optimized cache-oblivious
//! transpose. At LLM dimensions (2048×128) the cache behavior matters.
//!
//! ## Throughput metric
//!
//! Reports elements/s (rows × cols elements moved per transpose).

use aprender::autograd::Tensor;
use aprender_bench_compute::deterministic_f32;
use aprender_bench_compute::sizes::TRANSPOSE_SIZES;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_transpose_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_aprender");

    for &(rows, cols) in TRANSPOSE_SIZES {
        let data = deterministic_f32(rows * cols);
        let t = Tensor::new(&data, &[rows, cols]);
        let label = format!("{rows}x{cols}");
        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(BenchmarkId::new("tensor", &label), &label, |bench, _| {
            bench.iter(|| black_box(t.transpose()));
        });
    }

    group.finish();
}

fn bench_transpose_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_ndarray");

    for &(rows, cols) in TRANSPOSE_SIZES {
        let data = deterministic_f32(rows * cols);
        let arr = ndarray::Array2::from_shape_vec((rows, cols), data).unwrap();
        let label = format!("{rows}x{cols}");
        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(BenchmarkId::new("ndarray", &label), &label, |bench, _| {
            // ndarray .t() is a view (zero-cost), so we force a contiguous copy
            // to match aprender's semantics (new allocation + data copy)
            bench.iter(|| black_box(arr.t().to_owned()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_transpose_aprender, bench_transpose_ndarray,);
criterion_main!(benches);
