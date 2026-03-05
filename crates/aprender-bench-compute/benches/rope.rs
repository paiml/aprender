//! Rotary Position Embedding (RoPE) benchmarks: aprender vs scalar reference.
//!
//! RoPE runs once per layer per token on both Q and K:
//! For 7B (32 layers): 64 RoPE applications per token.
//! At 100 tok/s → 6400 RoPE/s. Must be fast.
//!
//! aprender's implementation is scalar (nested loops over batch × seq × heads × half_dim).
//! Reference: same math using ndarray vectorized ops.
//!
//! ## Throughput metric
//!
//! Reports elements/s (seq_len × num_heads × head_dim elements rotated).

use aprender::autograd::Tensor;
use aprender::nn::RotaryPositionEmbedding;
use aprender_bench_compute::deterministic_f32;
use aprender_bench_compute::sizes::ROPE_SIZES;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_rope_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_aprender");

    for &(seq_len, num_heads, head_dim) in ROPE_SIZES {
        let total = seq_len * num_heads * head_dim;
        let data = deterministic_f32(total);
        let x = Tensor::new(&data, &[1, seq_len, num_heads, head_dim]);
        let rope = RotaryPositionEmbedding::new(head_dim, seq_len.max(2048));
        let position_ids: Vec<usize> = (0..seq_len).collect();

        let label = format!("{seq_len}x{num_heads}x{head_dim}");
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::new("trueno", &label), &label, |bench, _| {
            bench.iter(|| black_box(rope.apply(&x, &position_ids)));
        });
    }

    group.finish();
}

fn bench_rope_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_ndarray");

    for &(seq_len, num_heads, head_dim) in ROPE_SIZES {
        let total = seq_len * num_heads * head_dim;
        let half_dim = head_dim / 2;

        // Precompute cos/sin cache (same as aprender)
        let base: f32 = 10000.0;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let max_pos = seq_len.max(2048);
        let cos_cache: Vec<f32> = (0..max_pos)
            .flat_map(|pos| inv_freq.iter().map(move |&freq| (pos as f32 * freq).cos()))
            .collect();
        let sin_cache: Vec<f32> = (0..max_pos)
            .flat_map(|pos| inv_freq.iter().map(move |&freq| (pos as f32 * freq).sin()))
            .collect();

        let data = deterministic_f32(total);
        let label = format!("{seq_len}x{num_heads}x{head_dim}");
        group.throughput(Throughput::Elements(total as u64));

        group.bench_with_input(BenchmarkId::new("scalar", &label), &label, |bench, _| {
            bench.iter(|| {
                let mut output = vec![0.0f32; total];
                for s in 0..seq_len {
                    for h in 0..num_heads {
                        for i in 0..half_dim {
                            let cos_val = cos_cache[s * half_dim + i];
                            let sin_val = sin_cache[s * half_dim + i];

                            let idx1 = s * num_heads * head_dim + h * head_dim + 2 * i;
                            let idx2 = idx1 + 1;

                            let x1 = data[idx1];
                            let x2 = data[idx2];

                            output[idx1] = x1 * cos_val - x2 * sin_val;
                            output[idx2] = x1 * sin_val + x2 * cos_val;
                        }
                    }
                }
                black_box(&output);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rope_aprender, bench_rope_ndarray,);
criterion_main!(benches);
