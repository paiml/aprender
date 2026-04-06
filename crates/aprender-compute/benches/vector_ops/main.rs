#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Benchmarks for Vector operations comparing Scalar vs SSE2 vs AVX2 backends
//!
//! This benchmark suite verifies SIMD performance improvements across backends.
//!
//! # Benchmark Methodology
//!
//! - Tests multiple vector sizes: 100, 1000, 10000 elements
//! - Compares Scalar, SSE2, and AVX2 backends explicitly
//! - Uses Criterion for statistical analysis
//! - Each benchmark measures throughput (elements/second)
//!
//! # Performance Goals
//!
//! Expected SSE2 speedup over Scalar:
//! - Small vectors (100):   1.5-2x (some overhead from SIMD setup)
//! - Medium vectors (1K):   3-4x (optimal SIMD utilization)
//! - Large vectors (10K+):  3-4x (memory bandwidth limited)
//!
//! Expected AVX2 speedup over SSE2:
//! - Element-wise ops:      2x (8-wide vs 4-wide SIMD)
//! - Dot product:           2x+ (FMA acceleration)

mod activations;
mod arithmetic;
mod math;
mod reductions;

use criterion::{criterion_group, criterion_main};

/// Generate test data for benchmarks
pub fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.5).collect()
}

criterion_group!(
    benches,
    arithmetic::bench_add,
    arithmetic::bench_sub,
    arithmetic::bench_mul,
    arithmetic::bench_scale,
    arithmetic::bench_div,
    arithmetic::bench_fma,
    reductions::bench_dot,
    reductions::bench_sum,
    reductions::bench_max,
    reductions::bench_min,
    reductions::bench_argmax,
    reductions::bench_argmin,
    activations::bench_relu,
    activations::bench_sigmoid,
    activations::bench_gelu,
    activations::bench_swish,
    activations::bench_tanh,
    activations::bench_softmax,
    activations::bench_log_softmax,
    activations::bench_clip,
    math::bench_norm_l1,
    math::bench_norm_l2,
    math::bench_norm_linf,
    math::bench_abs,
    math::bench_exp,
    math::bench_ln,
    math::bench_log2,
    math::bench_log10,
    math::bench_sqrt,
    math::bench_recip,
);
criterion_main!(benches);
