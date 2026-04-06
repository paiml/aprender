#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! GPU Performance Benchmarks
//!
//! **EMPIRICAL FINDINGS (RTX 4090, 2025-11-23):**
//!
//! - GPU FAILS for vector operations due to 3.5ms fixed transfer overhead
//! - GPU WINS for matrix operations (81x speedup on 1000x1000 matmul)
//! - SIMD VASTLY SUPERIOR for vector ops (2000x+ faster than GPU)
//!
//! See: docs/gpu-benchmark-report-2025-11-23.md for complete analysis

#![cfg(feature = "gpu")]

mod activation_benches;
mod activation_nonlinear_benches;
mod matrix_benches;
mod softmax_benches;
mod vector_benches;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    vector_benches::bench_gpu_vec_add,
    vector_benches::bench_gpu_dot,
    matrix_benches::bench_gpu_matmul,
    activation_benches::bench_gpu_relu,
    activation_benches::bench_gpu_leaky_relu,
    activation_benches::bench_gpu_elu,
    activation_benches::bench_gpu_clip,
    activation_nonlinear_benches::bench_gpu_sigmoid,
    activation_nonlinear_benches::bench_gpu_tanh,
    activation_nonlinear_benches::bench_gpu_swish,
    activation_nonlinear_benches::bench_gpu_gelu,
    softmax_benches::bench_gpu_softmax,
    softmax_benches::bench_gpu_log_softmax
);
criterion_main!(benches);
