//! Shared utilities for aprender-bench-compute benchmarks.
//!
//! Provides deterministic data generators and canonical LLM sizes for
//! head-to-head benchmarks of aprender compute primitives vs ndarray.
//!
//! # Design
//!
//! Follows the `aprender-bench-tokenizer` pattern:
//! - Head-to-head A/B comparison against a reference implementation
//! - Criterion with `Throughput` metrics
//! - Canonical LLM-relevant sizes (not arbitrary micro sizes)
//! - Scaling analysis across input dimensions
//!
//! # Reference: ndarray (pure Rust)
//!
//! ndarray without BLAS is pure Rust — same constraint as aprender/trueno.
//! This makes the comparison fair: both are Rust-native, no external BLAS.
//!
//! # LLM size targets
//!
//! | Model | Hidden | FFN | Heads | KV Heads |
//! |-------|--------|-----|-------|----------|
//! | 1.5B  | 1536   | 8960 | 12   | 2        |
//! | 7B    | 4096   | 11008 | 32  | 32       |
//! | 13B   | 5120   | 13824 | 40  | 40       |
//!
//! # Measured results (GH-379, GH-380–389)
//!
//! | Operation | Size | aprender | ndarray | Ratio | Issue |
//! |-----------|------|----------|---------|-------|-------|
//! | Matvec | 1x4096x11008 | 880 M | 3.0 G | 0.29x | #380 |
//! | RMSNorm | 4096 | 1.33 G | 2.62 G | 0.51x | #381 |
//! | LayerNorm | 4096 | 704 M | 1.67 G | 0.42x | #382 |
//! | ReLU | 11008 | 11.4 G | 18.0 G | 0.64x | #383 |
//! | Softmax | 32000 | 281 M | 312 M | 0.90x | #384 |
//! | Add | 4096 | 8.3 G | 10.9 G | 0.76x | #387 |
//! | Mul scalar | 4096 | 10.3 G | 15.3 G | 0.67x | #387 |
//! | Transpose | 2048×128 | 459M | 6.79G | 0.068x | #388 |
//! | Transpose | 4096² | 241M | 500M | 0.48x | #388 |
//! | RoPE (1tok) | 1×32×128 | 2.1G | 916M | 2.3x ✓ | #389 ✓ |
//! | RoPE (prefill) | 512×32×128 | 2.1G | 840M | 2.5x ✓ | #389 ✓ |
//! | GELU | 11008 | 135 M | 134 M | parity | — |
//! | SiLU | 11008 | 470 M | 468 M | parity | — |
//! | Matmul (sq) | 1024³ | 71 G | 40 G | 1.8x ✓ | — |

pub mod sizes;

use aprender::autograd::Tensor;

/// Generate a deterministic f32 vector of given length.
///
/// Uses a simple LCG-style pattern for reproducibility across runs.
/// Values in [-0.5, 0.5] to match typical weight distributions.
#[must_use]
pub fn deterministic_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
        .collect()
}

/// Generate a deterministic aprender Tensor.
#[must_use]
pub fn deterministic_tensor(rows: usize, cols: usize) -> Tensor {
    let data = deterministic_f32(rows * cols);
    Tensor::new(&data, &[rows, cols])
}

/// Generate a deterministic ndarray Array2.
#[must_use]
pub fn deterministic_ndarray(rows: usize, cols: usize) -> ndarray::Array2<f32> {
    let data = deterministic_f32(rows * cols);
    ndarray::Array2::from_shape_vec((rows, cols), data).expect("shape should match data length")
}

/// Generate a deterministic 1D ndarray.
#[must_use]
pub fn deterministic_ndarray_1d(len: usize) -> ndarray::Array1<f32> {
    let data = deterministic_f32(len);
    ndarray::Array1::from_vec(data)
}
