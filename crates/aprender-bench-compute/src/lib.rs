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
//! # Measured results (GH-379, GH-380–387)
//!
//! | Operation | Size | aprender | ndarray | Ratio | Issue |
//! |-----------|------|----------|---------|-------|-------|
//! | Matvec | 1x4096x11008 | 880 M | 3.0 G | 0.29x | #380 |
//! | RMSNorm | 4096 | 574 M | 2.65 G | 0.22x | #381 |
//! | LayerNorm | 4096 | 392 M | 1.66 G | 0.24x | #382 |
//! | ReLU | 11008 | 8.2 G | 17.8 G | 0.46x | #383 |
//! | Softmax | 32000 | 198 M | 307 M | 0.64x | #384 |
//! | Add | 4096 | 5.7 G | 10.6 G | 0.54x | #387 |
//! | Mul scalar | 4096 | 6.5 G | 16.2 G | 0.40x | #387 |
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
