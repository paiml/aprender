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
//! | Matvec (small) | 1x1536x1536 | 53.5 G | 2.57 G | 20.8x ✓ | #380 ✓ |
//! | Matvec (LLM) | 1x4096x11008 | 9.4 G | 2.93 G | 3.2x ✓ | #380 ✓ |
//! | Matvec (FFN) | 1x4096x4096 | 17.9 G | 2.05 G | 8.7x ✓ | #380 ✓ |
//! | RMSNorm | 4096 | 6.01 G | 2.54 G | 2.37x ✓ | #381 ✓ |
//! | LayerNorm | 4096 | 3.46 G | 1.70 G | 2.03x ✓ | #382 ✓ |
//! | ReLU | 11008 | 16.3 G | 17.4 G | 0.94x | #383 |
//! | Softmax | 32000 | 477 M | 311 M | 1.53x ✓ | #384 ✓ |
//! | Add | 4096 | 13.9 G | 11.8 G | 1.17x ✓ | #387 ✓ |
//! | Mul scalar | 4096 | 14.7 G | 17.2 G | 0.86x | #387 |
//! | Transpose | 2048×128 | 3.71G | 6.77G | 0.55x | #388 |
//! | Transpose | 4096² | 486M | 550M | 0.88x | #388 |
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
