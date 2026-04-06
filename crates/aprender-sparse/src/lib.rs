#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::disallowed_methods,
        clippy::float_cmp,
        clippy::panic,
        clippy::cast_precision_loss,
        clippy::unnecessary_wraps
    )
)]
//! Trueno Sparse: Sparse matrix formats and operations
//!
//! Provides CSR, COO, and BSR sparse matrix formats with SIMD-accelerated
//! SpMV (sparse matrix-vector multiply) and SpMM (sparse matrix-dense matrix multiply).
//!
//! # Design
//!
//! - **Provable contracts**: Format invariants validated at construction time
//! - **Backward error bounded**: Numerical accuracy follows LAProof bounds
//! - **SIMD dispatch**: Scalar → AVX2 → AVX-512 runtime selection
//! - **GPU ready**: Formats are GPU-transfer-friendly (contiguous arrays)
//!
//! # Quick Start
//!
//! ```
//! use trueno_sparse::{CsrMatrix, CooMatrix, SparseOps};
//!
//! // Build from COO (triplets)
//! let coo = CooMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1.0_f32, 2.0, 3.0]).unwrap();
//! let csr = CsrMatrix::from_coo(&coo);
//!
//! // SpMV: y = A * x
//! let x = vec![1.0_f32, 1.0, 1.0];
//! let mut y = vec![0.0_f32; 3];
//! csr.spmv(1.0, &x, 0.0, &mut y);
//! assert!((y[0] - 1.0).abs() < 1e-6);
//! assert!((y[1] - 2.0).abs() < 1e-6);
//! assert!((y[2] - 3.0).abs() < 1e-6);
//! ```
//!
//! # References
//!
//! - Merrill & Garland, "Merge-Based Parallel SpMV", PPoPP 2016
//! - LAProof (Princeton): formal backward error bounds for CSR SpMV

mod bsr;
mod coo;
mod csr;
mod error;
mod ops;
mod sell;
mod spgemm;
mod validate;

pub use bsr::BsrMatrix;
pub use coo::CooMatrix;
pub use csr::CsrMatrix;
pub use error::SparseError;
#[cfg(target_arch = "x86_64")]
pub use ops::Avx2Backend;
#[cfg(target_arch = "aarch64")]
pub use ops::NeonBackend;
pub use ops::{ScalarBackend, SparseBackend, SparseOps};
pub use sell::SellMatrix;
pub use spgemm::spgemm;
pub use validate::validate_csr_invariants;

#[cfg(test)]
mod tests;
