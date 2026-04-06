use super::*;
use crate::Vector;

// ===== Internal Implementation Tests (DISABLED - PMAT-018) =====
// These tests referenced internal methods (matmul_naive, matmul_simd, microkernels)
// that are now properly encapsulated in ops/arithmetic.rs.
// The public API tests above provide equivalent coverage.
// TODO: Move internal tests to ops/arithmetic.rs if needed.

#[cfg(internal_matrix_tests)]
mod internal_tests;

mod constructors;
mod conv_property_tests;
mod coverage;
mod matmul;
mod ml_primitives;
mod property_tests;
mod transpose;
