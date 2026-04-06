//! BLIS Tests - Shattered for Maintainability
//!
//! Test modules:
//! - `gemm_ref` - Scalar reference GEMM tests
//! - `jidoka` - Jidoka guard tests
//! - `microkernel` - Microkernel tests
//! - `packing` - Matrix packing tests
//! - `blis_gemm` - BLIS GEMM integration tests
//! - `profiler_heijunka` - Profiler and Heijunka tests
//! - `falsification` - Falsification tests (F21-F30)
//! - `numerical` - Numerical stability tests (F31-F42)

mod blis_gemm;
mod coverage_contract;
mod falsification;
mod gemm_ref;
mod jidoka;
mod microkernel;
mod numerical;
mod packing;
mod profiler_heijunka;
mod uninit_falsify;
mod validate_and_parallel;
