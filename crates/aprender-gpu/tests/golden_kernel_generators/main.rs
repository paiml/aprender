//! Golden Kernel Generator Tests (Popperian Falsification)
//!
//! These tests are LOCKED as immutable guardians of kernel correctness.
//! To modify: First demonstrate a falsifying test case (black swan).
//!
//! These tests verify the INTENT of each kernel generator, ensuring correct
//! PTX structure and instruction patterns.
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test golden_kernel_generators --features cuda`

#![cfg(feature = "cuda")]

mod gemm;
mod params_and_boundaries;
mod softmax_and_norm;
