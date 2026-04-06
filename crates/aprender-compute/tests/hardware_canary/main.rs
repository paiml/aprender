#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Hardware Canary Tests
//!
//! These tests verify that the hardware is correctly detected and available.
//! If these tests fail, it indicates a configuration problem (e.g., RUSTFLAGS not set).
//!
//! ## Purpose
//!
//! These are "canary in the coal mine" tests that will fail loudly if:
//! 1. AVX-512 is not detected on a Threadripper (SIMD Canary)
//! 2. CUDA is not available on an NVIDIA GPU (GPU Canary)
//!
//! ## Running
//!
//! ```bash
//! # With native CPU features (required for AVX-512)
//! RUSTFLAGS="-C target-cpu=native" cargo test --test hardware_canary --all-features
//!
//! # This is what `make coverage` does - always use make coverage
//! ```

mod gpu_canary;
mod hardware_report;
mod simd_canary;
