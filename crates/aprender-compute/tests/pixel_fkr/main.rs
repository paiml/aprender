#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! TRUENO-SPEC-013: Pixel FKR (Falsification Kernel Regression) Test Suites
//!
//! Visual pixel-level regression tests using Popperian falsification methodology.
//! Each suite renders compute outputs as data and compares against golden baselines.
//!
//! # Test Suites (SPEC Section 3.5)
//! - scalar-pixel-fkr: Baseline truth (pure Rust, no SIMD/GPU)
//! - simd-pixel-fkr: SSE2/AVX2/AVX-512/NEON vs scalar baseline
//! - wgpu-pixel-fkr: WGSL compute shaders vs scalar baseline
//! - ptx-pixel-fkr: CUDA PTX kernels vs scalar baseline
//!
//! # Running
//! ```bash
//! make pixel-fkr-all         # All suites
//! make pixel-scalar-fkr      # Baseline only
//! ```
//!
//! # Academic Foundation
//! - Alipour et al. (ESEC/FSE 2021): Pixel comparison catches bugs unit tests miss
//! - Lidbury et al. (PLDI 2015): Randomized inputs expose corner cases

mod helpers;
mod scalar_baselines;
mod simd_validation;
mod summary;
#[cfg(feature = "gpu")]
mod wgpu_validation;
