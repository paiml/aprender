//! CGP: Compute-GPU-Profile — Unified Performance Analysis Library
//!
//! Provides profiling, roofline modeling, regression detection, and Muda (waste)
//! analysis for scalar, SIMD, wgpu, and CUDA workloads.

// `.unwrap()` is banned in production via workspace `.clippy.toml`, but is
// idiomatic in `#[cfg(test)]` assertions. Allow it only under `cfg(test)`.
#![cfg_attr(test, allow(clippy::disallowed_methods))]

pub mod analysis;
pub mod doctor;
pub mod metrics;
pub mod profilers;
