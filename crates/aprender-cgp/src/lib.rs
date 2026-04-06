//! CGP: Compute-GPU-Profile — Unified Performance Analysis Library
//!
//! Provides profiling, roofline modeling, regression detection, and Muda (waste)
//! analysis for scalar, SIMD, wgpu, and CUDA workloads.

pub mod analysis;
pub mod doctor;
pub mod metrics;
pub mod profilers;
