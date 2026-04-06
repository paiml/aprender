//! CUDA-accelerated quantized model
//!
//! This module provides GPU-accelerated inference for quantized models
//! using NVIDIA CUDA.
//!
//! # Architecture
//!
//! `OwnedQuantizedModelCuda` wraps an `OwnedQuantizedModel` with a CUDA executor
//! for GPU-accelerated matrix operations.

// Re-export from cuda module
#[cfg(feature = "cuda")]
pub use super::cuda::OwnedQuantizedModelCuda;
