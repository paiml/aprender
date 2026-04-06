//! GPU-Resident Tensor Operations (f32 specialization)
//!
//! This module contains all f32-specialized operations for `GpuResidentTensor`.
//! Operations include GEMM, Softmax, LayerNorm, GELU, etc.
//!
//! ## Design
//!
//! All operations:
//! - Stay on GPU (no implicit host transfers)
//! - Use kernel caching for performance
//! - Support both synchronous and stream-based async variants
//!
//! ## Usage
//!
//! ```ignore
//! let result = tensor.matmul(&ctx, &other, m, n, k)?;
//! let activated = result.gelu(&ctx)?;
//! ```
//!
//! ## Submodules
//!
//! - [`gemm`] - Matrix multiplication (naive, tiled, WMMA)
//! - [`elementwise`] - Softmax, add, scale, layout transforms
//! - [`composite`] - LayerNorm, GELU, bias, linear, conv1d, fused ops

mod composite;
mod elementwise;
mod gemm;
