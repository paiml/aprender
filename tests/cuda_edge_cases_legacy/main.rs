#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! GPU Edge-Case Testing with trueno-cuda-edge
//!
//! This test suite applies trueno-cuda-edge's falsification frameworks to
//! trueno's GPU compute primitives, verifying:
//!
//! - Null pointer handling for device memory
//! - Quantization boundaries for SIMD operations
//! - PTX verification for generated kernels
//! - Shared memory constraints for GPU backends
//! - Supervision patterns for GPU worker management
//!
//! Tests run without actual GPU hardware by validating pure-Rust
//! type system guarantees and configuration.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod lifecycle_and_quant;
mod null_and_shmem;
mod ptx_and_supervisor;
