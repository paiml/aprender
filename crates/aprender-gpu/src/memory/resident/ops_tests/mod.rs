//! GPU-Resident Tensor Operations Tests (PMAT-018: Coverage Killer Remediation)
//!
//! Comprehensive tests for all f32-specialized operations on `GpuResidentTensor`.
//! These tests exercise the actual CUDA kernel paths to achieve coverage.

#![cfg(all(test, feature = "cuda"))]

mod attention;
mod elementwise;
mod linear_conv;
mod matmul;
mod weights_and_tensor;
