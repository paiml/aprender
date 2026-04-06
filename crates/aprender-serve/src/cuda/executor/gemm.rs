//! Multi-stream async execution and GEMM/GEMV operations
//!
//! This module implements:
//! - PARITY-038: Multi-Stream Async Execution
//! - GEMM operations (tiled, optimized, fused)
//! - GEMV operations for M=1 token generation
//! - Softmax kernel
//! - Q4K/Q5K/Q6K GEMV with direct weight transfer

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

include!("fused.rs");
include!("gemm_tests.rs");
