//! Cached quantized GEMV methods for Q4K/Q5K/Q6K weights
//!
//! This module implements:
//! - PAR-005: Cached GEMV Methods (avoid per-call weight transfers)
//! - Q4K/Q5K/Q6K GEMV with GPU-cached weights
//! - Tiled, chunked, and coalesced GEMV variants
//! - DP4A SIMD-accelerated GEMV
//! - Fused RMSNorm + Q4K GEMV
//! - Batched Q4K/Q6K GEMV

#![allow(clippy::wildcard_imports)] // Internal module organization uses super::*

use super::*;

include!("residual.rs");
include!("quantized_03.rs");
