//! Q4K quantized GEMV operations
//!
//! This module implements Q4_K dequantization and matrix-vector multiplication
//! for efficient inference with 4-bit quantized weights.

#![allow(clippy::wildcard_imports)]
#![allow(clippy::too_many_arguments)]

use super::*;

include!("q4k_tiled_gemv.rs");
include!("q4k_tests_gemv_cached.rs");
