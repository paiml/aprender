//! Basic quantized GEMV operations (Q6K, Q8, Q5, Q4)
//!
//! This module implements the core quantized matrix-vector multiplication
//! for Q6_K, Q8_0, Q5_0, Q4_0, Q4_1, and Q5_K quantization formats.

#![allow(clippy::wildcard_imports)]
#![allow(clippy::too_many_arguments)]

use super::*;

include!("device.rs");
include!("q8_gemv_tests.rs");
