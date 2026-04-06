//! Phase 5 falsification tests for normalization kernels.
//!
//! Covers Softmax, `RMSNorm`, `LayerNorm`, and `BatchNorm` with 24 tests total.
//! Each test targets a specific mathematical invariant that would break
//! if the implementation contains a common bug class.

mod common;

use proptest::prelude::*;
use provable_contracts::kernels::batchnorm::*;
use provable_contracts::kernels::layernorm::*;
use provable_contracts::kernels::rmsnorm::*;
use provable_contracts::kernels::softmax::*;

// Part 1: Softmax + RMSNorm (11 tests)
include!("includes/falsify_norm_sm_rms.rs");

// Part 2: LayerNorm + BatchNorm (13 tests)
include!("includes/falsify_norm_ln_bn.rs");
