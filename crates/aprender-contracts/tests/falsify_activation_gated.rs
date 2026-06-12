//! Phase 5 falsification tests for activation and gated kernels.
//!
//! Covers: Activation (`ReLU`, GELU, `SiLU`), `SiLU` Standalone, `SwiGLU`,
//! Cross-Entropy, and `RoPE` kernels.
//!
//! 28 tests total: FALSIFY-ACT-001..006, FALSIFY-SI-001..006,
//! FALSIFY-SG-001..006, FALSIFY-CE-001..006, FALSIFY-RP-001..004.

// Integration-test binary (includes proptest! macros that unwrap internally):
// `.unwrap()` is idiomatic here and the lib`s cfg(test) allow does not reach it.
#![allow(clippy::disallowed_methods)]

mod common;

use proptest::prelude::*;

#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::activation::relu_avx2;
use provable_contracts::kernels::activation::{gelu_scalar, relu_scalar, silu_scalar};

#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::silu_standalone::silu_standalone_avx2;
use provable_contracts::kernels::silu_standalone::{sigmoid_scalar, silu_standalone_scalar};

#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::swiglu::swiglu_avx2;
use provable_contracts::kernels::swiglu::swiglu_scalar;

#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::cross_entropy::cross_entropy_avx2;
use provable_contracts::kernels::cross_entropy::{cross_entropy_scalar, log_softmax_scalar};

#[cfg(target_arch = "x86_64")]
use provable_contracts::kernels::rope::rope_avx2;
use provable_contracts::kernels::rope::rope_scalar;

// Part 1: Activation + SiLU Standalone (12 tests)
include!("includes/falsify_actgate_act_silu.rs");

// Part 2: SwiGLU + Cross-Entropy + RoPE (16 tests)
include!("includes/falsify_actgate_swi_ce_rope.rs");
