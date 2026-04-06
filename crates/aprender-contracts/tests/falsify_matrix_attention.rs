//! Phase 5 falsification tests for matrix and attention kernels.
//!
//! Covers Matmul, Attention, GQA, and Flash Attention kernels.
//! 20 tests total: FALSIFY-MM-001..005, FALSIFY-ATT-001..005,
//! FALSIFY-GQ-001..006, FALSIFY-FA-001..004.

mod common;

use proptest::prelude::*;
use provable_contracts::kernels::attention::*;
use provable_contracts::kernels::flash_attention::*;
use provable_contracts::kernels::gqa::*;
use provable_contracts::kernels::matmul::*;

// Part 1: Matmul + Attention (10 tests)
include!("includes/falsify_matatt_matmul_att.rs");

// Part 2: GQA + Flash Attention (10 tests)
include!("includes/falsify_matatt_gqa_flash.rs");
