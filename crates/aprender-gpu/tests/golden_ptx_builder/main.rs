//! Golden PTX Builder Tests (Popperian Falsification)
//!
//! These tests are LOCKED as immutable guardians of PTX correctness.
//! To modify: First demonstrate a falsifying test case (black swan).
//!
//! These tests verify the INTENT of each PTX instruction, not just string presence.
//! Each test generates PTX and verifies the exact instruction format.
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test golden_ptx_builder --features cuda`

#![cfg(feature = "cuda")]

mod arithmetic_and_integer;
mod comparison_and_memory;
mod math_bits_conversion;
mod special_regs_and_structure;
mod warp_and_control;
