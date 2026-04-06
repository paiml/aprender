//! Advanced PTX Builder Golden Tests
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE
//!
//! These tests are LOCKED as immutable guardians of PTX correctness.
//! To modify: First demonstrate a falsifying test case (black swan).
//!
//! Tests for less commonly used but important PTX operations.
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test golden_ptx_advanced --features cuda`

#![cfg(feature = "cuda")]

mod arithmetic_and_atomics;
mod conversions_and_registers;
mod memory_and_bitwise;
mod predicates_and_moves;
mod tensor_core_and_debug;
