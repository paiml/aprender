#![allow(clippy::disallowed_methods)]
//! M001-M010: Measurement Tools (cbtop) Falsification Tests
//!
//! Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md S9.4
//!
//! These tests verify the cbtop measurement tool works correctly.
//! Measurement tools must be accurate and reliable for optimization.
//!
//! FALSIFICATION: If measurement is unreliable, optimization is impossible.

use std::process::Command;

include!("includes/falsification_measurement_part_01.rs");
include!("includes/falsification_measurement_scoring.rs");
