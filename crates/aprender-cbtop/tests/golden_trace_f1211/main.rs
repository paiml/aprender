#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-029: Golden Trace Comparison
//!
//! F1211-F1220: Golden trace comparison falsification tests
//!
//! These tests verify the golden trace module for:
//! - Trace capture and storage
//! - Comparison and delta calculation
//! - Regression detection
//! - Export/import functionality

mod additional_tests;
mod golden_tests;
