#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-025: Cache Efficiency Analysis
//!
//! F1121-F1130: Cache analysis falsification tests
//!
//! These tests verify the cache efficiency analysis module for:
//! - Cache level classification
//! - Working set estimation
//! - Tiling recommendations
//! - Bandwidth prediction

mod analysis_tests;
mod cache_tests;
