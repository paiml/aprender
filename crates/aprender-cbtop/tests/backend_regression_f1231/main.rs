#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-031: Cross-Backend Regression Detector
//!
//! F1231-F1240: Backend regression falsification tests
//!
//! These tests verify the backend regression detector for:
//! - Cross-backend comparison
//! - Size cliff detection
//! - Best backend recommendation
//! - Transfer overhead analysis

mod config_and_additional;
mod regression_core;
