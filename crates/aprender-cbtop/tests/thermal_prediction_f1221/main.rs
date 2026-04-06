#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-030: Thermal Trend Prediction
//!
//! F1221-F1230: Thermal prediction falsification tests
//!
//! These tests verify the thermal prediction module for:
//! - Trend prediction accuracy
//! - Throttle risk calculation
//! - Cooldown recommendations
//! - Thermal-latency correlation

mod additional_tests;
mod thermal_tests;
