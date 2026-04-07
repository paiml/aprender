//! Shared utilities for the Batuta stack.
//!
//! Provides consistent formatting, system information, and display helpers
//! used across pmat, trueno-viz, aprender, and other Batuta crates.
//!
//! # Modules
//!
//! - [`fmt`] — Byte, percentage, duration, and number formatting
//! - [`sys`] — System information (cgroup detection, CPU info)
//! - [`display`] — Column alignment, truncation, and builder traits

pub mod compression;
pub mod display;
pub mod fmt;
pub mod math;
pub mod sys;
