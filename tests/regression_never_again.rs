#![allow(clippy::disallowed_methods)]
//! Bug Regression Test Suite -- Never Again
//!
//! Each test encodes a specific historical bug from the showcase spec.
//! Test names include bug numbers for traceability.
//!
//! These tests use synthetic data and do NOT require model files,
//! so they run in every `cargo test`.

use std::path::Path;
use tempfile::NamedTempFile;

include!("includes/regression_never_again_part_01.rs");
include!("includes/regression_never_again_part_02.rs");
