//! Tests for ensemble merging
//!
//! This module contains all tests for the ensemble merging functionality,
//! organized into logical submodules:
//!
//! - `common`: Shared test utilities and helper functions
//! - `weighted_average`: Tests for weighted/uniform average merging
//! - `slerp`: Tests for iterative SLERP merging
//! - `hierarchical`: Tests for hierarchical merging strategies
//! - `ties_dare`: Tests for TIES and DARE ensemble methods
//! - `error_cases`: Tests for error handling and edge cases
//! - `property`: Property-based tests using proptest
//! - `config`: Tests for configuration defaults and utilities

mod common;
mod config;
mod error_cases;
mod hierarchical;
mod property;
mod slerp;
mod ties_dare;
mod weighted_average;
