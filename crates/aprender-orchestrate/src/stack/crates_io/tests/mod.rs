//! Tests for the crates_io module.
//!
//! Split into submodules:
//! - `helpers` - Test helper functions
//! - `unit_tests` - Basic unit tests (CRATES-001 through CRATES-006)
//! - `client_tests` - CratesIoClient and MockCratesIoClient tests (CRATES-007 through CRATES-010)
//! - `proptests` - Property-based tests
//! - `serialization_tests` - Serialization roundtrip and edge case tests (CRATES-011, CRATES-012)

mod helpers;

#[cfg(test)]
mod client_tests;
#[cfg(test)]
mod proptests;
#[cfg(test)]
mod serialization_tests;
#[cfg(test)]
mod unit_tests;
