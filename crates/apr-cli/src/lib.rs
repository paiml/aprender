//! apr-cli library
//!
//! This library exposes the error module for use in tests.
//! The binary is the primary interface.

pub mod error;
#[cfg(feature = "inference")]
pub mod federation;
