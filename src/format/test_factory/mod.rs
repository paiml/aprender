//! Test Factory - Pygmy Model Builders (T-COV-95)
//!
//! Implements the "Active Pygmy" pattern from realizar for creating minimal
//! valid model files in memory without needing real model files on disk.
//!
//! # Dr. Popper's "Minimum Viable Predictor"
//!
//! A tiny model that:
//! 1. Has valid tensor layout
//! 2. Has valid quantized/unquantized weights
//! 3. Exercises all code paths in format loading/conversion
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::test_factory::{build_pygmy_safetensors, build_pygmy_apr};
//!
//! // Create minimal SafeTensors in memory
//! let st_bytes = build_pygmy_safetensors();
//!
//! // Create minimal APR v2 in memory
//! let apr_bytes = build_pygmy_apr();
//! ```

// Conversion test harness (rosetta-testing.md spec)
#[cfg(test)]
pub(crate) mod harness;

// Unit tests for builder functions
#[cfg(test)]
mod tests;

// Re-export all public builder types and functions
mod builders;
pub use builders::*;
