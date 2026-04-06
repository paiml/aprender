//! Training — top-level training module re-exports
//!
//! Provides a unified training entry point, separated from inference
//! to prevent feedback loops (MTD-09: Feedback Loop Detection).

pub use crate::train::*;
