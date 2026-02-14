//! APR Converter Error Tests - Extreme TDD
//! PMAT-197: Split from tests.rs for file size reduction
//!
//! Contains: import errors, export format, merge strategy,
//! sharded index, quantization, and coverage boost tests (part 1).

#[allow(unused_imports)]
use super::super::*;

// ============================================================================
// GH-129: Import error message tests
// ============================================================================

#[cfg(test)]
#[path = "errors_tests.rs"]
mod tests;
