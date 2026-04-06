#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Comprehensive Integration Test Suite
//!
//! This integration test serves as the definitive release gate for Trueno.
//! It must pass with 100% success before any release is tagged.
//!
//! Requirements:
//! - Tests ALL features currently supported
//! - Uses property-based testing for mathematical correctness
//! - Contributes to code coverage
//! - Runs under 30 seconds (enforced)
//! - Included in pre-commit hooks
//!
//! Coverage:
//! - All 87 vector operations
//! - All matrix operations (matmul, transpose)
//! - All backends (Scalar, SSE2, AVX2, etc.)
//! - Error handling and edge cases
//! - Mathematical properties and invariants

mod backend_tests;
mod matrix_ops;
mod smoke_performance;
mod vector_ops;
