//! Book example validation tests
//!
//! This module contains all tests that validate code examples in documentation.
//! Every example in the book MUST have a corresponding test here.
//!
//! ## Structure
//!
//! - `grammar_generation/` - Tests for grammar-based generation chapters
//! - `mutation_testing/` - Tests for mutation testing chapters
//! - `verification/` - Tests for verification oracle chapters
//! - `ml_pipeline/` - Tests for ML training pipeline chapters
//!
//! ## CI Enforcement
//!
//! The book build will FAIL if any test in this module fails.
//! This is **Poka-Yoke** (error-proofing) - we cannot publish broken examples.

mod grammar_generation;
mod integrations;
mod ml_pipeline;
mod mutation_testing;
mod verification;
