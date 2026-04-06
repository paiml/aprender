//! Book example validation tests
//!
//! This module contains all tests that validate code examples in the EXTREME TDD book.
//! Every example in the book MUST have a corresponding test here.
//!
//! ## Structure
//!
//! - `ml_fundamentals/` - Tests for Machine Learning Fundamentals chapters
//! - `case_studies/` - Tests for case study chapters
//!
//! ## CI Enforcement
//!
//! The book build will FAIL if any test in this module fails.
//! This is **Poka-Yoke** (error-proofing) - we cannot publish broken examples.

mod case_studies;
mod ml_fundamentals;
