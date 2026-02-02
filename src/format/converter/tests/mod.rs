//! APR Converter Tests - Extreme TDD
//! PMAT-197: Split from monolithic tests.rs for file size reduction
//!
//! Test organization:
//! - core.rs: Basic converter tests (source parsing, name mapping, quantization)
//! - errors.rs: Import errors and coverage boost (part 1)
//! - coverage.rs: Coverage boost (part 2) and internal helper tests
//! - pmat.rs: PMAT/GH issue specific regression tests
//!
//! # Harness Policy (Audit Round 3, Item #1)
//!
//! All new conversion tests MUST use `ConversionTestHarness`. See `core.rs` header.

mod core;
mod coverage;
mod errors;
mod pmat;
mod pmat_round19;
