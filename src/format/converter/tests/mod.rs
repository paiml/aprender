//! APR Converter Tests - Extreme TDD
//! PMAT-197: Split from monolithic tests.rs for file size reduction
//!
//! Test organization:
//! - core.rs: Basic converter tests (source parsing, name mapping, quantization)
//! - errors.rs: Import errors and coverage boost (part 1)
//! - coverage_types.rs: Type/helper coverage (TensorExpectation, Architecture, etc.)
//! - coverage_functions.rs: Function coverage (export, merge, write, import, lint)
//! - coverage_falsification.rs: Falsification tests (PMAT-197..205)
//! - pmat.rs: PMAT/GH issue specific regression tests
//!
//! # Harness Policy (Audit Round 3, Item #1)
//!
//! All new conversion tests MUST use `ConversionTestHarness`. See `core.rs` header.

mod core;
mod coverage_falsification;
mod coverage_functions;
mod coverage_types;
mod errors;
mod gh202_layout;
mod infer_config;
mod pmat;
mod pmat_round19;
mod pure_functions;
mod sharded_import;
mod tokenizer_parse;
