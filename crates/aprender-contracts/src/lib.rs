//! # provable-contracts
//!
//! Papers to Math to Contracts in Code.
//!
//! A Rust library for converting peer-reviewed research papers into
//! mathematically provable kernel implementations via YAML contract
//! intermediaries with Kani bounded model checking verification.
//!
//! ## Modules
//!
//! - [`schema`] — Parse and validate YAML kernel contracts
//! - [`scaffold`] — Generate Rust trait stubs + failing tests from contracts
//! - [`kani`] — Generate `#[kani::proof]` harnesses from contracts
//! - [`probar`] — Generate probar property-based tests from contracts
//! - [`audit`] — Trace paper→equation→contract→test→proof chain
//! - [`binding`] — Map contract equations to implementation functions
//! - [`diff`] — Detect drift between contract versions
//! - [`coverage`] — Cross-contract obligation coverage report
//! - [`generate`] — End-to-end codegen to disk
//! - [`graph`] — Contract dependency graph and cycle detection
//! - [`latex`] — LaTeX conversion for contract math notation
//! - [`book_gen`] — mdBook page generation for contracts
//! - [`lean_gen`] — Lean 4 definition and theorem stub generation
//! - [`lint`] — Contract quality gate: validate + audit + score in one pass
//! - [`kernels`] — Scalar, AVX2, and PTX kernel implementations

// `.unwrap()` is banned in production via workspace `.clippy.toml`, but is
// idiomatic in `#[cfg(test)]` assertions (including the `include!`-ed
// `*_tests.rs` modules). Allow it only under `cfg(test)`.
#![cfg_attr(test, allow(clippy::disallowed_methods))]

pub mod audit;
pub(crate) mod auto_exempt;
pub mod binding;
pub mod book_gen;
pub mod build_helper;
pub mod codegen;
pub mod coq_gen;
pub mod coverage;
pub mod diff;
pub mod doc_integrity;
pub mod error;
pub mod explain;
pub mod extract;
pub mod flux_gen;
pub mod fuzz_gen;
pub mod generate;
pub mod graph;
pub mod inert;
pub mod infer;
pub mod invariant_gen;
pub mod kani_gen;
pub mod kernels;
pub mod latex;
pub mod lean_gen;
pub mod lint;
pub mod mirai_gen;
pub mod obligation_matrix;
pub mod pipeline;
pub mod probar_gen;
pub mod proof_status;
pub mod query;
pub mod readme_gen;
pub mod reverse_coverage;
pub mod roofline;
pub mod scaffold;
pub mod schema;
pub mod scoring;
pub mod tla_gen;
pub mod traits;
