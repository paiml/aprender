//! # `provable-contracts-macros` — compatibility facade
//!
//! Renamed to [`aprender-contracts-macros`] during the APR-MONO consolidation.
//! The five attribute macros (`contract`, `requires`, `ensures`, `invariant`,
//! `must_contract`) are re-exported unchanged, so
//! `use provable_contracts_macros::requires;` keeps resolving.
//!
//! [`aprender-contracts-macros`]: https://crates.io/crates/aprender-contracts-macros
//!
//! This crate is deliberately NOT `proc-macro = true` — a proc-macro crate may
//! export nothing but its own `#[proc_macro*]` functions, so it cannot forward
//! anyone else's. A plain library re-exporting them works, and downstream code
//! can still *invoke* them through this path; `compat/invoke.rs` compiles an
//! invocation of each of the five to prove it.
//!
//! Bound by `contracts/provable-contracts-facade-v1.yaml`.

pub use upstream::*;
