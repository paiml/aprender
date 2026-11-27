//! Aprender — Next-generation ML framework in pure Rust.
//!
//! This facade crate re-exports `aprender-core` so that
//! `use aprender::*` works whether you depend on `aprender`
//! or `aprender-core` directly.
//!
//! Install the CLI: `cargo install aprender`

// PMAT-124: enforce `// SAFETY:` on every unsafe block in the facade crate.
#![deny(clippy::undocumented_unsafe_blocks)]

pub use aprender_ml::*;

// The README's library snippet compiles as a doctest. A wrong module path in it fails `cargo test --doc`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
