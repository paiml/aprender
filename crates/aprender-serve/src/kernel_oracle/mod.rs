//! KTEST-02 (KTEST-001 §3.2): the f64 scalar oracle and the per-element margin harness.
//!
//! A kernel is judged element by element: mᵢ = |ŷᵢ − yᵢ| / Bᵢ, where yᵢ is the oracle's f64 value
//! and Bᵢ is the bound the kernel's declared error model gives for that element. The kernel passes
//! iff max mᵢ ≤ 1 (plus the NaN/Inf rules in [`margin`]). A global NMSE is reported next to it so
//! receipts stay comparable with llama.cpp's `test-backend-ops`, but NMSE never decides: one wrong
//! tile in a million outputs moves it by ~1e-11 (F-12, tested in [`margin`]).
//!
//! The oracle is independent by construction: it is std-only and names nothing outside this
//! module, so it cannot call the optimized backend it judges (§0.3 "the producer is never the
//! gate"). It is a test-only module of this crate rather than its own crate because a path-only
//! dev-dependency is stripped at publish, so `src/` may not name one;
//! `kernel_oracle_scan` reads every `.rs` file under this directory and refuses any other path.
//!
//! Quantization error is not the kernel's (§0.4): feed the oracle the dequantized weights and, for
//! a kernel that quantizes its activations, the round-tripped activations it actually consumed.

pub mod error_model;
pub mod margin;
pub mod oracle;

pub use error_model::{Bound, Dtype, ErrorModel, Refusal};
pub use margin::{judge, Failure, Report, Verdict};
