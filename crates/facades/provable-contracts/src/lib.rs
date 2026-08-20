//! # `provable-contracts` — compatibility facade
//!
//! This crate was renamed to [`aprender-contracts`] during the APR-MONO
//! consolidation. It stays on crates.io so existing dependents keep compiling:
//! it re-exports the real crate's public surface verbatim and adds nothing of
//! its own.
//!
//! ```toml
//! # migrate at your convenience; no source change is needed either way
//! aprender-contracts = "0.63"
//! ```
//!
//! [`aprender-contracts`]: https://crates.io/crates/aprender-contracts
//!
//! # The promise, and how it is held
//!
//! Every path that resolved through `provable_contracts::…` at 0.3.1 resolves
//! to the same item today. That is checked, not hoped: the 28 example programs
//! published *inside* `provable-contracts 0.3.1` are vendored verbatim under
//! `compat/0.3.1/` and compiled against this crate by
//! `scripts/check_facade_compat.sh` in CI. They call into 20 of the
//! re-exported modules by name, so a drifted *signature* — not merely a
//! removed export — fails the build.
//!
//! Bound by `contracts/provable-contracts-facade-v1.yaml`.

pub use upstream::*;
