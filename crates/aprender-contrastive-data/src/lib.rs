//! Deterministic, leakage-safe contrastive data construction.
//!
//! # Contract: contrastive-pair-protocol-v1.yaml
//!
//! This crate owns contrastive/Siamese **data construction** as a general capability —
//! class buckets, balanced few-shot selection, bounded positive/negative pair sampling,
//! typed split roles, dataset fingerprints, and the cross-split leakage checks that make
//! all of the above trustworthy. SetFit is its first consumer, not its owner (D-01/D-03).
//!
//! # The bytes boundary (D-04)
//!
//! The public API is **bytes-in / bytes-out and typed values**. This crate performs no
//! filesystem access, opens no sockets, and exposes no path-shaped parameters — not even
//! in its tests. `apr-cli` owns every filesystem adapter.
//!
//! That is not stylistic. The destination for these artifacts is object storage behind a
//! serverless consumer, where a manifest is an S3 object rather than a file; a crate
//! whose API speaks in `&Path` forces such a consumer to be a rewrite instead of a
//! wrapper. The boundary is **enforced**, not asserted: `make contrastive-data-boundary`
//! compares the resolved dependency closure against a positive allowlist and bans
//! `std::fs`/`std::net`/`std::path`/`Path`/`PathBuf` throughout `src/`, and
//! `OBLIG-CPP-BYTES-BOUNDARY` in the contract names that check.
//!
//! # Determinism (D-20)
//!
//! Every random decision is a pure function of its draw ordinal, obtained from the
//! counter-based Philox generator in `aprender-rand` (library name `trueno_rand`) rather
//! than from a stateful stream. Worker-count independence is therefore *structural*: draw
//! *i* cannot depend on how many draws preceded it, because nothing precedes it.
//!
//! ```
//! use trueno_rand::Philox4x32;
//!
//! // The same (key, counter) always yields the same block ...
//! let key = [0xdead_beef_u32, 0x0000_002a];
//! let draw_7 = Philox4x32::generate_at(key, [7, 0, 0, 0]);
//! assert_eq!(draw_7, Philox4x32::generate_at(key, [7, 0, 0, 0]));
//!
//! // ... and a different ordinal is an independent draw, with no carried state.
//! assert_ne!(draw_7, Philox4x32::generate_at(key, [8, 0, 0, 0]));
//! ```
//!
//! The RNG obligations (key derivation, counter mapping, the frozen domain-string table,
//! and the bounded-draw derivation) are stated **inline** in
//! `contracts/contrastive-pair-protocol-v1.yaml`. They deliberately do not cite an
//! external RNG contract: no such file exists in this repository, and a dangling
//! cross-reference is worse than an inline statement.
//!
//! # Why modules, not a flat re-export surface
//!
//! Consumers use module paths — `aprender_contrastive_data::split::Split`,
//! `::pairs::CanonicalPair` — and this file re-exports exactly one name. That is
//! deliberate. The complete module skeleton is declared here, once, so that each
//! subsequent unit of work edits only its own module file and never contends on
//! `lib.rs`. Without that, several independent workstreams would serialize behind a
//! single re-export list for no engineering reason. The cost is one extra path segment
//! at the call site; the benefit is that the module tree is also the ownership map.

/// Typed failure surface shared by every boundary in this crate.
pub mod error;

/// Labeled-example schema and strict JSONL parse/encode over `&[u8]`.
///
/// Implemented by plan 02-03.
pub mod schema;

/// Exact and normalized content hashes plus the dataset fingerprint.
///
/// Implemented by plan 02-03.
pub mod hash;

/// Typestate split roles: `Split<Train>`, `Split<Validation>`, `Split<Test>`,
/// `Split<CompatibilityTest>`.
///
/// Implemented by plan 02-03.
pub mod split;

/// The attested, profile-parameterized dataset a consumer must present before canonical
/// splits are exposed.
///
/// Implemented by plan 02-06.
pub mod prepared;

/// Dataset identity attestation and its re-derivation from supplied buffers.
///
/// Implemented by plan 02-06.
pub mod attestation;

/// Cross-split duplicate coalescing and the deterministic exclusion record.
///
/// Implemented by plan 02-03.
pub mod dedup;

/// Append-only access ledger: which splits were touched, under which profile.
///
/// Implemented by plan 02-04.
pub mod ledger;

/// Domain-separated Philox key derivation and the bounded-draw primitive.
///
/// Implemented by plan 02-04.
pub mod rng;

/// Sorted per-class buckets over a selection pool.
///
/// Implemented by plan 02-05.
pub mod buckets;

/// Balanced few-shot selection and the ordered selected-ID manifest model.
///
/// Implemented by plan 02-05.
pub mod select;

/// Bounded pair sampling: canonical pairs, capacity math, budget resolution, and the
/// singleton and degenerate-layout policies.
///
/// Implemented by plan 02-07.
pub mod pairs;

/// Canonical serialization and semantic hashing for every manifest in the protocol.
///
/// Implemented by plan 02-07.
pub mod manifest;

pub use error::ContrastiveDataError;
