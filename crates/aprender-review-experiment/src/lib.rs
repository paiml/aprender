//! REX-001 review-lane experiment (spec `docs/specifications/review-experiment-protocol.md`).
//!
//! - [`prereg`]: the REX-00 pre-registration lock (`rex-prereg-v1`).
//! - [`stats`]: the frozen analysis code — Wilson, McNemar exact, seeded
//!   bootstrap, Holm, H6 interference.
//! - [`corpus`]: the REX-02 review corpus (`review-corpus-v1`).
//! - [`contamination`]: the sealed-test leak check (`review-corpus-contamination-v1`).
//! - [`receipt`]: the REX-03 receipt schema, verdict parser and admissibility
//!   (`review-experiment-receipt-v1`).
//! - [`score`]: the REX-03 scorer (§2.3 metrics, §3 hypothesis inputs).
//! - [`harness`]: the REX-03 client that drives a resident `apr serve`.

pub mod build_corpus;
pub mod contamination;
pub mod corpus;
pub mod harness;
pub mod prereg;
pub mod receipt;
pub mod score;
pub mod stats;
