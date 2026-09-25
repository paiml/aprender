//! REX-001 review-lane experiment (spec `docs/specifications/review-experiment-protocol.md`).
//!
//! - [`prereg`]: the REX-00 pre-registration lock (`rex-prereg-v1`).
//! - [`stats`]: the frozen analysis code — Wilson, McNemar exact, seeded
//!   bootstrap, Holm, H6 interference.
//! - [`corpus`]: the REX-02 review corpus (`review-corpus-v1`).
//! - [`contamination`]: the sealed-test leak check (`review-corpus-contamination-v1`).

pub mod build_corpus;
pub mod contamination;
pub mod corpus;
pub mod prereg;
pub mod stats;
