//! REX-001 review-lane experiment (spec `docs/specifications/review-experiment-protocol.md`).
//!
//! - [`prereg`]: the REX-00 pre-registration lock (`rex-prereg-v1`).
//! - [`stats`]: the frozen analysis code — Wilson, McNemar exact, seeded
//!   bootstrap, Holm, H6 interference.

pub mod prereg;
pub mod stats;
