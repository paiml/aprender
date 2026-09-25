//! PRM-001 PROMETHEUS review-lane experiment (spec `docs/specifications/PRM-001-prometheus.md`; was REX-001).
//!
//! - [`prereg`]: the REX-00 pre-registration lock (`rex-prereg-v2`).
//! - [`stats`]: the frozen analysis code — Wilson, McNemar exact, seeded
//!   bootstrap, Holm, H6 interference.
//! - [`corpus`]: the REX-02 review corpus (`review-corpus-v1`).
//! - [`contamination`]: the sealed-test leak check (`review-corpus-contamination-v1`).
//! - [`receipt`]: the REX-03 receipt schema, verdict parser and admissibility
//!   (`review-experiment-receipt-v1`).
//! - [`score`]: the REX-03 scorer (§2.3 metrics, §3 hypothesis inputs).
//! - [`harness`]: the REX-03 client that drives a resident `apr serve`.
//! - [`admission`]: the REX-04 per-cell admission file (`rex-cell-admission-v1`).
//! - [`pilot`]: the REX-05 pilot projection and the §2.2 sample-size rule.
//! - [`ledger`]: the REX-07 shadow-lane `review-ledger-v1` rows and coverage.
//! - [`ladder`]: the REX-09 shadow → tripwire → vote promotion gates (H4/H5).
//! - [`ratchet`]: the REX-10 `review-lane-perf-ratchet-v1` p95 ratchet (§5.1).
//! - [`champion`]: the REX-11 §5.4 champion/challenger promotion gate.
//! - [`fewshot`]: the REX-11 B1 challengers: prompt versions and leak-proof retrieval few-shot.
//! - [`b2`]: the REX-12 B2 loop: verb-gated row status and the teacher-logit dataset receipt.
//! - [`datacard`]: the PRA-001 T14 Croissant + Datasheet card over the agent-trace index (`trace-datacard-v1`).

pub mod admission;
pub mod b2;
pub mod build_corpus;
pub mod champion;
pub mod cluster;
pub mod contamination;
pub mod corpus;
pub mod datacard;
pub mod dedup;
pub mod fewshot;
pub mod harness;
pub mod kappa_probe;
pub mod ladder;
pub mod ledger;
pub mod pilot;
pub mod pool;
pub mod prereg;
pub mod ratchet;
pub mod receipt;
pub mod score;
pub mod secret;
pub mod sparse_logits;
pub mod stats;
