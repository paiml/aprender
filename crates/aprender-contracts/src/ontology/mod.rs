//! ONT-001 ontology gates (ONT-6 onward).
//!
//! - [`arming`] — which gates enter a repo's meet (`armed_gates` in `contracts/lint-baseline.json`, monotone; §3.9).
//! - [`sigma`] — Σ, the ontology's own declaration (`contracts/ontology.yaml`; §4.1).
//! - [`verdict`] — the one verdict lattice every `pv lint` gate reports into (§3.4).

pub mod arming;
pub mod sigma;
pub mod verdict;
