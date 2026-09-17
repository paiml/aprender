//! ONT-001 ontology gates (ONT-6 onward).
//!
//! - [`arming`] — which gates enter a repo's meet (`armed_gates` in `contracts/lint-baseline.json`, monotone; §3.9).
//! - [`verdict`] — the one verdict lattice every `pv lint` gate reports into (§3.4).

pub mod arming;
pub mod verdict;
