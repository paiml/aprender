//! ONT-001 ontology gates (ONT-6 onward).
//!
//! - [`arming`] — which gates enter a repo's meet (`armed_gates` in `contracts/lint-baseline.json`, monotone; §3.9).
//! - [`extract`] — extractors: each entity type becomes RDF (§3.7; ONT-4b implements `pv_contract`).
//! - [`rdf`] — the deterministic graph and its N-Triples writer (R-15; no blank nodes).
//! - [`shapes`] — the in-house SHACL-Core-subset validator and the Turtle export (§3.6; ONT-4b).
//! - [`sigma`] — Σ, the ontology's own declaration (`contracts/ontology.yaml`; §4.1).
//! - [`verdict`] — the one verdict lattice every `pv lint` gate reports into (§3.4).

pub mod arming;
pub mod extract;
pub mod rdf;
pub mod shapes;
pub mod sigma;
pub mod verdict;
