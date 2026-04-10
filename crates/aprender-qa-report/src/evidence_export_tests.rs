use super::*;

// FALSIFY-EVIDENCE-001: Round-trip integrity
//
// Falsification hypothesis: "JSON round-trip corrupts evidence data"
// If from_json(to_json(export)) != export semantically, implementation is broken.

include!("evidence_export_tests_part_a.rs");
include!("evidence_export_tests_part_b.rs");
