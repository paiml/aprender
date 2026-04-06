use super::*;
use std::path::PathBuf;

// BH-MOD-001 through BH-MOD-008: Hunt core, ensemble, classification, detection
include!("tests_hunt.rs");

// BH-MOD-009 through BH-MOD-021: Coverage gap tests — mutations, lcov, patterns, unsafe
include!("tests_coverage.rs");

// BH-MOD-022 through BH-MOD-038: Pattern detection, clippy, scan, categorization
include!("tests_patterns.rs");

// BH-MOD-039 through BH-MOD-063: Hunt modes, spec, tickets, integration
include!("tests_modes.rs");
