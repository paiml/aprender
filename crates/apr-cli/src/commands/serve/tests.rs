// ============================================================================
// EXTREME TDD Tests - APR-SPEC §4.15.9 Falsification Checklist
// ============================================================================
//
// This test module implements the 100-point falsification checklist from
// APR-SPEC §4.15.9. Tests are written FIRST (TDD), then implementation
// is added to make them pass.
//
// Coverage targets:
// - Server Lifecycle (SL01-SL10): 10 points
// - Health & Readiness (HR01-HR10): 10 points
// - Metrics Accuracy (MA01-MA10): 10 points
// - Error Handling (EH01-EH10): 10 points
// - Concurrency (CC01-CC10): 10 points

// File-level imports from serve module (needed for inner mod tests via use super::*)
#[allow(unused_imports)]
use super::*;

#[cfg(test)]
#[path = "tests_tests.rs"]
mod tests;
