#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! 200-Point Falsification Validation Tests
//!
//! Implements CBTOP-SPEC-001 §13: Popperian Falsification Protocol
//!
//! Each test attempts to REFUTE the system's claims. If a test passes,
//! it means we failed to falsify that property - a corroboration.
//!
//! F-series IDs map to spec sections:
//! - F001-F020: ComputeBrick Core Invariants
//! - F021-F040: BrickBudget Verification
//! - F041-F060: Backend Equivalence
//! - F061-F080: TUI Rendering
//! - F081-F100: Performance Metrics
//! - F101-F120: Error Handling
//! - F121-F140: Memory Safety
//! - F141-F160: Concurrency
//! - F161-F180: Integration
//! - F181-F200: Jidoka (Built-in Quality)

mod budget_and_collectors;
mod core_invariants;
mod quality_and_integration;
