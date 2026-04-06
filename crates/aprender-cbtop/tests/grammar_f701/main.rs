#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Grammar of ComputeBlock Falsification Tests (F701-F720)
//!
//! PMAT-018: Test the Grammar of ComputeBlock DSL per §32.
//!
//! # Falsification Criteria
//!
//! | ID | Claim | Test | Pass Criteria |
//! |----|-------|------|---------------|
//! | F701 | Builder rejects incomplete spec | Build without workload | Returns Err |
//! | F702 | Strategy fallback works | Request GPU on CPU-only | Falls back to CPU |
//! | F703 | Resource scaling honors limits | Request 1TB memory | Error/Cap applied |
//! | F704 | Composition output consistent | Batch(1) vs None | Identical output |
//! | F710 | Identity transform is no-op | Apply Identity | Output == Input |
//! | F711 | Scale domain validation | Domain(10, 0) | Returns Err |
//! | F719 | Builder immutability | Reuse builder | Independent instances |

mod dsl_tests;
mod grammar_tests;
