#![allow(clippy::disallowed_methods)]
//! Mutation Testing Verification Tests (MUT-01 to MUT-15)
//!
//! These tests verify mutation testing infrastructure and create mutation-sensitive
//! tests as specified in spec v3.0.0 Part V: Mutation Testing Integration.
//!
//! Mutation testing operationalizes Popper's falsificationism by asking:
//! "If we mutate the code, do the tests fail?"
//!
//! Citation: DeMillo, R.A., Lipton, R.J., & Sayward, F.G. (1978).
//! Hints on Test Data Selection: Help for the Practicing Programmer.
//! IEEE Computer, 11(4), 34-41.

use std::path::Path;

include!("includes/mutation_testing_part_01.rs");
include!("includes/mutation_testing_part_02.rs");
