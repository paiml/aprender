//! Extended coverage tests for the stress testing framework.
//!
//! Split into submodules for maintainability:
//! - `clone_debug`: Clone and Debug derive coverage tests
//! - `fields_defaults`: Field access, default values, and boundary tests
//! - `runner_rng`: StressTestRunner, StressRng, and verification tests

use std::time::Duration;

use super::*;

mod clone_debug;
mod fields_defaults;
mod runner_rng;
