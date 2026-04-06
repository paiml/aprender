//! # simular
//!
//! Unified Simulation Engine for the Sovereign AI Stack.
//!
//! A falsifiable, reproducible simulation framework implementing:
//! - Toyota Production System (TPS): Jidoka, Poka-Yoke, Heijunka, Kaizen
//! - JPL Mission-Critical Verification: Power of 10 rules
//! - Popperian Falsification: Null hypothesis testing
//!
//! ## Example
//!
//! ```rust
//! use simular::prelude::*;
//!
//! // Create a deterministic simulation
//! let config = SimConfig::builder()
//!     .seed(42)
//!     .build();
//! ```

#![forbid(unsafe_code)]
// Production code: deny unwrap/expect/panic (tests/examples inherit allows from Cargo.toml)
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// pedantic + nursery set in Cargo.toml [lints.clippy] — individual allows there too
#![allow(
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suspicious_operation_groupings,  // False positive for variance = E[X²] - E[X]²
    clippy::imprecise_flops,   // Numerical code choices are intentional
    clippy::no_effect_underscore_binding,
    clippy::missing_const_for_fn,  // Many functions can't be const in stable Rust
    clippy::needless_range_loop,   // Sometimes range loops are clearer
    clippy::manual_midpoint,       // Manual midpoint is intentional in numerical code
    clippy::manual_is_multiple_of, // is_multiple_of is unstable in Rust 1.85
)]
// Test code is allowed to use unwrap/expect/panic for concise assertions
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreadable_literal,
        clippy::default_constructed_unit_structs,
        clippy::map_unwrap_or,
        clippy::single_char_pattern,
        clippy::needless_collect,
        clippy::missing_docs_in_private_items,
    )
)]

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

pub mod cli;
pub mod config;
pub mod demos;
pub mod discovery;
pub mod domains;
pub mod edd;
pub mod engine;
pub mod error;
pub mod falsification;
pub mod orbit;
pub mod renderers;
pub mod replay;
pub mod scenarios;
#[cfg(feature = "tui")]
pub mod tui;
pub mod visualization;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::config::{SimConfig, SimConfigBuilder};
    pub use crate::engine::jidoka::{JidokaGuard, JidokaViolation};
    pub use crate::engine::rng::SimRng;
    pub use crate::engine::{SimEngine, SimState, SimTime};
    pub use crate::error::{SimError, SimResult};
    pub use crate::falsification::{FalsifiableHypothesis, NHSTResult};
}

/// Re-export for public API
pub use error::{SimError, SimResult};
