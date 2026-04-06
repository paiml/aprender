//! TUI module for simular.
//!
//! This module contains reusable TUI application state and logic
//! extracted from bin/*.rs to enable testing.
//!
//! The actual terminal I/O remains in the binaries, but all testable
//! state management and business logic lives here.
//!
//! # `ComputeBlocks` Integration (SIMULAR-CB-001)
//!
//! The `compute_blocks` submodule provides SIMD-optimized visualization
//! widgets from presentar-terminal for simulation metrics:
//!
//! - `EnergySparkline`: Energy conservation history
//! - `MomentumSparkline`: Angular momentum conservation history
//! - `FrameBudgetTrend`: Heijunka frame budget utilization

#[cfg(feature = "tui")]
pub mod compute_blocks;
#[cfg(feature = "tui")]
pub mod orbit_app;
#[cfg(feature = "tui")]
pub mod tsp_app;

#[cfg(test)]
#[cfg(feature = "tui")]
mod tests;
