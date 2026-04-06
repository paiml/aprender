//! ExecutionGraph - Execution Path Graph for Profiling
//!
//! PAR-201: Captures the full execution hierarchy for profiling analysis.
//!
//! Split into submodules:
//! - `core`: Struct definition, basic graph operations, scope management
//! - `export`: DOT, CSR, TUI tree, ASCII tree visualization
//! - `analysis`: Critical path, slack, roofline, ping-pong detection

mod analysis;
mod core;
mod export;

pub use self::core::ExecutionGraph;
