//! Compliance Checking Module
//!
//! Per `PROBAR-SPEC-WASM-001`, this module provides compliance checking
//! for WASM threading, state synchronization patterns, and pmat integration.
//!
//! ## Usage
//!
//! ```bash
//! probar comply --wasm-threading ./src
//! probar comply --pmat ./src
//! ```
//!
//! ## Modules
//!
//! - `tarantula`: Spectrum-based fault localization
//! - `wasm_threading`: WASM threading compliance checks
//! - `pmat_bridge`: Integration with pmat for SATD/complexity analysis

pub mod pmat_bridge;
pub mod tarantula;
pub mod wasm_threading;

pub use pmat_bridge::{PmatBridge, PmatResult};
pub use tarantula::{TarantulaEngine, TarantulaReport};
pub use wasm_threading::{
    ComplianceCheck, ComplianceResult, ComplianceStatus, WasmThreadingCompliance,
};
