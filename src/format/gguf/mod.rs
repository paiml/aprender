//! GGUF Import/Export (spec §7.2)
//!
//! Pure Rust reader/writer for GGUF format (llama.cpp compatible).
//! WASM compatible - no C/C++ dependencies.
//!
//! # Format Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Magic: "GGUF" (4 bytes)                 │
//! │ Version: u32 (currently 3)              │
//! │ Tensor count: u64                       │
//! │ Metadata KV count: u64                  │
//! ├─────────────────────────────────────────┤
//! │ Metadata KV pairs                       │
//! ├─────────────────────────────────────────┤
//! │ Tensor info array                       │
//! ├─────────────────────────────────────────┤
//! │ Tensor data (aligned)                   │
//! └─────────────────────────────────────────┘
//! ```
//!
//! Reference: Gerganov, G. (2023). GGUF Format.

// Submodules (PMAT-199: split from monolithic gguf.rs)
pub mod types;
pub mod reader;
pub mod dequant;
pub mod api;

// Re-exports for backward compatibility
pub use types::*;
pub use reader::*;
pub use dequant::*;
pub use api::*;

// Module-level imports for test access via `use super::*;`
#[allow(unused_imports)]
use std::collections::BTreeMap;
#[allow(unused_imports)]
use std::io::{self, Read, Write};
#[allow(unused_imports)]
use std::fs::File;
#[allow(unused_imports)]
use std::path::Path;
#[allow(unused_imports)]
use crate::error::{AprenderError, Result};


#[cfg(test)]
mod tests;
