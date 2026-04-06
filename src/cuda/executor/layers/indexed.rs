//! Indexed layer operations for optimized decode path
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-043: transformer_layer_indexed (hot path for decode)
//! - Private helpers for indexed operations

#![allow(clippy::wildcard_imports)]

use super::super::*;

include!("gemv_dispatch.rs");
include!("indexed_03.rs");
