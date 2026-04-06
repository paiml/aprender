//! Batched forward pass operations for multi-sequence inference
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-111: forward_batched_to_token_ids
//! - PAR-121: forward_batched_to_token_ids_graphed

#![allow(clippy::wildcard_imports)]

use super::super::*;

include!("rmsnorm_ptr.rs");
include!("batched_03.rs");
