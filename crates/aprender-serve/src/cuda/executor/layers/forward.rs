//! Forward pass operations for transformer inference
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-023: forward_all_layers_gpu
//! - PAR-023: forward_all_layers_gpu_to_logits

#![allow(clippy::wildcard_imports)]

use super::super::*;

include!("forward_utils.rs");
include!("forward_forward.rs");
