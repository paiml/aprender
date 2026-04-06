//! CUDA Graph-captured forward pass operations
//!
//! Extracted from layer.rs (PMAT-802) to reduce module size.
//! Contains:
//! - PAR-054: forward_all_layers_gpu_to_logits_graphed
//! - PAR-062: gpu_argmax
//! - PAR-062: forward_graphed_replay_to_token_id

#![allow(clippy::wildcard_imports)]

use super::super::*;

include!("forward_workspace_captured.rs");
include!("graphed_03.rs");
