//! Attention mechanisms: incremental attention, flash decoding, tensor core attention
//!
//! This module implements:
//! - PAR-023: GPU-Resident Incremental Attention
//! - PAR-118: Flash Decoding for parallel KV processing
//! - PAR-065: Tensor Core Attention
//! - Batched attention for multi-sequence processing

use super::*;

include!("gemm_fp16_tensorcore.rs");
include!("incremental_attention_tests.rs");
