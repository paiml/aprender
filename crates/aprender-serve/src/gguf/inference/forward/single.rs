//! Single-token forward pass with KV cache
//!
//! Contains forward_single_with_cache and forward_single_with_cache_adaptive.
//! These are the decode-phase entry points for autoregressive generation.

use crate::brick::BrickProfiler;
use crate::error::Result;
use crate::gguf::ops;
#[cfg(feature = "gpu")]
use crate::gguf::DispatchMetrics;
use crate::gguf::{
    InferenceScratchBuffer, OwnedQuantizedKVCache, OwnedQuantizedLayer, OwnedQuantizedModel,
    GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K,
};

include!("results.rs");
include!("forward_single_profiled.rs");
