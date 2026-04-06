//! Token generation for OwnedQuantizedModel
//!
//! Contains generate, generate_with_cache, generate_with_cache_streaming,
//! generate_with_scratch, and sampling methods.

use crate::brick::BrickProfiler;
use crate::error::{RealizarError, Result};
use crate::gguf::ops;
#[cfg(feature = "gpu")]
use crate::gguf::DispatchMetrics;
use crate::gguf::{
    InferenceScratchBuffer, OwnedQuantizedKVCache, OwnedQuantizedModel, QuantizedGenerateConfig,
};
use rand::Rng;

include!("generate_scratch.rs");
include!("generation_argmax_basic.rs");
