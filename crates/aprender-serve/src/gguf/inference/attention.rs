//! Attention computation for OwnedQuantizedModel
//!
//! Contains apply_rope (rotary position embeddings), causal_attention,
//! and KV-cache based attention variants with GQA support.

// RealizarError used by transitively included attention_gqa.rs
#[allow(unused_imports)]
use crate::error::{RealizarError, Result};
use crate::gguf::{GGUFConfig, OwnedQuantizedModel};

include!("flash_attention_dispatch.rs");
