//! Forward pass implementations for OwnedQuantizedModel
//!
//! This module contains all forward pass variants:
//! - `core.rs`: Basic forward and forward_cached (prefill)
//! - `single.rs`: Single-token forward with cache (decode)
//! - `batch.rs`: Batched forward pass variants

mod batch;
mod core;
mod encoder_decoder;

mod forward_qwen3_moe;
mod forward_qwen3_moe_gpu;
mod forward_qwen3_moe_traced;
mod single;
mod traced;

// PMAT-395: Re-export encoder-decoder types
pub use encoder_decoder::EncoderOutput;

// PP-ARCH-001 §9.2 (#3422): shared attention/FFN blocks as free fns, reachable
// by every architecture's forward rather than only through OwnedQuantizedModel.
#[allow(unused_imports)]
pub(crate) use batch::{
    online_softmax, standard_single_head_attention, standard_softmax, tiled_single_head_attention,
};
#[cfg(feature = "gpu")]
#[allow(unused_imports)]
pub(crate) use batch::{parallel_batched_qk_scores, reshape_for_parallel_heads};
#[allow(unused_imports)]
pub(crate) use single::{ffn_gated_activate, first_token_attention, post_norm_in_place};

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod core_tests;
#[cfg(test)]
mod encoder_decoder_tests;
#[cfg(test)]
mod single_tests;
