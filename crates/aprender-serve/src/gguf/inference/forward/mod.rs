//! Forward pass implementations for OwnedQuantizedModel
//!
//! This module contains all forward pass variants:
//! - `core.rs`: Basic forward and forward_cached (prefill)
//! - `single.rs`: Single-token forward with cache (decode)
//! - `batch.rs`: Batched forward pass variants

mod batch;
mod core;
mod encoder_decoder;
mod single;

// PMAT-395: Re-export encoder-decoder types
pub use encoder_decoder::EncoderOutput;

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod core_tests;
#[cfg(test)]
mod encoder_decoder_tests;
#[cfg(test)]
mod single_tests;
