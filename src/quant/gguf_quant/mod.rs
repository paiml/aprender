//! GGUF-compatible quantization formats (Q4_0, Q8_0)
//!
//! Implements quantization formats compatible with llama.cpp and GGUF:
//! - Q4_0: 4-bit quantization with per-block f16 scale (32 elements/block)
//! - Q8_0: 8-bit quantization with per-block f16 scale (32 elements/block)
//!
//! Block structure:
//! - Q4_0: 2 bytes scale (f16) + 16 bytes data (32 × 4-bit) = 18 bytes/block
//! - Q8_0: 2 bytes scale (f16) + 32 bytes data (32 × 8-bit) = 34 bytes/block

mod q4_0;
mod q8_0;
mod quant_type;

#[cfg(test)]
mod tests;

pub use q4_0::Q4_0;
pub use q8_0::Q8_0;
pub use quant_type::GGUFQuantType;

/// GGUF block size (standard for llama.cpp)
pub const GGUF_BLOCK_SIZE: usize = 32;
