//! Pre-trained model architectures for inference.
//!
//! This module provides ready-to-use model implementations that combine
//! the primitives from `nn` into complete architectures.
//!
//! # Available Models
//!
//! - [`qwen2::Qwen2Model`] - Qwen2-0.5B-Instruct decoder-only transformer
//!
//! # Design Philosophy
//!
//! Models follow the "assembly pattern" - they compose existing primitives
//! (attention, normalization, feedforward) rather than duplicating code.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Model Architecture                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   ┌─────────────┐    ┌─────────────────────┐    ┌────────────┐  │
//! │   │ Embedding   │ -> │  N × DecoderLayer   │ -> │ LM Head    │  │
//! │   │ (vocab→d)   │    │  (GQA + FFN + Norm) │    │ (d→vocab)  │  │
//! │   └─────────────┘    └─────────────────────┘    └────────────┘  │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - Bai et al. (2023). "Qwen Technical Report"
//! - Vaswani et al. (2017). "Attention Is All You Need"

pub mod qwen2;

pub use qwen2::Qwen2Model;
