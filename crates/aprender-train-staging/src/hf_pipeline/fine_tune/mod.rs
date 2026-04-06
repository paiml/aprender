//! Fine-tuning configuration for HuggingFace models
//!
//! Bridges the HF pipeline with LoRA/QLoRA adapters for efficient fine-tuning.
//!
//! # References
//!
//! [1] Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language
//!     Models." arXiv:2106.09685
//!
//! [2] Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized
//!     LLMs." arXiv:2305.14314

mod config;
mod memory;
mod method;

#[cfg(test)]
mod tests;

pub use config::FineTuneConfig;
pub use memory::{MemoryRequirement, MixedPrecision};
pub use method::FineTuneMethod;
