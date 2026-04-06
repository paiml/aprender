//! LoRA (Low-Rank Adaptation) layer implementation
//!
//! LoRA enables parameter-efficient fine-tuning by adding trainable low-rank
//! decomposition matrices to frozen pretrained weights.
//!
//! For a frozen weight matrix W ∈ ℝ^(d_out × d_in), LoRA adds:
//! ΔW = B @ A where A ∈ ℝ^(r × d_in) and B ∈ ℝ^(d_out × r)
//!
//! Forward pass: y = (W + α·B·A) @ x = W@x + α·(B@(A@x))
//! where α is a scaling factor (typically alpha/r)

mod core;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod falsify_tests;

pub use self::core::{LoRALayer, LoRAScaling};
