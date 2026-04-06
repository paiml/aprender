//! QuantizedBrick Implementation (PMAT-013)
//!
//! Implements quantized weight support for ComputeBricks per cbtop spec S17.
//!
//! # Supported Formats
//!
//! | Format | Bits/Weight | Memory | Perplexity Delta |
//! |--------|------------|--------|------------------|
//! | Q4_0   | 4.0        | 25%    | ~0.5%            |
//! | Q4_K   | 4.5        | 28%    | ~0.3%            |
//! | Q5_K   | 5.5        | 34%    | ~0.1%            |
//! | Q8_0   | 8.0        | 50%    | ~0.01%           |
//!
//! # Citations
//!
//! - [Dettmers et al. 2022] "LLM.int8(): 8-bit Matrix Multiplication" NeurIPS
//! - [Frantar et al. 2023] "GPTQ: Accurate Post-Training Quantization" ICLR
//! - [Lin et al. 2023] "AWQ: Activation-aware Weight Quantization" MLSys

// Allow non-camel-case for GGML standard quantization type names
#![allow(non_camel_case_types)]

mod format;
mod gguf;
mod weights;

pub use format::{DequantStrategy, QuantFormat};
pub use gguf::{GgufError, GgufHeader, GgufLoader, GgufResult, GgufTensorInfo, GgufValue};
pub use weights::{LayerQuantStats, QuantStats, QuantizedWeights};

use std::fmt;

/// QuantizedBrick wraps compute operations with quantized weights.
///
/// Per cbtop spec S17.2.
#[derive(Debug, Clone)]
pub struct QuantizedBrick {
    /// Brick name
    pub name: String,
    /// Quantized weights for this brick
    pub weights: Option<QuantizedWeights>,
    /// Dequantization strategy
    pub dequant_strategy: DequantStrategy,
    /// Performance budget (tokens per second)
    pub budget_tok_per_sec: Option<u64>,
}

impl QuantizedBrick {
    /// Create a new quantized brick.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            weights: None,
            dequant_strategy: DequantStrategy::default(),
            budget_tok_per_sec: None,
        }
    }

    /// Set quantized weights.
    pub fn with_weights(mut self, weights: QuantizedWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set dequantization strategy.
    pub fn with_dequant_strategy(mut self, strategy: DequantStrategy) -> Self {
        self.dequant_strategy = strategy;
        self
    }

    /// Set performance budget.
    pub fn with_budget(mut self, tok_per_sec: u64) -> Self {
        self.budget_tok_per_sec = Some(tok_per_sec);
        self
    }

    /// Get memory footprint (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.weights.as_ref().map_or(0, |w| w.memory_bytes())
    }

    /// Get effective bits per weight.
    pub fn bits_per_weight(&self) -> f64 {
        self.weights.as_ref().map_or(0.0, |w| w.actual_bits_per_weight())
    }

    /// Get quantization format.
    pub fn format(&self) -> Option<QuantFormat> {
        self.weights.as_ref().map(|w| w.format)
    }

    /// Check if weights are loaded.
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }
}

impl fmt::Display for QuantizedBrick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuantizedBrick[{}]", self.name)?;
        if let Some(weights) = &self.weights {
            write!(
                f,
                " format={} weights={} memory={:.2}MB",
                weights.format,
                weights.num_weights(),
                weights.memory_bytes() as f64 / 1_000_000.0
            )?;
        }
        Ok(())
    }
}

/// GGML tensor type to QuantFormat mapping.
pub fn ggml_type_to_format(ggml_type: u32) -> Option<QuantFormat> {
    match ggml_type {
        0 => Some(QuantFormat::F32),
        1 => Some(QuantFormat::F16),
        2 => Some(QuantFormat::Q4_0),
        3 => Some(QuantFormat::Q4_K), // Q4_1 in GGML, map to Q4_K
        8 => Some(QuantFormat::Q8_0),
        12 => Some(QuantFormat::Q4_K),
        13 => Some(QuantFormat::Q5_K),
        14 => Some(QuantFormat::Q6_K),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
