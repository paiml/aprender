//! Quantization format definitions and properties.

// Allow non-camel-case for GGML standard quantization type names
#![allow(non_camel_case_types)]

use std::fmt;

/// Supported quantization formats for ComputeBricks.
///
/// Based on GGML/llama.cpp quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// Full precision (baseline)
    F32,
    /// Half precision
    F16,
    /// Brain float 16
    BF16,

    /// 4-bit quantization, no scales per block (32 values/block)
    Q4_0,
    /// 4-bit quantization with K-quants (256 values/super-block, 6-bit scales)
    Q4_K,
    /// 5-bit quantization with K-quants
    Q5_K,
    /// 6-bit quantization with K-quants
    Q6_K,
    /// 8-bit quantization, simple (32 values/block)
    Q8_0,

    /// GPTQ format (ExLlama compatible)
    Gptq { bits: u8, group_size: u16 },
    /// AWQ format (activation-aware)
    Awq { bits: u8 },
}

impl QuantFormat {
    /// Effective bits per weight (including scales and metadata).
    pub fn bits_per_weight(&self) -> f64 {
        match self {
            QuantFormat::F32 => 32.0,
            QuantFormat::F16 => 16.0,
            QuantFormat::BF16 => 16.0,
            QuantFormat::Q4_0 => 4.5, // 4 bits + scale overhead
            QuantFormat::Q4_K => 4.5, // 4 bits + 6-bit scales
            QuantFormat::Q5_K => 5.5,
            QuantFormat::Q6_K => 6.5,
            QuantFormat::Q8_0 => 8.5, // 8 bits + scale
            QuantFormat::Gptq { bits, .. } => *bits as f64 + 0.5,
            QuantFormat::Awq { bits } => *bits as f64 + 0.5,
        }
    }

    /// Memory ratio compared to F16 (lower is better).
    pub fn memory_ratio(&self) -> f64 {
        self.bits_per_weight() / 16.0
    }

    /// Expected perplexity delta compared to F16 (lower is better).
    ///
    /// Based on llama.cpp benchmarks.
    pub fn expected_ppl_delta(&self) -> f64 {
        match self {
            QuantFormat::F32 => 0.0,
            QuantFormat::F16 => 0.0,
            QuantFormat::BF16 => 0.01,
            QuantFormat::Q4_0 => 0.5,
            QuantFormat::Q4_K => 0.3,
            QuantFormat::Q5_K => 0.1,
            QuantFormat::Q6_K => 0.05,
            QuantFormat::Q8_0 => 0.01,
            QuantFormat::Gptq { bits, .. } => match bits {
                4 => 0.4,
                8 => 0.02,
                _ => 0.5,
            },
            QuantFormat::Awq { bits } => match bits {
                4 => 0.2, // AWQ tends to have better quality
                _ => 0.3,
            },
        }
    }

    /// Block size (number of weights per quantization block).
    pub fn block_size(&self) -> usize {
        match self {
            QuantFormat::F32 | QuantFormat::F16 | QuantFormat::BF16 => 1,
            QuantFormat::Q4_0 | QuantFormat::Q8_0 => 32,
            QuantFormat::Q4_K | QuantFormat::Q5_K | QuantFormat::Q6_K => 256, // Super-block
            QuantFormat::Gptq { group_size, .. } => *group_size as usize,
            QuantFormat::Awq { .. } => 128,
        }
    }

    /// Bytes per block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            QuantFormat::F32 => 4,
            QuantFormat::F16 | QuantFormat::BF16 => 2,
            QuantFormat::Q4_0 => 18,  // 2 (scale) + 16 (32 x 4-bit)
            QuantFormat::Q4_K => 144, // 2+2+12+128 (super-block)
            QuantFormat::Q5_K => 176, // 2+2+12+128+32
            QuantFormat::Q6_K => 210, // 128+64+16+2
            QuantFormat::Q8_0 => 34,  // 2 (scale) + 32 (8-bit values)
            QuantFormat::Gptq { bits, group_size } => {
                let data_bytes = (*group_size as usize * *bits as usize).div_ceil(8);
                data_bytes + 4 // + scale/zero
            }
            QuantFormat::Awq { bits } => {
                let data_bytes = (128 * *bits as usize).div_ceil(8);
                data_bytes + 4
            }
        }
    }
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantFormat::F32 => write!(f, "F32"),
            QuantFormat::F16 => write!(f, "F16"),
            QuantFormat::BF16 => write!(f, "BF16"),
            QuantFormat::Q4_0 => write!(f, "Q4_0"),
            QuantFormat::Q4_K => write!(f, "Q4_K"),
            QuantFormat::Q5_K => write!(f, "Q5_K"),
            QuantFormat::Q6_K => write!(f, "Q6_K"),
            QuantFormat::Q8_0 => write!(f, "Q8_0"),
            QuantFormat::Gptq { bits, group_size } => {
                write!(f, "GPTQ-{}bit-g{}", bits, group_size)
            }
            QuantFormat::Awq { bits } => write!(f, "AWQ-{}bit", bits),
        }
    }
}

/// Dequantization strategy.
///
/// Controls when and how quantized weights are dequantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantStrategy {
    /// Fused: dequantize during matmul (best for GPU, saves memory bandwidth)
    Fused,
    /// Prefetch: dequantize ahead of compute (good for pipelining)
    Prefetch { lookahead_blocks: usize },
    /// On-demand: dequantize per block (lowest memory footprint)
    OnDemand,
}

impl Default for DequantStrategy {
    fn default() -> Self {
        DequantStrategy::Fused
    }
}
