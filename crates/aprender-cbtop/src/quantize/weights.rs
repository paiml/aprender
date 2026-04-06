//! Quantized weight storage and statistics.

use std::collections::HashMap;
use std::fmt;

use super::format::QuantFormat;

/// Quantized weight storage for a single layer.
#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    /// Quantization format
    pub format: QuantFormat,
    /// Raw quantized data
    pub data: Vec<u8>,
    /// Shape: [rows, cols] for 2D weights
    pub shape: (usize, usize),
    /// Layer name (for debugging)
    pub layer_name: String,
}

impl QuantizedWeights {
    /// Create new quantized weights.
    pub fn new(format: QuantFormat, data: Vec<u8>, shape: (usize, usize), name: &str) -> Self {
        Self {
            format,
            data,
            shape,
            layer_name: name.to_string(),
        }
    }

    /// Total number of weights.
    pub fn num_weights(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Memory footprint if stored as F16.
    pub fn f16_memory_bytes(&self) -> usize {
        self.num_weights() * 2
    }

    /// Compression ratio (F16 / quantized).
    pub fn compression_ratio(&self) -> f64 {
        self.f16_memory_bytes() as f64 / self.memory_bytes() as f64
    }

    /// Effective bits per weight (actual).
    pub fn actual_bits_per_weight(&self) -> f64 {
        (self.data.len() * 8) as f64 / self.num_weights() as f64
    }
}

/// Per-layer quantization statistics.
#[derive(Debug, Clone)]
pub struct LayerQuantStats {
    /// Layer name
    pub name: String,
    /// Quantization format
    pub format: QuantFormat,
    /// Weight count
    pub weights: usize,
    /// Memory bytes
    pub memory_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f64,
}

/// Quantization statistics for a model or layer.
#[derive(Debug, Clone, Default)]
pub struct QuantStats {
    /// Total weights across all layers
    pub total_weights: usize,
    /// Total memory (bytes) for quantized weights
    pub total_memory_bytes: usize,
    /// Memory if stored as F16
    pub f16_memory_bytes: usize,
    /// Weights per format
    pub weights_by_format: HashMap<QuantFormat, usize>,
    /// Memory per format
    pub memory_by_format: HashMap<QuantFormat, usize>,
    /// Per-layer stats
    pub layer_stats: Vec<LayerQuantStats>,
}

impl QuantStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add layer statistics.
    pub fn add_layer(&mut self, weights: &QuantizedWeights) {
        self.total_weights += weights.num_weights();
        self.total_memory_bytes += weights.memory_bytes();
        self.f16_memory_bytes += weights.f16_memory_bytes();

        *self.weights_by_format.entry(weights.format).or_default() += weights.num_weights();
        *self.memory_by_format.entry(weights.format).or_default() += weights.memory_bytes();

        self.layer_stats.push(LayerQuantStats {
            name: weights.layer_name.clone(),
            format: weights.format,
            weights: weights.num_weights(),
            memory_bytes: weights.memory_bytes(),
            compression_ratio: weights.compression_ratio(),
        });
    }

    /// Overall compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.total_memory_bytes == 0 {
            1.0
        } else {
            self.f16_memory_bytes as f64 / self.total_memory_bytes as f64
        }
    }

    /// Effective bits per weight (average).
    pub fn avg_bits_per_weight(&self) -> f64 {
        if self.total_weights == 0 {
            0.0
        } else {
            (self.total_memory_bytes * 8) as f64 / self.total_weights as f64
        }
    }

    /// Dominant format (most weights).
    pub fn dominant_format(&self) -> Option<QuantFormat> {
        self.weights_by_format
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(format, _)| *format)
    }
}

impl fmt::Display for QuantStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quantization Statistics")?;
        writeln!(f, "======================")?;
        writeln!(f, "Total Weights: {}", self.total_weights)?;
        writeln!(
            f,
            "Total Memory: {:.2} MB (quantized)",
            self.total_memory_bytes as f64 / 1_000_000.0
        )?;
        writeln!(
            f,
            "F16 Memory: {:.2} MB (baseline)",
            self.f16_memory_bytes as f64 / 1_000_000.0
        )?;
        writeln!(f, "Compression: {:.2}x", self.compression_ratio())?;
        writeln!(f, "Avg Bits/Weight: {:.2}", self.avg_bits_per_weight())?;

        if !self.weights_by_format.is_empty() {
            writeln!(f)?;
            writeln!(f, "By Format:")?;
            for (format, weights) in &self.weights_by_format {
                let memory = self.memory_by_format.get(format).unwrap_or(&0);
                writeln!(
                    f,
                    "  {}: {} weights, {:.2} MB",
                    format,
                    weights,
                    *memory as f64 / 1_000_000.0
                )?;
            }
        }

        Ok(())
    }
}
