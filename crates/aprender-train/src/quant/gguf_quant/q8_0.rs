//! Q8_0 quantization format

use super::GGUF_BLOCK_SIZE;
use serde::{Deserialize, Serialize};

/// Q8_0 quantized tensor (GGUF format)
///
/// 8-bit quantization with per-block f16 scale factors.
/// Each block: 32 values → 34 bytes (2 scale + 32 data)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Q8_0 {
    /// Per-block scale factors (stored as f32, converted to f16 on export)
    pub scales: Vec<f32>,
    /// 8-bit quantized data (1 byte per value)
    pub data: Vec<i8>,
    /// Original number of elements
    pub len: usize,
}

impl Q8_0 {
    /// Quantize f32 values to Q8_0 format
    pub fn quantize(values: &[f32]) -> Self {
        let len = values.len();
        let num_blocks = len.div_ceil(GGUF_BLOCK_SIZE);

        let mut scales = Vec::with_capacity(num_blocks);
        let mut data = Vec::with_capacity(len);

        for block_idx in 0..num_blocks {
            let start = block_idx * GGUF_BLOCK_SIZE;
            let end = (start + GGUF_BLOCK_SIZE).min(len);
            let block = &values[start..end];

            // Compute scale: max absolute value / 127 (8-bit signed: -128 to 127)
            let max_abs = block
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            let scale = if max_abs < 1e-10 { 1e-10 } else { max_abs / 127.0 };
            scales.push(scale);

            // Quantize block
            for &val in block {
                let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
                data.push(q);
            }

            // Pad incomplete blocks with zeros
            let padding = GGUF_BLOCK_SIZE - block.len();
            data.extend(std::iter::repeat_n(0i8, padding));
        }

        // Trim padding from last block
        data.truncate(len);

        Self { scales, data, len }
    }

    /// Dequantize Q8_0 back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.len);

        for (i, &q) in self.data.iter().enumerate() {
            let block_idx = i / GGUF_BLOCK_SIZE;
            let scale = self.scales[block_idx];
            result.push(f32::from(q) * scale);
        }

        result
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.scales.len() * 4 + self.data.len()
    }

    /// Get GGUF-format memory (with f16 scales)
    pub fn gguf_bytes(&self) -> usize {
        self.scales.len() * 2 + self.data.len()
    }

    /// Get compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        let original = self.len * 4;
        original as f32 / self.gguf_bytes() as f32
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.scales.len()
    }
}
