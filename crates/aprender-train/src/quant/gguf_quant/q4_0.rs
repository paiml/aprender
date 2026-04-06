//! Q4_0 quantization format

use super::GGUF_BLOCK_SIZE;
use serde::{Deserialize, Serialize};

/// Q4_0 quantized tensor (GGUF format)
///
/// 4-bit quantization with per-block f16 scale factors.
/// Each block: 32 values → 18 bytes (2 scale + 16 data)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Q4_0 {
    /// Per-block scale factors (stored as f32, converted to f16 on export)
    pub scales: Vec<f32>,
    /// Packed 4-bit data (2 values per byte, 16 bytes per block)
    pub data: Vec<u8>,
    /// Original number of elements
    pub len: usize,
}

impl Q4_0 {
    /// Quantize f32 values to Q4_0 format
    pub fn quantize(values: &[f32]) -> Self {
        let len = values.len();
        let num_blocks = len.div_ceil(GGUF_BLOCK_SIZE);

        let mut scales = Vec::with_capacity(num_blocks);
        let mut data = Vec::with_capacity(num_blocks * 16); // 16 bytes per block

        for block_idx in 0..num_blocks {
            let start = block_idx * GGUF_BLOCK_SIZE;
            let end = (start + GGUF_BLOCK_SIZE).min(len);
            let block = &values[start..end];

            // Compute scale: max absolute value / 7 (4-bit signed: -8 to 7)
            let max_abs = block
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            let scale = if max_abs < 1e-10 { 1e-10 } else { max_abs / 7.0 };
            scales.push(scale);

            // Quantize block (pad with zeros if incomplete)
            let mut block_data = [0u8; 16];
            for i in 0..GGUF_BLOCK_SIZE {
                let val = if start + i < end { block[i] } else { 0.0 };

                // Quantize to [-8, 7] range
                let q = ((val / scale).round().clamp(-8.0, 7.0) as i8) & 0x0F;

                // Pack 2 values per byte
                if i % 2 == 0 {
                    block_data[i / 2] = (q as u8) & 0x0F;
                } else {
                    block_data[i / 2] |= ((q as u8) & 0x0F) << 4;
                }
            }
            data.extend_from_slice(&block_data);
        }

        Self { scales, data, len }
    }

    /// Dequantize Q4_0 back to f32
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.len);
        let num_blocks = self.scales.len();

        for block_idx in 0..num_blocks {
            let scale = self.scales[block_idx];
            let start = block_idx * GGUF_BLOCK_SIZE;
            let block_len = (self.len - start).min(GGUF_BLOCK_SIZE);

            for i in 0..block_len {
                let byte_idx = block_idx * 16 + i / 2;
                let byte = self.data[byte_idx];

                // Extract 4-bit value
                let nibble = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };

                // Sign extend from 4-bit
                let q = if nibble & 0x08 != 0 { (nibble | 0xF0) as i8 } else { nibble as i8 };

                result.push(f32::from(q) * scale);
            }
        }

        result
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.scales.len() * 4 + self.data.len() // scales as f32 for now
    }

    /// Get GGUF-format memory (with f16 scales)
    pub fn gguf_bytes(&self) -> usize {
        self.scales.len() * 2 + self.data.len() // 2 bytes per f16 scale
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
