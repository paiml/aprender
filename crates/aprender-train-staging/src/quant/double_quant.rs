//! Double quantization for QLoRA (ENT-LoRA-008)
//!
//! Quantizes the FP32 absmax scale factors from 4-bit quantization to 8-bit,
//! saving ~0.37 bits/param (~0.5 GB for a 7B model).
//!
//! Two-level quantization:
//! - Level 1: Values → 4-bit with FP32 scales (standard, 64-element blocks)
//! - Level 2: FP32 scales → 8-bit unsigned with FP32 super-scales (256-scale blocks)
//!
//! Memory per 64 values:
//! - Without double quant: 32 bytes (data) + 4 bytes (scale) = 36 bytes = 4.50 bits/param
//! - With double quant: 32 bytes (data) + 1 byte (scale) + 0.016 bytes (super) ≈ 33 bytes = 4.13 bits/param

use serde::{Deserialize, Serialize};

use super::quant4bit::{quantize_4bit, BLOCK_SIZE};

/// Block size for second-level scale quantization (256 scales per super-block)
pub const DOUBLE_QUANT_BLOCK_SIZE: usize = 256;

/// Double-quantized 4-bit representation
///
/// Same 4-bit packed data as `Quantized4Bit`, but scale factors are stored as
/// 8-bit unsigned values with a second-level FP32 super-scale per group of 256.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleQuantized4Bit {
    /// 8-bit quantized scale factors (one per first-level block)
    pub quantized_scales: Vec<u8>,
    /// Super-scale factors (one FP32 per DOUBLE_QUANT_BLOCK_SIZE first-level blocks)
    pub super_scales: Vec<f32>,
    /// Quantized data: 2 values per byte (4 bits each) — identical to Quantized4Bit
    pub data: Vec<u8>,
    /// Original number of elements
    pub len: usize,
    /// Number of first-level blocks
    pub num_blocks: usize,
}

impl DoubleQuantized4Bit {
    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.quantized_scales.len() // 1 byte per quantized scale
            + self.super_scales.len() * 4 // 4 bytes per f32 super-scale
            + self.data.len() // packed 4-bit data
    }

    /// Compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.len * 4;
        let compressed_bytes = self.memory_bytes();
        if compressed_bytes == 0 {
            return 1.0;
        }
        original_bytes as f32 / compressed_bytes as f32
    }

    /// Memory saved compared to single quantization (bytes)
    pub fn double_quant_savings(&self) -> usize {
        // Single quant: num_blocks * 4 bytes per f32 scale
        // Double quant: num_blocks * 1 byte + super_scales * 4 bytes
        let single_scale_bytes = self.num_blocks * 4;
        let double_scale_bytes = self.quantized_scales.len() + self.super_scales.len() * 4;
        single_scale_bytes.saturating_sub(double_scale_bytes)
    }
}

/// Quantize values to 4-bit with double quantization of scale factors
///
/// First applies standard 4-bit quantization, then quantizes the resulting
/// FP32 scale factors to 8-bit with a second-level block size of 256.
pub fn quantize_4bit_double(values: &[f32]) -> DoubleQuantized4Bit {
    // Step 1: Standard 4-bit quantization
    let single = quantize_4bit(values);
    let num_blocks = single.scales.len();

    // Step 2: Double-quantize the scale factors
    let num_super_blocks = num_blocks.div_ceil(DOUBLE_QUANT_BLOCK_SIZE);
    let mut quantized_scales = Vec::with_capacity(num_blocks);
    let mut super_scales = Vec::with_capacity(num_super_blocks);

    for sb in 0..num_super_blocks {
        let start = sb * DOUBLE_QUANT_BLOCK_SIZE;
        let end = (start + DOUBLE_QUANT_BLOCK_SIZE).min(num_blocks);
        let scale_block = &single.scales[start..end];

        // Super-scale = max of this block of scales
        // Scales are always non-negative (absmax / 7)
        let max_scale =
            scale_block.iter().copied().max_by(f32::total_cmp).unwrap_or(1e-8).max(1e-8); // avoid division by zero
        super_scales.push(max_scale);

        // Quantize each scale to u8: q = round(scale / max_scale * 255)
        for &scale in scale_block {
            let normalized = scale / max_scale;
            let q = (normalized * 255.0).round().clamp(0.0, 255.0) as u8;
            quantized_scales.push(q);
        }
    }

    DoubleQuantized4Bit {
        quantized_scales,
        super_scales,
        data: single.data,
        len: single.len,
        num_blocks,
    }
}

/// Dequantize double-quantized 4-bit values back to f32
pub fn dequantize_4bit_double(dq: &DoubleQuantized4Bit) -> Vec<f32> {
    // Step 1: Reconstruct FP32 scales from double-quantized representation
    let scales = reconstruct_scales(dq);

    // Step 2: Standard 4-bit dequantization using reconstructed scales
    let mut result = Vec::with_capacity(dq.len);

    for block_idx in 0..dq.num_blocks {
        let scale = scales[block_idx];
        let start = block_idx * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(dq.len);
        let block_len = end - start;

        for i in 0..block_len {
            let byte_idx = usize::midpoint(start, i);
            let byte = dq.data[byte_idx];

            let q_val = if (start + i).is_multiple_of(2) {
                let nibble = (byte >> 4) & 0x0F;
                if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                }
            } else {
                let nibble = byte & 0x0F;
                if nibble & 0x08 != 0 {
                    (nibble | 0xF0) as i8
                } else {
                    nibble as i8
                }
            };

            result.push(f32::from(q_val) * scale);
        }
    }

    result
}

/// Reconstruct FP32 scales from double-quantized representation
fn reconstruct_scales(dq: &DoubleQuantized4Bit) -> Vec<f32> {
    let mut scales = Vec::with_capacity(dq.num_blocks);

    for (i, &q_scale) in dq.quantized_scales.iter().enumerate() {
        let super_idx = i / DOUBLE_QUANT_BLOCK_SIZE;
        let super_scale = dq.super_scales[super_idx];
        // Dequantize: scale = q / 255 * super_scale
        let scale = f32::from(q_scale) / 255.0 * super_scale;
        scales.push(scale);
    }

    scales
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::quant::{dequantize_4bit, quantize_4bit};
    use proptest::prelude::*;

    // ========================================================================
    // SPEC REQUIREMENT: dequantize(double_quant(x)) within 1% of dequantize(single_quant(x))
    // ========================================================================

    #[test]
    fn test_ent_lora_008_double_quant_within_1pct_of_single() {
        // Realistic transformer weight distribution
        let values: Vec<f32> = (0..4096).map(|i| ((i as f32 * 0.1).sin() * 2.0)).collect();

        let single = quantize_4bit(&values);
        let single_deq = dequantize_4bit(&single);

        let double = quantize_4bit_double(&values);
        let double_deq = dequantize_4bit_double(&double);

        assert_eq!(single_deq.len(), double_deq.len());

        for (i, (s, d)) in single_deq.iter().zip(double_deq.iter()).enumerate() {
            let diff = (s - d).abs();
            let tolerance = s.abs() * 0.01 + 1e-6; // 1% relative + small absolute
            assert!(
                diff <= tolerance,
                "Double quant diverged at [{i}]: single={s}, double={d}, diff={diff}, tol={tolerance}"
            );
        }
    }

    #[test]
    fn test_ent_lora_008_memory_savings() {
        // 7B model params: ~7 billion. Test with representative block count.
        // 65536 values = 1024 blocks of 64
        let values: Vec<f32> = (0..65536).map(|i| (i as f32 * 0.01).sin()).collect();

        let single = quantize_4bit(&values);
        let double = quantize_4bit_double(&values);

        let single_bytes = single.memory_bytes();
        let double_bytes = double.memory_bytes();

        // Double quant should use fewer bytes
        assert!(
            double_bytes < single_bytes,
            "Double quant ({double_bytes}B) should be smaller than single ({single_bytes}B)"
        );

        // Savings should be ~3 bytes per block (4 bytes f32 → 1 byte u8)
        let savings = double.double_quant_savings();
        assert!(savings > 0, "Should have positive savings, got {savings}");

        // Savings per param: (savings * 8 bits) / num_values
        let savings_bits_per_param = (savings as f64 * 8.0) / values.len() as f64;
        assert!(
            savings_bits_per_param > 0.3,
            "Expected ~0.37 bits/param savings, got {savings_bits_per_param:.3}"
        );
    }

    #[test]
    fn test_ent_lora_008_round_trip_preserves_length() {
        let values: Vec<f32> = (0..200).map(|i| i as f32 * 0.5).collect();
        let dq = quantize_4bit_double(&values);
        let result = dequantize_4bit_double(&dq);
        assert_eq!(result.len(), values.len());
    }

    #[test]
    fn test_ent_lora_008_zeros() {
        let values = vec![0.0; 128];
        let dq = quantize_4bit_double(&values);
        let result = dequantize_4bit_double(&dq);

        for val in result {
            assert!(val.abs() < 1e-6, "Zero input should dequantize to ~0, got {val}");
        }
    }

    #[test]
    fn test_ent_lora_008_compression_ratio_better_than_single() {
        let values: Vec<f32> = (0..65536).map(|i| (i as f32 * 0.01).cos()).collect();

        let single = quantize_4bit(&values);
        let double = quantize_4bit_double(&values);

        assert!(
            double.compression_ratio() > single.compression_ratio(),
            "Double quant ratio ({:.2}) should exceed single ({:.2})",
            double.compression_ratio(),
            single.compression_ratio()
        );
    }

    #[test]
    fn test_ent_lora_008_small_input() {
        // Fewer values than one scale block
        let values = vec![1.0, -2.0, 3.0, -4.0];
        let dq = quantize_4bit_double(&values);
        let result = dequantize_4bit_double(&dq);
        assert_eq!(result.len(), 4);
        assert_eq!(dq.num_blocks, 1);
        assert_eq!(dq.super_scales.len(), 1);
    }

    #[test]
    fn test_ent_lora_008_scale_reconstruction_accuracy() {
        // Verify scale factors survive double quantization well
        let values: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.05).sin() * 5.0).collect();

        let single = quantize_4bit(&values);
        let double = quantize_4bit_double(&values);
        let reconstructed = reconstruct_scales(&double);

        assert_eq!(reconstructed.len(), single.scales.len());

        for (i, (orig, recon)) in single.scales.iter().zip(reconstructed.iter()).enumerate() {
            let diff = (orig - recon).abs();
            let tolerance = orig.abs() * 0.01 + 1e-8; // 1% relative
            assert!(
                diff <= tolerance,
                "Scale [{i}] diverged: orig={orig}, recon={recon}, diff={diff}"
            );
        }
    }

    // ========================================================================
    // PROPERTY TESTS
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        #[test]
        fn prop_double_quant_within_1pct(
            n in (64usize..1024).prop_map(|n| n - (n % 64)), // multiple of 64
            magnitude in 0.1f32..10.0,
        ) {
            let values: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.1).sin() * magnitude))
                .collect();

            let single_deq = dequantize_4bit(&quantize_4bit(&values));
            let double_deq = dequantize_4bit_double(&quantize_4bit_double(&values));

            prop_assert_eq!(single_deq.len(), double_deq.len());

            for (s, d) in single_deq.iter().zip(double_deq.iter()) {
                let diff = (s - d).abs();
                let tolerance = s.abs() * 0.01 + 1e-5;
                prop_assert!(
                    diff <= tolerance,
                    "single={s}, double={d}, diff={diff}, tol={tolerance}"
                );
            }
        }

        #[test]
        fn prop_double_quant_preserves_length(n in 1usize..2048) {
            let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let dq = quantize_4bit_double(&values);
            let result = dequantize_4bit_double(&dq);
            prop_assert_eq!(result.len(), n);
        }

        #[test]
        fn prop_double_quant_uses_less_memory(
            n in (256usize..8192).prop_map(|n| n - (n % 64)),
        ) {
            let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
            let single = quantize_4bit(&values);
            let double = quantize_4bit_double(&values);

            // Double quant saves 3 bytes per scale (f32→u8) minus super-scale overhead
            // For large enough inputs this should always be positive
            if single.scales.len() > DOUBLE_QUANT_BLOCK_SIZE {
                prop_assert!(double.memory_bytes() < single.memory_bytes());
            }
        }
    }
}
