//! QLoRA (Quantized LoRA) implementation
//!
//! QLoRA reduces memory usage by storing frozen base weights in 4-bit quantized format.
//! During forward pass, weights are dequantized on-the-fly.
//!
//! Memory savings: ~75% for base weights (4-bit vs 32-bit)
//! LoRA adapters remain in full precision for training.

use crate::autograd::matmul;
use crate::lora::LoRALayer;
use crate::quant::{
    dequantize_4bit, dequantize_4bit_double, quantize_4bit, quantize_4bit_double,
    DoubleQuantized4Bit, Quantized4Bit,
};
use crate::Tensor;

/// QLoRA layer with 4-bit quantized base weight
///
/// Memory-efficient variant of LoRALayer that stores the frozen base weight
/// in 4-bit quantized format, reducing memory usage by ~75%.
/// Optionally uses double quantization (ENT-LoRA-008) for additional savings.
pub struct QLoRALayer {
    /// Quantized base weight (4-bit, single quantization)
    base_weight_quantized: Quantized4Bit,
    /// Double-quantized base weight (ENT-LoRA-008, None if not enabled)
    base_weight_double: Option<DoubleQuantized4Bit>,
    /// LoRA matrix A [r * d_in] - full precision, trainable
    lora_a: Tensor,
    /// LoRA matrix B [d_out * r] - full precision, trainable
    lora_b: Tensor,
    /// Output dimension
    d_out: usize,
    /// Input dimension
    d_in: usize,
    /// LoRA rank
    rank: usize,
    /// Scaling factor (alpha/rank)
    scale: f32,
    /// Whether the adapter is merged (not supported for quantized weights)
    merged: bool,
}

impl QLoRALayer {
    /// Create QLoRA layer from existing LoRA layer
    ///
    /// # Arguments
    /// * `lora_layer` - Existing LoRALayer to convert
    ///
    /// # Returns
    /// QLoRALayer with quantized base weight
    pub fn from_lora(lora_layer: LoRALayer) -> Self {
        let base_weight_data = lora_layer.base_weight().data().to_vec();
        let base_weight_quantized = quantize_4bit(&base_weight_data);

        Self {
            base_weight_quantized,
            base_weight_double: None,
            lora_a: lora_layer.lora_a().clone(),
            lora_b: lora_layer.lora_b().clone(),
            d_out: lora_layer.d_out(),
            d_in: lora_layer.d_in(),
            rank: lora_layer.rank(),
            scale: lora_layer.scale(),
            merged: false,
        }
    }

    /// Create QLoRA layer from LoRALayer with double quantization (ENT-LoRA-008)
    pub fn from_lora_double_quant(lora_layer: LoRALayer) -> Self {
        let base_weight_data = lora_layer.base_weight().data().to_vec();
        let base_weight_quantized = quantize_4bit(&base_weight_data);
        let base_weight_double = Some(quantize_4bit_double(&base_weight_data));

        Self {
            base_weight_quantized,
            base_weight_double,
            lora_a: lora_layer.lora_a().clone(),
            lora_b: lora_layer.lora_b().clone(),
            d_out: lora_layer.d_out(),
            d_in: lora_layer.d_in(),
            rank: lora_layer.rank(),
            scale: lora_layer.scale(),
            merged: false,
        }
    }

    /// Create QLoRA layer directly with quantized base weight
    ///
    /// # Arguments
    /// * `base_weight` - Base weight to quantize [d_out * d_in]
    /// * `d_out` - Output dimension
    /// * `d_in` - Input dimension
    /// * `rank` - LoRA rank
    /// * `alpha` - LoRA alpha parameter
    pub fn new(base_weight: Tensor, d_out: usize, d_in: usize, rank: usize, alpha: f32) -> Self {
        // Create LoRALayer first, then convert
        let lora_layer = LoRALayer::new(base_weight, d_out, d_in, rank, alpha);
        Self::from_lora(lora_layer)
    }

    /// Forward pass with on-the-fly dequantization
    ///
    /// # Arguments
    /// * `x` - Input tensor `[d_in]`
    ///
    /// # Returns
    /// Output tensor `[d_out]`
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Contract: inference-pipeline-v1.yaml precondition (pv codegen)
        // contract_pre_prefill_phase!(x.data()); // TODO: macro not yet generated

        assert_eq!(x.len(), self.d_in, "Input size must match d_in");

        // Dequantize base weight on-the-fly (use double quant path if available)
        let base_weight_data = if let Some(ref dq) = self.base_weight_double {
            dequantize_4bit_double(dq)
        } else {
            dequantize_4bit(&self.base_weight_quantized)
        };
        let base_weight = Tensor::new(ndarray::arr1(&base_weight_data), false);

        // Base forward: W @ x
        let base_output = matmul(&base_weight, x, self.d_out, self.d_in, 1);

        if self.merged {
            base_output
        } else {
            // LoRA forward: scale * (B @ (A @ x))
            let lora_out_a = matmul(&self.lora_a, x, self.rank, self.d_in, 1);
            let lora_out_b = matmul(&self.lora_b, &lora_out_a, self.d_out, self.rank, 1);

            // Scale and add
            let mut scaled_lora_data = lora_out_b.data().to_owned();
            for val in &mut scaled_lora_data {
                *val *= self.scale;
            }
            let scaled_lora = Tensor::new(scaled_lora_data, false);

            let mut result_data = base_output.data().to_owned();
            for (i, val) in result_data.iter_mut().enumerate() {
                *val += scaled_lora.data()[i];
            }
            Tensor::new(result_data, base_output.requires_grad())
        }
    }

    /// Get reference to LoRA A matrix
    pub fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    /// Get mutable reference to LoRA A matrix
    pub fn lora_a_mut(&mut self) -> &mut Tensor {
        &mut self.lora_a
    }

    /// Get reference to LoRA B matrix
    pub fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    /// Get mutable reference to LoRA B matrix
    pub fn lora_b_mut(&mut self) -> &mut Tensor {
        &mut self.lora_b
    }

    /// Get trainable parameters (A and B)
    pub fn trainable_params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get output dimension
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// Get input dimension
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let base_unquantized_bytes = self.d_out * self.d_in * 4; // f32
        let base_quantized_bytes = if let Some(ref dq) = self.base_weight_double {
            dq.memory_bytes()
        } else {
            self.base_weight_quantized.memory_bytes()
        };
        let lora_a_bytes = self.lora_a.len() * 4;
        let lora_b_bytes = self.lora_b.len() * 4;

        MemoryStats {
            base_unquantized_bytes,
            base_quantized_bytes,
            lora_bytes: lora_a_bytes + lora_b_bytes,
            total_bytes: base_quantized_bytes + lora_a_bytes + lora_b_bytes,
            compression_ratio: base_unquantized_bytes as f32 / base_quantized_bytes.max(1) as f32,
        }
    }

    /// Check if merged (always false for quantized layers)
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Merge adapter into dequantized base weight, returning full-precision f32 weights
    ///
    /// Computes: `dequantize(base_4bit) + scale * B @ A`
    ///
    /// This produces a merged weight matrix that can be exported as SafeTensors or GGUF.
    /// The result is a flat Vec<f32> of shape [d_out, d_in] in row-major layout.
    pub fn merge_to_f32(&self) -> Vec<f32> {
        // Dequantize base weight (use double quant path if available)
        let mut merged = if let Some(ref dq) = self.base_weight_double {
            dequantize_4bit_double(dq)
        } else {
            dequantize_4bit(&self.base_weight_quantized)
        };

        // Compute scale * B @ A and add to merged weights
        // A: [rank, d_in], B: [d_out, rank]
        // B @ A = [d_out, d_in]
        let a_data = self.lora_a.data();
        let b_data = self.lora_b.data();

        for row in 0..self.d_out {
            for col in 0..self.d_in {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    let b_val = b_data[row * self.rank + r];
                    let a_val = a_data[r * self.d_in + col];
                    sum += b_val * a_val;
                }
                merged[row * self.d_in + col] += self.scale * sum;
            }
        }

        merged
    }

    /// Get reference to quantized base weights
    pub fn base_weight_quantized(&self) -> &Quantized4Bit {
        &self.base_weight_quantized
    }

    /// Check if double quantization is enabled (ENT-LoRA-008)
    pub fn is_double_quantized(&self) -> bool {
        self.base_weight_double.is_some()
    }
}

/// Memory usage statistics for QLoRA layer
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Base weight size if unquantized (bytes)
    pub base_unquantized_bytes: usize,
    /// Base weight size quantized (bytes)
    pub base_quantized_bytes: usize,
    /// LoRA adapters size (bytes)
    pub lora_bytes: usize,
    /// Total memory usage (bytes)
    pub total_bytes: usize,
    /// Compression ratio for base weights
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Mathematical correctness validation
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        /// Memory savings should be consistent with dimensions
        #[test]
        fn prop_qlora_memory_savings_consistent(
            d in 8usize..32,
            rank in 1usize..8,
            alpha in 1.0f32..32.0
        ) {
            let size = d * d;
            let base_weight = Tensor::from_vec(vec![0.5; size], false);
            let qlora = QLoRALayer::new(base_weight, d, d, rank, alpha);

            let stats = qlora.memory_stats();

            // Quantized should always be smaller than unquantized
            prop_assert!(stats.base_quantized_bytes <= stats.base_unquantized_bytes);

            // Compression ratio should be > 1.0
            prop_assert!(stats.compression_ratio >= 1.0);

            // Total bytes = quantized base + lora
            prop_assert_eq!(
                stats.total_bytes,
                stats.base_quantized_bytes + stats.lora_bytes
            );

            // LoRA bytes = (d*rank + d*rank) * 4 bytes
            let expected_lora_bytes = (d * rank + d * rank) * 4;
            prop_assert_eq!(stats.lora_bytes, expected_lora_bytes);
        }

        /// LoRA parameters should be preserved during quantization
        #[test]
        fn prop_lora_params_preserved_after_quantization(
            d_out in 4usize..16,
            d_in in 4usize..16,
            rank in 1usize..4,
            alpha in 1.0f32..16.0
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let lora = LoRALayer::new(base_weight.clone(), d_out, d_in, rank, alpha);

            let qlora = QLoRALayer::from_lora(lora.clone());

            // Dimensions should match exactly
            prop_assert_eq!(qlora.d_out(), lora.d_out());
            prop_assert_eq!(qlora.d_in(), lora.d_in());
            prop_assert_eq!(qlora.rank(), lora.rank());

            // Scale should match
            prop_assert!((qlora.scale() - lora.scale()).abs() < 1e-6);

            // LoRA A and B data should be identical
            prop_assert_eq!(qlora.lora_a().data().len(), lora.lora_a().data().len());
            prop_assert_eq!(qlora.lora_b().data().len(), lora.lora_b().data().len());

            for (a, b) in qlora.lora_a().data().iter().zip(lora.lora_a().data().iter()) {
                prop_assert!((a - b).abs() < 1e-6);
            }
            for (a, b) in qlora.lora_b().data().iter().zip(lora.lora_b().data().iter()) {
                prop_assert!((a - b).abs() < 1e-6);
            }
        }

        /// Quantization error should be bounded
        #[test]
        fn prop_quantization_error_bounded(
            d in 8usize..24,
        ) {
            let size = d * d;
            // Use values in reasonable range for 4-bit quantization
            let base_weight = Tensor::from_vec(
                (0..size).map(|i| ((i % 16) as f32 - 8.0) * 0.1).collect(),
                false
            );
            let lora = LoRALayer::new(base_weight.clone(), d, d, 2, 4.0);
            let qlora = QLoRALayer::from_lora(lora.clone());

            // Forward with same input
            let x = Tensor::from_vec(vec![0.1; d], true);
            let lora_out = lora.forward(&x);
            let qlora_out = qlora.forward(&x);

            // Outputs should be close (quantization introduces some error)
            prop_assert_eq!(lora_out.len(), qlora_out.len());
            for i in 0..lora_out.len() {
                let diff = (lora_out.data()[i] - qlora_out.data()[i]).abs();
                // Allow 30% relative error due to 4-bit quantization
                let max_diff = lora_out.data()[i].abs() * 0.3 + 0.5;
                prop_assert!(
                    diff < max_diff,
                    "Quantization error {} > {} at index {}",
                    diff, max_diff, i
                );
            }
        }

        /// Forward output dimensions should always be correct
        #[test]
        fn prop_forward_dimensions_correct(
            d_out in 4usize..16,
            d_in in 4usize..16,
            rank in 1usize..4,
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let qlora = QLoRALayer::new(base_weight, d_out, d_in, rank, 4.0);

            let x = Tensor::from_vec(vec![0.5; d_in], true);
            let output = qlora.forward(&x);

            prop_assert_eq!(output.len(), d_out);
        }

        /// Trainable params should have correct dimensions
        #[test]
        fn prop_trainable_params_dimensions(
            d_out in 4usize..16,
            d_in in 4usize..16,
            rank in 1usize..4,
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let mut qlora = QLoRALayer::new(base_weight, d_out, d_in, rank, 4.0);

            let params = qlora.trainable_params();
            prop_assert_eq!(params.len(), 2);

            // A: [rank * d_in]
            prop_assert_eq!(params[0].len(), rank * d_in);
            // B: [d_out * rank]
            prop_assert_eq!(params[1].len(), d_out * rank);

            // Both should require grad
            prop_assert!(params[0].requires_grad());
            prop_assert!(params[1].requires_grad());
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_qlora_creation() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let qlora = QLoRALayer::new(base_weight, 2, 2, 1, 2.0);

        assert_eq!(qlora.rank(), 1);
        assert_eq!(qlora.d_out(), 2);
        assert_eq!(qlora.d_in(), 2);
        assert_abs_diff_eq!(qlora.scale(), 2.0, epsilon = 1e-6); // alpha/rank = 2/1
        assert!(!qlora.is_merged());
    }

    #[test]
    fn test_qlora_forward_matches_lora() {
        // Test that QLoRA forward pass approximates LoRA
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight.clone(), 2, 2, 1, 1.0);
        *lora.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *lora.lora_b_mut().data_mut() = ndarray::arr1(&[0.3, 0.3]);

        let mut qlora = QLoRALayer::new(base_weight, 2, 2, 1, 1.0);
        *qlora.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *qlora.lora_b_mut().data_mut() = ndarray::arr1(&[0.3, 0.3]);

        let x = Tensor::from_vec(vec![2.0, 3.0], true);

        let lora_output = lora.forward(&x);
        let qlora_output = qlora.forward(&x);

        // Outputs should be close (within quantization error)
        assert_eq!(lora_output.len(), qlora_output.len());
        for i in 0..lora_output.len() {
            let diff = (lora_output.data()[i] - qlora_output.data()[i]).abs();
            assert!(
                diff < 0.2,
                "Output mismatch at {}: {} vs {} (diff: {})",
                i,
                lora_output.data()[i],
                qlora_output.data()[i],
                diff
            );
        }
    }

    #[test]
    fn test_qlora_memory_savings() {
        // Test with large enough weight to show compression (use perfect square)
        let d = 16; // 16x16 = 256 elements
        let size = d * d;
        let base_weight = Tensor::from_vec(vec![1.0; size], false);
        let qlora = QLoRALayer::new(base_weight, d, d, 8, 16.0);

        let stats = qlora.memory_stats();

        // Should see significant memory savings
        assert!(
            stats.base_quantized_bytes < stats.base_unquantized_bytes,
            "Quantized should use less memory"
        );

        // Compression ratio should be > 6x
        assert!(
            stats.compression_ratio > 6.0,
            "Compression ratio {} should be > 6.0",
            stats.compression_ratio
        );
    }

    #[test]
    fn test_qlora_trainable_params() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let mut qlora = QLoRALayer::new(base_weight, 2, 2, 2, 4.0);

        let params = qlora.trainable_params();
        assert_eq!(params.len(), 2);

        // Both should be trainable
        assert!(params[0].requires_grad());
        assert!(params[1].requires_grad());
    }

    #[test]
    fn test_qlora_from_lora() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
        let lora = LoRALayer::new(base_weight, 3, 2, 2, 8.0);

        let qlora = QLoRALayer::from_lora(lora);

        assert_eq!(qlora.rank(), 2);
        assert_eq!(qlora.d_out(), 3);
        assert_eq!(qlora.d_in(), 2);
        assert_abs_diff_eq!(qlora.scale(), 4.0, epsilon = 1e-6); // 8/2 = 4
    }

    #[test]
    fn test_qlora_merge_to_f32_dimensions() {
        let d_out = 8;
        let d_in = 16;
        let base_weight = Tensor::from_vec(vec![1.0; d_out * d_in], false);
        let qlora = QLoRALayer::new(base_weight, d_out, d_in, 4, 8.0);

        let merged = qlora.merge_to_f32();
        assert_eq!(merged.len(), d_out * d_in);
    }

    #[test]
    fn test_qlora_merge_to_f32_includes_adapter() {
        let d_out = 4;
        let d_in = 4;
        let base_weight = Tensor::from_vec(vec![0.0; d_out * d_in], false);
        let mut qlora = QLoRALayer::new(base_weight, d_out, d_in, 2, 2.0);

        // Set adapter weights to known values
        *qlora.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        *qlora.lora_b_mut().data_mut() = ndarray::arr1(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);

        let merged = qlora.merge_to_f32();

        // Base is all zeros (within quant error), adapter should contribute non-zero values
        let adapter_contribution: f32 = merged.iter().map(|v| v.abs()).sum();
        assert!(adapter_contribution > 0.0, "Merged weights should include adapter contribution");
    }

    #[test]
    fn test_qlora_merge_to_f32_equivalence_with_lora() {
        // For small values, QLoRA merge should approximate LoRA merge
        let d_out = 4;
        let d_in = 4;
        let base_data = vec![
            0.5, 0.3, -0.2, 0.1, 0.4, -0.1, 0.6, 0.2, -0.3, 0.5, 0.1, -0.4, 0.2, 0.3, -0.5, 0.6,
        ];
        let base_weight = Tensor::from_vec(base_data.clone(), false);
        let mut lora = LoRALayer::new(base_weight.clone(), d_out, d_in, 2, 4.0);

        let a_data = vec![0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.1, 0.1];
        let b_data = vec![0.3, -0.1, 0.2, 0.1, -0.2, 0.3, 0.1, -0.1];
        *lora.lora_a_mut().data_mut() = ndarray::arr1(&a_data);
        *lora.lora_b_mut().data_mut() = ndarray::arr1(&b_data);

        let mut qlora = QLoRALayer::from_lora(lora.clone());
        // Copy same adapter weights
        *qlora.lora_a_mut().data_mut() = ndarray::arr1(&a_data);
        *qlora.lora_b_mut().data_mut() = ndarray::arr1(&b_data);

        // Merge LoRA in-place and extract base weight as the merged result
        lora.merge();
        let lora_merged: Vec<f32> = lora.base_weight().data().to_vec();
        let qlora_merged = qlora.merge_to_f32();

        assert_eq!(lora_merged.len(), qlora_merged.len());
        for i in 0..lora_merged.len() {
            let diff = (lora_merged[i] - qlora_merged[i]).abs();
            assert!(
                diff < 0.5,
                "Merge difference too large at {i}: lora={}, qlora={}, diff={diff}",
                lora_merged[i],
                qlora_merged[i]
            );
        }
    }

    #[test]
    fn test_qlora_large_matrix() {
        // Test with realistic transformer dimensions
        let d_model = 256;
        let base_weight = Tensor::from_vec(vec![1.0; d_model * d_model], false);
        let qlora = QLoRALayer::new(base_weight, d_model, d_model, 16, 32.0);

        let x = Tensor::from_vec(vec![0.5; d_model], true);
        let output = qlora.forward(&x);

        assert_eq!(output.len(), d_model);

        // Check memory savings
        let stats = qlora.memory_stats();
        let savings_percent =
            (1.0 - stats.base_quantized_bytes as f32 / stats.base_unquantized_bytes as f32) * 100.0;

        assert!(savings_percent > 70.0, "Should save > 70% memory, got {savings_percent}%");
    }

    // ========================================================================
    // ENT-LoRA-008: Double Quantization Tests
    // ========================================================================

    #[test]
    fn test_ent_lora_008_double_quant_creation() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let lora = LoRALayer::new(base_weight, 2, 2, 1, 2.0);
        let qlora = QLoRALayer::from_lora_double_quant(lora);

        assert!(qlora.is_double_quantized());
        assert_eq!(qlora.d_out(), 2);
        assert_eq!(qlora.d_in(), 2);
    }

    #[test]
    fn test_ent_lora_008_double_quant_forward_close_to_single() {
        let d = 64;
        let base_weight =
            Tensor::from_vec((0..d * d).map(|i| ((i as f32 * 0.1).sin() * 2.0)).collect(), false);
        let lora = LoRALayer::new(base_weight, d, d, 4, 8.0);

        let single = QLoRALayer::from_lora(lora.clone());
        let double = QLoRALayer::from_lora_double_quant(lora);

        let x = Tensor::from_vec(vec![0.1; d], true);
        let single_out = single.forward(&x);
        let double_out = double.forward(&x);

        assert_eq!(single_out.len(), double_out.len());
        for i in 0..single_out.len() {
            let diff = (single_out.data()[i] - double_out.data()[i]).abs();
            let tol = single_out.data()[i].abs() * 0.01 + 0.1;
            assert!(
                diff <= tol,
                "Forward output diverged at [{i}]: single={}, double={}, diff={diff}",
                single_out.data()[i],
                double_out.data()[i]
            );
        }
    }

    #[test]
    fn test_ent_lora_008_single_quant_not_double() {
        let base_weight = Tensor::from_vec(vec![1.0; 16], false);
        let qlora = QLoRALayer::new(base_weight, 4, 4, 2, 4.0);
        assert!(!qlora.is_double_quantized());
    }

    #[test]
    fn test_ent_lora_008_double_quant_memory_stats() {
        let d = 256;
        let base_weight = Tensor::from_vec(vec![1.0; d * d], false);
        let lora = LoRALayer::new(base_weight, d, d, 16, 32.0);

        let single = QLoRALayer::from_lora(lora.clone());
        let double = QLoRALayer::from_lora_double_quant(lora);

        let single_stats = single.memory_stats();
        let double_stats = double.memory_stats();

        // Double quant should use less memory for base weights
        assert!(
            double_stats.base_quantized_bytes <= single_stats.base_quantized_bytes,
            "Double quant ({}) should use <= memory than single ({})",
            double_stats.base_quantized_bytes,
            single_stats.base_quantized_bytes
        );
    }

    // ========================================================================
    // COVERAGE GAP TESTS — double-quant merge, accessors, merged forward
    // ========================================================================

    #[test]
    fn test_qlora_merge_to_f32_double_quant() {
        // Covers the double quantization path in merge_to_f32()
        let d_out = 8;
        let d_in = 8;
        let base_weight = Tensor::from_vec(
            (0..d_out * d_in).map(|i| (i as f32 * 0.2).sin() * 0.5).collect(),
            false,
        );
        let lora = LoRALayer::new(base_weight, d_out, d_in, 2, 4.0);
        let qlora_dq = QLoRALayer::from_lora_double_quant(lora);

        assert!(qlora_dq.is_double_quantized());

        let merged = qlora_dq.merge_to_f32();
        assert_eq!(merged.len(), d_out * d_in);

        // All values should be finite
        for val in &merged {
            assert!(val.is_finite(), "Merged weight must be finite, got {val}");
        }
    }

    #[test]
    fn test_qlora_merge_to_f32_single_vs_double_close() {
        // Verify single-quant and double-quant merge paths produce similar results
        let d_out = 8;
        let d_in = 8;
        let base_data: Vec<f32> =
            (0..d_out * d_in).map(|i| (i as f32 * 0.15).cos() * 0.3).collect();
        let base_weight = Tensor::from_vec(base_data, false);

        let lora = LoRALayer::new(base_weight, d_out, d_in, 2, 4.0);
        let single = QLoRALayer::from_lora(lora.clone());
        let double = QLoRALayer::from_lora_double_quant(lora);

        let merged_single = single.merge_to_f32();
        let merged_double = double.merge_to_f32();

        assert_eq!(merged_single.len(), merged_double.len());
        for i in 0..merged_single.len() {
            let diff = (merged_single[i] - merged_double[i]).abs();
            let tol = merged_single[i].abs() * 0.05 + 0.2;
            assert!(
                diff <= tol,
                "merge_to_f32 single vs double diverged at [{i}]: single={}, double={}, diff={diff}",
                merged_single[i],
                merged_double[i]
            );
        }
    }

    #[test]
    fn test_qlora_base_weight_quantized_accessor() {
        // Covers base_weight_quantized() accessor
        let d = 8;
        let base_weight = Tensor::from_vec(vec![1.0; d * d], false);
        let qlora = QLoRALayer::new(base_weight, d, d, 2, 4.0);

        let quantized = qlora.base_weight_quantized();
        // The quantized struct should have data — just verify it exists and has content
        assert!(quantized.memory_bytes() > 0, "Quantized base weight should use memory");
    }

    #[test]
    fn test_qlora_double_quant_forward_with_known_adapter() {
        // Covers the double-quant forward path with explicit adapter weights
        let d_out = 4;
        let d_in = 4;
        let base_weight = Tensor::from_vec(vec![0.5; d_out * d_in], false);
        let lora = LoRALayer::new(base_weight, d_out, d_in, 2, 4.0);
        let mut qlora = QLoRALayer::from_lora_double_quant(lora);

        assert!(qlora.is_double_quantized());

        // Set non-zero adapter weights
        let a_data: Vec<f32> = (0..2 * d_in).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let b_data: Vec<f32> = (0..d_out * 2).map(|i| (i as f32 * 0.2).cos() * 0.3).collect();
        *qlora.lora_a_mut().data_mut() = ndarray::Array1::from_vec(a_data);
        *qlora.lora_b_mut().data_mut() = ndarray::Array1::from_vec(b_data);

        let x = Tensor::from_vec(vec![1.0; d_in], true);
        let output = qlora.forward(&x);

        assert_eq!(output.len(), d_out);
        for val in output.data() {
            assert!(val.is_finite(), "Forward output must be finite, got {val}");
        }
    }

    #[test]
    fn test_qlora_memory_stats_double_quant() {
        // Covers the double-quant path in memory_stats()
        let d = 16;
        let base_weight = Tensor::from_vec(vec![1.0; d * d], false);
        let lora = LoRALayer::new(base_weight, d, d, 4, 8.0);
        let qlora = QLoRALayer::from_lora_double_quant(lora);

        let stats = qlora.memory_stats();

        // Basic sanity checks
        assert!(stats.base_quantized_bytes > 0);
        assert!(stats.lora_bytes > 0);
        assert_eq!(stats.total_bytes, stats.base_quantized_bytes + stats.lora_bytes);
        assert!(stats.compression_ratio >= 1.0);
        assert_eq!(stats.base_unquantized_bytes, d * d * 4);
    }

    #[test]
    fn test_qlora_memory_stats_clone_and_debug() {
        // Covers Clone and Debug derives on MemoryStats
        let base_weight = Tensor::from_vec(vec![1.0; 16], false);
        let qlora = QLoRALayer::new(base_weight, 4, 4, 2, 4.0);

        let stats = qlora.memory_stats();
        let stats_clone = stats.clone();

        assert_eq!(stats.total_bytes, stats_clone.total_bytes);
        assert_eq!(stats.lora_bytes, stats_clone.lora_bytes);
        assert_eq!(stats.base_quantized_bytes, stats_clone.base_quantized_bytes);

        let debug_str = format!("{stats_clone:?}");
        assert!(debug_str.contains("MemoryStats"));
    }

    #[test]
    fn test_qlora_lora_a_mut_and_lora_b_mut() {
        // Explicitly covers lora_a_mut() and lora_b_mut() on QLoRALayer
        let base_weight = Tensor::from_vec(vec![1.0; 4], false);
        let mut qlora = QLoRALayer::new(base_weight, 2, 2, 1, 2.0);

        // Mutate A
        *qlora.lora_a_mut().data_mut() = ndarray::arr1(&[10.0, 20.0]);
        assert_abs_diff_eq!(qlora.lora_a().data()[0], 10.0, epsilon = 1e-6);
        assert_abs_diff_eq!(qlora.lora_a().data()[1], 20.0, epsilon = 1e-6);

        // Mutate B
        *qlora.lora_b_mut().data_mut() = ndarray::arr1(&[30.0, 40.0]);
        assert_abs_diff_eq!(qlora.lora_b().data()[0], 30.0, epsilon = 1e-6);
        assert_abs_diff_eq!(qlora.lora_b().data()[1], 40.0, epsilon = 1e-6);
    }
}
