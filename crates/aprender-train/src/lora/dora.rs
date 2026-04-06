//! DoRA (Weight-Decomposed Low-Rank Adaptation) — ENT-LoRA-011
//!
//! DoRA decomposes the weight into magnitude `m` and direction `V/||V||`, then applies
//! LoRA only to the direction component:
//!
//!   W' = m * (V + scale * B @ A) / ||V + scale * B @ A||
//!
//! This achieves +1-3% accuracy over standard LoRA on many benchmarks (ICML 2024 Oral).
//!
//! Reference: Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation."

use crate::lora::{LoRALayer, LoRAScaling};
use crate::Tensor;

/// DoRA layer: magnitude-direction decomposed LoRA
///
/// The base weight W is decomposed into:
/// - `magnitude`: column norms `m = ||W_col||` for each output neuron
/// - `direction`: normalized columns `V = W / m`
///
/// LoRA is applied to direction only, preserving the magnitude structure.
pub struct DoRALayer {
    /// Per-output-neuron magnitudes [d_out], trainable
    magnitude: Tensor,
    /// Underlying LoRA layer (applied to direction)
    lora: LoRALayer,
    /// Cached column norms of (V + scale * B @ A) for forward
    d_out: usize,
    d_in: usize,
}

impl DoRALayer {
    /// Create a DoRA layer from a base weight
    ///
    /// Decomposes W into magnitude and direction, creates LoRA on the direction.
    pub fn new(
        base_weight: Tensor,
        d_out: usize,
        d_in: usize,
        rank: usize,
        alpha: f32,
        scaling: LoRAScaling,
    ) -> Self {
        // Compute per-row magnitudes: m[i] = ||W[i, :]||
        let magnitude_data: Vec<f32> = (0..d_out)
            .map(|row| {
                let row_start = row * d_in;
                let row_end = row_start + d_in;
                let row_norm_sq: f32 = base_weight
                    .data()
                    .slice(ndarray::s![row_start..row_end])
                    .iter()
                    .map(|x| x * x)
                    .sum();
                row_norm_sq.sqrt().max(1e-8)
            })
            .collect();
        let magnitude = Tensor::from_vec(magnitude_data, true); // trainable

        // Create LoRA layer on the base weight (direction component)
        let lora = LoRALayer::new_with_scaling(base_weight, d_out, d_in, rank, alpha, scaling);

        Self { magnitude, lora, d_out, d_in }
    }

    /// Forward pass: m * normalize(V + scale * B @ A) @ x
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.len(), self.d_in, "Input size must match d_in");

        // Compute V + scale * B @ A direction matrix
        // For efficiency, compute (W + scale * B @ A) @ x first, then normalize
        let lora_output = self.lora.forward(x); // (W + scale*B@A) @ x

        // Normalize per row and apply magnitude
        // Row norms of the effective weight matrix
        let row_norms = self.compute_effective_row_norms();

        let mut result = lora_output.data().to_owned();
        for (i, val) in result.iter_mut().enumerate() {
            let norm = row_norms[i].max(1e-8);
            *val = self.magnitude.data()[i] * (*val / norm);
        }

        Tensor::new(result, self.magnitude.requires_grad())
    }

    /// Compute row norms of the effective weight matrix (W + scale * B @ A)
    fn compute_effective_row_norms(&self) -> Vec<f32> {
        let base = self.lora.base_weight().data();
        let scale = self.lora.scale();
        let a_data = self.lora.lora_a().data();
        let b_data = self.lora.lora_b().data();
        let rank = self.lora.rank();

        let mut norms = vec![0.0f32; self.d_out];
        for row in 0..self.d_out {
            let mut row_norm_sq = 0.0f32;
            for col in 0..self.d_in {
                let base_val = base[row * self.d_in + col];
                // Compute (B @ A)[row, col]
                let mut ba_val = 0.0f32;
                for r in 0..rank {
                    ba_val += b_data[row * rank + r] * a_data[r * self.d_in + col];
                }
                let effective = base_val + scale * ba_val;
                row_norm_sq += effective * effective;
            }
            norms[row] = row_norm_sq.sqrt();
        }
        norms
    }

    /// Merge DoRA into a single weight matrix for inference
    pub fn merge_to_f32(&self) -> Vec<f32> {
        let row_norms = self.compute_effective_row_norms();
        let base = self.lora.base_weight().data();
        let scale = self.lora.scale();
        let a_data = self.lora.lora_a().data();
        let b_data = self.lora.lora_b().data();
        let rank = self.lora.rank();

        let mut merged = vec![0.0f32; self.d_out * self.d_in];
        for row in 0..self.d_out {
            let m = self.magnitude.data()[row];
            let norm = row_norms[row].max(1e-8);
            for col in 0..self.d_in {
                let base_val = base[row * self.d_in + col];
                let mut ba_val = 0.0f32;
                for r in 0..rank {
                    ba_val += b_data[row * rank + r] * a_data[r * self.d_in + col];
                }
                merged[row * self.d_in + col] = m * (base_val + scale * ba_val) / norm;
            }
        }
        merged
    }

    /// Get trainable parameters (magnitude + LoRA A + LoRA B)
    pub fn trainable_params(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.magnitude];
        params.extend(self.lora.trainable_params());
        params
    }

    /// Get the magnitude vector
    pub fn magnitude(&self) -> &Tensor {
        &self.magnitude
    }

    /// Get the underlying LoRA layer
    pub fn lora(&self) -> &LoRALayer {
        &self.lora
    }

    /// Trainable param count: magnitude (d_out) + LoRA A (r*d_in) + LoRA B (d_out*r)
    pub fn trainable_param_count(&self) -> usize {
        self.d_out + self.lora.rank() * self.d_in + self.d_out * self.lora.rank()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn test_ent_lora_011_dora_creation() {
        let base = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let dora = DoRALayer::new(base, 2, 2, 1, 2.0, LoRAScaling::Standard);
        assert_eq!(dora.d_out, 2);
        assert_eq!(dora.d_in, 2);
        assert!(dora.magnitude().len() == 2);
    }

    #[test]
    fn test_ent_lora_011_dora_magnitude_init() {
        // Identity matrix: each row has norm 1.0
        let base = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let dora = DoRALayer::new(base, 2, 2, 1, 2.0, LoRAScaling::Standard);
        assert_abs_diff_eq!(dora.magnitude().data()[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(dora.magnitude().data()[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ent_lora_011_dora_forward_dimensions() {
        let base = Tensor::from_vec(vec![1.0; 12], false);
        let dora = DoRALayer::new(base, 3, 4, 2, 4.0, LoRAScaling::RsLoRA);
        let x = Tensor::from_vec(vec![0.5; 4], true);
        let out = dora.forward(&x);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_ent_lora_011_dora_trainable_count() {
        let base = Tensor::from_vec(vec![1.0; 16], false);
        let dora = DoRALayer::new(base, 4, 4, 2, 4.0, LoRAScaling::Standard);
        // magnitude: 4 + A: 2*4=8 + B: 4*2=8 = 20
        assert_eq!(dora.trainable_param_count(), 20);
    }

    #[test]
    fn test_ent_lora_011_dora_merge_dimensions() {
        let base = Tensor::from_vec(vec![1.0; 12], false);
        let dora = DoRALayer::new(base, 3, 4, 2, 4.0, LoRAScaling::Standard);
        let merged = dora.merge_to_f32();
        assert_eq!(merged.len(), 12);
    }

    #[test]
    fn test_ent_lora_011_dora_trainable_params() {
        let base = Tensor::from_vec(vec![1.0; 16], false);
        let mut dora = DoRALayer::new(base, 4, 4, 2, 4.0, LoRAScaling::Standard);
        let params = dora.trainable_params();
        // magnitude + A + B = 3 tensors
        assert_eq!(params.len(), 3);
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(50))]

        #[test]
        fn prop_dora_forward_finite(
            d_out in 2usize..8,
            d_in in 2usize..8,
            rank in 1usize..4,
        ) {
            let base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
            let dora = DoRALayer::new(base, d_out, d_in, rank, 4.0, LoRAScaling::Standard);
            let x = Tensor::from_vec(vec![0.1; d_in], true);
            let out = dora.forward(&x);
            prop_assert_eq!(out.len(), d_out);
            for val in out.data() {
                prop_assert!(val.is_finite(), "Output must be finite, got {val}");
            }
        }

        #[test]
        fn prop_dora_merge_finite(
            d_out in 2usize..8,
            d_in in 2usize..8,
            rank in 1usize..4,
        ) {
            let base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
            let dora = DoRALayer::new(base, d_out, d_in, rank, 4.0, LoRAScaling::Standard);
            let merged = dora.merge_to_f32();
            prop_assert_eq!(merged.len(), d_out * d_in);
            for val in &merged {
                prop_assert!(val.is_finite());
            }
        }
    }
}
