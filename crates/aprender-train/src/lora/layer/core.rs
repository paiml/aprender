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
//!
//! Dropout placement (PMAT-879): matching HF PEFT `lora.Linear.forward`, dropout
//! is applied to the INPUT `x` *before* the down-projection `A`:
//!
//! `y = W@x + scale · B(A(dropout(x)))`
//!
//! Dropout is active only in training mode; in eval mode (the default) it is the
//! identity, so inference output is unchanged. Inverted dropout scales the
//! surviving activations by `1/(1-p)` so the expected value is preserved.

use crate::autograd::matmul;
use crate::autograd::ops::{add, scale};
use crate::Tensor;
use std::cell::Cell;

/// LoRA scaling mode (ENT-LoRA-004)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoRAScaling {
    /// Standard: scale = alpha / rank
    Standard,
    /// rsLoRA: scale = alpha / sqrt(rank) — rank-stable, default for rank > 16
    RsLoRA,
}

impl LoRAScaling {
    /// Compute the scaling factor
    ///
    /// # Panics
    /// Panics if rank is zero
    pub fn compute(self, alpha: f32, rank: usize) -> f32 {
        assert!(rank > 0, "LoRA rank must be > 0");
        match self {
            Self::Standard => alpha / rank as f32,
            Self::RsLoRA => alpha / (rank as f32).sqrt(),
        }
    }
}

/// LoRA layer: adds trainable low-rank adaptation to a frozen base weight
#[derive(Clone)]
pub struct LoRALayer {
    /// Frozen base weight matrix stored as 1D [d_out * d_in]
    base_weight: Tensor,
    /// LoRA matrix A stored as 1D [r * d_in] - downprojection
    lora_a: Tensor,
    /// LoRA matrix B stored as 1D [d_out * r] - upprojection
    lora_b: Tensor,
    /// Output dimension
    d_out: usize,
    /// Input dimension
    d_in: usize,
    /// LoRA rank
    rank: usize,
    /// Scaling factor (alpha/rank)
    scale: f32,
    /// Whether the adapter is merged into base_weight
    merged: bool,
    /// LoRA dropout probability applied to the input `x` in the LoRA branch
    /// (PMAT-879). `0.0` disables dropout (identity). Active only in training mode.
    dropout: f32,
    /// Training mode flag. `false` (eval) is the default so inference output is
    /// deterministic and dropout-free, matching PEFT (`nn.Dropout` is identity in eval).
    training: bool,
    /// Base seed for the deterministic dropout RNG (PMAT-879).
    dropout_seed: u64,
    /// Per-forward counter that advances the dropout RNG so successive training
    /// steps draw fresh, but reproducible, masks. Interior mutability keeps
    /// `forward(&self)` (shared borrow) while remaining deterministic for a seed.
    dropout_step: Cell<u64>,
}

impl LoRALayer {
    /// Create a new LoRA layer
    ///
    /// # Arguments
    /// * `base_weight` - Frozen pretrained weight [d_out * d_in]
    /// * `d_out` - Output dimension
    /// * `d_in` - Input dimension
    /// * `rank` - LoRA rank (typically 4, 8, 16, 32, or 64)
    /// * `alpha` - LoRA scaling parameter (often same as rank)
    ///
    /// # Returns
    /// LoRA layer with randomly initialized A (Gaussian) and zero-initialized B
    pub fn new(base_weight: Tensor, d_out: usize, d_in: usize, rank: usize, alpha: f32) -> Self {
        assert!(rank > 0, "LoRA rank must be > 0");
        assert_eq!(base_weight.len(), d_out * d_in, "Base weight size must match d_out * d_in");

        // Initialize A with small Gaussian noise, B with zeros (standard LoRA init)
        // This ensures that initially ΔW = B·A = 0
        let lora_a_data: Vec<f32> = (0..rank * d_in)
            .map(|i| {
                // Simple deterministic "random" init for reproducibility in tests
                let x = (i as f32 * 0.1).sin();
                x * 0.01 // Small values
            })
            .collect();
        let lora_a = Tensor::from_vec(lora_a_data, true);

        let lora_b = Tensor::zeros(d_out * rank, true);

        let scale = alpha / rank as f32;

        Self {
            base_weight,
            lora_a,
            lora_b,
            d_out,
            d_in,
            rank,
            scale,
            merged: false,
            dropout: 0.0,
            training: false,
            dropout_seed: 0,
            dropout_step: Cell::new(0),
        }
    }

    /// Create a new LoRA layer with explicit scaling mode (ENT-LoRA-004)
    ///
    /// Use `LoRAScaling::RsLoRA` for rank-stable training (recommended for rank > 16).
    pub fn new_with_scaling(
        base_weight: Tensor,
        d_out: usize,
        d_in: usize,
        rank: usize,
        alpha: f32,
        scaling: LoRAScaling,
    ) -> Self {
        let mut layer = Self::new(base_weight, d_out, d_in, rank, alpha);
        layer.scale = scaling.compute(alpha, rank);
        layer
    }

    /// Override the LoRA scaling factor.
    ///
    /// Used when restoring a serialized adapter whose `scale` was produced by a
    /// non-Standard mode (e.g. rsLoRA, where `scale = alpha / sqrt(rank)` rather than
    /// the `alpha / rank` that [`LoRALayer::new`] recomputes). The adapter stores the
    /// resulting scale *value*, not the scaling mode, so restoration sets it directly.
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the LoRA dropout probability (PMAT-879).
    ///
    /// Matches HF PEFT `lora_dropout`: dropout is applied to the input `x` before
    /// the down-projection `A`. `p` is clamped to `[0.0, 1.0)`; `0.0` disables
    /// dropout. Dropout is only active in training mode (see [`LoRALayer::train`]).
    #[must_use]
    pub fn with_dropout(mut self, p: f32) -> Self {
        // Clamp to [0, 1): p == 1.0 would zero everything and divide by zero in
        // the inverted-dropout scale, which is never a valid configuration.
        self.dropout = p.clamp(0.0, 0.999_999);
        self
    }

    /// Set the deterministic dropout RNG seed (PMAT-879).
    ///
    /// With a fixed seed, dropout masks are fully reproducible, which makes the
    /// training-mode forward path testable.
    #[must_use]
    pub fn with_dropout_seed(mut self, seed: u64) -> Self {
        self.dropout_seed = seed;
        self
    }

    /// Switch the layer to training mode. Dropout is active when `dropout > 0.0`.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switch the layer to evaluation mode (the default). Dropout is the identity,
    /// so inference output is unchanged — matching PEFT `nn.Dropout` in eval.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Set training/eval mode explicitly.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Whether the layer is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// LoRA dropout probability.
    pub fn dropout(&self) -> f32 {
        self.dropout
    }

    /// Apply inverted dropout to the LoRA-branch input (PMAT-879).
    ///
    /// Returns `x` unchanged when not in training mode or when `p == 0.0` (the
    /// identity), exactly mirroring PEFT's `nn.Dropout`/`nn.Identity` placement.
    /// Otherwise each element is independently zeroed with probability `p` and the
    /// survivors are scaled by `1/(1-p)` so the expectation is preserved.
    ///
    /// Uses a deterministic RNG seeded from `(dropout_seed, dropout_step)` so the
    /// mask is reproducible for a given seed while advancing per forward call.
    fn apply_input_dropout(&self, x: &Tensor) -> Tensor {
        if !self.training || self.dropout <= 0.0 {
            return x.clone();
        }

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let step = self.dropout_step.get();
        self.dropout_step.set(step.wrapping_add(1));

        // Mix seed and step so each forward draws a fresh, reproducible mask.
        let mixed = self.dropout_seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = StdRng::seed_from_u64(mixed);

        let keep = 1.0 - self.dropout;
        let inv_keep = 1.0 / keep;

        let dropped: Vec<f32> = x
            .data()
            .iter()
            .map(|&v| if rng.random::<f32>() < self.dropout { 0.0 } else { v * inv_keep })
            .collect();

        Tensor::from_vec(dropped, x.requires_grad())
    }

    /// Forward pass: y = W@x + scale * (B @ (A @ dropout(x)))
    ///
    /// # Arguments
    /// * `x` - Input tensor `[d_in]`
    ///
    /// # Returns
    /// Output tensor `[d_out]`
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.len(), self.d_in, "Input size must match d_in");

        // Base forward: W @ x [d_out, d_in] @ [d_in, 1] -> [d_out, 1]
        let base_output = matmul(&self.base_weight, x, self.d_out, self.d_in, 1);

        if self.merged {
            // If merged, W already includes LoRA adaptation
            base_output
        } else {
            // LoRA forward: scale * (B @ (A @ dropout(x)))
            // PEFT placement: dropout is applied to the INPUT x before A.
            // In eval mode (default) or with dropout == 0.0 this is the identity.
            let dropped_x = self.apply_input_dropout(x);

            // Step 1: A @ dropout(x) [r, d_in] @ [d_in, 1] -> [r, 1]
            let lora_out_a = matmul(&self.lora_a, &dropped_x, self.rank, self.d_in, 1);

            // Step 2: B @ (A @ x) [d_out, r] @ [r, 1] -> [d_out, 1]
            let lora_out_b = matmul(&self.lora_b, &lora_out_a, self.d_out, self.rank, 1);

            // Step 3: scale * LoRA output.
            //
            // PMAT-931: route the scale through the autograd-aware `scale` op
            // instead of rebuilding the tensor with `Tensor::new(.., false)`,
            // which SEVERS the backward edge to lora_a/lora_b (the same
            // graph-severing class as the PMAT-921/922 sweep). Without this the
            // adapter receives no gradient and LoRA fine-tuning silently fails
            // to train.
            let scaled_lora = scale(&lora_out_b, self.scale);

            // Step 4: base + LoRA, again through the autograd-aware `add` op so
            // the result keeps a live backward op reaching both the frozen base
            // matmul (no-grad, dropped) AND the trainable LoRA branch.
            add(&base_output, &scaled_lora)
        }
    }

    /// Merge LoRA weights into base weight: W' = W + scale * (B @ A)
    ///
    /// After merging, forward pass only uses W' (more efficient).
    /// This is typically done for inference.
    pub fn merge(&mut self) {
        if self.merged {
            return; // Already merged
        }

        // Compute B @ A [d_out, r] @ [r, d_in] -> [d_out, d_in]
        let ba = matmul(&self.lora_b, &self.lora_a, self.d_out, self.rank, self.d_in);

        // Scale and add to base weight: W' = W + scale * B @ A
        for (i, val) in self.base_weight.data_mut().iter_mut().enumerate() {
            *val += self.scale * ba.data()[i];
        }

        self.merged = true;
    }

    /// Unmerge LoRA weights from base weight: W = W' - scale * (B @ A)
    ///
    /// Reverses the merge operation. Useful for continuing training or
    /// switching adapters.
    pub fn unmerge(&mut self) {
        if !self.merged {
            return; // Not merged
        }

        // Compute B @ A
        let ba = matmul(&self.lora_b, &self.lora_a, self.d_out, self.rank, self.d_in);

        // Subtract from base weight: W = W' - scale * B @ A
        for (i, val) in self.base_weight.data_mut().iter_mut().enumerate() {
            *val -= self.scale * ba.data()[i];
        }

        self.merged = false;
    }

    /// Get reference to base weight matrix
    pub fn base_weight(&self) -> &Tensor {
        &self.base_weight
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

    /// Check if LoRA is merged
    pub fn is_merged(&self) -> bool {
        self.merged
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
}
