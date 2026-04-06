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

use crate::autograd::matmul;
use crate::Tensor;

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

        Self { base_weight, lora_a, lora_b, d_out, d_in, rank, scale, merged: false }
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

    /// Forward pass: y = W@x + scale * (B @ (A @ x))
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
            // LoRA forward: scale * (B @ (A @ x))
            // Step 1: A @ x [r, d_in] @ [d_in, 1] -> [r, 1]
            let lora_out_a = matmul(&self.lora_a, x, self.rank, self.d_in, 1);

            // Step 2: B @ (A @ x) [d_out, r] @ [r, 1] -> [d_out, 1]
            let lora_out_b = matmul(&self.lora_b, &lora_out_a, self.d_out, self.rank, 1);

            // Step 3: scale * LoRA output
            let mut scaled_lora_data = lora_out_b.data().to_owned();
            for val in &mut scaled_lora_data {
                *val *= self.scale;
            }
            let scaled_lora = Tensor::new(scaled_lora_data, false);

            // Step 4: base + LoRA
            let mut result_data = base_output.data().to_owned();
            for (i, val) in result_data.iter_mut().enumerate() {
                *val += scaled_lora.data()[i];
            }
            Tensor::new(result_data, base_output.requires_grad())
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
