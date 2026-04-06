//! Memory estimation for model loading

/// Memory estimation for model loading
#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    /// Memory for model weights
    pub weights: u64,
    /// Memory for activations during forward pass
    pub activations: u64,
    /// Memory for gradients (0 for frozen teacher)
    pub gradients: u64,
}

impl MemoryEstimate {
    /// Total memory required
    #[must_use]
    pub fn total(&self) -> u64 {
        self.weights + self.activations + self.gradients
    }

    /// Check if model fits in available memory
    #[must_use]
    pub fn fits_in(&self, available: u64) -> bool {
        self.total() <= available
    }

    /// Create estimate for FP32 model
    #[must_use]
    pub fn fp32(param_count: u64, batch_size: usize, seq_len: usize, hidden_size: usize) -> Self {
        Self {
            weights: param_count * 4,
            activations: (batch_size * seq_len * hidden_size * 4) as u64,
            gradients: 0, // Frozen teacher
        }
    }

    /// Create estimate for FP16 model
    #[must_use]
    pub fn fp16(param_count: u64, batch_size: usize, seq_len: usize, hidden_size: usize) -> Self {
        Self {
            weights: param_count * 2,
            activations: (batch_size * seq_len * hidden_size * 2) as u64,
            gradients: 0,
        }
    }

    /// Create estimate for INT4/Q4 model
    #[must_use]
    pub fn int4(param_count: u64, batch_size: usize, seq_len: usize, hidden_size: usize) -> Self {
        Self {
            weights: param_count / 2, // 4-bit = 0.5 bytes per param
            // Activations still in FP16 for compute
            activations: (batch_size * seq_len * hidden_size * 2) as u64,
            gradients: 0,
        }
    }
}
