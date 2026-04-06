
impl Linear {
    /// Create a new linear layer
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Errors
    ///
    /// Returns error if either dimension is zero
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let linear = Linear::new(512, 2048)?;
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "in_features and out_features must be > 0".to_string(),
            });
        }

        // Initialize weights to zero (will be loaded from model)
        let weight = vec![0.0; in_features * out_features];
        // Initialize bias to zero
        let bias = vec![0.0; out_features];

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
        })
    }

    /// Forward pass through linear layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape `[batch, in_features]` or `[in_features]`
    ///
    /// # Returns
    ///
    /// Output tensor with shape `[batch, out_features]` or `[out_features]`
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input last dimension doesn't match `in_features`
    /// - Input is empty
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let input = Tensor::from_vec(vec![2, 512], data)?;
    /// let output = linear.forward(&input)?;
    /// assert_eq!(output.shape(), &[2, 2048]);
    /// ```
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.in_features {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match in_features {}",
                    last_dim, self.in_features
                ),
            });
        }

        let data = input.data();
        let total_size = data.len();
        let num_rows = total_size / self.in_features;

        let mut output = Vec::with_capacity(num_rows * self.out_features);

        // For each input row, compute: output = input * weight + bias
        for row_idx in 0..num_rows {
            let input_start = row_idx * self.in_features;
            let input_row = &data[input_start..input_start + self.in_features];

            // Matrix-vector multiplication: output[j] = sum(input[i] * weight[i][j]) + bias[j]
            for j in 0..self.out_features {
                let mut sum = self.bias[j];
                for (i, &input_val) in input_row.iter().enumerate() {
                    sum += input_val * self.weight[i * self.out_features + j];
                }
                output.push(sum);
            }
        }

        // Construct output shape
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        // Debug assertion for numerical stability - catch exploding activations early
        debug_assert!(
            output.iter().all(|&x| x.is_finite()),
            "Linear layer produced NaN or Inf values - check for exploding gradients/activations"
        );

        Tensor::from_vec(output_shape, output)
    }

    /// Get input features
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get weight matrix (for loading from model)
    #[must_use]
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// Get bias vector (for loading from model)
    #[must_use]
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }
}

// =============================================================================
// BENCH-SPRINT-002: QuantizedLinear (Q4_K Inference)
// Per benchmark-model-runners-spec.md v2.0: Inline dequantization for 8x
// memory bandwidth reduction vs f32.
//
// References:
// - [21] Dettmers et al. (2022) - "LLM.int8(): 8-bit Matrix Multiplication"
// - [22] Frantar et al. (2022) - "GPTQ: Post-Training Quantization"
// =============================================================================

/// Q4_K Quantized Linear Layer
///
/// Memory-efficient linear layer using 4-bit K-quantization (Q4_K) format.
/// Achieves ~8x memory reduction vs f32 by storing weights as quantized bytes
/// and performing inline dequantization during matrix-vector multiplication.
///
/// # Performance Characteristics
///
/// - Memory: 4.5 bits/weight (vs 32 bits for f32) → ~7x reduction
/// - Compute: Fused dequant+dot avoids intermediate f32 tensor
/// - Bandwidth: Memory-bound, not compute-bound (per memory wall analysis)
///
/// # Format
///
/// Q4_K uses super-blocks of 256 values:
/// - 144 bytes per super-block
/// - Contains: d (scale), dmin (min), 12-byte scales, 128-byte quantized values
///
/// # Example
///
/// ```rust,ignore
/// // Create from raw Q4_K bytes loaded from GGUF model
/// let layer = QuantizedLinear::new(4096, 4096, q4k_bytes, bias)?;
/// let output = layer.forward(&activations)?;
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Input features (must be multiple of 256 for Q4_K)
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Q4_K quantized weight bytes [out_features * bytes_per_row]
    weight_bytes: Vec<u8>,
    /// Bias vector [out_features]
    bias: Vec<f32>,
    /// Bytes per output row (super_blocks_per_row * 144)
    bytes_per_row: usize,
}

impl QuantizedLinear {
    /// Create a new Q4_K quantized linear layer
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features (should align to 256 for efficiency)
    /// * `out_features` - Number of output features
    /// * `weight_bytes` - Raw Q4_K quantized weight data
    /// * `bias` - Bias vector
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Dimensions are zero
    /// - Weight bytes don't match expected size
    /// - Bias length doesn't match out_features
    pub fn new(
        in_features: usize,
        out_features: usize,
        weight_bytes: Vec<u8>,
        bias: Vec<f32>,
    ) -> Result<Self> {
        // Q4_K: 144 bytes per super-block of 256 values
        const SUPER_BLOCK_VALUES: usize = 256;
        const SUPER_BLOCK_BYTES: usize = 144;

        if in_features == 0 || out_features == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "in_features and out_features must be > 0".to_string(),
            });
        }

        if bias.len() != out_features {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Bias length {} doesn't match out_features {}",
                    bias.len(),
                    out_features
                ),
            });
        }

        let super_blocks_per_row = in_features.div_ceil(SUPER_BLOCK_VALUES);
        let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;
        let expected_bytes = out_features * bytes_per_row;

        if weight_bytes.len() != expected_bytes {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Weight bytes {} doesn't match expected {} ({}x{})",
                    weight_bytes.len(),
                    expected_bytes,
                    out_features,
                    bytes_per_row
                ),
            });
        }

        Ok(Self {
            in_features,
            out_features,
            weight_bytes,
            bias,
            bytes_per_row,
        })
    }

    /// Forward pass through quantized linear layer
    ///
    /// Uses fused dequantization+dot product for memory efficiency.
    /// Per llama.cpp optimization: inline dequant avoids 8x memory traffic penalty.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape `[batch, in_features]` or `[in_features]`
    ///
    /// # Returns
    ///
    /// Output tensor with shape `[batch, out_features]` or `[out_features]`
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input tensor is empty
    /// - Input last dimension doesn't match `in_features`
    /// - Quantization format error during fused dequant+dot
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        use crate::quantize::fused_q4k_dot_simd;

        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.in_features {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match in_features {}",
                    last_dim, self.in_features
                ),
            });
        }

        let data = input.data();
        let total_size = data.len();
        let num_rows = total_size / self.in_features;

        let mut output = Vec::with_capacity(num_rows * self.out_features);

        // For each input row, compute output using fused Q4_K dequant+dot
        for row_idx in 0..num_rows {
            let input_start = row_idx * self.in_features;
            let input_row = &data[input_start..input_start + self.in_features];

            // For each output column, use fused dequant+dot
            for j in 0..self.out_features {
                let weight_start = j * self.bytes_per_row;
                let weight_row =
                    &self.weight_bytes[weight_start..weight_start + self.bytes_per_row];

                // Fused dequantization + dot product (SIMD-accelerated)
                let dot = fused_q4k_dot_simd(weight_row, input_row)?;
                output.push(dot + self.bias[j]);
            }
        }

        // Construct output shape
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        // Handle degenerate case: 1D input produces 1D output
        if output_shape.is_empty() {
            output_shape.push(self.out_features);
        }

        Tensor::from_vec(output_shape, output)
    }

    /// Get input features
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get weight bytes (for model inspection)
    #[must_use]
    pub fn weight_bytes(&self) -> &[u8] {
        &self.weight_bytes
    }

    /// Get bias vector
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Memory usage in bytes (for diagnostics)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.weight_bytes.len() + self.bias.len() * std::mem::size_of::<f32>()
    }
}

/// Fused LayerNorm + Linear layer
///
/// Combines layer normalization and linear transformation in a single pass
/// to reduce memory bandwidth by avoiding intermediate tensor writes.
///
/// # Algorithm
///
/// Standard (two passes):
/// ```text
/// norm_out = LayerNorm(input)   // Full tensor written to memory
/// output = Linear(norm_out)      // Full tensor read from memory
/// ```
///
/// Fused (single pass):
/// ```text
/// For each row:
///   1. Compute mean and variance (in registers)
///   2. For each output column:
///      a. Normalize input in registers
///      b. Apply linear transformation directly
/// ```
///
/// This reduces memory traffic by ~50% for the LayerNorm→Linear pattern.
///
/// # References
///
/// - "Fused Operations for Deep Learning" - NVIDIA, 2019
#[derive(Debug, Clone)]
pub struct FusedLayerNormLinear {
    /// Feature dimension (must match between LayerNorm and Linear input)
    feature_dim: usize,
    /// Output dimension
    out_features: usize,
    /// LayerNorm epsilon
    eps: f32,
    /// LayerNorm weight (gamma)
    norm_weight: Vec<f32>,
    /// LayerNorm bias (beta)
    norm_bias: Vec<f32>,
    /// Linear weight matrix [feature_dim, out_features]
    linear_weight: Vec<f32>,
    /// Linear bias vector [out_features]
    linear_bias: Vec<f32>,
}
