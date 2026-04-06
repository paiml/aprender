
impl FusedLayerNormLinear {
    /// Create a new fused LayerNorm+Linear layer
    ///
    /// # Arguments
    ///
    /// * `feature_dim` - Input feature dimension (normalized dimension)
    /// * `out_features` - Output dimension of linear layer
    /// * `eps` - LayerNorm epsilon for numerical stability
    ///
    /// # Errors
    ///
    /// Returns error if feature_dim or out_features is zero
    pub fn new(feature_dim: usize, out_features: usize, eps: f32) -> Result<Self> {
        if feature_dim == 0 || out_features == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "feature_dim and out_features must be > 0".to_string(),
            });
        }

        Ok(Self {
            feature_dim,
            out_features,
            eps,
            norm_weight: vec![1.0; feature_dim],
            norm_bias: vec![0.0; feature_dim],
            linear_weight: vec![0.0; feature_dim * out_features],
            linear_bias: vec![0.0; out_features],
        })
    }

    /// Forward pass with fused LayerNorm + Linear
    ///
    /// Computes `Linear(LayerNorm(input))` in a single pass without
    /// materializing the intermediate normalized tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor `[batch, feature_dim]` or `[feature_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[batch, out_features]` or `[out_features]`
    ///
    /// # Errors
    ///
    /// Returns error if input dimensions don't match feature_dim
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.feature_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match feature_dim {}",
                    last_dim, self.feature_dim
                ),
            });
        }

        let data = input.data();
        let num_rows = data.len() / self.feature_dim;
        let mut output = Vec::with_capacity(num_rows * self.out_features);

        for row_idx in 0..num_rows {
            let row_start = row_idx * self.feature_dim;
            let row = &data[row_start..row_start + self.feature_dim];

            // Step 1: Compute mean (in registers)
            #[allow(clippy::cast_precision_loss)]
            let mean: f32 = row.iter().sum::<f32>() / self.feature_dim as f32;

            // Step 2: Compute variance (in registers)
            #[allow(clippy::cast_precision_loss)]
            let variance: f32 = row
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.feature_dim as f32;

            let inv_std = 1.0 / (variance + self.eps).sqrt();

            // Step 3: Fused normalize + linear for each output column
            // This avoids writing normalized values to memory
            for j in 0..self.out_features {
                let mut sum = self.linear_bias[j];
                for (i, &x) in row.iter().enumerate() {
                    // Normalize in registers
                    let normalized = (x - mean) * inv_std;
                    let transformed = normalized * self.norm_weight[i] + self.norm_bias[i];
                    // Apply linear weight immediately
                    sum += transformed * self.linear_weight[i * self.out_features + j];
                }
                output.push(sum);
            }
        }

        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        Tensor::from_vec(output_shape, output)
    }

    /// Parallel forward pass using rayon
    ///
    /// Parallelizes over rows for multi-core utilization.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input tensor is empty
    /// - Last dimension doesn't match feature_dim
    pub fn forward_parallel(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        use rayon::prelude::*;

        let shape = input.shape();
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor cannot be empty".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.feature_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Last dimension {} doesn't match feature_dim {}",
                    last_dim, self.feature_dim
                ),
            });
        }

        let data = input.data();
        let num_rows = data.len() / self.feature_dim;

        let output: Vec<f32> = (0..num_rows)
            .into_par_iter()
            .flat_map(|row_idx| {
                let row_start = row_idx * self.feature_dim;
                let row = &data[row_start..row_start + self.feature_dim];

                // Compute mean and variance
                #[allow(clippy::cast_precision_loss)]
                let mean: f32 = row.iter().sum::<f32>() / self.feature_dim as f32;

                #[allow(clippy::cast_precision_loss)]
                let variance: f32 = row
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / self.feature_dim as f32;

                let inv_std = 1.0 / (variance + self.eps).sqrt();

                // Fused normalize + linear
                (0..self.out_features)
                    .map(|j| {
                        let mut sum = self.linear_bias[j];
                        for (i, &x) in row.iter().enumerate() {
                            let normalized = (x - mean) * inv_std;
                            let transformed = normalized * self.norm_weight[i] + self.norm_bias[i];
                            sum += transformed * self.linear_weight[i * self.out_features + j];
                        }
                        sum
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        Tensor::from_vec(output_shape, output)
    }

    /// Get feature dimension
    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Get output features
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get mutable reference to LayerNorm weight (gamma)
    #[must_use]
    pub fn norm_weight_mut(&mut self) -> &mut [f32] {
        &mut self.norm_weight
    }

    /// Get mutable reference to LayerNorm bias (beta)
    #[must_use]
    pub fn norm_bias_mut(&mut self) -> &mut [f32] {
        &mut self.norm_bias
    }

    /// Get mutable reference to Linear weight
    #[must_use]
    pub fn linear_weight_mut(&mut self) -> &mut [f32] {
        &mut self.linear_weight
    }

    /// Get mutable reference to Linear bias
    #[must_use]
    pub fn linear_bias_mut(&mut self) -> &mut [f32] {
        &mut self.linear_bias
    }
}

/// Feed-forward network (FFN)
///
/// Two-layer feed-forward network with GELU activation:
/// ```text
/// FFN(x) = Linear2(GELU(Linear1(x)))
/// ```
///
/// Typically used in transformer blocks with:
/// - `hidden_dim` = model dimension (e.g., 768, 512)
/// - `intermediate_dim` = expansion (typically 4x `hidden_dim`)
///
/// # References
///
/// Standard transformer FFN from "Attention is All You Need"
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// First linear layer (expansion)
    fc1: Linear,
    /// Second linear layer (projection)
    fc2: Linear,
    /// Hidden dimension
    hidden_dim: usize,
    /// Intermediate dimension
    intermediate_dim: usize,
}

impl FeedForward {
    /// Create a new feed-forward network
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Input/output dimension
    /// * `intermediate_dim` - Intermediate dimension (typically 4x `hidden_dim`)
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are zero
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let ffn = FeedForward::new(768, 3072)?; // GPT-2 style (4x expansion)
    /// ```
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Result<Self> {
        let fc1 = Linear::new(hidden_dim, intermediate_dim)?;
        let fc2 = Linear::new(intermediate_dim, hidden_dim)?;

        Ok(Self {
            fc1,
            fc2,
            hidden_dim,
            intermediate_dim,
        })
    }

    /// Forward pass through feed-forward network
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with shape `[..., hidden_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor with shape `[..., hidden_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape doesn't match `hidden_dim`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let input = Tensor::from_vec(vec![2, 768], data)?;
    /// let output = ffn.forward(&input)?;
    /// assert_eq!(output.shape(), &[2, 768]);
    /// ```
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // fc1: [hidden_dim] -> [intermediate_dim]
        let hidden = self.fc1.forward(input)?;

        // GELU activation
        let activated = gelu(&hidden)?;

        // fc2: [intermediate_dim] -> [hidden_dim]
        self.fc2.forward(&activated)
    }

    /// Get hidden dimension
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get intermediate dimension
    #[must_use]
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate_dim
    }

    /// Get mutable reference to first linear layer (for loading weights)
    #[must_use]
    pub fn fc1_mut(&mut self) -> &mut Linear {
        &mut self.fc1
    }

    /// Get mutable reference to second linear layer (for loading weights)
    #[must_use]
    pub fn fc2_mut(&mut self) -> &mut Linear {
        &mut self.fc2
    }
}

#[cfg(test)]
mod tests;
