
impl ScaledRoPE {
    /// Create a new scaled `RoPE` layer
    ///
    /// # Arguments
    ///
    /// * `dim` - Embedding dimension (must be even)
    /// * `base` - Base frequency (typically 10000)
    /// * `scaling` - Scaling method to use
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn new(dim: usize, base: f32, scaling: RopeScalingType) -> Result<Self> {
        if dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be > 0".to_string(),
            });
        }
        if !dim.is_multiple_of(2) {
            return Err(RealizarError::InvalidShape {
                reason: "dim must be even for RoPE".to_string(),
            });
        }

        let (scaled_base, mscale, inv_freq) = Self::compute_frequencies(dim, base, &scaling);

        Ok(Self {
            dim,
            original_base: base,
            scaled_base,
            scaling,
            inv_freq,
            mscale,
        })
    }

    /// Create scaled `RoPE` with default base (10000)
    ///
    /// # Errors
    ///
    /// Returns error if `dim` is zero or odd
    pub fn with_default_base(dim: usize, scaling: RopeScalingType) -> Result<Self> {
        Self::new(dim, 10000.0, scaling)
    }

    /// Compute inverse frequencies with scaling applied
    fn compute_frequencies(
        dim: usize,
        base: f32,
        scaling: &RopeScalingType,
    ) -> (f32, f32, Vec<f32>) {
        let half_dim = dim / 2;

        // Compute scaled base and mscale based on scaling type
        #[allow(clippy::cast_precision_loss)]
        let (scaled_base, mscale) = match scaling {
            RopeScalingType::None | RopeScalingType::Linear { .. } => (base, 1.0),
            RopeScalingType::Ntk { scale } => {
                // NTK formula: base' = base * scale^(dim / (dim - 2))
                let dim_f = dim as f32;
                let exponent = dim_f / (dim_f - 2.0);
                let ntk_base = base * scale.powf(exponent);
                (ntk_base, 1.0)
            },
            RopeScalingType::DynamicNtk {
                original_max_len,
                target_max_len,
            } => {
                // Dynamic NTK uses scale = target / original
                let scale = (*target_max_len as f32) / (*original_max_len as f32);
                let dim_f = dim as f32;
                let exponent = dim_f / (dim_f - 2.0);
                let ntk_base = base * scale.powf(exponent);
                (ntk_base, 1.0)
            },
            RopeScalingType::Yarn {
                original_max_len,
                target_max_len,
                attn_factor,
                beta_fast,
                beta_slow,
            } => {
                // YaRN combines NTK with attention scaling
                let scale = (*target_max_len as f32) / (*original_max_len as f32);
                let dim_f = dim as f32;

                // Compute NTK-style base modification
                // YaRN uses a smoother interpolation based on frequency
                let exponent = dim_f / (dim_f - 2.0);
                let ntk_base = base * scale.powf(exponent);

                // Compute mscale (attention factor)
                // Default: sqrt(1 + ln(scale) / ln(original_max_len))
                let mscale = if *attn_factor > 0.0 {
                    *attn_factor
                } else {
                    let log_scale = scale.ln();
                    let log_orig = (*original_max_len as f32).ln();
                    (1.0 + log_scale / log_orig).sqrt()
                };

                // The beta parameters affect the interpolation ramp
                // but are applied per-frequency in the forward pass
                let _ = (beta_fast, beta_slow); // Used in forward

                (ntk_base, mscale)
            },
        };

        // Compute inverse frequencies with scaled base
        let mut inv_freq = Vec::with_capacity(half_dim);

        #[allow(clippy::cast_precision_loss)]
        for i in 0..half_dim {
            let exponent = -2.0 * (i as f32) / (dim as f32);
            inv_freq.push(scaled_base.powf(exponent));
        }

        (scaled_base, mscale, inv_freq)
    }

    /// Apply scaled rotary embeddings to input at given position
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with last dimension equal to `dim`
    /// * `position` - Position index for computing rotation angles
    ///
    /// # Returns
    ///
    /// Tensor with same shape as input, with scaled rotary embeddings applied
    ///
    /// # Errors
    ///
    /// Returns error if input's last dimension doesn't match `dim`
    pub fn forward(&self, input: &Tensor<f32>, position: usize) -> Result<Tensor<f32>> {
        let shape = input.shape();

        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Input tensor must have at least 1 dimension".to_string(),
            });
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Expected last dimension {}, got {}", self.dim, last_dim),
            });
        }

        let data = input.data();
        let num_vectors = data.len() / self.dim;
        let mut output = Vec::with_capacity(data.len());

        // Compute effective position based on scaling type
        #[allow(clippy::cast_precision_loss)]
        let effective_pos = match &self.scaling {
            RopeScalingType::None
            | RopeScalingType::Ntk { .. }
            | RopeScalingType::DynamicNtk { .. }
            | RopeScalingType::Yarn { .. } => position as f32,
            RopeScalingType::Linear { scale } => (position as f32) / scale,
        };

        // Compute sin/cos for this position
        let half_dim = self.dim / 2;
        let mut cos_vals = Vec::with_capacity(half_dim);
        let mut sin_vals = Vec::with_capacity(half_dim);

        // Apply YaRN interpolation ramp if applicable
        #[allow(clippy::cast_precision_loss)]
        for (i, inv_f) in self.inv_freq.iter().enumerate() {
            let angle = inv_f * effective_pos;

            // For YaRN, apply interpolation ramp based on frequency index
            let (cos_val, sin_val) = if let RopeScalingType::Yarn {
                original_max_len,
                target_max_len,
                beta_fast,
                beta_slow,
                ..
            } = &self.scaling
            {
                // Compute wavelength for this frequency
                let freq = 1.0 / inv_f;
                let wavelength = 2.0 * std::f32::consts::PI * freq;

                // Compute interpolation factor
                let low_freq_wavelen = (*original_max_len as f32) / *beta_slow;
                let high_freq_wavelen = (*original_max_len as f32) / *beta_fast;

                let ramp = if wavelength < high_freq_wavelen {
                    0.0 // Full extrapolation (use NTK-scaled frequency)
                } else if wavelength > low_freq_wavelen {
                    1.0 // Full interpolation (use linear-scaled position)
                } else {
                    // Linear ramp between extrapolation and interpolation
                    (wavelength - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
                };

                // Interpolate between NTK angle and linear angle
                let scale = (*target_max_len as f32) / (*original_max_len as f32);
                let linear_pos = effective_pos / scale;

                // Original frequency from unscaled base
                let orig_inv_f = self
                    .original_base
                    .powf(-2.0 * (i as f32) / (self.dim as f32));
                let linear_angle = orig_inv_f * linear_pos;

                // Interpolate angles
                let final_angle = angle * (1.0 - ramp) + linear_angle * ramp;

                (final_angle.cos(), final_angle.sin())
            } else {
                (angle.cos(), angle.sin())
            };

            cos_vals.push(cos_val);
            sin_vals.push(sin_val);
        }

        // Apply rotation to each vector (with mscale for YaRN)
        for vec_idx in 0..num_vectors {
            let offset = vec_idx * self.dim;

            for i in 0..half_dim {
                let x0 = data[offset + 2 * i];
                let x1 = data[offset + 2 * i + 1];
                let cos_val = cos_vals[i];
                let sin_val = sin_vals[i];

                // Apply 2D rotation with mscale
                let y0 = (x0 * cos_val - x1 * sin_val) * self.mscale;
                let y1 = (x0 * sin_val + x1 * cos_val) * self.mscale;

                output.push(y0);
                output.push(y1);
            }
        }

        Tensor::from_vec(shape.to_vec(), output)
    }

    /// Get embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get original base frequency
    #[must_use]
    pub fn original_base(&self) -> f32 {
        self.original_base
    }

    /// Get scaled base frequency
    #[must_use]
    pub fn scaled_base(&self) -> f32 {
        self.scaled_base
    }

    /// Get scaling configuration
    #[must_use]
    pub fn scaling(&self) -> &RopeScalingType {
        &self.scaling
    }

    /// Get inverse frequencies
    #[must_use]
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// Get attention scaling factor (mscale)
    #[must_use]
    pub fn mscale(&self) -> f32 {
        self.mscale
    }

    /// Compute effective context length multiplier
    ///
    /// Returns the factor by which the original context length is extended
    #[must_use]
    pub fn context_length_multiplier(&self) -> f32 {
        match &self.scaling {
            RopeScalingType::None => 1.0,
            RopeScalingType::Linear { scale } | RopeScalingType::Ntk { scale } => *scale,
            RopeScalingType::DynamicNtk {
                original_max_len,
                target_max_len,
            }
            | RopeScalingType::Yarn {
                original_max_len,
                target_max_len,
                ..
            } => (*target_max_len as f32) / (*original_max_len as f32),
        }
    }
}

/// Attention with Linear Biases (`ALiBi`)
///
/// Replaces traditional position embeddings by adding a static, non-learned
/// bias to query-key attention scores. The bias is proportional to the distance
/// between positions, with head-specific slopes that enable better length
/// extrapolation.
///
/// # Algorithm
///
/// For each attention head h, `ALiBi` adds the following bias to attention scores:
///
/// ```text
/// bias[i, j] = -m[h] * |i - j|
/// ```
///
/// where m[h] is the head-specific slope computed as:
/// - For powers of 2: m[h] = 2^(-8h/n) where n is the number of heads
/// - For non-powers of 2: interpolation between adjacent powers of 2
///
/// # References
///
/// - "Train Short, Test Long: Attention with Linear Biases Enables Input Length
///   Extrapolation" - Press et al., ICLR 2022
/// - <https://arxiv.org/abs/2108.12409>
///
/// # Example
///
/// ```rust,ignore
/// use realizar::layers::ALiBi;
///
/// // Create ALiBi for 8 attention heads
/// let alibi = ALiBi::new(8)?;
///
/// // Get bias matrix for sequence length 10
/// let bias = alibi.get_bias(10)?;
///
/// // Add to attention scores before softmax
/// // scores: [seq_len, seq_len, num_heads]
/// // scores = scores + bias
/// ```
#[derive(Debug, Clone)]
pub struct ALiBi {
    /// Number of attention heads
    num_heads: usize,
    /// Head-specific slopes (one per head)
    slopes: Vec<f32>,
}
