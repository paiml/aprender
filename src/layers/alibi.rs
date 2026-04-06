
impl ALiBi {
    /// Create a new `ALiBi` layer
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    ///
    /// # Errors
    ///
    /// Returns error if `num_heads` is zero
    pub fn new(num_heads: usize) -> Result<Self> {
        if num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }

        // Compute slopes for each head
        let slopes = Self::compute_slopes(num_heads);

        Ok(Self { num_heads, slopes })
    }

    /// Compute head-specific slopes following `ALiBi` paper algorithm
    ///
    /// For powers of 2: m[h] = 2^(-8h/n)
    /// For non-powers of 2: interpolate between adjacent powers of 2
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        // Find closest power of 2
        let closest_power_of_2 = if num_heads.is_power_of_two() {
            num_heads
        } else {
            num_heads.next_power_of_two() / 2
        };

        #[allow(clippy::cast_precision_loss)]
        let ratio = 8.0 / (closest_power_of_2 as f32);

        let mut slopes = Vec::with_capacity(num_heads);

        // Compute slopes for power of 2 heads
        for i in 0..closest_power_of_2.min(num_heads) {
            #[allow(clippy::cast_precision_loss)]
            let exponent = -(i as f32) * ratio;
            slopes.push(2_f32.powf(exponent));
        }

        // If not power of 2, add extra slopes with step=2
        if num_heads > closest_power_of_2 {
            #[allow(clippy::cast_precision_loss)]
            let extra_ratio = 4.0 / (closest_power_of_2 as f32);

            for i in 0..(num_heads - closest_power_of_2) {
                #[allow(clippy::cast_precision_loss)]
                let exponent = -((2 * i + 1) as f32) * extra_ratio;
                slopes.push(2_f32.powf(exponent));
            }
        }

        slopes
    }

    /// Get bias matrix for a given sequence length
    ///
    /// Returns a tensor of shape `[seq_len, seq_len, num_heads]` where:
    /// ```text
    /// bias[i, j, h] = -slopes[h] * abs(i - j)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Sequence length for computing bias
    ///
    /// # Returns
    ///
    /// Tensor of shape `[seq_len, seq_len, num_heads]` containing position biases
    ///
    /// # Errors
    ///
    /// Returns error if `seq_len` is zero
    pub fn get_bias(&self, seq_len: usize) -> Result<Tensor<f32>> {
        if seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "seq_len must be > 0".to_string(),
            });
        }

        let total_size = seq_len * seq_len * self.num_heads;
        let mut data = Vec::with_capacity(total_size);

        // Compute bias for each position pair and head
        for i in 0..seq_len {
            for j in 0..seq_len {
                for &slope in &self.slopes {
                    #[allow(clippy::cast_precision_loss)]
                    let distance = (i as f32 - j as f32).abs();
                    let bias = -slope * distance;
                    data.push(bias);
                }
            }
        }

        Tensor::from_vec(vec![seq_len, seq_len, self.num_heads], data)
    }

    /// Get number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head-specific slopes
    #[must_use]
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }
}
