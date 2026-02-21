use super::attention_helpers::ALiBi;
#[allow(clippy::wildcard_imports)]
use super::*;

impl ALiBi {
    /// Create `ALiBi` with specified number of attention heads.
    ///
    /// Slopes follow geometric sequence: 2^(-8/n), 2^(-16/n), ...
    #[must_use]
    pub fn new(num_heads: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);
        Self { num_heads, slopes }
    }

    /// Compute slopes using the formula from the paper.
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        // For power-of-2 heads, use geometric sequence
        // For non-power-of-2, interpolate
        let closest_pow2 = (num_heads as f32).log2().ceil() as u32;
        let base = 2.0_f32.powf(-(8.0 / 2.0_f32.powi(closest_pow2 as i32)));

        let mut slopes = Vec::with_capacity(num_heads);

        if num_heads.is_power_of_two() {
            for i in 0..num_heads {
                slopes.push(base.powi((i + 1) as i32));
            }
        } else {
            // Interpolate for non-power-of-2
            let extra_base = 2.0_f32.powf(-(8.0 / 2.0_f32.powi(closest_pow2 as i32 - 1)));
            let num_extra = 2 * num_heads - 2_usize.pow(closest_pow2);

            for i in 0..num_extra {
                slopes.push(extra_base.powi(((i + 1) * 2) as i32));
            }
            for i in num_extra..num_heads {
                slopes.push(base.powi((i - num_extra + 1) as i32));
            }
        }

        slopes
    }

    /// Compute `ALiBi` bias matrix for given sequence length.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Current sequence length
    ///
    /// # Returns
    ///
    /// Bias tensor [`num_heads`, `seq_len`, `seq_len`] to add to attention scores.
    #[must_use]
    pub fn compute_bias(&self, seq_len: usize) -> Tensor {
        let mut bias = vec![0.0; self.num_heads * seq_len * seq_len];

        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let distance = (i as i32 - j as i32).abs() as f32;
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    bias[idx] = -slope * distance;
                }
            }
        }

        Tensor::new(&bias, &[self.num_heads, seq_len, seq_len])
    }

    /// Apply `ALiBi` to attention scores.
    ///
    /// # Arguments
    ///
    /// * `scores` - Attention scores [batch, `num_heads`, `seq_len`, `seq_len`]
    ///
    /// # Returns
    ///
    /// Scores with `ALiBi` bias applied.
    #[must_use]
    pub fn apply(&self, scores: &Tensor) -> Tensor {
        let shape = scores.shape();
        assert!(shape.len() == 4, "Expected 4D tensor");
        assert_eq!(shape[1], self.num_heads, "num_heads mismatch");

        let (batch, _, seq_len, _) = (shape[0], shape[1], shape[2], shape[3]);
        let bias = self.compute_bias(seq_len);

        // Add bias (broadcast over batch)
        let mut output = scores.data().to_vec();

        for b in 0..batch {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let score_idx = b * self.num_heads * seq_len * seq_len
                            + h * seq_len * seq_len
                            + i * seq_len
                            + j;
                        let bias_idx = h * seq_len * seq_len + i * seq_len + j;
                        output[score_idx] += bias.data()[bias_idx];
                    }
                }
            }
        }

        Tensor::new(&output, shape)
    }

    /// Get slopes for each head.
    #[must_use]
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// Get number of heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

#[cfg(test)]
mod tests;
