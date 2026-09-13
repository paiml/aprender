
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
    /// For powers of 2: m[h] = 2^(-8(h+1)/n)
    /// For non-powers of 2: interpolate between adjacent powers of 2
    ///
    /// Matches Press et al. 2021 and llama.cpp ggml `soft_max_ext`
    /// (`m0 = 2^(-8/n)`, `slope = m0^(h+1)`). For n=8: slopes are
    /// 0.5, 0.25, ..., 2^(-8) = 0.00390625. The `+ 1` is load-bearing:
    /// without it head 0 carries slope 1.0 (an extra factor of `m0`).
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

        // Compute slopes for power of 2 heads.
        // slope[h] = 2^(-8(h+1)/n) — the (i + 1) is load-bearing (PMAT-858):
        // without it, head 0 gets 2^0 = 1.0 (an extra factor of m0 = 2^(8/n))
        // instead of the correct 2^(-8/n).
        for i in 0..closest_power_of_2.min(num_heads) {
            #[allow(clippy::cast_precision_loss)]
            let exponent = -((i + 1) as f32) * ratio;
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

// ============================================================================
// PMAT-858: ALiBi slope exponent falsifier
//
// Defect (since fixed): compute_slopes used exponent = -h * (8/n), giving slope[h] = 2^(-8h/n).
// For h=0 that is 2^0 = 1.0 — every head carried an extra factor of m0 = 2^(8/n)
// vs the reference. The correct ALiBi slope (Press et al. 2021 + llama.cpp ggml
// soft_max_ext: m0 = 2^(-8/n), slope = m0^(h+1)) is slope[h] = 2^(-8(h+1)/n).
//
// RED   (buggy code): slopes()[0] == 1.0
// GREEN (fixed code): slopes()[0] == 0.5 for n=8, slopes()[7] == 2^(-8).
// ============================================================================
#[cfg(test)]
mod pmat_858_alibi_slope_falsifier {
    use super::ALiBi;

    /// Reference slope per Press et al. 2021 / llama.cpp ggml:
    /// m0 = 2^(-8/n), slope[h] = m0^(h+1) = 2^(-8(h+1)/n).
    fn reference_slope(h: usize, num_heads: usize) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let exponent = -8.0 * ((h + 1) as f32) / (num_heads as f32);
        2_f32.powf(exponent)
    }

    #[test]
    fn falsifier_alibi_slopes_power_of_two_match_reference() {
        // n=8: reference slopes are 2^(-1), 2^(-2), ..., 2^(-8).
        let alibi = ALiBi::new(8).expect("8 heads is valid");
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 8);

        // Headline falsifier: head 0 must be 0.5, NOT the buggy 1.0.
        assert!(
            (slopes[0] - 0.5).abs() < 1e-7,
            "PMAT-858: slopes[0] must be 0.5 (2^(-8/8)), got {} (buggy code yields 1.0)",
            slopes[0]
        );
        // Last head: 2^(-8) = 0.003_906_25 (buggy code yields 2^(-7) = 0.007_812_5).
        assert!(
            (slopes[7] - 0.003_906_25).abs() < 1e-7,
            "PMAT-858: slopes[7] must be 2^(-8) = 0.00390625, got {}",
            slopes[7]
        );

        // Every head must match the closed-form reference exactly (within fp tol).
        for (h, &s) in slopes.iter().enumerate() {
            let want = reference_slope(h, 8);
            assert!(
                (s - want).abs() < 1e-7,
                "PMAT-858: slopes[{h}] = {s}, reference 2^(-8(h+1)/8) = {want}"
            );
        }
    }

    #[test]
    fn falsifier_alibi_slopes_single_head() {
        // n=1: slope[0] = 2^(-8) (NOT 2^0 = 1.0).
        let alibi = ALiBi::new(1).expect("1 head is valid");
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 1);
        assert!(
            (slopes[0] - 0.003_906_25).abs() < 1e-7,
            "PMAT-858: single-head slope must be 2^(-8) = 0.00390625, got {}",
            slopes[0]
        );
    }

    #[test]
    fn falsifier_alibi_slopes_non_power_of_two_match_paper() {
        // n=12 exercises the interpolation branch. The original ALiBi paper
        // get_slopes(12) = get_slopes_power_of_2(8) followed by the even
        // entries of get_slopes(16):
        //   [2^-1, 2^-2, .., 2^-8, 2^-0.5, 2^-1.5, 2^-2.5, 2^-3.5]
        let alibi = ALiBi::new(12).expect("12 heads is valid");
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 12);

        let expected: [f32; 12] = [
            // power-of-2 block (closest_power_of_2 = 8): 2^(-8(h+1)/8)
            0.5,
            0.25,
            0.125,
            0.062_5,
            0.031_25,
            0.015_625,
            0.007_812_5,
            0.003_906_25,
            // interpolation block: 2^(-4(2i+1)/8) = 2^(-(2i+1)/2)
            2_f32.powf(-0.5),
            2_f32.powf(-1.5),
            2_f32.powf(-2.5),
            2_f32.powf(-3.5),
        ];
        for (h, (&got, &want)) in slopes.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "PMAT-858: slopes[{h}] = {got}, paper get_slopes(12)[{h}] = {want}"
            );
        }

        // First head of the power-of-2 block must still be 0.5, not 1.0.
        assert!(
            (slopes[0] - 0.5).abs() < 1e-7,
            "PMAT-858: non-power-of-2 head 0 must be 0.5, got {}",
            slopes[0]
        );
    }

    #[test]
    fn falsifier_alibi_slopes_strictly_below_one() {
        // With the fix, no slope can be >= 1.0 (the buggy head-0 = 1.0 is gone).
        for n in [1usize, 2, 4, 8, 12, 16, 32] {
            let alibi = ALiBi::new(n).expect("valid head count");
            for (h, &s) in alibi.slopes().iter().enumerate() {
                assert!(s > 0.0, "PMAT-858: slope[{h}] (n={n}) must be > 0, got {s}");
                assert!(
                    s < 1.0,
                    "PMAT-858: slope[{h}] (n={n}) must be < 1.0 (buggy head 0 = 1.0), got {s}"
                );
            }
        }
    }
}
