
impl SpecAugment {
    /// Create `SpecAugment` with default parameters.
    ///
    /// Default: 2 frequency masks (F=27), 2 time masks (T=100)
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_freq_masks: 2,
            freq_mask_param: 27,
            num_time_masks: 2,
            time_mask_param: 100,
            mask_value: 0.0,
        }
    }

    /// Create with custom parameters.
    #[must_use]
    pub fn with_params(
        num_freq_masks: usize,
        freq_mask_param: usize,
        num_time_masks: usize,
        time_mask_param: usize,
    ) -> Self {
        Self {
            num_freq_masks,
            freq_mask_param,
            num_time_masks,
            time_mask_param,
            mask_value: 0.0,
        }
    }

    /// Set the mask value.
    #[must_use]
    pub fn with_mask_value(mut self, value: f32) -> Self {
        self.mask_value = value;
        self
    }

    /// Apply `SpecAugment` to a spectrogram.
    ///
    /// # Arguments
    ///
    /// * `spec` - Spectrogram as flat vector [`freq_bins` * `time_steps`]
    /// * `freq_bins` - Number of frequency bins
    /// * `time_steps` - Number of time steps
    ///
    /// # Returns
    ///
    /// Augmented spectrogram.
    #[must_use]
    pub fn apply(&self, spec: &[f32], freq_bins: usize, time_steps: usize) -> Vec<f32> {
        let mut result = spec.to_vec();
        let mut rng = rand::thread_rng();

        // Apply frequency masks
        for _ in 0..self.num_freq_masks {
            let f = rng.gen_range(0..=self.freq_mask_param.min(freq_bins));
            let f0 = rng.gen_range(0..freq_bins.saturating_sub(f).max(1));

            for freq in f0..f0 + f {
                if freq < freq_bins {
                    for t in 0..time_steps {
                        let idx = freq * time_steps + t;
                        if idx < result.len() {
                            result[idx] = self.mask_value;
                        }
                    }
                }
            }
        }

        // Apply time masks
        for _ in 0..self.num_time_masks {
            let t = rng.gen_range(0..=self.time_mask_param.min(time_steps));
            let t0 = rng.gen_range(0..time_steps.saturating_sub(t).max(1));

            for time in t0..t0 + t {
                if time < time_steps {
                    for freq in 0..freq_bins {
                        let idx = freq * time_steps + time;
                        if idx < result.len() {
                            result[idx] = self.mask_value;
                        }
                    }
                }
            }
        }

        result
    }

    /// Apply only frequency masking.
    #[must_use]
    pub fn freq_mask(&self, spec: &[f32], freq_bins: usize, time_steps: usize) -> Vec<f32> {
        let mut result = spec.to_vec();
        let mut rng = rand::thread_rng();

        for _ in 0..self.num_freq_masks {
            let f = rng.gen_range(0..=self.freq_mask_param.min(freq_bins));
            let f0 = rng.gen_range(0..freq_bins.saturating_sub(f).max(1));

            for freq in f0..f0 + f {
                if freq < freq_bins {
                    for t in 0..time_steps {
                        let idx = freq * time_steps + t;
                        if idx < result.len() {
                            result[idx] = self.mask_value;
                        }
                    }
                }
            }
        }

        result
    }

    /// Apply only time masking.
    #[must_use]
    pub fn time_mask(&self, spec: &[f32], freq_bins: usize, time_steps: usize) -> Vec<f32> {
        let mut result = spec.to_vec();
        let mut rng = rand::thread_rng();

        for _ in 0..self.num_time_masks {
            let t = rng.gen_range(0..=self.time_mask_param.min(time_steps));
            let t0 = rng.gen_range(0..time_steps.saturating_sub(t).max(1));

            for time in t0..t0 + t {
                if time < time_steps {
                    for freq in 0..freq_bins {
                        let idx = freq * time_steps + time;
                        if idx < result.len() {
                            result[idx] = self.mask_value;
                        }
                    }
                }
            }
        }

        result
    }

    #[must_use]
    pub fn num_freq_masks(&self) -> usize {
        self.num_freq_masks
    }

    #[must_use]
    pub fn num_time_masks(&self) -> usize {
        self.num_time_masks
    }
}

/// `RandAugment`: Automated data augmentation policy (Cubuk et al., 2020).
///
/// Applies N random augmentations from a set, each with magnitude M.
/// Simpler than `AutoAugment` with fewer hyperparameters.
///
/// # Reference
///
/// - Cubuk, E., et al. (2020). Randaugment: Practical automated data
///   augmentation with a reduced search space. CVPR.
#[derive(Debug, Clone)]
pub struct RandAugment {
    /// Number of augmentations to apply
    n: usize,
    /// Magnitude of augmentations (0-30 scale)
    m: usize,
    /// Available augmentation types
    augmentations: Vec<AugmentationType>,
}

/// Types of image augmentations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AugmentationType {
    Identity,
    Rotate,
    TranslateX,
    TranslateY,
    ShearX,
    ShearY,
    Brightness,
    Contrast,
    Sharpness,
    Posterize,
    Solarize,
    Equalize,
}

impl Default for RandAugment {
    fn default() -> Self {
        Self::new(2, 9)
    }
}

impl RandAugment {
    /// Create `RandAugment` with N operations and magnitude M.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of augmentations to apply
    /// * `m` - Magnitude (0-30)
    #[must_use]
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            n,
            m: m.min(30),
            augmentations: vec![
                AugmentationType::Identity,
                AugmentationType::Rotate,
                AugmentationType::TranslateX,
                AugmentationType::TranslateY,
                AugmentationType::Brightness,
                AugmentationType::Contrast,
                AugmentationType::Sharpness,
            ],
        }
    }

    /// Set custom augmentation types.
    #[must_use]
    pub fn with_augmentations(mut self, augs: Vec<AugmentationType>) -> Self {
        self.augmentations = augs;
        self
    }

    /// Get N random augmentation types.
    #[must_use]
    pub fn sample_augmentations(&self) -> Vec<AugmentationType> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut selected = Vec::with_capacity(self.n);

        for _ in 0..self.n {
            if let Some(&aug) = self.augmentations.choose(&mut rng) {
                selected.push(aug);
            }
        }

        selected
    }

    /// Get magnitude as normalized value [0, 1].
    #[must_use]
    pub fn normalized_magnitude(&self) -> f32 {
        self.m as f32 / 30.0
    }

    /// Apply a single augmentation to image data (simplified).
    ///
    /// # Arguments
    ///
    /// * `image` - Flat image vector `[C*H*W]`
    /// * `aug` - Augmentation type
    /// * `h` - Image height
    /// * `w` - Image width
    #[must_use]
    pub fn apply_single(
        &self,
        image: &[f32],
        aug: AugmentationType,
        h: usize,
        w: usize,
    ) -> Vec<f32> {
        let mag = self.normalized_magnitude();
        let mut result = image.to_vec();

        match aug {
            AugmentationType::Brightness => {
                let factor = 1.0 + (mag - 0.5) * 2.0; // [0, 2]
                for v in &mut result {
                    *v = (*v * factor).clamp(0.0, 1.0);
                }
            }
            AugmentationType::Contrast => {
                let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
                let factor = 1.0 + (mag - 0.5) * 2.0;
                for v in &mut result {
                    *v = ((*v - mean) * factor + mean).clamp(0.0, 1.0);
                }
            }
            AugmentationType::Rotate => {
                // Simplified: just flip for magnitude > 0.5
                if mag > 0.5 {
                    result.reverse();
                }
            }
            AugmentationType::TranslateX => {
                let shift = ((mag - 0.5) * w as f32 * 0.3) as i32;
                Self::shift_horizontal(&mut result, h, w, shift);
            }
            AugmentationType::TranslateY => {
                let shift = ((mag - 0.5) * h as f32 * 0.3) as i32;
                Self::shift_vertical(&mut result, h, w, shift);
            }
            // Identity and others: no operation
            AugmentationType::Identity
            | AugmentationType::ShearX
            | AugmentationType::ShearY
            | AugmentationType::Sharpness
            | AugmentationType::Posterize
            | AugmentationType::Solarize
            | AugmentationType::Equalize => {}
        }

        result
    }

    fn shift_horizontal(data: &mut [f32], h: usize, w: usize, shift: i32) {
        if shift == 0 {
            return;
        }
        let channels = data.len() / (h * w);
        for c in 0..channels {
            for y in 0..h {
                let row_start = c * h * w + y * w;
                let row: Vec<f32> = (0..w)
                    .map(|x| {
                        let src_x = (x as i32 - shift).rem_euclid(w as i32) as usize;
                        data[row_start + src_x]
                    })
                    .collect();
                data[row_start..row_start + w].copy_from_slice(&row);
            }
        }
    }

    fn shift_vertical(data: &mut [f32], h: usize, w: usize, shift: i32) {
        if shift == 0 {
            return;
        }
        let channels = data.len() / (h * w);
        for c in 0..channels {
            for x in 0..w {
                let col: Vec<f32> = (0..h)
                    .map(|y| {
                        let src_y = (y as i32 - shift).rem_euclid(h as i32) as usize;
                        data[c * h * w + src_y * w + x]
                    })
                    .collect();
                for (y, &val) in col.iter().enumerate() {
                    data[c * h * w + y * w + x] = val;
                }
            }
        }
    }

    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }
}

#[cfg(test)]
mod tests;
