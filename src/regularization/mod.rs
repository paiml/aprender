//! Regularization techniques for neural network training.
//!
//! # Techniques
//! - Mixup: Interpolate between training samples
//! - Label Smoothing: Soft targets instead of hard labels
//! - `CutMix`: Cut and paste patches between images

use crate::primitives::Vector;
use rand::Rng;

/// Mixup data augmentation (Zhang et al., 2018).
/// Creates virtual training examples: x' = `λx_i` + (1-λ)x_j
#[derive(Debug, Clone)]
pub struct Mixup {
    alpha: f32,
}

impl Mixup {
    /// Create new Mixup with alpha parameter for Beta distribution.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Sample mixing coefficient from Beta(alpha, alpha).
    #[must_use]
    pub fn sample_lambda(&self) -> f32 {
        if self.alpha <= 0.0 {
            return 1.0;
        }
        sample_beta(self.alpha, self.alpha)
    }

    /// Mix two samples: x' = λ*x1 + (1-λ)*x2
    #[must_use]
    pub fn mix_samples(&self, x1: &Vector<f32>, x2: &Vector<f32>, lambda: f32) -> Vector<f32> {
        let mixed: Vec<f32> = x1
            .as_slice()
            .iter()
            .zip(x2.as_slice().iter())
            .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
            .collect();
        Vector::from_slice(&mixed)
    }

    /// Mix labels: y' = λ*y1 + (1-λ)*y2
    #[must_use]
    pub fn mix_labels(&self, y1: &Vector<f32>, y2: &Vector<f32>, lambda: f32) -> Vector<f32> {
        self.mix_samples(y1, y2, lambda)
    }

    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

/// Label smoothing for soft targets.
/// Converts hard labels to: (1-ε)y + ε/K
#[derive(Debug, Clone)]
pub struct LabelSmoothing {
    epsilon: f32,
}

impl LabelSmoothing {
    /// Create label smoothing with smoothing factor ε.
    #[must_use]
    pub fn new(epsilon: f32) -> Self {
        assert!((0.0..1.0).contains(&epsilon));
        Self { epsilon }
    }

    /// Smooth a one-hot label vector.
    #[must_use]
    pub fn smooth(&self, label: &Vector<f32>) -> Vector<f32> {
        let n_classes = label.len();
        let smoothed: Vec<f32> = label
            .as_slice()
            .iter()
            .map(|&y| (1.0 - self.epsilon) * y + self.epsilon / n_classes as f32)
            .collect();
        Vector::from_slice(&smoothed)
    }

    /// Create smoothed one-hot from class index.
    #[must_use]
    pub fn smooth_index(&self, class_idx: usize, n_classes: usize) -> Vector<f32> {
        let mut result = vec![self.epsilon / n_classes as f32; n_classes];
        result[class_idx] = 1.0 - self.epsilon + self.epsilon / n_classes as f32;
        Vector::from_slice(&result)
    }

    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

/// Cross-entropy loss with label smoothing.
#[must_use]
pub fn cross_entropy_with_smoothing(logits: &Vector<f32>, target_idx: usize, epsilon: f32) -> f32 {
    let n_classes = logits.len();
    let probs = softmax(logits.as_slice());

    let mut loss = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        let target = if i == target_idx {
            1.0 - epsilon + epsilon / n_classes as f32
        } else {
            epsilon / n_classes as f32
        };
        loss -= target * p.max(1e-10).ln();
    }
    loss
}

/// Sample from Beta distribution using Gamma samples.
fn sample_beta(alpha: f32, beta: f32) -> f32 {
    let mut rng = rand::thread_rng();
    let x = sample_gamma(alpha, &mut rng);
    let y = sample_gamma(beta, &mut rng);
    let sum = x + y;
    // With extreme shape parameters (e.g. 0.01), f32 gamma samples can
    // underflow to 0.0, producing 0/0 = NaN. Return 0.5 in that case
    // (unbiased midpoint, correct for the symmetric alpha==beta case).
    if sum <= 0.0 {
        return 0.5;
    }
    (x / sum).clamp(0.0, 1.0)
}

fn sample_gamma(shape: f32, rng: &mut impl Rng) -> f32 {
    // Marsaglia and Tsang's method
    if shape < 1.0 {
        return sample_gamma(1.0 + shape, rng) * rng.gen::<f32>().powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f32 = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 {
            let u: f32 = rng.gen();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

fn sample_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// `CutMix` data augmentation (Yun et al., 2019).
///
/// Cuts a rectangular region from one image and pastes onto another.
#[derive(Debug, Clone)]
pub struct CutMix {
    alpha: f32,
}

impl CutMix {
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Sample lambda and bounding box for cutmix.
    #[must_use]
    pub fn sample(&self, height: usize, width: usize) -> CutMixParams {
        // Alpha <= 0 means no mixing: lambda = 1.0, empty box
        if self.alpha <= 0.0 {
            return CutMixParams {
                lambda: 1.0,
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0,
            };
        }
        let lambda = sample_beta(self.alpha, self.alpha);

        // Sample bounding box
        let ratio = (1.0 - lambda).sqrt();
        let rh = (height as f32 * ratio) as usize;
        let rw = (width as f32 * ratio) as usize;

        let mut rng = rand::thread_rng();
        let cx = rng.gen_range(0..width);
        let cy = rng.gen_range(0..height);

        let x1 = cx.saturating_sub(rw / 2);
        let y1 = cy.saturating_sub(rh / 2);
        let x2 = (cx + rw / 2).min(width);
        let y2 = (cy + rh / 2).min(height);

        // Actual lambda based on box area
        let actual_lambda = 1.0 - ((x2 - x1) * (y2 - y1)) as f32 / (height * width) as f32;

        CutMixParams {
            lambda: actual_lambda,
            x1,
            y1,
            x2,
            y2,
        }
    }

    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

/// Parameters for a `CutMix` operation.
#[derive(Debug, Clone)]
pub struct CutMixParams {
    pub lambda: f32,
    pub x1: usize,
    pub y1: usize,
    pub x2: usize,
    pub y2: usize,
}

impl CutMixParams {
    /// Apply cutmix to two flat image vectors `[C*H*W]`.
    #[must_use]
    pub fn apply(
        &self,
        img1: &[f32],
        img2: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f32> {
        let mut result = img1.to_vec();

        for c in 0..channels {
            for y in self.y1..self.y2 {
                for x in self.x1..self.x2 {
                    let idx = c * height * width + y * width + x;
                    if idx < result.len() {
                        result[idx] = img2[idx];
                    }
                }
            }
        }
        result
    }
}

/// Stochastic Depth (Huang et al., 2016).
///
/// Randomly drops entire residual blocks during training.
#[derive(Debug, Clone)]
pub struct StochasticDepth {
    drop_prob: f32,
    mode: DropMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DropMode {
    /// All samples in batch are dropped together
    Batch,
    /// Each sample dropped independently
    Row,
}

impl StochasticDepth {
    #[must_use]
    pub fn new(drop_prob: f32, mode: DropMode) -> Self {
        assert!((0.0..1.0).contains(&drop_prob));
        Self { drop_prob, mode }
    }

    /// Apply stochastic depth: returns true if should keep (not drop).
    #[must_use]
    pub fn should_keep(&self, training: bool) -> bool {
        if !training || self.drop_prob == 0.0 {
            return true;
        }
        rand::thread_rng().gen::<f32>() >= self.drop_prob
    }

    /// Compute survival probability for linear decay schedule.
    #[must_use]
    pub fn linear_decay(depth: usize, total_depth: usize, max_drop: f32) -> f32 {
        1.0 - (depth as f32 / total_depth as f32) * max_drop
    }

    #[must_use]
    pub fn drop_prob(&self) -> f32 {
        self.drop_prob
    }

    #[must_use]
    pub fn mode(&self) -> DropMode {
        self.mode
    }
}

/// R-Drop regularization (Liang et al., 2021).
///
/// Forces consistency between two forward passes with different dropout masks.
/// Adds bidirectional KL divergence loss between the two outputs.
///
/// Loss = CE(p1, y) + CE(p2, y) + α * (KL(p1||p2) + KL(p2||p1)) / 2
///
/// # Reference
/// Liang, X., et al. (2021). R-Drop: Regularized Dropout for Neural Networks.
#[derive(Debug, Clone)]
pub struct RDrop {
    alpha: f32,
}

impl RDrop {
    /// Create R-Drop with regularization weight alpha.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        assert!(alpha >= 0.0, "Alpha must be non-negative");
        Self { alpha }
    }

    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Compute KL divergence: KL(p || q) = Σ p * log(p / q)
    #[must_use]
    pub fn kl_divergence(&self, p: &[f32], q: &[f32]) -> f32 {
        assert_eq!(p.len(), q.len());
        let eps = 1e-10;
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                let pi = pi.max(eps);
                let qi = qi.max(eps);
                pi * (pi / qi).ln()
            })
            .sum()
    }

    /// Compute bidirectional KL divergence (symmetric).
    #[must_use]
    pub fn symmetric_kl(&self, p: &[f32], q: &[f32]) -> f32 {
        (self.kl_divergence(p, q) + self.kl_divergence(q, p)) / 2.0
    }

    /// Compute R-Drop regularization loss between two forward passes.
    #[must_use]
    pub fn compute_loss(&self, logits1: &[f32], logits2: &[f32]) -> f32 {
        let p1 = softmax_slice(logits1);
        let p2 = softmax_slice(logits2);
        self.alpha * self.symmetric_kl(&p1, &p2)
    }
}

fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// `SpecAugment`: Data augmentation for speech recognition (Park et al., 2019).
///
/// Applies time warping, frequency masking, and time masking to spectrograms.
///
/// # Methods
///
/// - **Time Warping**: Warps spectrogram along time axis
/// - **Frequency Masking**: Masks F consecutive frequency channels
/// - **Time Masking**: Masks T consecutive time steps
///
/// # Reference
///
/// - Park, D., et al. (2019). `SpecAugment`: A Simple Data Augmentation Method
///   for Automatic Speech Recognition. Interspeech.
#[derive(Debug, Clone)]
pub struct SpecAugment {
    /// Number of frequency masks to apply
    num_freq_masks: usize,
    /// Maximum size of frequency mask
    freq_mask_param: usize,
    /// Number of time masks to apply
    num_time_masks: usize,
    /// Maximum size of time mask
    time_mask_param: usize,
    /// Mask value (usually 0 or mean)
    mask_value: f32,
}

impl Default for SpecAugment {
    fn default() -> Self {
        Self::new()
    }
}

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
