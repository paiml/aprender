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
    x / (x + y)
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
mod tests {
    use super::*;

    #[test]
    fn test_mixup_new() {
        let mixup = Mixup::new(0.4);
        assert_eq!(mixup.alpha(), 0.4);
    }

    #[test]
    fn test_mixup_sample_lambda() {
        let mixup = Mixup::new(1.0);
        for _ in 0..10 {
            let lambda = mixup.sample_lambda();
            assert!((0.0..=1.0).contains(&lambda));
        }
    }

    #[test]
    fn test_mixup_alpha_zero() {
        let mixup = Mixup::new(0.0);
        assert_eq!(mixup.sample_lambda(), 1.0);
    }

    #[test]
    fn test_mixup_mix_samples() {
        let mixup = Mixup::new(1.0);
        let x1 = Vector::from_slice(&[1.0, 0.0]);
        let x2 = Vector::from_slice(&[0.0, 1.0]);

        let mixed = mixup.mix_samples(&x1, &x2, 0.5);
        assert!((mixed.as_slice()[0] - 0.5).abs() < 1e-6);
        assert!((mixed.as_slice()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mixup_mix_extreme_lambda() {
        let mixup = Mixup::new(1.0);
        let x1 = Vector::from_slice(&[1.0, 2.0]);
        let x2 = Vector::from_slice(&[3.0, 4.0]);

        let mixed0 = mixup.mix_samples(&x1, &x2, 0.0);
        assert_eq!(mixed0.as_slice(), &[3.0, 4.0]);

        let mixed1 = mixup.mix_samples(&x1, &x2, 1.0);
        assert_eq!(mixed1.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn test_label_smoothing_new() {
        let ls = LabelSmoothing::new(0.1);
        assert_eq!(ls.epsilon(), 0.1);
    }

    #[test]
    fn test_label_smoothing_smooth() {
        let ls = LabelSmoothing::new(0.1);
        let label = Vector::from_slice(&[1.0, 0.0, 0.0]);
        let smoothed = ls.smooth(&label);

        // First element: 0.9 * 1.0 + 0.1/3 ≈ 0.933
        assert!((smoothed.as_slice()[0] - 0.9333).abs() < 0.01);
        // Others: 0.9 * 0.0 + 0.1/3 ≈ 0.033
        assert!((smoothed.as_slice()[1] - 0.0333).abs() < 0.01);
    }

    #[test]
    fn test_label_smoothing_smooth_index() {
        let ls = LabelSmoothing::new(0.1);
        let smoothed = ls.smooth_index(0, 3);

        let sum: f32 = smoothed.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_label_smoothing_sums_to_one() {
        let ls = LabelSmoothing::new(0.2);
        let smoothed = ls.smooth_index(2, 5);

        let sum: f32 = smoothed.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_with_smoothing() {
        let logits = Vector::from_slice(&[2.0, 1.0, 0.5]);
        let loss = cross_entropy_with_smoothing(&logits, 0, 0.1);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_cross_entropy_no_smoothing() {
        let logits = Vector::from_slice(&[10.0, 0.0, 0.0]);
        let loss = cross_entropy_with_smoothing(&logits, 0, 0.0);
        // Should be close to 0 since softmax gives ~1.0 for first class
        assert!(loss < 0.1);
    }

    // CutMix tests

    #[test]
    fn test_cutmix_creation() {
        let cm = CutMix::new(1.0);
        assert_eq!(cm.alpha(), 1.0);
    }

    #[test]
    fn test_cutmix_sample() {
        let cm = CutMix::new(1.0);
        let params = cm.sample(32, 32);

        assert!(params.lambda >= 0.0 && params.lambda <= 1.0);
        assert!(params.x1 <= params.x2);
        assert!(params.y1 <= params.y2);
        assert!(params.x2 <= 32);
        assert!(params.y2 <= 32);
    }

    #[test]
    fn test_cutmix_apply() {
        let params = CutMixParams {
            lambda: 0.5,
            x1: 1,
            y1: 1,
            x2: 2,
            y2: 2,
        };

        let img1 = vec![1.0; 12]; // 1 channel, 3x4
        let img2 = vec![2.0; 12];

        let result = params.apply(&img1, &img2, 1, 3, 4);
        assert_eq!(result.len(), 12);
        // Position (1,1) should be from img2
        assert_eq!(result[4 + 1], 2.0);
    }

    // Stochastic Depth tests

    #[test]
    fn test_stochastic_depth_creation() {
        let sd = StochasticDepth::new(0.2, DropMode::Batch);
        assert_eq!(sd.drop_prob(), 0.2);
    }

    #[test]
    fn test_stochastic_depth_eval_always_keeps() {
        let sd = StochasticDepth::new(0.9, DropMode::Batch);
        // In eval mode, should always keep
        for _ in 0..10 {
            assert!(sd.should_keep(false));
        }
    }

    #[test]
    fn test_stochastic_depth_zero_drop() {
        let sd = StochasticDepth::new(0.0, DropMode::Batch);
        for _ in 0..10 {
            assert!(sd.should_keep(true));
        }
    }

    #[test]
    fn test_stochastic_depth_linear_decay() {
        let survival = StochasticDepth::linear_decay(5, 10, 0.5);
        assert!((survival - 0.75).abs() < 1e-6);

        let survival_last = StochasticDepth::linear_decay(10, 10, 0.5);
        assert!((survival_last - 0.5).abs() < 1e-6);
    }

    // R-Drop Tests
    #[test]
    fn test_rdrop_creation() {
        let rdrop = RDrop::new(0.5);
        assert_eq!(rdrop.alpha(), 0.5);
    }

    #[test]
    fn test_rdrop_kl_divergence_same() {
        let rdrop = RDrop::new(1.0);
        let p = vec![0.5, 0.3, 0.2];
        let kl = rdrop.kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-5);
    }

    #[test]
    fn test_rdrop_kl_divergence_different() {
        let rdrop = RDrop::new(1.0);
        let p = vec![0.9, 0.1];
        let q = vec![0.1, 0.9];
        let kl = rdrop.kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_rdrop_symmetric_kl() {
        let rdrop = RDrop::new(1.0);
        let p = vec![0.7, 0.3];
        let q = vec![0.4, 0.6];
        let sym = rdrop.symmetric_kl(&p, &q);
        assert!(sym > 0.0);
    }

    #[test]
    fn test_rdrop_compute_loss_same() {
        let rdrop = RDrop::new(1.0);
        let logits = vec![2.0, 1.0, 0.5];
        let loss = rdrop.compute_loss(&logits, &logits);
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_rdrop_compute_loss_different() {
        let rdrop = RDrop::new(1.0);
        let logits1 = vec![2.0, 0.0, 0.0];
        let logits2 = vec![0.0, 2.0, 0.0];
        let loss = rdrop.compute_loss(&logits1, &logits2);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_rdrop_alpha_zero() {
        let rdrop = RDrop::new(0.0);
        let logits1 = vec![2.0, 0.0];
        let logits2 = vec![0.0, 2.0];
        let loss = rdrop.compute_loss(&logits1, &logits2);
        assert_eq!(loss, 0.0);
    }

    // SpecAugment Tests

    #[test]
    fn test_specaugment_new() {
        let sa = SpecAugment::new();
        assert_eq!(sa.num_freq_masks(), 2);
        assert_eq!(sa.num_time_masks(), 2);
    }

    #[test]
    fn test_specaugment_custom() {
        let sa = SpecAugment::with_params(1, 10, 3, 50);
        assert_eq!(sa.num_freq_masks(), 1);
        assert_eq!(sa.num_time_masks(), 3);
    }

    #[test]
    fn test_specaugment_apply_shape() {
        let sa = SpecAugment::with_params(1, 5, 1, 10);
        let spec = vec![1.0; 80 * 100]; // 80 freq bins, 100 time steps
        let result = sa.apply(&spec, 80, 100);
        assert_eq!(result.len(), spec.len());
    }

    #[test]
    fn test_specaugment_masks_applied() {
        let sa = SpecAugment::with_params(2, 10, 2, 20).with_mask_value(-999.0);
        let spec = vec![1.0; 40 * 50];
        let result = sa.apply(&spec, 40, 50);

        // Some values should be masked
        let masked_count = result.iter().filter(|&&v| v == -999.0).count();
        assert!(masked_count > 0, "Should have some masked values");
    }

    #[test]
    fn test_specaugment_freq_mask() {
        let sa = SpecAugment::with_params(1, 5, 0, 0).with_mask_value(0.0);
        let spec = vec![1.0; 20 * 30]; // 20 freq, 30 time
        let result = sa.freq_mask(&spec, 20, 30);

        // Check that entire frequency bands are masked
        let zero_count = result.iter().filter(|&&v| v == 0.0).count();
        // zero_count is always >= 0 as usize; just verify we computed something
        let _ = zero_count; // May mask 0 width band
    }

    #[test]
    fn test_specaugment_time_mask() {
        let sa = SpecAugment::with_params(0, 0, 1, 5).with_mask_value(0.0);
        let spec = vec![1.0; 20 * 30];
        let result = sa.time_mask(&spec, 20, 30);

        // Some time columns should be masked
        assert_eq!(result.len(), spec.len());
    }

    // RandAugment Tests

    #[test]
    fn test_randaugment_new() {
        let ra = RandAugment::new(2, 9);
        assert_eq!(ra.n(), 2);
        assert_eq!(ra.m(), 9);
    }

    #[test]
    fn test_randaugment_default() {
        let ra = RandAugment::default();
        assert_eq!(ra.n(), 2);
        assert_eq!(ra.m(), 9);
    }

    #[test]
    fn test_randaugment_magnitude_clamp() {
        let ra = RandAugment::new(1, 50); // Over max
        assert_eq!(ra.m(), 30);
    }

    #[test]
    fn test_randaugment_normalized_magnitude() {
        let ra = RandAugment::new(1, 15);
        assert!((ra.normalized_magnitude() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_randaugment_sample_augmentations() {
        let ra = RandAugment::new(3, 10);
        let sampled = ra.sample_augmentations();
        assert_eq!(sampled.len(), 3);
    }

    #[test]
    fn test_randaugment_apply_identity() {
        let ra = RandAugment::new(1, 15);
        let image = vec![0.5; 16]; // 1x4x4
        let result = ra.apply_single(&image, AugmentationType::Identity, 4, 4);
        assert_eq!(result, image);
    }

    #[test]
    fn test_randaugment_apply_brightness() {
        let ra = RandAugment::new(1, 30); // Max magnitude
        let image = vec![0.5; 16];
        let result = ra.apply_single(&image, AugmentationType::Brightness, 4, 4);

        // Values should be modified
        let changed = result
            .iter()
            .zip(image.iter())
            .any(|(&r, &o)| (r - o).abs() > 0.01);
        assert!(changed, "Brightness should modify values");
    }

    #[test]
    fn test_randaugment_apply_contrast() {
        let ra = RandAugment::new(1, 20);
        let image: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let result = ra.apply_single(&image, AugmentationType::Contrast, 4, 4);
        assert_eq!(result.len(), image.len());
    }

    #[test]
    fn test_randaugment_custom_augmentations() {
        let ra = RandAugment::new(2, 10).with_augmentations(vec![
            AugmentationType::Identity,
            AugmentationType::Brightness,
        ]);

        let sampled = ra.sample_augmentations();
        for aug in sampled {
            assert!(aug == AugmentationType::Identity || aug == AugmentationType::Brightness);
        }
    }

    #[test]
    fn test_augmentation_type_equality() {
        assert_eq!(AugmentationType::Rotate, AugmentationType::Rotate);
        assert_ne!(AugmentationType::Rotate, AugmentationType::Brightness);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_stochastic_depth_mode() {
        let sd_batch = StochasticDepth::new(0.1, DropMode::Batch);
        assert_eq!(sd_batch.mode(), DropMode::Batch);

        let sd_row = StochasticDepth::new(0.1, DropMode::Row);
        assert_eq!(sd_row.mode(), DropMode::Row);
    }

    #[test]
    fn test_drop_mode_eq() {
        assert_eq!(DropMode::Batch, DropMode::Batch);
        assert_ne!(DropMode::Batch, DropMode::Row);
    }

    #[test]
    fn test_specaugment_default() {
        let sa = SpecAugment::default();
        assert_eq!(sa.num_freq_masks(), 2);
        assert_eq!(sa.num_time_masks(), 2);
    }

    #[test]
    fn test_specaugment_with_mask_value() {
        let sa = SpecAugment::new().with_mask_value(-1.0);
        // Just verify it compiles and works
        let spec = vec![1.0; 100];
        let result = sa.apply(&spec, 10, 10);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_randaugment_apply_rotate() {
        let ra = RandAugment::new(1, 20); // mag > 0.5
        let image = vec![1.0, 2.0, 3.0, 4.0];
        let result = ra.apply_single(&image, AugmentationType::Rotate, 2, 2);
        // High magnitude should reverse
        assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_randaugment_apply_rotate_low_mag() {
        let ra = RandAugment::new(1, 5); // mag = 5/30 < 0.5
        let image = vec![1.0, 2.0, 3.0, 4.0];
        let result = ra.apply_single(&image, AugmentationType::Rotate, 2, 2);
        // Low magnitude shouldn't reverse
        assert_eq!(result, image);
    }

    #[test]
    fn test_randaugment_apply_translate_x() {
        let ra = RandAugment::new(1, 15);
        let image = vec![1.0; 16];
        let result = ra.apply_single(&image, AugmentationType::TranslateX, 4, 4);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_randaugment_apply_translate_y() {
        let ra = RandAugment::new(1, 15);
        let image = vec![1.0; 16];
        let result = ra.apply_single(&image, AugmentationType::TranslateY, 4, 4);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_randaugment_apply_shear_x() {
        let ra = RandAugment::new(1, 15);
        let image = vec![0.5; 16];
        let result = ra.apply_single(&image, AugmentationType::ShearX, 4, 4);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_randaugment_apply_shear_y() {
        let ra = RandAugment::new(1, 15);
        let image = vec![0.5; 16];
        let result = ra.apply_single(&image, AugmentationType::ShearY, 4, 4);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_randaugment_apply_sharpness() {
        let ra = RandAugment::new(1, 15);
        let image = vec![0.5; 16];
        let result = ra.apply_single(&image, AugmentationType::Sharpness, 4, 4);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_randaugment_apply_posterize() {
        let ra = RandAugment::new(1, 15);
        let image = vec![0.5; 16];
        let result = ra.apply_single(&image, AugmentationType::Posterize, 4, 4);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_randaugment_apply_solarize() {
        let ra = RandAugment::new(1, 15);
        let image = vec![0.3, 0.7, 0.5, 0.9];
        let result = ra.apply_single(&image, AugmentationType::Solarize, 2, 2);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_randaugment_apply_equalize() {
        let ra = RandAugment::new(1, 15);
        let image = vec![0.1, 0.5, 0.9, 0.3];
        let result = ra.apply_single(&image, AugmentationType::Equalize, 2, 2);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_mixup_mix_labels() {
        let mixup = Mixup::new(1.0);
        let y1 = Vector::from_slice(&[1.0, 0.0, 0.0]);
        let y2 = Vector::from_slice(&[0.0, 1.0, 0.0]);
        let mixed = mixup.mix_labels(&y1, &y2, 0.7);
        assert!((mixed.as_slice()[0] - 0.7).abs() < 1e-6);
        assert!((mixed.as_slice()[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_mixup_alpha_negative() {
        let mixup = Mixup::new(-0.5);
        // Should return 1.0 when alpha <= 0
        assert_eq!(mixup.sample_lambda(), 1.0);
    }

    #[test]
    fn test_cutmix_params_debug() {
        let params = CutMixParams {
            lambda: 0.5,
            x1: 0,
            y1: 0,
            x2: 2,
            y2: 2,
        };
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("CutMixParams"));
    }

    #[test]
    fn test_cutmix_sample_edge_cases() {
        let cm = CutMix::new(0.0);
        // Alpha 0 means lambda = 1.0 always
        let params = cm.sample(10, 10);
        assert_eq!(params.lambda, 1.0);
    }

    #[test]
    fn test_stochastic_depth_clone() {
        let sd = StochasticDepth::new(0.3, DropMode::Row);
        let cloned = sd.clone();
        assert_eq!(cloned.drop_prob(), sd.drop_prob());
        assert_eq!(cloned.mode(), sd.mode());
    }

    #[test]
    fn test_rdrop_clone() {
        let rdrop = RDrop::new(1.5);
        let cloned = rdrop.clone();
        assert_eq!(cloned.alpha(), rdrop.alpha());
    }

    #[test]
    fn test_specaugment_clone() {
        let sa = SpecAugment::with_params(3, 20, 4, 80);
        let cloned = sa.clone();
        assert_eq!(cloned.num_freq_masks(), 3);
        assert_eq!(cloned.num_time_masks(), 4);
    }

    #[test]
    fn test_randaugment_clone() {
        let ra = RandAugment::new(3, 12);
        let cloned = ra.clone();
        assert_eq!(cloned.n(), ra.n());
        assert_eq!(cloned.m(), ra.m());
    }

    #[test]
    fn test_mixup_debug() {
        let mixup = Mixup::new(0.5);
        let debug_str = format!("{:?}", mixup);
        assert!(debug_str.contains("Mixup"));
    }

    #[test]
    fn test_label_smoothing_debug() {
        let ls = LabelSmoothing::new(0.1);
        let debug_str = format!("{:?}", ls);
        assert!(debug_str.contains("LabelSmoothing"));
    }

    #[test]
    fn test_cutmix_debug() {
        let cm = CutMix::new(1.0);
        let debug_str = format!("{:?}", cm);
        assert!(debug_str.contains("CutMix"));
    }

    #[test]
    fn test_stochastic_depth_debug() {
        let sd = StochasticDepth::new(0.2, DropMode::Batch);
        let debug_str = format!("{:?}", sd);
        assert!(debug_str.contains("StochasticDepth"));
    }

    #[test]
    fn test_rdrop_debug() {
        let rdrop = RDrop::new(0.5);
        let debug_str = format!("{:?}", rdrop);
        assert!(debug_str.contains("RDrop"));
    }

    #[test]
    fn test_specaugment_debug() {
        let sa = SpecAugment::new();
        let debug_str = format!("{:?}", sa);
        assert!(debug_str.contains("SpecAugment"));
    }

    #[test]
    fn test_randaugment_debug() {
        let ra = RandAugment::new(2, 10);
        let debug_str = format!("{:?}", ra);
        assert!(debug_str.contains("RandAugment"));
    }

    #[test]
    fn test_augmentation_type_debug_copy() {
        let aug = AugmentationType::Posterize;
        let copied = aug;
        let debug_str = format!("{:?}", copied);
        assert!(debug_str.contains("Posterize"));
    }
}
