//! Dropout regularization.
//!
//! Dropout randomly zeroes elements during training to prevent co-adaptation
//! of neurons and reduce overfitting.
//!
//! # Reference
//!
//! - Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural
//!   networks from overfitting. JMLR.

use super::module::Module;
use crate::autograd::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

/// Dropout regularization layer.
///
/// During training, randomly zeroes elements with probability `p`.
/// During evaluation, returns input unchanged.
///
/// The output is scaled by `1/(1-p)` during training to maintain
/// expected values (inverted dropout).
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Module, Dropout};
/// use aprender::autograd::Tensor;
///
/// let mut dropout = Dropout::new(0.5);
/// let x = Tensor::ones(&[10, 10]);
///
/// dropout.train();
/// let y_train = dropout.forward(&x);  // ~50% zeros, rest scaled by 2
///
/// dropout.eval();
/// let y_eval = dropout.forward(&x);   // same as input
/// ```
pub struct Dropout {
    /// Probability of element being zeroed
    p: f32,

    /// Whether in training mode
    training: bool,

    /// Random number generator (Mutex for thread safety)
    rng: Mutex<StdRng>,
}

impl Dropout {
    /// Create a new Dropout layer.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of element being zeroed (must be in [0, 1))
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in [0, 1).
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {p}",
        );

        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    /// Create a new Dropout layer with a specific seed for reproducibility.
    pub fn with_seed(p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {p}",
        );

        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Get the dropout probability.
    pub fn probability(&self) -> f32 {
        self.p
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let mut rng = self.rng.lock().expect("Dropout RNG lock poisoned");
        let scale = 1.0 / (1.0 - self.p);

        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| {
                if rng.gen::<f32>() < self.p {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect();

        Tensor::new(&data, input.shape())
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout")
            .field("p", &self.p)
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

/// 2D Dropout (Spatial Dropout).
///
/// Drops entire channels randomly during training. This is more effective
/// for convolutional networks where adjacent pixels are highly correlated.
///
/// # Shape
///
/// - Input: `(N, C, H, W)` or `(N, C, L)` for 1D
/// - Output: Same as input
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Module, Dropout2d};
/// use aprender::autograd::Tensor;
///
/// let mut dropout = Dropout2d::new(0.5);
/// let x = Tensor::ones(&[4, 64, 32, 32]);
///
/// dropout.train();
/// let y = dropout.forward(&x);  // ~50% of channels are zeroed
/// ```
///
/// # Reference
///
/// - Tompson, J., et al. (2015). Efficient Object Localization Using
///   Convolutional Networks. CVPR.
pub struct Dropout2d {
    /// Probability of channel being zeroed
    p: f32,
    /// Whether in training mode
    training: bool,
    /// Random number generator
    rng: Mutex<StdRng>,
}

impl Dropout2d {
    /// Create a new Dropout2d layer.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of channel being zeroed (must be in [0, 1))
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {p}",
        );

        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    /// Create Dropout2d with a specific seed for reproducibility.
    pub fn with_seed(p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {p}",
        );

        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Get the dropout probability.
    pub fn probability(&self) -> f32 {
        self.p
    }
}

impl Module for Dropout2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let shape = input.shape();
        assert!(
            shape.len() >= 3,
            "Dropout2d expects at least 3D input [N, C, ...], got {}D",
            shape.len()
        );

        let batch_size = shape[0];
        let num_channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        let mut rng = self.rng.lock().expect("Dropout2d RNG lock poisoned");
        let scale = 1.0 / (1.0 - self.p);

        // Generate channel masks for each sample in batch
        let mut channel_masks: Vec<bool> = Vec::with_capacity(batch_size * num_channels);
        for _ in 0..(batch_size * num_channels) {
            channel_masks.push(rng.gen::<f32>() >= self.p);
        }

        let input_data = input.data();
        let mut output = vec![0.0; input_data.len()];

        for n in 0..batch_size {
            for c in 0..num_channels {
                let keep = channel_masks[n * num_channels + c];
                for s in 0..spatial_size {
                    let idx = n * num_channels * spatial_size + c * spatial_size + s;
                    output[idx] = if keep { input_data[idx] * scale } else { 0.0 };
                }
            }
        }

        Tensor::new(&output, shape)
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Dropout2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout2d")
            .field("p", &self.p)
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

/// Alpha Dropout for SELU activations.
///
/// Maintains self-normalizing properties when used with SELU activation.
/// Randomly sets elements to negative saturation value instead of zero.
///
/// # Reference
///
/// - Klambauer, G., et al. (2017). Self-Normalizing Neural Networks. NeurIPS.
pub struct AlphaDropout {
    /// Probability of element being dropped
    p: f32,
    /// Whether in training mode
    training: bool,
    /// Random number generator
    rng: Mutex<StdRng>,
}

// SELU parameters (Klambauer et al., 2017)
const ALPHA: f32 = 1.673_263_2;
const SCALE: f32 = 1.050_701;

impl AlphaDropout {
    /// Create a new AlphaDropout layer.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {p}",
        );

        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    /// Create AlphaDropout with a specific seed.
    pub fn with_seed(p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {p}",
        );

        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Module for AlphaDropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let mut rng = self.rng.lock().expect("AlphaDropout RNG lock poisoned");

        // Compute affine transformation parameters to maintain mean and variance
        let alpha_p = -ALPHA * SCALE;
        let a = ((1.0 - self.p) * (1.0 + self.p * alpha_p.powi(2))).powf(-0.5);
        let b = -a * alpha_p * self.p;

        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| {
                if rng.gen::<f32>() < self.p {
                    a * alpha_p + b
                } else {
                    a * x + b
                }
            })
            .collect();

        Tensor::new(&data, input.shape())
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for AlphaDropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlphaDropout")
            .field("p", &self.p)
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_eval_mode() {
        let mut dropout = Dropout::new(0.5);
        dropout.eval();

        let x = Tensor::ones(&[10, 10]);
        let y = dropout.forward(&x);

        // In eval mode, output should equal input
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_dropout_train_mode_zeros() {
        let dropout = Dropout::with_seed(0.5, 42);

        let x = Tensor::ones(&[100]);
        let y = dropout.forward(&x);

        // Should have some zeros
        let num_zeros = y.data().iter().filter(|&&v| v == 0.0).count();
        assert!(num_zeros > 0, "Expected some zeros in dropout output");
        assert!(num_zeros < 100, "Expected some non-zeros in dropout output");
    }

    #[test]
    fn test_dropout_scaling() {
        let dropout = Dropout::with_seed(0.5, 42);

        let x = Tensor::ones(&[100]);
        let y = dropout.forward(&x);

        // Non-zero elements should be scaled by 2 (1 / (1 - 0.5))
        for &val in y.data() {
            assert!(val == 0.0 || (val - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dropout_zero_probability() {
        let dropout = Dropout::new(0.0);

        let x = Tensor::ones(&[10, 10]);
        let y = dropout.forward(&x);

        // With p=0, output should equal input
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_dropout_expected_value() {
        // With large samples, mean should be approximately preserved
        let dropout = Dropout::with_seed(0.3, 42);

        let x = Tensor::ones(&[10000]);
        let y = dropout.forward(&x);

        let mean: f32 = y.data().iter().sum::<f32>() / y.numel() as f32;

        // Expected value should be close to 1.0 (original value)
        assert!(
            (mean - 1.0).abs() < 0.1,
            "Mean {} should be close to 1.0",
            mean
        );
    }

    #[test]
    fn test_dropout_reproducible() {
        let dropout1 = Dropout::with_seed(0.5, 42);
        let dropout2 = Dropout::with_seed(0.5, 42);

        let x = Tensor::ones(&[100]);
        let y1 = dropout1.forward(&x);
        let y2 = dropout2.forward(&x);

        assert_eq!(y1.data(), y2.data());
    }

    #[test]
    fn test_dropout_train_eval_toggle() {
        let mut dropout = Dropout::new(0.5);

        assert!(dropout.training());

        dropout.eval();
        assert!(!dropout.training());

        dropout.train();
        assert!(dropout.training());
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn test_dropout_invalid_probability_high() {
        Dropout::new(1.0);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn test_dropout_invalid_probability_negative() {
        Dropout::new(-0.1);
    }

    // Dropout2d tests

    #[test]
    fn test_dropout2d_eval_mode() {
        let mut dropout = Dropout2d::new(0.5);
        dropout.eval();

        let x = Tensor::ones(&[4, 64, 8, 8]);
        let y = dropout.forward(&x);

        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_dropout2d_shape() {
        let dropout = Dropout2d::with_seed(0.5, 42);

        let x = Tensor::ones(&[4, 64, 8, 8]);
        let y = dropout.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_dropout2d_drops_entire_channels() {
        let dropout = Dropout2d::with_seed(0.5, 42);

        let x = Tensor::ones(&[1, 16, 4, 4]); // 1 batch, 16 channels, 4x4
        let y = dropout.forward(&x);

        // Check that entire channels are either all zeros or all scaled
        let y_data = y.data();
        for c in 0..16 {
            let channel_start = c * 16;
            let channel_end = channel_start + 16;
            let channel_data = &y_data[channel_start..channel_end];

            // Either all zeros or all ~2.0 (scaled by 1/(1-0.5))
            let first_val = channel_data[0];
            for &val in channel_data {
                assert!(
                    (val - first_val).abs() < 1e-5,
                    "Channel should have uniform values, got {} and {}",
                    first_val,
                    val
                );
            }
        }
    }

    #[test]
    fn test_dropout2d_3d_input() {
        let dropout = Dropout2d::with_seed(0.5, 42);

        let x = Tensor::ones(&[4, 64, 100]); // 3D input (e.g., for Conv1d)
        let y = dropout.forward(&x);

        assert_eq!(y.shape(), &[4, 64, 100]);
    }

    #[test]
    fn test_dropout2d_reproducible() {
        let dropout1 = Dropout2d::with_seed(0.5, 42);
        let dropout2 = Dropout2d::with_seed(0.5, 42);

        let x = Tensor::ones(&[4, 16, 8, 8]);
        let y1 = dropout1.forward(&x);
        let y2 = dropout2.forward(&x);

        assert_eq!(y1.data(), y2.data());
    }

    // AlphaDropout tests

    #[test]
    fn test_alpha_dropout_eval_mode() {
        let mut dropout = AlphaDropout::new(0.5);
        dropout.eval();

        let x = Tensor::ones(&[100]);
        let y = dropout.forward(&x);

        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_alpha_dropout_shape() {
        let dropout = AlphaDropout::with_seed(0.5, 42);

        let x = Tensor::ones(&[32, 64]);
        let y = dropout.forward(&x);

        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_alpha_dropout_not_zeros() {
        // AlphaDropout doesn't produce zeros, it uses negative saturation value
        let dropout = AlphaDropout::with_seed(0.5, 42);

        let x = Tensor::zeros(&[1000]);
        let y = dropout.forward(&x);

        // Some values should be non-zero (the dropped ones become negative saturation)
        let has_non_zero = y.data().iter().any(|&v| v != 0.0);
        assert!(
            has_non_zero,
            "AlphaDropout should produce non-zero dropped values"
        );
    }

    #[test]
    fn test_alpha_dropout_train_eval_toggle() {
        let mut dropout = AlphaDropout::new(0.5);

        assert!(dropout.training());

        dropout.eval();
        assert!(!dropout.training());

        dropout.train();
        assert!(dropout.training());
    }
}
