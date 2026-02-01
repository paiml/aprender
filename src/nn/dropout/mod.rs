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
    #[must_use]
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
    #[must_use]
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
    /// Forward pass with dropout.
    ///
    /// # Panics
    /// Panics if the RNG mutex is poisoned (indicates prior thread panic - unrecoverable).
    /// This is acceptable per Toyota Way: mutex poisoning means the system is already
    /// in a catastrophic state from a prior panic.
    #[allow(clippy::expect_used)]
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
    #[must_use]
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
    #[must_use]
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
    /// Forward pass with 2D spatial dropout.
    ///
    /// # Panics
    /// Panics if the RNG mutex is poisoned (unrecoverable system state).
    #[allow(clippy::expect_used)]
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
/// - Klambauer, G., et al. (2017). Self-Normalizing Neural Networks. `NeurIPS`.
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
    /// Create a new `AlphaDropout` layer.
    #[must_use]
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

    /// Create `AlphaDropout` with a specific seed.
    #[must_use]
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
    /// Forward pass with alpha dropout (SELU-preserving).
    ///
    /// # Panics
    /// Panics if the RNG mutex is poisoned (unrecoverable system state).
    #[allow(clippy::expect_used)]
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

/// `DropBlock` regularization (Ghiasi et al., 2018).
///
/// Drops contiguous regions (blocks) rather than individual elements.
/// More effective for convolutional networks than standard dropout.
///
/// # Reference
/// Ghiasi, G., et al. (2018). `DropBlock`: A regularization technique for CNNs.
pub struct DropBlock {
    block_size: usize,
    p: f32,
    training: bool,
    rng: Mutex<StdRng>,
}

impl DropBlock {
    /// Create `DropBlock` with given block size and drop probability.
    #[must_use]
    pub fn new(block_size: usize, p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Drop probability must be in [0, 1)"
        );
        assert!(block_size > 0, "Block size must be > 0");
        Self {
            block_size,
            p,
            training: true,
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    /// Create with specific seed.
    #[must_use]
    pub fn with_seed(block_size: usize, p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Drop probability must be in [0, 1)"
        );
        Self {
            block_size,
            p,
            training: true,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn p(&self) -> f32 {
        self.p
    }
}

impl Module for DropBlock {
    /// Forward pass with block dropout.
    ///
    /// # Panics
    /// Panics if the RNG mutex is poisoned (unrecoverable system state).
    #[allow(clippy::range_plus_one, clippy::expect_used)]
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let shape = input.shape();
        if shape.len() != 4 {
            // Not 4D (N, C, H, W), fall back to regular dropout
            return apply_dropout(input, self.p, &self.rng);
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let block_size = self.block_size.min(h).min(w);

        // Compute gamma (probability of dropping a center point)
        let gamma = self.p / (block_size * block_size) as f32 * (h * w) as f32
            / ((h - block_size + 1) * (w - block_size + 1)) as f32;

        let mut rng = self.rng.lock().expect("DropBlock RNG lock");
        let mut mask = vec![1.0_f32; n * c * h * w];

        // Sample block centers and create mask
        for batch in 0..n {
            for ch in 0..c {
                for i in 0..(h - block_size + 1) {
                    for j in 0..(w - block_size + 1) {
                        if rng.gen::<f32>() < gamma {
                            // Drop block
                            for bi in 0..block_size {
                                for bj in 0..block_size {
                                    let idx =
                                        batch * c * h * w + ch * h * w + (i + bi) * w + (j + bj);
                                    mask[idx] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize by keep ratio
        let kept: f32 = mask.iter().sum();
        let total = mask.len() as f32;
        let norm = total / kept.max(1.0);

        let data: Vec<f32> = input
            .data()
            .iter()
            .zip(mask.iter())
            .map(|(&x, &m)| x * m * norm)
            .collect();

        Tensor::new(&data, shape)
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

impl std::fmt::Debug for DropBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DropBlock")
            .field("block_size", &self.block_size)
            .field("p", &self.p)
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

/// `DropConnect` regularization (Wan et al., 2013).
///
/// Drops weights instead of activations. Each weight has probability p
/// of being set to zero during training.
///
/// More general than Dropout - Dropout is `DropConnect` with identity weight matrix.
///
/// # Reference
/// Wan, L., et al. (2013). Regularization of Neural Networks using `DropConnect`. ICML.
pub struct DropConnect {
    /// Probability of weight being zeroed
    p: f32,
    /// Whether in training mode
    training: bool,
    /// Random number generator
    rng: Mutex<StdRng>,
}

impl DropConnect {
    /// Create new `DropConnect` with drop probability.
    #[must_use]
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Drop probability must be in [0, 1), got {p}"
        );
        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::from_entropy()),
        }
    }

    /// Create `DropConnect` with specific seed.
    #[must_use]
    pub fn with_seed(p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Drop probability must be in [0, 1), got {p}"
        );
        Self {
            p,
            training: true,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn probability(&self) -> f32 {
        self.p
    }

    /// Apply `DropConnect` to weight matrix.
    /// Returns masked weights (zeros some weights during training).
    ///
    /// # Panics
    /// Panics if the RNG mutex is poisoned (unrecoverable system state).
    #[allow(clippy::expect_used)]
    pub fn apply_to_weights(&self, weights: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return weights.clone();
        }

        let mut rng = self.rng.lock().expect("DropConnect RNG lock");
        let scale = 1.0 / (1.0 - self.p);

        let data: Vec<f32> = weights
            .data()
            .iter()
            .map(|&w| {
                if rng.gen::<f32>() < self.p {
                    0.0
                } else {
                    w * scale
                }
            })
            .collect();

        Tensor::new(&data, weights.shape())
    }
}

impl Module for DropConnect {
    /// Forward pass with connection dropout.
    ///
    /// # Panics
    /// Panics if the RNG mutex is poisoned (unrecoverable system state).
    #[allow(clippy::expect_used)]
    fn forward(&self, input: &Tensor) -> Tensor {
        // DropConnect typically applied to weights, but for Module interface
        // we apply element-wise like dropout (for flexibility)
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let mut rng = self.rng.lock().expect("DropConnect RNG lock");
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

impl std::fmt::Debug for DropConnect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DropConnect")
            .field("p", &self.p)
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

/// Apply dropout to a tensor with the given probability.
///
/// # Panics
/// Panics if the RNG mutex is poisoned (unrecoverable system state).
#[allow(clippy::expect_used)]
fn apply_dropout(input: &Tensor, p: f32, rng: &Mutex<StdRng>) -> Tensor {
    let mut rng = rng.lock().expect("RNG lock");
    let scale = 1.0 / (1.0 - p);
    let data: Vec<f32> = input
        .data()
        .iter()
        .map(|&x| if rng.gen::<f32>() < p { 0.0 } else { x * scale })
        .collect();
    Tensor::new(&data, input.shape())
}

#[cfg(test)]
mod tests;
