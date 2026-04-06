#[allow(clippy::wildcard_imports)]
use super::*;

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
            rng: Mutex::new(StdRng::from_os_rng()),
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

    /// Create a block dropout mask by sampling block centers and zeroing rectangular regions.
    fn create_block_mask(
        rng: &mut impl rand::Rng,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        block_size: usize,
        gamma: f32,
    ) -> Vec<f32> {
        let mut mask = vec![1.0_f32; n * c * h * w];
        let plane_size = h * w;
        for batch in 0..n {
            for ch in 0..c {
                let base = batch * c * plane_size + ch * plane_size;
                Self::sample_block_centers(rng, &mut mask, base, h, w, block_size, gamma);
            }
        }
        mask
    }

    /// Sample block centers for one spatial plane and zero the block regions.
    fn sample_block_centers(
        rng: &mut impl rand::Rng,
        mask: &mut [f32],
        base: usize,
        h: usize,
        w: usize,
        block_size: usize,
        gamma: f32,
    ) {
        for i in 0..=(h - block_size) {
            for j in 0..=(w - block_size) {
                if rng.random::<f32>() < gamma {
                    for bi in 0..block_size {
                        for bj in 0..block_size {
                            mask[base + (i + bi) * w + (j + bj)] = 0.0;
                        }
                    }
                }
            }
        }
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
            return apply_dropout(input, self.p, &self.rng);
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let block_size = self.block_size.min(h).min(w);
        let gamma = self.p / (block_size * block_size) as f32 * (h * w) as f32
            / ((h - block_size + 1) * (w - block_size + 1)) as f32;

        let mut rng = self.rng.lock().expect("DropBlock RNG lock");
        let mask = Self::create_block_mask(&mut rng, n, c, h, w, block_size, gamma);

        let kept: f32 = mask.iter().sum();
        let norm = (mask.len() as f32) / kept.max(1.0);

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
            rng: Mutex::new(StdRng::from_os_rng()),
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
                if rng.random::<f32>() < self.p {
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
                if rng.random::<f32>() < self.p {
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
        .map(|&x| {
            if rng.random::<f32>() < p {
                0.0
            } else {
                x * scale
            }
        })
        .collect();
    Tensor::new(&data, input.shape())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_dropout_contract.rs"]
mod tests_dropout_contract;
