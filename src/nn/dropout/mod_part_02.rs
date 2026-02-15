
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
