//! K-Fold cross-validation splitter

/// K-Fold cross-validation splitter
#[derive(Clone, Debug)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    seed: u64,
}

impl KFold {
    /// Create a new KFold splitter
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits, shuffle: true, seed: 42 }
    }

    /// Set random seed for shuffling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Disable shuffling
    pub fn without_shuffle(mut self) -> Self {
        self.shuffle = false;
        self
    }

    /// Generate train/test indices for each fold
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            // Simple LCG-based shuffle for reproducibility
            let mut rng_state = self.seed;
            for i in (1..n_samples).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state >> 33) as usize % (i + 1);
                indices.swap(i, j);
            }
        }

        let fold_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        let mut folds = Vec::with_capacity(self.n_splits);
        let mut start = 0;

        for i in 0..self.n_splits {
            let extra = usize::from(i < remainder);
            let end = start + fold_size + extra;

            let test_indices: Vec<usize> = indices[start..end].to_vec();
            let train_indices: Vec<usize> =
                indices[..start].iter().chain(indices[end..].iter()).copied().collect();

            folds.push((train_indices, test_indices));
            start = end;
        }

        folds
    }
}
