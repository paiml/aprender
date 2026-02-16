//! Self-Supervised Learning Pretext Tasks.
//!
//! Pretext tasks generate labels from data itself for representation learning.
//!
//! # Tasks
//! - Masked Token Prediction (MLM-style)
//! - Rotation Prediction
//! - Contrastive Instance Discrimination
//! - Jigsaw Puzzle

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Masked Token Prediction task.
/// Randomly masks tokens and predicts the original values.
#[derive(Debug, Clone)]
pub struct MaskedPrediction {
    mask_prob: f32,
    mask_token_id: usize,
}

impl MaskedPrediction {
    #[must_use]
    pub fn new(mask_prob: f32, mask_token_id: usize) -> Self {
        assert!((0.0..1.0).contains(&mask_prob));
        Self {
            mask_prob,
            mask_token_id,
        }
    }

    /// Apply masking to input sequence, returns (`masked_input`, `mask_positions`).
    #[must_use]
    pub fn apply(&self, input: &[usize], seed: u64) -> (Vec<usize>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut masked = input.to_vec();
        let mut positions = Vec::new();

        for (i, _) in input.iter().enumerate() {
            if rng.gen::<f32>() < self.mask_prob {
                masked[i] = self.mask_token_id;
                positions.push(i);
            }
        }
        (masked, positions)
    }

    #[must_use]
    pub fn mask_prob(&self) -> f32 {
        self.mask_prob
    }
}

/// Rotation Prediction for images (0°, 90°, 180°, 270°).
#[derive(Debug, Clone)]
pub struct RotationPrediction;

impl RotationPrediction {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Rotate flat image `[C*H*W]` by rotation class (0-3).
    #[must_use]
    pub fn rotate(&self, image: &[f32], h: usize, w: usize, c: usize, rot: usize) -> Vec<f32> {
        let mut result = vec![0.0; image.len()];
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = ch * h * w + y * w + x;
                    let (ny, nx) = match rot % 4 {
                        0 => (y, x),
                        1 => (x, h - 1 - y),
                        2 => (h - 1 - y, w - 1 - x),
                        3 => (w - 1 - x, y),
                        _ => unreachable!(),
                    };
                    let dst_idx = ch * h * w + ny * w + nx;
                    if dst_idx < result.len() {
                        result[dst_idx] = image[src_idx];
                    }
                }
            }
        }
        result
    }

    /// Generate random rotation and return (`rotated_image`, label).
    #[must_use]
    pub fn generate_task(
        &self,
        image: &[f32],
        h: usize,
        w: usize,
        c: usize,
        seed: u64,
    ) -> (Vec<f32>, usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let rot = rng.gen_range(0..4);
        (self.rotate(image, h, w, c, rot), rot)
    }
}

impl Default for RotationPrediction {
    fn default() -> Self {
        Self::new()
    }
}

/// Jigsaw Puzzle task - shuffle patches and predict permutation.
#[derive(Debug, Clone)]
pub struct JigsawPuzzle {
    grid_size: usize,
    num_permutations: usize,
}

impl JigsawPuzzle {
    #[must_use]
    pub fn new(grid_size: usize, num_permutations: usize) -> Self {
        Self {
            grid_size,
            num_permutations,
        }
    }

    /// Shuffle patches and return (`shuffled_patches`, `permutation_idx`).
    #[must_use]
    pub fn generate_task(
        &self,
        image: &[f32],
        h: usize,
        w: usize,
        c: usize,
        seed: u64,
    ) -> (Vec<Vec<f32>>, usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let ph = h / self.grid_size;
        let pw = w / self.grid_size;

        // Extract patches
        let mut patches = Vec::new();
        for gy in 0..self.grid_size {
            for gx in 0..self.grid_size {
                let mut patch = vec![0.0; c * ph * pw];
                for ch in 0..c {
                    for py in 0..ph {
                        for px in 0..pw {
                            let src_y = gy * ph + py;
                            let src_x = gx * pw + px;
                            let src = ch * h * w + src_y * w + src_x;
                            let dst = ch * ph * pw + py * pw + px;
                            if src < image.len() && dst < patch.len() {
                                patch[dst] = image[src];
                            }
                        }
                    }
                }
                patches.push(patch);
            }
        }

        // Shuffle
        let perm_idx = rng.gen_range(0..self.num_permutations);
        for i in (1..patches.len()).rev() {
            let j = rng.gen_range(0..=i);
            patches.swap(i, j);
        }

        (patches, perm_idx)
    }

    #[must_use]
    pub fn grid_size(&self) -> usize {
        self.grid_size
    }
}

/// Contrastive Instance Discrimination.
#[derive(Debug, Clone)]
pub struct ContrastiveTask {
    temperature: f32,
}

impl ContrastiveTask {
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Compute `InfoNCE` loss for positive pair against negatives.
    #[must_use]
    pub fn info_nce_loss(&self, anchor: &[f32], positive: &[f32], negatives: &[Vec<f32>]) -> f32 {
        let pos_sim = cosine_sim(anchor, positive) / self.temperature;

        let mut neg_sims: Vec<f32> = negatives
            .iter()
            .map(|n| cosine_sim(anchor, n) / self.temperature)
            .collect();
        neg_sims.push(pos_sim);

        let max_sim = neg_sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let log_sum_exp: f32 = neg_sims
            .iter()
            .map(|&s| (s - max_sim).exp())
            .sum::<f32>()
            .ln()
            + max_sim;

        -pos_sim + log_sum_exp
    }

    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

/// `SimCLR` framework (Chen et al., 2020).
///
/// Simple framework for contrastive learning of visual representations:
/// 1. Apply two random augmentations to create positive pairs
/// 2. Use NT-Xent loss (normalized temperature-scaled cross-entropy)
///
/// # Reference
///
/// Chen, T., et al. (2020). A Simple Framework for Contrastive Learning
/// of Visual Representations. ICML.
#[derive(Debug, Clone)]
pub struct SimCLR {
    /// Temperature for NT-Xent loss
    temperature: f32,
    /// Projection dimension
    projection_dim: usize,
}

impl SimCLR {
    /// Create `SimCLR` framework.
    #[must_use]
    pub fn new(temperature: f32, projection_dim: usize) -> Self {
        Self {
            temperature,
            projection_dim,
        }
    }

    /// Compute NT-Xent loss for a batch of positive pairs.
    ///
    /// # Arguments
    ///
    /// * `z_i` - First augmented view embeddings `[batch_size, dim]`
    /// * `z_j` - Second augmented view embeddings `[batch_size, dim]`
    ///
    /// For each sample i, its positive pair is `z_j[i]` and negatives are
    /// all other samples in the batch.
    #[must_use]
    pub fn nt_xent_loss(&self, z_i: &[Vec<f32>], z_j: &[Vec<f32>]) -> f32 {
        let batch_size = z_i.len();
        assert_eq!(batch_size, z_j.len());
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            // Positive similarity: z_i[i] and z_j[i]
            let pos_sim = cosine_sim(&z_i[i], &z_j[i]) / self.temperature;

            // Negative similarities: z_i[i] with all z_j[k] where k != i
            // and z_i[i] with all z_i[k] where k != i
            let mut sims = vec![pos_sim];
            for k in 0..batch_size {
                if k != i {
                    sims.push(cosine_sim(&z_i[i], &z_j[k]) / self.temperature);
                    sims.push(cosine_sim(&z_i[i], &z_i[k]) / self.temperature);
                }
            }

            // Log-sum-exp trick for numerical stability
            let max_sim = sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let log_sum_exp: f32 =
                sims.iter().map(|&s| (s - max_sim).exp()).sum::<f32>().ln() + max_sim;

            total_loss += -pos_sim + log_sum_exp;
        }

        // Also compute loss for z_j as anchor
        for j in 0..batch_size {
            let pos_sim = cosine_sim(&z_j[j], &z_i[j]) / self.temperature;

            let mut sims = vec![pos_sim];
            for k in 0..batch_size {
                if k != j {
                    sims.push(cosine_sim(&z_j[j], &z_i[k]) / self.temperature);
                    sims.push(cosine_sim(&z_j[j], &z_j[k]) / self.temperature);
                }
            }

            let max_sim = sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let log_sum_exp: f32 =
                sims.iter().map(|&s| (s - max_sim).exp()).sum::<f32>().ln() + max_sim;

            total_loss += -pos_sim + log_sum_exp;
        }

        total_loss / (2.0 * batch_size as f32)
    }

    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    #[must_use]
    pub fn projection_dim(&self) -> usize {
        self.projection_dim
    }
}

/// `MoCo` (Momentum Contrast) framework (He et al., 2020).
///
/// Uses a momentum encoder and a queue of negative samples:
/// - Slowly-evolving momentum encoder provides consistent targets
/// - Queue allows large number of negatives without large batch size
///
/// # Reference
///
/// He, K., et al. (2020). Momentum Contrast for Unsupervised Visual
/// Representation Learning. CVPR.
#[derive(Debug, Clone)]
pub struct MoCo {
    /// Temperature for `InfoNCE`
    temperature: f32,
    /// Momentum coefficient (0.999 typical)
    momentum: f32,
    /// Queue size
    queue_size: usize,
    /// Embedding dimension
    #[allow(dead_code)]
    dim: usize,
    /// Queue of negative keys
    queue: Vec<Vec<f32>>,
    /// Queue pointer
    queue_ptr: usize,
}

impl MoCo {
    /// Create `MoCo` framework.
    #[must_use]
    pub fn new(temperature: f32, momentum: f32, queue_size: usize, dim: usize) -> Self {
        Self {
            temperature,
            momentum,
            queue_size,
            dim,
            queue: Vec::with_capacity(queue_size),
            queue_ptr: 0,
        }
    }

    /// Update momentum encoder parameters.
    ///
    /// ```text
    /// θ_k = m * θ_k + (1 - m) * θ_q
    /// ```
    pub fn momentum_update(&self, encoder_params: &[f32], momentum_params: &mut [f32]) {
        assert_eq!(encoder_params.len(), momentum_params.len());
        for (m, e) in momentum_params.iter_mut().zip(encoder_params) {
            *m = self.momentum * *m + (1.0 - self.momentum) * e;
        }
    }

    /// Enqueue new keys and dequeue oldest.
    pub fn update_queue(&mut self, keys: &[Vec<f32>]) {
        for key in keys {
            if self.queue.len() < self.queue_size {
                self.queue.push(key.clone());
            } else {
                self.queue[self.queue_ptr].clone_from(key);
                self.queue_ptr = (self.queue_ptr + 1) % self.queue_size;
            }
        }
    }

    /// Compute contrastive loss.
    ///
    /// # Arguments
    ///
    /// * `queries` - Query embeddings from encoder
    /// * `keys` - Key embeddings from momentum encoder (positive)
    #[must_use]
    pub fn contrastive_loss(&self, queries: &[Vec<f32>], keys: &[Vec<f32>]) -> f32 {
        let batch_size = queries.len();
        if batch_size == 0 || self.queue.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let pos_sim = cosine_sim(&queries[i], &keys[i]) / self.temperature;

            // Negatives from queue
            let mut sims = vec![pos_sim];
            for neg in &self.queue {
                sims.push(cosine_sim(&queries[i], neg) / self.temperature);
            }

            let max_sim = sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let log_sum_exp: f32 =
                sims.iter().map(|&s| (s - max_sim).exp()).sum::<f32>().ln() + max_sim;

            total_loss += -pos_sim + log_sum_exp;
        }

        total_loss / batch_size as f32
    }

    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    #[must_use]
    pub fn momentum(&self) -> f32 {
        self.momentum
    }
}

#[path = "self_supervised_part_02.rs"]
mod self_supervised_part_02;
pub use self_supervised_part_02::*;

#[path = "self_supervised_part_03.rs"]
mod self_supervised_part_03;
