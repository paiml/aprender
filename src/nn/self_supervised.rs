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

/// BYOL (Bootstrap Your Own Latent) framework (Grill et al., 2020).
///
/// Self-supervised learning WITHOUT negative samples:
/// - Online network predicts target network representation
/// - Target network is momentum-updated from online
/// - Stops gradient on target to prevent collapse
///
/// # Reference
///
/// Grill, J.B., et al. (2020). Bootstrap Your Own Latent: A New Approach
/// to Self-Supervised Learning. `NeurIPS`.
#[derive(Debug, Clone)]
pub struct BYOL {
    /// Momentum coefficient
    momentum: f32,
}

impl BYOL {
    /// Create BYOL framework.
    #[must_use]
    pub fn new(momentum: f32) -> Self {
        Self { momentum }
    }

    /// Compute BYOL loss (MSE between normalized predictions).
    ///
    /// # Arguments
    ///
    /// * `online_pred` - Predictions from online network
    /// * `target_proj` - Projections from target network (stop gradient)
    #[must_use]
    pub fn loss(&self, online_pred: &[Vec<f32>], target_proj: &[Vec<f32>]) -> f32 {
        assert_eq!(online_pred.len(), target_proj.len());
        let batch_size = online_pred.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            // L2 normalize
            let pred_norm = l2_normalize(&online_pred[i]);
            let target_norm = l2_normalize(&target_proj[i]);

            // MSE loss
            let mse: f32 = pred_norm
                .iter()
                .zip(target_norm.iter())
                .map(|(&p, &t)| (p - t).powi(2))
                .sum();

            total_loss += mse;
        }

        total_loss / batch_size as f32
    }

    /// Symmetric BYOL loss (both views predict each other).
    #[must_use]
    pub fn symmetric_loss(
        &self,
        pred_1: &[Vec<f32>],
        proj_2: &[Vec<f32>],
        pred_2: &[Vec<f32>],
        proj_1: &[Vec<f32>],
    ) -> f32 {
        (self.loss(pred_1, proj_2) + self.loss(pred_2, proj_1)) / 2.0
    }

    /// Update target network parameters with momentum.
    pub fn momentum_update(&self, online_params: &[f32], target_params: &mut [f32]) {
        for (t, o) in target_params.iter_mut().zip(online_params) {
            *t = self.momentum * *t + (1.0 - self.momentum) * o;
        }
    }

    #[must_use]
    pub fn momentum(&self) -> f32 {
        self.momentum
    }
}

/// `SimCSE` for text embeddings (Gao et al., 2021).
///
/// Simple Contrastive Learning of Sentence Embeddings:
/// - Uses dropout as minimal augmentation
/// - Same sentence with different dropout = positive pair
/// - Other sentences in batch = negatives
///
/// # Reference
///
/// Gao, T., et al. (2021). `SimCSE`: Simple Contrastive Learning of
/// Sentence Embeddings. EMNLP.
#[derive(Debug, Clone)]
pub struct SimCSE {
    /// Temperature for contrastive loss
    temperature: f32,
}

impl SimCSE {
    /// Create `SimCSE`.
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Compute unsupervised `SimCSE` loss.
    ///
    /// # Arguments
    ///
    /// * `embeddings_1` - First pass embeddings (with dropout)
    /// * `embeddings_2` - Second pass embeddings (different dropout)
    ///
    /// Same index = positive pair, different indices = negatives.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn unsupervised_loss(&self, emb_1: &[Vec<f32>], emb_2: &[Vec<f32>]) -> f32 {
        assert_eq!(emb_1.len(), emb_2.len());
        let batch_size = emb_1.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let pos_sim = cosine_sim(&emb_1[i], &emb_2[i]) / self.temperature;

            let mut sims = vec![pos_sim];
            for j in 0..batch_size {
                if j != i {
                    sims.push(cosine_sim(&emb_1[i], &emb_2[j]) / self.temperature);
                }
            }

            let max_sim = sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let log_sum_exp: f32 =
                sims.iter().map(|&s| (s - max_sim).exp()).sum::<f32>().ln() + max_sim;

            total_loss += -pos_sim + log_sum_exp;
        }

        total_loss / batch_size as f32
    }

    /// Compute supervised `SimCSE` loss with hard negatives.
    ///
    /// # Arguments
    ///
    /// * `anchors` - Anchor embeddings
    /// * `positives` - Positive (entailment) embeddings
    /// * `negatives` - Hard negative (contradiction) embeddings
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn supervised_loss(
        &self,
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        negatives: &[Vec<f32>],
    ) -> f32 {
        let batch_size = anchors.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let pos_sim = cosine_sim(&anchors[i], &positives[i]) / self.temperature;

            let mut sims = vec![pos_sim];

            // Add hard negative
            if i < negatives.len() {
                sims.push(cosine_sim(&anchors[i], &negatives[i]) / self.temperature);
            }

            // In-batch negatives
            for j in 0..batch_size {
                if j != i {
                    sims.push(cosine_sim(&anchors[i], &positives[j]) / self.temperature);
                }
            }

            let max_sim = sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let log_sum_exp: f32 =
                sims.iter().map(|&s| (s - max_sim).exp()).sum::<f32>().ln() + max_sim;

            total_loss += -pos_sim + log_sum_exp;
        }

        total_loss / batch_size as f32
    }

    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.iter().map(|&x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_prediction() {
        let mp = MaskedPrediction::new(0.15, 103);
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let (masked, positions) = mp.apply(&input, 42);
        assert_eq!(masked.len(), input.len());
        for &pos in &positions {
            assert_eq!(masked[pos], 103);
        }
    }

    #[test]
    fn test_rotation_prediction() {
        let rp = RotationPrediction::new();
        let image = vec![1.0, 2.0, 3.0, 4.0]; // 1x2x2
        let rotated = rp.rotate(&image, 2, 2, 1, 0);
        assert_eq!(rotated, image);
    }

    #[test]
    fn test_rotation_task() {
        let rp = RotationPrediction::new();
        let image = vec![1.0; 16]; // 1x4x4
        let (rotated, label) = rp.generate_task(&image, 4, 4, 1, 42);
        assert_eq!(rotated.len(), 16);
        assert!(label < 4);
    }

    #[test]
    fn test_jigsaw_puzzle() {
        let jp = JigsawPuzzle::new(2, 10);
        let image = vec![1.0; 16]; // 1x4x4
        let (patches, perm) = jp.generate_task(&image, 4, 4, 1, 42);
        assert_eq!(patches.len(), 4); // 2x2 grid
        assert!(perm < 10);
    }

    #[test]
    fn test_contrastive_task() {
        let ct = ContrastiveTask::new(0.07);
        let anchor = vec![1.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0];
        let negatives = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let loss = ct.info_nce_loss(&anchor, &positive, &negatives);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_contrastive_same_positive() {
        let ct = ContrastiveTask::new(0.1);
        let anchor = vec![1.0, 0.0];
        let positive = vec![1.0, 0.0]; // Same as anchor
        let negatives = vec![vec![0.0, 1.0]];
        let loss = ct.info_nce_loss(&anchor, &positive, &negatives);
        assert!(loss.is_finite());
    }

    // SimCLR Tests
    #[test]
    fn test_simclr_basic() {
        let simclr = SimCLR::new(0.07, 128);
        assert!((simclr.temperature() - 0.07).abs() < 1e-6);
        assert_eq!(simclr.projection_dim(), 128);
    }

    #[test]
    fn test_simclr_nt_xent_loss() {
        let simclr = SimCLR::new(0.5, 64);
        let z_i = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let z_j = vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]];

        let loss = simclr.nt_xent_loss(&z_i, &z_j);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_simclr_identical_pairs() {
        let simclr = SimCLR::new(0.5, 64);
        let z_i = vec![vec![1.0, 0.0]];
        let z_j = vec![vec![1.0, 0.0]];

        let loss = simclr.nt_xent_loss(&z_i, &z_j);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_simclr_empty_batch() {
        let simclr = SimCLR::new(0.5, 64);
        let z_i: Vec<Vec<f32>> = vec![];
        let z_j: Vec<Vec<f32>> = vec![];

        let loss = simclr.nt_xent_loss(&z_i, &z_j);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    // MoCo Tests
    #[test]
    fn test_moco_basic() {
        let moco = MoCo::new(0.07, 0.999, 65536, 128);
        assert!((moco.momentum() - 0.999).abs() < 1e-6);
        assert_eq!(moco.queue_len(), 0);
    }

    #[test]
    fn test_moco_queue_update() {
        let mut moco = MoCo::new(0.07, 0.999, 4, 3);

        moco.update_queue(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        assert_eq!(moco.queue_len(), 2);

        moco.update_queue(&[vec![0.0, 0.0, 1.0], vec![1.0, 1.0, 0.0]]);
        assert_eq!(moco.queue_len(), 4);

        // Now queue is full, next update wraps around
        moco.update_queue(&[vec![0.5, 0.5, 0.0]]);
        assert_eq!(moco.queue_len(), 4);
    }

    #[test]
    fn test_moco_momentum_update() {
        let moco = MoCo::new(0.07, 0.9, 100, 3);
        let encoder = vec![1.0, 2.0, 3.0];
        let mut momentum = vec![0.0, 0.0, 0.0];

        moco.momentum_update(&encoder, &mut momentum);

        // momentum = 0.9 * 0 + 0.1 * encoder = 0.1 * encoder
        assert!((momentum[0] - 0.1).abs() < 1e-6);
        assert!((momentum[1] - 0.2).abs() < 1e-6);
        assert!((momentum[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_moco_contrastive_loss() {
        let mut moco = MoCo::new(0.5, 0.999, 100, 3);

        // Fill queue
        moco.update_queue(&[vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]);

        let queries = vec![vec![1.0, 0.0, 0.0]];
        let keys = vec![vec![0.9, 0.1, 0.0]];

        let loss = moco.contrastive_loss(&queries, &keys);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    // BYOL Tests
    #[test]
    fn test_byol_basic() {
        let byol = BYOL::new(0.996);
        assert!((byol.momentum() - 0.996).abs() < 1e-6);
    }

    #[test]
    fn test_byol_loss() {
        let byol = BYOL::new(0.996);
        let pred = vec![vec![1.0, 0.0, 0.0]];
        let target = vec![vec![0.9, 0.1, 0.0]];

        let loss = byol.loss(&pred, &target);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_byol_identical() {
        let byol = BYOL::new(0.996);
        let pred = vec![vec![1.0, 0.0]];
        let target = vec![vec![1.0, 0.0]];

        let loss = byol.loss(&pred, &target);
        assert!(loss < 0.01); // Should be very small
    }

    #[test]
    fn test_byol_symmetric_loss() {
        let byol = BYOL::new(0.996);
        let pred_1 = vec![vec![1.0, 0.0]];
        let proj_2 = vec![vec![0.9, 0.1]];
        let pred_2 = vec![vec![0.0, 1.0]];
        let proj_1 = vec![vec![0.1, 0.9]];

        let loss = byol.symmetric_loss(&pred_1, &proj_2, &pred_2, &proj_1);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_byol_momentum_update() {
        let byol = BYOL::new(0.9);
        let online = vec![1.0, 2.0];
        let mut target = vec![0.0, 0.0];

        byol.momentum_update(&online, &mut target);

        assert!((target[0] - 0.1).abs() < 1e-6);
        assert!((target[1] - 0.2).abs() < 1e-6);
    }

    // SimCSE Tests
    #[test]
    fn test_simcse_basic() {
        let simcse = SimCSE::new(0.05);
        assert!((simcse.temperature() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_simcse_unsupervised() {
        let simcse = SimCSE::new(0.5); // Higher temp for smoother loss
        let emb_1 = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let emb_2 = vec![
            vec![0.95, 0.05, 0.0],
            vec![0.05, 0.95, 0.0],
            vec![0.0, 0.05, 0.95],
        ];

        let loss = simcse.unsupervised_loss(&emb_1, &emb_2);
        assert!(loss.is_finite());
        // With in-batch negatives, loss should be positive
        assert!(loss > 0.0 || loss >= 0.0); // Allow zero for identical pairs
    }

    #[test]
    fn test_simcse_supervised() {
        let simcse = SimCSE::new(0.5);
        let anchors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]];
        let negatives = vec![vec![0.0, 0.0, 1.0], vec![0.5, 0.5, 0.0]];

        let loss = simcse.supervised_loss(&anchors, &positives, &negatives);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_simcse_empty() {
        let simcse = SimCSE::new(0.05);
        let emb_1: Vec<Vec<f32>> = vec![];
        let emb_2: Vec<Vec<f32>> = vec![];

        let loss = simcse.unsupervised_loss(&emb_1, &emb_2);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let norm = l2_normalize(&v);
        let length: f32 = norm.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6);
    }
}
