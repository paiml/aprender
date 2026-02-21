#[allow(clippy::wildcard_imports)]
use super::*;

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

pub(super) fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.iter().map(|&x| x / norm).collect()
}
