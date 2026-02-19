
/// ONE PATH: Delegates to `Tensor::mul_scalar` (UCBD §4).
fn scale_tensor(x: &Tensor, scale: f32) -> Tensor {
    x.mul_scalar(scale)
}

/// ONE PATH: Delegates to `nn::transformer::transpose_last_two` (UCBD §4).
fn transpose_last_two(x: &Tensor) -> Tensor {
    crate::nn::transformer::transpose_last_two(x)
}

/// ONE PATH: Delegates to `nn::transformer::matmul_batched` (UCBD §4, SIMD-accelerated).
fn batched_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    crate::nn::transformer::matmul_batched(a, b)
}

/// ONE PATH: Delegates to `nn::functional::softmax` (UCBD §4).
fn softmax(x: &Tensor, dim: i32) -> Tensor {
    crate::nn::functional::softmax(x, dim)
}

fn dropout(x: &Tensor, p: f32) -> Tensor {
    // Dropout with scaling
    if p <= 0.0 {
        return x.clone();
    }

    let data = x.data();
    let scale = 1.0 / (1.0 - p);
    let mut output = Vec::with_capacity(data.len());

    // Simple deterministic "dropout" for reproducibility
    // In production, use proper random dropout
    for (i, &val) in data.iter().enumerate() {
        if (i % 100) as f32 / 100.0 < p {
            output.push(0.0);
        } else {
            output.push(val * scale);
        }
    }

    Tensor::new(&output, x.shape())
}

/// ONE PATH: Delegates to `nn::transformer::reshape_from_attention` (UCBD §4).
fn concat_heads(x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
    let embed_dim = x.shape()[1] * x.shape()[3]; // num_heads * head_dim
    crate::nn::transformer::reshape_from_attention(x, batch_size, seq_len, embed_dim)
}

/// ONE PATH: Delegates to `nn::functional::gelu` (UCBD §4).
fn gelu(x: &Tensor) -> Tensor {
    crate::nn::functional::gelu(x)
}

/// ONE PATH: Delegates to `nn::functional::layer_norm` (UCBD §4).
fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    crate::nn::functional::layer_norm(x, weight, bias, eps)
}

// ==================== Contrastive Loss ====================

/// `InfoNCE` contrastive loss for learning embeddings.
///
/// Given anchor, positive, and negative examples, learns embeddings
/// where similar items are close and dissimilar items are far.
///
/// # Reference
///
/// Oord, A. v. d., et al. (2018). Representation learning with contrastive predictive coding.
#[derive(Debug)]
pub struct ContrastiveLoss {
    /// Temperature parameter for softmax
    temperature: f32,
}

impl ContrastiveLoss {
    /// Create a new contrastive loss with default temperature.
    #[must_use]
    pub fn new() -> Self {
        Self { temperature: 0.07 }
    }

    /// Create with custom temperature.
    #[must_use]
    pub fn with_temperature(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Compute `InfoNCE` loss.
    ///
    /// # Arguments
    ///
    /// * `anchor` - Anchor embeddings [batch, dim]
    /// * `positive` - Positive (similar) embeddings [batch, dim]
    /// * `negatives` - Negative embeddings [batch, `num_negatives`, dim] (optional, uses in-batch)
    ///
    /// # Returns
    ///
    /// Scalar loss value.
    #[must_use]
    pub fn forward(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negatives: Option<&Tensor>,
    ) -> Tensor {
        // Compute similarity between anchor and positive
        let pos_sim = cosine_similarity_batch(anchor, positive);
        let pos_sim = div_scalar(&pos_sim, self.temperature);

        // Compute similarities with negatives
        let neg_sims = if let Some(negs) = negatives {
            // Explicit negatives provided
            let sims = cosine_similarity_many(anchor, negs);
            div_scalar(&sims, self.temperature)
        } else {
            // Use in-batch negatives (other positives become negatives)
            let all_sims = cosine_similarity_matrix(anchor, positive);
            div_scalar(&all_sims, self.temperature)
        };

        // InfoNCE loss: -log(exp(pos_sim) / sum(exp(all_sims)))
        info_nce_loss(&pos_sim, &neg_sims)
    }
}

impl Default for ContrastiveLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Triplet Loss for metric learning with margin.
///
/// Given anchor, positive (similar), and negative (dissimilar) examples,
/// minimizes: max(0, d(anchor, positive) - d(anchor, negative) + margin)
///
/// # Reference
///
/// Schroff, F., et al. (2015). `FaceNet`: A Unified Embedding for Face Recognition and Clustering.
#[derive(Debug, Clone)]
pub struct TripletLoss {
    /// Margin for the triplet loss
    margin: f32,
    /// Distance metric to use
    distance: TripletDistance,
}

/// Distance metric for triplet loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TripletDistance {
    /// Euclidean (L2) distance
    Euclidean,
    /// Squared Euclidean distance (faster, no sqrt)
    SquaredEuclidean,
    /// Cosine distance (1 - `cosine_similarity`)
    Cosine,
}
