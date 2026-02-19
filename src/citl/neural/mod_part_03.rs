
fn scale_tensor(x: &Tensor, scale: f32) -> Tensor {
    let data = x.data();
    let scaled: Vec<f32> = data.iter().map(|&v| v * scale).collect();
    Tensor::new(&scaled, x.shape())
}

fn transpose_last_two(x: &Tensor) -> Tensor {
    // Transpose last two dimensions: [..., A, B] -> [..., B, A]
    let shape = x.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return x.clone();
    }

    let a = shape[ndim - 2];
    let b = shape[ndim - 1];
    let batch_dims: usize = shape[..ndim - 2].iter().product();

    let data = x.data();
    let mut output = vec![0.0f32; data.len()];

    for batch in 0..batch_dims {
        let offset = batch * a * b;
        for i in 0..a {
            for j in 0..b {
                output[offset + j * a + i] = data[offset + i * b + j];
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = b;
    new_shape[ndim - 1] = a;

    Tensor::new(&output, &new_shape)
}

fn batched_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    // Batched matrix multiplication: [..., M, K] @ [..., K, N] -> [..., M, N]
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];
    let batch_dims: usize = a_shape[..a_shape.len() - 2].iter().product();

    let a_data = a.data();
    let b_data = b.data();
    let mut output = vec![0.0f32; batch_dims * m * n];

    for batch in 0..batch_dims {
        let a_offset = batch * m * k;
        let b_offset = batch * k * n;
        let out_offset = batch * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_data[a_offset + i * k + l] * b_data[b_offset + l * n + j];
                }
                output[out_offset + i * n + j] = sum;
            }
        }
    }

    let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);

    Tensor::new(&output, &out_shape)
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

fn concat_heads(x: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
    // Concatenate attention heads: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim]
    let shape = x.shape();
    let num_heads = shape[1];
    let head_dim = shape[3];
    let embed_dim = num_heads * head_dim;

    let data = x.data();
    let mut output = Vec::with_capacity(batch_size * seq_len * embed_dim);

    for b in 0..batch_size {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    output.push(data[idx]);
                }
            }
        }
    }

    Tensor::new(&output, &[batch_size, seq_len, embed_dim])
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
