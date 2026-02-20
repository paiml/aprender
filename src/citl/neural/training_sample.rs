
impl TripletLoss {
    /// Create a new triplet loss with default margin (1.0) and Euclidean distance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            margin: 1.0,
            distance: TripletDistance::Euclidean,
        }
    }

    /// Create triplet loss with custom margin.
    #[must_use]
    pub fn with_margin(margin: f32) -> Self {
        Self {
            margin,
            distance: TripletDistance::Euclidean,
        }
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_distance(mut self, distance: TripletDistance) -> Self {
        self.distance = distance;
        self
    }

    /// Get the margin value.
    #[must_use]
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// Get the distance metric.
    #[must_use]
    pub fn distance_metric(&self) -> TripletDistance {
        self.distance
    }

    /// Compute triplet loss for a batch.
    ///
    /// # Arguments
    ///
    /// * `anchor` - Anchor embeddings [batch, dim]
    /// * `positive` - Positive (similar) embeddings [batch, dim]
    /// * `negative` - Negative (dissimilar) embeddings [batch, dim]
    ///
    /// # Returns
    ///
    /// Mean triplet loss over the batch.
    #[must_use]
    pub fn forward(&self, anchor: &Tensor, positive: &Tensor, negative: &Tensor) -> Tensor {
        let batch_size = anchor.shape()[0];
        let dim = anchor.shape()[1];

        let anchor_data = anchor.data();
        let positive_data = positive.data();
        let negative_data = negative.data();

        let mut total_loss = 0.0f32;

        for i in 0..batch_size {
            let a_slice = &anchor_data[i * dim..(i + 1) * dim];
            let p_slice = &positive_data[i * dim..(i + 1) * dim];
            let n_slice = &negative_data[i * dim..(i + 1) * dim];

            let d_ap = self.compute_distance(a_slice, p_slice);
            let d_an = self.compute_distance(a_slice, n_slice);

            // Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            let loss = (d_ap - d_an + self.margin).max(0.0);
            total_loss += loss;
        }

        Tensor::new(&[total_loss / batch_size as f32], &[1])
    }

    /// Compute distance between two vectors based on the distance metric.
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.distance {
            TripletDistance::Euclidean => {
                let va = Vector::from_slice(a);
                let vb = Vector::from_slice(b);
                va.sub(&vb).and_then(|diff| diff.norm_l2()).unwrap_or(0.0)
            }
            TripletDistance::SquaredEuclidean => {
                let va = Vector::from_slice(a);
                let vb = Vector::from_slice(b);
                va.sub(&vb).and_then(|diff| diff.dot(&diff)).unwrap_or(0.0)
            }
            TripletDistance::Cosine => {
                let va = Vector::from_slice(a);
                let vb = Vector::from_slice(b);
                let dot = va.dot(&vb).unwrap_or(0.0);
                let norm_a = va.norm_l2().unwrap_or(1.0);
                let norm_b = vb.norm_l2().unwrap_or(1.0);
                let cosine = dot / (norm_a * norm_b + 1e-8);
                1.0 - cosine // Cosine distance
            }
        }
    }

    /// Compute pairwise distances for hard negative mining.
    ///
    /// Returns a matrix of shape [batch, batch] where entry (i, j) is the
    /// distance between embedding i and embedding j.
    #[must_use]
    pub fn pairwise_distances(&self, embeddings: &Tensor) -> Tensor {
        let batch_size = embeddings.shape()[0];
        let dim = embeddings.shape()[1];
        let data = embeddings.data();

        let mut distances = Vec::with_capacity(batch_size * batch_size);

        for i in 0..batch_size {
            let a = &data[i * dim..(i + 1) * dim];
            for j in 0..batch_size {
                let b = &data[j * dim..(j + 1) * dim];
                distances.push(self.compute_distance(a, b));
            }
        }

        Tensor::new(&distances, &[batch_size, batch_size])
    }

    /// Select hard negatives: for each anchor, find the closest negative.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - All embeddings `[batch, dim]`
    /// * `labels` - Class labels for each embedding `[batch]`
    ///
    /// # Returns
    ///
    /// Vector of (`anchor_idx`, `positive_idx`, `negative_idx`) triplets.
    #[must_use]
    pub fn mine_hard_triplets(
        &self,
        embeddings: &Tensor,
        labels: &[usize],
    ) -> Vec<(usize, usize, usize)> {
        let batch_size = embeddings.shape()[0];
        let distances = self.pairwise_distances(embeddings);
        let dist_data = distances.data();

        let mut triplets = Vec::new();

        for anchor_idx in 0..batch_size {
            let anchor_label = labels[anchor_idx];

            // Find hardest positive (farthest same-class sample)
            let mut best_positive_idx = anchor_idx;
            let mut best_positive_dist = f32::NEG_INFINITY;

            // Find hardest negative (closest different-class sample)
            let mut best_negative_idx = 0;
            let mut best_negative_dist = f32::INFINITY;

            for other_idx in 0..batch_size {
                if other_idx == anchor_idx {
                    continue;
                }

                let dist = dist_data[anchor_idx * batch_size + other_idx];
                let other_label = labels[other_idx];

                if other_label == anchor_label {
                    // Same class: track hardest positive (farthest)
                    if dist > best_positive_dist {
                        best_positive_dist = dist;
                        best_positive_idx = other_idx;
                    }
                } else {
                    // Different class: track hardest negative (closest)
                    if dist < best_negative_dist {
                        best_negative_dist = dist;
                        best_negative_idx = other_idx;
                    }
                }
            }

            // Only add valid triplets (where we found both positive and negative)
            if best_positive_idx != anchor_idx && best_negative_dist < f32::INFINITY {
                triplets.push((anchor_idx, best_positive_idx, best_negative_idx));
            }
        }

        triplets
    }

    /// Compute batch-hard triplet loss with online hard negative mining.
    ///
    /// For each anchor in the batch, selects the hardest positive (same class, farthest)
    /// and hardest negative (different class, closest).
    #[must_use]
    pub fn batch_hard_loss(&self, embeddings: &Tensor, labels: &[usize]) -> Tensor {
        let triplets = self.mine_hard_triplets(embeddings, labels);

        if triplets.is_empty() {
            return Tensor::new(&[0.0], &[1]);
        }

        let dim = embeddings.shape()[1];
        let data = embeddings.data();

        let mut total_loss = 0.0f32;
        let mut valid_count = 0;

        for (a_idx, p_idx, n_idx) in &triplets {
            let a = &data[a_idx * dim..(a_idx + 1) * dim];
            let p = &data[p_idx * dim..(p_idx + 1) * dim];
            let n = &data[n_idx * dim..(n_idx + 1) * dim];

            let d_ap = self.compute_distance(a, p);
            let d_an = self.compute_distance(a, n);

            let loss = (d_ap - d_an + self.margin).max(0.0);
            if loss > 0.0 {
                total_loss += loss;
                valid_count += 1;
            }
        }

        let mean_loss = if valid_count > 0 {
            total_loss / valid_count as f32
        } else {
            0.0
        };

        Tensor::new(&[mean_loss], &[1])
    }
}

impl Default for TripletLoss {
    fn default() -> Self {
        Self::new()
    }
}

fn div_scalar(x: &Tensor, scalar: f32) -> Tensor {
    scale_tensor(x, 1.0 / scalar)
}

/// ONE PATH: Each element delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
fn cosine_similarity_batch(a: &Tensor, b: &Tensor) -> Tensor {
    let shape_a = a.shape();
    let batch_size = shape_a[0];
    let dim = shape_a[1];

    let a_data = a.data();
    let b_data = b.data();
    let mut output = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let a_slice = &a_data[i * dim..(i + 1) * dim];
        let b_slice = &b_data[i * dim..(i + 1) * dim];
        output.push(crate::nn::functional::cosine_similarity_slice(a_slice, b_slice));
    }

    Tensor::new(&output, &[batch_size])
}

/// ONE PATH: Each element delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
fn cosine_similarity_many(anchor: &Tensor, negatives: &Tensor) -> Tensor {
    let a_shape = anchor.shape();
    let n_shape = negatives.shape();
    let batch_size = a_shape[0];
    let num_negatives = n_shape[1];
    let dim = a_shape[1];

    let a_data = anchor.data();
    let n_data = negatives.data();
    let mut output = Vec::with_capacity(batch_size * num_negatives);

    for b in 0..batch_size {
        let a_slice = &a_data[b * dim..(b + 1) * dim];
        for n in 0..num_negatives {
            let n_start = b * num_negatives * dim + n * dim;
            let n_slice = &n_data[n_start..n_start + dim];
            output.push(crate::nn::functional::cosine_similarity_slice(a_slice, n_slice));
        }
    }

    Tensor::new(&output, &[batch_size, num_negatives])
}

/// ONE PATH: Each element delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
fn cosine_similarity_matrix(a: &Tensor, b: &Tensor) -> Tensor {
    let shape = a.shape();
    let batch_size = shape[0];
    let dim = shape[1];

    let a_data = a.data();
    let b_data = b.data();
    let mut output = Vec::with_capacity(batch_size * batch_size);

    for i in 0..batch_size {
        let a_slice = &a_data[i * dim..(i + 1) * dim];
        for j in 0..batch_size {
            let b_slice = &b_data[j * dim..(j + 1) * dim];
            output.push(crate::nn::functional::cosine_similarity_slice(a_slice, b_slice));
        }
    }

    Tensor::new(&output, &[batch_size, batch_size])
}

fn info_nce_loss(pos_sim: &Tensor, all_sims: &Tensor) -> Tensor {
    // InfoNCE loss: -log(exp(pos) / sum(exp(all)))
    let pos_data = pos_sim.data();
    let all_data = all_sims.data();
    let batch_size = pos_data.len();
    let num_sims = all_data.len() / batch_size;

    let mut total_loss = 0.0f32;

    for i in 0..batch_size {
        let pos = pos_data[i];

        // Compute log-sum-exp for numerical stability
        let all_slice = &all_data[i * num_sims..(i + 1) * num_sims];
        let max_val = all_slice.iter().copied().fold(pos, f32::max);

        let sum_exp: f32 =
            (pos - max_val).exp() + all_slice.iter().map(|&x| (x - max_val).exp()).sum::<f32>();

        let loss = -pos + max_val + sum_exp.ln();
        total_loss += loss;
    }

    Tensor::new(&[total_loss / batch_size as f32], &[1])
}

// ==================== Training Sample ====================

/// A training sample for the neural encoder.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// The error message
    pub error_message: String,
    /// Source code context
    pub source_context: String,
    /// Source language (e.g., "python", "rust")
    pub source_lang: String,
    /// Positive example (similar error)
    pub positive: Option<Box<TrainingSample>>,
    /// Error category for grouping
    pub category: String,
}

impl TrainingSample {
    /// Create a new training sample.
    #[must_use]
    pub fn new(error_message: &str, source_context: &str, source_lang: &str) -> Self {
        Self {
            error_message: error_message.to_string(),
            source_context: source_context.to_string(),
            source_lang: source_lang.to_string(),
            positive: None,
            category: String::new(),
        }
    }

    /// Set the positive example.
    #[must_use]
    pub fn with_positive(mut self, positive: TrainingSample) -> Self {
        self.positive = Some(Box::new(positive));
        self
    }

    /// Set the error category.
    #[must_use]
    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
        self
    }
}

#[cfg(test)]
mod tests;
