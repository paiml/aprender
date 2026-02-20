
impl CTCLoss {
    /// Create CTC loss with specified blank token index.
    #[must_use]
    pub fn new(blank_idx: usize) -> Self {
        Self { blank_idx }
    }

    #[must_use]
    pub fn blank_idx(&self) -> usize {
        self.blank_idx
    }

    /// Compute CTC loss using forward-backward algorithm.
    ///
    /// # Arguments
    /// * `log_probs` - Log probabilities [T, C] (time x classes)
    /// * `targets` - Target sequence (class indices, no blanks)
    /// * `input_length` - Length of input sequence
    /// * `target_length` - Length of target sequence
    #[must_use]
    pub fn forward(
        &self,
        log_probs: &[Vec<f32>],
        targets: &[usize],
        input_length: usize,
        target_length: usize,
    ) -> f32 {
        if target_length == 0 || input_length == 0 {
            return 0.0;
        }

        // Create extended labels with blanks: b, l1, b, l2, b, ...
        let extended_len = 2 * target_length + 1;
        let mut labels = vec![self.blank_idx; extended_len];
        for (i, &t) in targets.iter().take(target_length).enumerate() {
            labels[2 * i + 1] = t;
        }

        // Forward pass: alpha[t][s] = P(prefix up to s at time t)
        let neg_inf = f32::NEG_INFINITY;
        let mut alpha = vec![vec![neg_inf; extended_len]; input_length];

        // Initialize
        alpha[0][0] = log_probs[0][labels[0]];
        if extended_len > 1 {
            alpha[0][1] = log_probs[0][labels[1]];
        }

        // Forward recursion
        for t in 1..input_length {
            for s in 0..extended_len {
                let label = labels[s];
                let mut val = alpha[t - 1][s];

                if s > 0 {
                    val = log_sum_exp(val, alpha[t - 1][s - 1]);
                }

                // Skip blank (allow skipping to same label only if not blank and different)
                if s > 1 && label != self.blank_idx && labels[s - 2] != label {
                    val = log_sum_exp(val, alpha[t - 1][s - 2]);
                }

                alpha[t][s] = val + log_probs[t][label];
            }
        }

        // Final probability: sum of last two positions
        let last_t = input_length - 1;
        let last_s = extended_len - 1;
        let total = if extended_len > 1 {
            log_sum_exp(alpha[last_t][last_s], alpha[last_t][last_s - 1])
        } else {
            alpha[last_t][last_s]
        };

        -total // Negative log-likelihood
    }
}

/// Log-sum-exp for numerical stability: log(exp(a) + exp(b))
fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        b
    } else if b == f32::NEG_INFINITY {
        a
    } else if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Wasserstein (Earth Mover's) Distance Loss.
///
/// Measures the minimum cost to transform one distribution to another.
/// More stable for GAN training than cross-entropy.
///
/// For 1D sorted distributions: W1 = Σ|CDF1 - CDF2|
///
/// Reference: Arjovsky et al., "Wasserstein GAN" (2017)
#[must_use]
pub fn wasserstein_loss(real_scores: &Vector<f32>, fake_scores: &Vector<f32>) -> f32 {
    let real_mean: f32 = real_scores.as_slice().iter().sum::<f32>() / real_scores.len() as f32;
    let fake_mean: f32 = fake_scores.as_slice().iter().sum::<f32>() / fake_scores.len() as f32;
    fake_mean - real_mean
}

/// Wasserstein loss for discriminator (critic).
/// Maximizes distance between real and fake.
#[must_use]
pub fn wasserstein_discriminator_loss(real_scores: &Vector<f32>, fake_scores: &Vector<f32>) -> f32 {
    -wasserstein_loss(real_scores, fake_scores)
}

/// Wasserstein loss for generator.
/// Minimizes negative fake score.
#[must_use]
pub fn wasserstein_generator_loss(fake_scores: &Vector<f32>) -> f32 {
    -fake_scores.as_slice().iter().sum::<f32>() / fake_scores.len() as f32
}

/// Gradient penalty for WGAN-GP.
/// Enforces Lipschitz constraint via gradient norm penalty.
#[must_use]
pub fn gradient_penalty(gradients: &[f32], lambda: f32) -> f32 {
    let grad_norm: f32 = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
    lambda * (grad_norm - 1.0).powi(2)
}

/// Wasserstein Loss struct wrapper.
#[derive(Debug, Clone, Copy)]
pub struct WassersteinLoss {
    lambda_gp: f32,
}

impl WassersteinLoss {
    #[must_use]
    pub fn new(lambda_gp: f32) -> Self {
        Self { lambda_gp }
    }

    #[must_use]
    pub fn lambda_gp(&self) -> f32 {
        self.lambda_gp
    }

    #[must_use]
    pub fn discriminator_loss(&self, real: &Vector<f32>, fake: &Vector<f32>) -> f32 {
        wasserstein_discriminator_loss(real, fake)
    }

    #[must_use]
    pub fn generator_loss(&self, fake: &Vector<f32>) -> f32 {
        wasserstein_generator_loss(fake)
    }
}

impl Loss for WassersteinLoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        wasserstein_loss(y_pred, y_true)
    }

    fn name(&self) -> &'static str {
        "Wasserstein"
    }
}

#[cfg(test)]
mod tests;
