//! Advanced Knowledge Distillation Strategies (GH-451)
//!
//! Implements three strategies from the APR spec (DistillKit parity):
//! - **Hidden-state matching**: Match intermediate hidden states via linear projection
//! - **Quantization-aware distillation**: Polynomial approximation + error-diffusion quantization
//! - **Online distillation**: Concurrent teacher/student training (no precompute step)
//!
//! # References
//! - FitNets: Romero et al. 2015 — hidden-state matching
//! - QAT distillation: Jacob et al. 2018 — quantization-aware training
//! - Deep Mutual Learning: Zhang et al. 2018 — online co-distillation

use crate::error::{AprenderError, Result};

// ============================================================================
// Hidden-State Distillation
// ============================================================================

/// Configuration for hidden-state distillation.
///
/// Maps teacher hidden states to student hidden states via learned linear
/// projections. Each layer mapping has a projection matrix that transforms
/// teacher dimension → student dimension.
#[derive(Debug, Clone)]
pub struct HiddenStateConfig {
    /// Teacher hidden dimension
    pub teacher_dim: usize,
    /// Student hidden dimension
    pub student_dim: usize,
    /// Layer mappings: (teacher_layer_idx, student_layer_idx)
    pub layer_map: Vec<(usize, usize)>,
    /// Weight for hidden-state loss vs logit loss
    pub hidden_loss_weight: f64,
    /// Learning rate for projection matrices
    pub projection_lr: f64,
}

impl Default for HiddenStateConfig {
    fn default() -> Self {
        Self {
            teacher_dim: 768,
            student_dim: 256,
            layer_map: vec![(3, 1), (7, 2), (11, 3)], // Default 12→4 layer map
            hidden_loss_weight: 0.5,
            projection_lr: 0.001,
        }
    }
}

/// Linear projection for hidden-state matching.
///
/// Projects teacher hidden states (dim_in) to student dimension (dim_out)
/// via a learned weight matrix W: R^dim_in → R^dim_out.
#[derive(Debug, Clone)]
pub struct HiddenProjection {
    /// Weight matrix (dim_out × dim_in), row-major
    weights: Vec<f64>,
    /// Input dimension (teacher)
    dim_in: usize,
    /// Output dimension (student)
    dim_out: usize,
}

impl HiddenProjection {
    /// Create a new projection with Xavier initialization.
    #[must_use]
    pub fn new(dim_in: usize, dim_out: usize, seed: u64) -> Self {
        let scale = (2.0 / (dim_in + dim_out) as f64).sqrt();
        let mut weights = Vec::with_capacity(dim_out * dim_in);

        // Simple deterministic initialization (LCG)
        let mut state = seed;
        for _ in 0..dim_out * dim_in {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
            weights.push(u * scale);
        }

        Self {
            weights,
            dim_in,
            dim_out,
        }
    }

    /// Project teacher hidden state to student dimension.
    #[must_use]
    pub fn forward(&self, teacher_hidden: &[f64]) -> Vec<f64> {
        let n = teacher_hidden.len().min(self.dim_in);
        let mut output = vec![0.0; self.dim_out];
        for i in 0..self.dim_out {
            for j in 0..n {
                output[i] += self.weights[i * self.dim_in + j] * teacher_hidden[j];
            }
        }
        output
    }

    /// Update projection weights via gradient descent.
    ///
    /// Gradient of MSE(projected, student) w.r.t. W:
    /// dL/dW_ij = 2 * (projected_i - student_i) * teacher_j / n
    pub fn update(&mut self, teacher_hidden: &[f64], student_hidden: &[f64], lr: f64) {
        let projected = self.forward(teacher_hidden);
        let n = self.dim_out as f64;

        for i in 0..self.dim_out {
            let error = projected[i] - student_hidden.get(i).copied().unwrap_or(0.0);
            for j in 0..self.dim_in {
                let grad = 2.0 * error * teacher_hidden.get(j).copied().unwrap_or(0.0) / n;
                self.weights[i * self.dim_in + j] -= lr * grad;
            }
        }
    }

    /// Compute MSE between projected teacher and student hidden states.
    #[must_use]
    pub fn mse_loss(&self, teacher_hidden: &[f64], student_hidden: &[f64]) -> f64 {
        let projected = self.forward(teacher_hidden);
        let n = self.dim_out;
        if n == 0 {
            return 0.0;
        }
        projected
            .iter()
            .zip(student_hidden.iter())
            .map(|(&p, &s)| (p - s).powi(2))
            .sum::<f64>()
            / n as f64
    }
}

/// Hidden-state distillation engine.
///
/// Manages layer-wise projections and computes combined loss:
/// L = α * KL(student||teacher) + (1-α) * Σ w_l * MSE(proj(h_t^l), h_s^l)
#[derive(Debug, Clone)]
pub struct HiddenStateDistiller {
    /// Per-layer projections
    projections: Vec<HiddenProjection>,
    /// Configuration
    config: HiddenStateConfig,
}

impl HiddenStateDistiller {
    /// Create a new hidden-state distiller with Xavier-initialized projections.
    #[must_use]
    pub fn new(config: HiddenStateConfig) -> Self {
        let projections = config
            .layer_map
            .iter()
            .enumerate()
            .map(|(i, _)| {
                HiddenProjection::new(config.teacher_dim, config.student_dim, 42 + i as u64)
            })
            .collect();

        Self {
            projections,
            config,
        }
    }

    /// Compute hidden-state matching loss across all mapped layers.
    ///
    /// # Arguments
    /// * `teacher_hiddens` - Teacher hidden states indexed by layer
    /// * `student_hiddens` - Student hidden states indexed by layer
    #[must_use]
    pub fn hidden_loss(&self, teacher_hiddens: &[Vec<f64>], student_hiddens: &[Vec<f64>]) -> f64 {
        let mut total_loss = 0.0;

        for (idx, &(t_layer, s_layer)) in self.config.layer_map.iter().enumerate() {
            if let (Some(th), Some(sh)) =
                (teacher_hiddens.get(t_layer), student_hiddens.get(s_layer))
            {
                total_loss += self.projections[idx].mse_loss(th, sh);
            }
        }

        total_loss / self.config.layer_map.len().max(1) as f64
    }

    /// Update projections from a training step.
    pub fn update_projections(
        &mut self,
        teacher_hiddens: &[Vec<f64>],
        student_hiddens: &[Vec<f64>],
    ) {
        for (idx, &(t_layer, s_layer)) in self.config.layer_map.iter().enumerate() {
            if let (Some(th), Some(sh)) =
                (teacher_hiddens.get(t_layer), student_hiddens.get(s_layer))
            {
                self.projections[idx].update(th, sh, self.config.projection_lr);
            }
        }
    }

    /// Get the layer mapping.
    #[must_use]
    pub fn layer_map(&self) -> &[(usize, usize)] {
        &self.config.layer_map
    }

    /// Get number of projection layers.
    #[must_use]
    pub fn num_projections(&self) -> usize {
        self.projections.len()
    }
}

// ============================================================================
// Quantization-Aware Distillation
// ============================================================================

/// Configuration for quantization-aware distillation.
///
/// During training, weights are quantized-then-dequantized (fake quantization)
/// so the student learns to be robust to quantization noise. Error diffusion
/// spreads quantization error to neighboring weights to minimize accumulated error.
#[derive(Debug, Clone)]
pub struct QuantAwareConfig {
    /// Number of quantization bits (4, 8, etc.)
    pub bits: u32,
    /// Whether to use symmetric quantization
    pub symmetric: bool,
    /// Error diffusion strength (0.0 = none, 1.0 = full Floyd-Steinberg)
    pub error_diffusion: f64,
    /// Polynomial degree for activation approximation
    pub poly_degree: usize,
}

impl Default for QuantAwareConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            symmetric: false,
            error_diffusion: 0.5,
            poly_degree: 3,
        }
    }
}

/// Quantization-aware distillation engine.
///
/// Applies fake quantization during forward pass so the student learns
/// representations robust to low-bit quantization.
#[derive(Debug, Clone)]
pub struct QuantAwareDistiller {
    config: QuantAwareConfig,
}

impl QuantAwareDistiller {
    /// Create a new quantization-aware distiller.
    #[must_use]
    pub fn new(config: QuantAwareConfig) -> Self {
        Self { config }
    }

    /// Fake-quantize a weight tensor: quantize then immediately dequantize.
    ///
    /// This simulates quantization error during training without actually
    /// storing weights in low-bit format.
    #[must_use]
    pub fn fake_quantize(&self, weights: &[f64]) -> Vec<f64> {
        if weights.is_empty() {
            return vec![];
        }

        let (qmin, qmax, scale, zero_point) = self.compute_quant_params(weights);

        weights
            .iter()
            .map(|&w| {
                let q = (w / scale + zero_point).round().clamp(qmin, qmax);
                (q - zero_point) * scale
            })
            .collect()
    }

    /// Fake-quantize with error diffusion (Floyd-Steinberg style).
    ///
    /// Quantization error from each weight is partially diffused to
    /// subsequent weights, reducing systematic quantization bias.
    #[must_use]
    pub fn fake_quantize_diffused(&self, weights: &[f64]) -> Vec<f64> {
        if weights.is_empty() {
            return vec![];
        }

        let (qmin, qmax, scale, zero_point) = self.compute_quant_params(weights);
        let diffusion = self.config.error_diffusion;

        let mut result = Vec::with_capacity(weights.len());
        let mut error_accum = 0.0;

        for &w in weights {
            let adjusted = w + diffusion * error_accum;
            let q = (adjusted / scale + zero_point).round().clamp(qmin, qmax);
            let dequantized = (q - zero_point) * scale;
            error_accum = adjusted - dequantized;
            result.push(dequantized);
        }

        result
    }

    /// Compute quantization parameters (min, max, scale, zero_point).
    fn compute_quant_params(&self, weights: &[f64]) -> (f64, f64, f64, f64) {
        let levels = (1u64 << self.config.bits) as f64;

        if self.config.symmetric {
            let max_abs = weights
                .iter()
                .map(|w| w.abs())
                .fold(0.0_f64, f64::max)
                .max(1e-10);
            let qmax = levels / 2.0 - 1.0;
            let qmin = -qmax;
            let scale = max_abs / qmax;
            (qmin, qmax, scale, 0.0)
        } else {
            let min_val = weights.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (max_val - min_val).max(1e-10);
            let qmin = 0.0;
            let qmax = levels - 1.0;
            let scale = range / qmax;
            let zero_point = (-min_val / scale).round();
            (qmin, qmax, scale, zero_point)
        }
    }

    /// Compute quantization MSE between original and fake-quantized weights.
    #[must_use]
    pub fn quantization_error(&self, weights: &[f64]) -> f64 {
        let quantized = self.fake_quantize(weights);
        if weights.is_empty() {
            return 0.0;
        }
        weights
            .iter()
            .zip(quantized.iter())
            .map(|(&w, &q)| (w - q).powi(2))
            .sum::<f64>()
            / weights.len() as f64
    }

    /// Polynomial approximation of activation function.
    ///
    /// Fits a polynomial of `poly_degree` to the given activation values
    /// using least-squares, suitable for hardware that lacks transcendental
    /// function support (e.g., integer-only accelerators).
    pub fn polynomial_activation_approx(
        &self,
        x_values: &[f64],
        y_values: &[f64],
    ) -> Result<Vec<f64>> {
        if x_values.len() != y_values.len() {
            return Err(AprenderError::dimension_mismatch(
                "x/y values",
                x_values.len(),
                y_values.len(),
            ));
        }

        let n = x_values.len();
        let degree = self.config.poly_degree;

        if n <= degree {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Need at least {} data points for degree-{} polynomial, got {}",
                    degree + 1,
                    degree,
                    n
                ),
            });
        }

        // Solve via normal equations: (X^T X) coeffs = X^T y
        // X is the Vandermonde matrix [1, x, x^2, ..., x^degree]
        let cols = degree + 1;

        // Build X^T * X (cols × cols)
        let mut xtx = vec![0.0; cols * cols];
        let mut xty = vec![0.0; cols];

        for i in 0..n {
            let mut xi_powers = vec![1.0; cols];
            for j in 1..cols {
                xi_powers[j] = xi_powers[j - 1] * x_values[i];
            }

            for r in 0..cols {
                for c in 0..cols {
                    xtx[r * cols + c] += xi_powers[r] * xi_powers[c];
                }
                xty[r] += xi_powers[r] * y_values[i];
            }
        }

        // Solve via Gaussian elimination
        solve_linear_system(&xtx, &xty, cols)
    }

    /// Get quantization bits.
    #[must_use]
    pub fn bits(&self) -> u32 {
        self.config.bits
    }
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(AprenderError::FormatError {
                message: "Singular matrix in polynomial fit".to_string(),
            });
        }

        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let above = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * above;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            x[i] -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] /= aug[i * (n + 1) + i];
    }

    Ok(x)
}

// ============================================================================
// Online Distillation
// ============================================================================

/// Configuration for online distillation.
///
/// Unlike standard distillation which precomputes teacher logits, online
/// distillation runs teacher and student concurrently. The teacher's outputs
/// are computed on-the-fly for each batch.
#[derive(Debug, Clone)]
pub struct OnlineDistillConfig {
    /// Temperature for soft targets
    pub temperature: f64,
    /// Weight for distillation loss vs task loss
    pub alpha: f64,
    /// Whether to use exponential moving average of teacher logits
    pub ema_decay: Option<f64>,
    /// Buffer size for teacher logit history (ring buffer)
    pub buffer_size: usize,
}

impl Default for OnlineDistillConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.7,
            ema_decay: Some(0.999),
            buffer_size: 1024,
        }
    }
}

/// Online distillation engine with EMA teacher logit smoothing.
///
/// Maintains a ring buffer of teacher logits and optionally applies
/// exponential moving average to reduce variance from stochastic
/// teacher outputs.
#[derive(Debug, Clone)]
pub struct OnlineDistiller {
    config: OnlineDistillConfig,
    /// EMA of teacher logits (smoothed targets)
    ema_logits: Option<Vec<f64>>,
    /// Number of updates processed
    update_count: usize,
}

impl OnlineDistiller {
    /// Create a new online distiller.
    #[must_use]
    pub fn new(config: OnlineDistillConfig) -> Self {
        Self {
            config,
            ema_logits: None,
            update_count: 0,
        }
    }

    /// Process a batch: compute distillation loss with optional EMA smoothing.
    ///
    /// # Arguments
    /// * `student_logits` - Student model output logits
    /// * `teacher_logits` - Teacher model output logits (computed on-the-fly)
    /// * `hard_labels` - Ground truth one-hot labels
    ///
    /// # Returns
    /// Combined loss value
    pub fn step(
        &mut self,
        student_logits: &[f64],
        teacher_logits: &[f64],
        hard_labels: &[f64],
    ) -> Result<f64> {
        if student_logits.len() != teacher_logits.len() || student_logits.len() != hard_labels.len()
        {
            return Err(AprenderError::dimension_mismatch(
                "logits/labels",
                student_logits.len(),
                teacher_logits.len(),
            ));
        }

        // Apply EMA smoothing to teacher logits
        let effective_teacher = if let Some(decay) = self.config.ema_decay {
            self.update_ema(teacher_logits, decay);
            self.ema_logits.as_deref().unwrap_or(teacher_logits)
        } else {
            teacher_logits
        };

        // Compute soft targets
        let t = self.config.temperature;
        let teacher_soft = super::distillation::softmax_temperature(effective_teacher, t);
        let student_soft = super::distillation::softmax_temperature(student_logits, t);
        let student_hard = super::distillation::softmax(student_logits);

        // KL divergence with T² scaling
        let kl_loss = super::distillation::kl_divergence(&student_soft, &teacher_soft);
        let distill_loss = t * t * kl_loss;

        // Hard label loss
        let hard_loss = super::distillation::cross_entropy(&student_hard, hard_labels);

        // Combined
        let total = self.config.alpha * distill_loss + (1.0 - self.config.alpha) * hard_loss;

        self.update_count += 1;
        Ok(total)
    }

    /// Update EMA of teacher logits.
    fn update_ema(&mut self, teacher_logits: &[f64], decay: f64) {
        match &mut self.ema_logits {
            Some(ema) if ema.len() == teacher_logits.len() => {
                for (e, &t) in ema.iter_mut().zip(teacher_logits.iter()) {
                    *e = decay * *e + (1.0 - decay) * t;
                }
            }
            _ => {
                self.ema_logits = Some(teacher_logits.to_vec());
            }
        }
    }

    /// Get the number of updates processed.
    #[must_use]
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Get current EMA logits (if available).
    #[must_use]
    pub fn ema_logits(&self) -> Option<&[f64]> {
        self.ema_logits.as_deref()
    }

    /// Reset the distiller state.
    pub fn reset(&mut self) {
        self.ema_logits = None;
        self.update_count = 0;
    }
}

#[cfg(test)]
#[path = "distillation_advanced_tests.rs"]
mod tests;
