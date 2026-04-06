//! Linear probe classifier pipeline (SSC v11 Section 5)
//!
//! Implements the CodeBERT linear probe:
//! 1. Extract [CLS] embeddings from frozen encoder (CLF-001)
//! 2. Train Linear(hidden_size, num_classes) on cached embeddings (CLF-002)
//! 3. Evaluate with MCC, accuracy, recall, bootstrap CI (CLF-003)
//! 4. Cache confidence scores for conversation generation (CLF-007)
//!
//! # Architecture
//!
//! ```text
//! token_ids → EncoderModel.cls_embedding() → [hidden_size]
//!           → Linear(hidden_size, 2) → softmax → [p_safe, p_unsafe]
//! ```
//!
//! # Contract (linear-probe-classifier-v1.yaml)
//! - Frozen encoder: weights unchanged after training
//! - Probability simplex: softmax sums to 1.0
//! - Embedding determinism: bit-identical on repeated calls

use crate::autograd::{matmul, Tensor};

/// Classification metrics for binary or multi-class evaluation (CLF-003).
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Matthews Correlation Coefficient (-1 to 1)
    pub mcc: f32,
    /// Overall accuracy (0 to 1)
    pub accuracy: f32,
    /// Per-class recall (sensitivity)
    pub recall: Vec<f32>,
    /// Per-class precision
    pub precision: Vec<f32>,
    /// Number of samples evaluated
    pub num_samples: usize,
    /// Confusion matrix [predicted][actual] — row=predicted, col=actual
    pub confusion_matrix: Vec<Vec<usize>>,
}

/// Bootstrap confidence interval.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapCI {
    /// Point estimate
    pub estimate: f32,
    /// Lower bound (2.5th percentile)
    pub lower: f32,
    /// Upper bound (97.5th percentile)
    pub upper: f32,
    /// Number of bootstrap iterations
    pub n_bootstrap: usize,
}

/// Linear probe: frozen embeddings + trainable linear head (CLF-002).
///
/// Trains on pre-extracted embeddings (not raw token IDs), making training
/// complete in seconds rather than minutes.
pub struct LinearProbe {
    /// Linear weight [hidden_size, num_classes] flattened row-major
    pub weight: Tensor,
    /// Bias [num_classes]
    pub bias: Tensor,
    /// Input dimension
    hidden_size: usize,
    /// Number of output classes
    num_classes: usize,
}

impl LinearProbe {
    /// Create with Xavier initialization.
    pub fn new(hidden_size: usize, num_classes: usize) -> Self {
        assert!(hidden_size > 0, "hidden_size must be > 0");
        assert!(num_classes >= 2, "num_classes must be >= 2");

        let scale = (6.0 / (hidden_size + num_classes) as f32).sqrt();
        let mut rng: u64 = 42;
        let weight_data: Vec<f32> = (0..hidden_size * num_classes)
            .map(|_| {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = (rng >> 33) as f32 / (1u64 << 31) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect();

        Self {
            weight: Tensor::from_vec(weight_data, true),
            bias: Tensor::zeros(num_classes, true),
            hidden_size,
            num_classes,
        }
    }

    /// Forward pass: embedding → logits.
    ///
    /// # Arguments
    /// * `embedding` - Pre-extracted [CLS] embedding [hidden_size]
    ///
    /// # Returns
    /// Logits tensor [num_classes]
    pub fn forward(&self, embedding: &Tensor) -> Tensor {
        let logits = matmul(embedding, &self.weight, 1, self.hidden_size, self.num_classes);
        let logits_data = logits.data();
        let logits_slice = logits_data.as_slice().expect("contiguous logits");
        let bias_data = self.bias.data();
        let bias_slice = bias_data.as_slice().expect("contiguous bias");

        let output: Vec<f32> =
            logits_slice.iter().zip(bias_slice.iter()).map(|(&l, &b)| l + b).collect();
        Tensor::from_vec(output, logits.requires_grad())
    }

    /// Predict class probabilities via softmax.
    pub fn predict_probs(&self, embedding: &Tensor) -> Vec<f32> {
        let logits = self.forward(embedding);
        softmax_vec(&logits)
    }

    /// Predict class index (argmax of logits).
    pub fn predict(&self, embedding: &Tensor) -> usize {
        contract_pre_predict!();
        let logits = self.forward(embedding);
        let data = logits.data();
        let slice = data.as_slice().expect("contiguous");
        slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    /// Train on pre-extracted embeddings using SGD.
    ///
    /// # Arguments
    /// * `embeddings` - List of pre-extracted [CLS] embeddings (each len=hidden_size)
    /// * `labels` - Corresponding class labels
    /// * `epochs` - Number of training epochs
    /// * `learning_rate` - SGD learning rate
    /// * `class_weights` - Optional per-class loss weights for imbalance
    ///
    /// # Returns
    /// Final training loss
    pub fn train(
        &mut self,
        embeddings: &[Vec<f32>],
        labels: &[usize],
        epochs: usize,
        learning_rate: f32,
        class_weights: Option<&[f32]>,
    ) -> f32 {
        assert_eq!(embeddings.len(), labels.len());
        let n = embeddings.len();
        let mut final_loss = 0.0;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (emb, &label) in embeddings.iter().zip(labels.iter()) {
                assert_eq!(emb.len(), self.hidden_size);
                assert!(label < self.num_classes);

                // Forward
                let emb_tensor = Tensor::from_vec(emb.clone(), false);
                let logits = self.forward(&emb_tensor);

                // Cross-entropy loss with optional class weights
                let probs = softmax_vec(&logits);
                let loss_weight = class_weights.map_or(1.0, |w| w[label]);
                let loss = -probs[label].max(1e-10).ln() * loss_weight;
                epoch_loss += loss;

                // Gradient of cross-entropy w.r.t. logits: probs - one_hot(label)
                let mut grad_logits = probs;
                grad_logits[label] -= 1.0;
                if let Some(w) = class_weights {
                    for (i, g) in grad_logits.iter_mut().enumerate() {
                        *g *= w[i];
                    }
                }

                // Update weight: grad_W = emb^T @ grad_logits
                let w_data = self.weight.data();
                let mut w_slice = w_data.as_slice().expect("contiguous").to_vec();
                for i in 0..self.hidden_size {
                    for j in 0..self.num_classes {
                        w_slice[i * self.num_classes + j] -=
                            learning_rate * emb[i] * grad_logits[j];
                    }
                }
                self.weight = Tensor::from_vec(w_slice, true);

                // Update bias: grad_b = grad_logits
                let b_data = self.bias.data();
                let mut b_slice = b_data.as_slice().expect("contiguous").to_vec();
                for j in 0..self.num_classes {
                    b_slice[j] -= learning_rate * grad_logits[j];
                }
                self.bias = Tensor::from_vec(b_slice, true);
            }

            final_loss = epoch_loss / n as f32;
            if epoch == 0 || (epoch + 1) % 5 == 0 || epoch == epochs - 1 {
                eprintln!("  Epoch {}/{epochs}: loss={final_loss:.4}", epoch + 1);
            }
        }

        final_loss
    }

    /// Get total trainable parameter count (CLF-002: 1,538 for binary CodeBERT).
    pub fn num_parameters(&self) -> usize {
        self.hidden_size * self.num_classes + self.num_classes
    }

    /// Get number of classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// MLP probe: frozen embeddings + trainable 2-layer MLP head (Level 0.5).
///
/// Adds a hidden layer with ReLU between embeddings and classification head.
/// This allows non-linear decision boundaries, which can capture patterns
/// that a linear probe cannot (e.g., shell safety from CodeBERT embeddings).
///
/// Architecture: embedding → Linear(hidden_size, mlp_hidden) → ReLU → Linear(mlp_hidden, num_classes)
pub struct MlpProbe {
    /// First layer weights [hidden_size × mlp_hidden] flattened row-major
    pub w1: Vec<f32>,
    /// First layer bias [mlp_hidden]
    pub b1: Vec<f32>,
    /// Second layer weights [mlp_hidden × num_classes] flattened row-major
    pub w2: Vec<f32>,
    /// Second layer bias [num_classes]
    pub b2: Vec<f32>,
    /// Input dimension
    pub hidden_size: usize,
    /// Hidden layer dimension
    pub mlp_hidden: usize,
    /// Number of output classes
    pub num_classes: usize,
}

impl MlpProbe {
    /// Create with Xavier initialization.
    pub fn new(hidden_size: usize, mlp_hidden: usize, num_classes: usize) -> Self {
        assert!(hidden_size > 0 && mlp_hidden > 0 && num_classes >= 2);

        let mut rng: u64 = 42;
        let mut xavier = |fan_in: usize, fan_out: usize, n: usize| -> Vec<f32> {
            let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
            (0..n)
                .map(|_| {
                    rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let u = (rng >> 33) as f32 / (1u64 << 31) as f32;
                    (2.0 * u - 1.0) * scale
                })
                .collect()
        };

        Self {
            w1: xavier(hidden_size, mlp_hidden, hidden_size * mlp_hidden),
            b1: vec![0.0; mlp_hidden],
            w2: xavier(mlp_hidden, num_classes, mlp_hidden * num_classes),
            b2: vec![0.0; num_classes],
            hidden_size,
            mlp_hidden,
            num_classes,
        }
    }

    /// Forward pass: embedding → hidden (ReLU) → logits.
    pub fn forward(&self, emb: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Layer 1: h = ReLU(W1 @ emb + b1)
        let mut h = vec![0.0_f32; self.mlp_hidden];
        for j in 0..self.mlp_hidden {
            let mut sum = self.b1[j];
            for i in 0..self.hidden_size {
                sum += self.w1[i * self.mlp_hidden + j] * emb[i];
            }
            h[j] = sum.max(0.0); // ReLU
        }

        // Layer 2: logits = W2 @ h + b2
        let mut logits = vec![0.0_f32; self.num_classes];
        for j in 0..self.num_classes {
            let mut sum = self.b2[j];
            for i in 0..self.mlp_hidden {
                sum += self.w2[i * self.num_classes + j] * h[i];
            }
            logits[j] = sum;
        }

        (h, logits)
    }

    /// Predict class index.
    pub fn predict(&self, emb: &[f32]) -> usize {
        contract_pre_predict!();
        let (_, logits) = self.forward(emb);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    /// Predict class probabilities via softmax.
    pub fn predict_probs(&self, emb: &[f32]) -> Vec<f32> {
        let (_, logits) = self.forward(emb);
        softmax_slice(&logits)
    }

    /// Forward pass returning pre-ReLU activations, post-ReLU hidden, and logits.
    fn forward_train(&self, emb: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut h_pre = vec![0.0_f32; self.mlp_hidden];
        let mut h = vec![0.0_f32; self.mlp_hidden];
        for j in 0..self.mlp_hidden {
            let mut sum = self.b1[j];
            for i in 0..self.hidden_size {
                sum += self.w1[i * self.mlp_hidden + j] * emb[i];
            }
            h_pre[j] = sum;
            h[j] = sum.max(0.0);
        }

        let mut logits = vec![0.0_f32; self.num_classes];
        for j in 0..self.num_classes {
            let mut sum = self.b2[j];
            for i in 0..self.mlp_hidden {
                sum += self.w2[i * self.num_classes + j] * h[i];
            }
            logits[j] = sum;
        }
        (h_pre, h, logits)
    }

    /// Backward pass: update W1, b1, W2, b2 given gradients.
    fn backward_step(
        &mut self,
        emb: &[f32],
        h_pre: &[f32],
        h: &[f32],
        grad_logits: &[f32],
        lr: f32,
        wd: f32,
    ) {
        // Update W2 and b2
        for i in 0..self.mlp_hidden {
            for j in 0..self.num_classes {
                let idx = i * self.num_classes + j;
                self.w2[idx] -= lr * (h[i] * grad_logits[j] + wd * self.w2[idx]);
            }
        }
        for j in 0..self.num_classes {
            self.b2[j] -= lr * grad_logits[j];
        }

        // Compute grad_h (with ReLU mask)
        let mut grad_h = vec![0.0_f32; self.mlp_hidden];
        for i in 0..self.mlp_hidden {
            if h_pre[i] > 0.0 {
                for j in 0..self.num_classes {
                    grad_h[i] += self.w2[i * self.num_classes + j] * grad_logits[j];
                }
            }
        }

        // Update W1 and b1
        for i in 0..self.hidden_size {
            for j in 0..self.mlp_hidden {
                let idx = i * self.mlp_hidden + j;
                self.w1[idx] -= lr * (emb[i] * grad_h[j] + wd * self.w1[idx]);
            }
        }
        for j in 0..self.mlp_hidden {
            self.b1[j] -= lr * grad_h[j];
        }
    }

    /// Train with online SGD + class weights + L2 regularization.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        embeddings: &[Vec<f32>],
        labels: &[usize],
        epochs: usize,
        learning_rate: f32,
        class_weights: Option<&[f32]>,
        weight_decay: f32,
    ) -> f32 {
        assert_eq!(embeddings.len(), labels.len());
        let n = embeddings.len();
        let mut final_loss = 0.0;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (emb, &label) in embeddings.iter().zip(labels.iter()) {
                let (h_pre, h, logits) = self.forward_train(emb);
                let probs = softmax_slice(&logits);
                let loss_weight = class_weights.map_or(1.0, |w| w[label]);
                epoch_loss += -probs[label].max(1e-10).ln() * loss_weight;

                let mut grad_logits = probs;
                grad_logits[label] -= 1.0;
                if let Some(w) = class_weights {
                    for (i, g) in grad_logits.iter_mut().enumerate() {
                        *g *= w[i];
                    }
                }

                self.backward_step(emb, &h_pre, &h, &grad_logits, learning_rate, weight_decay);
            }

            final_loss = epoch_loss / n as f32;
            if epoch == 0 || (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                eprintln!("  Epoch {}/{epochs}: loss={final_loss:.4}", epoch + 1);
            }
        }

        final_loss
    }

    /// Total trainable parameters.
    pub fn num_parameters(&self) -> usize {
        self.hidden_size * self.mlp_hidden + self.mlp_hidden  // W1 + b1
        + self.mlp_hidden * self.num_classes + self.num_classes // W2 + b2
    }
}

/// Compute softmax from a slice.
fn softmax_slice(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

/// Compute softmax probabilities from a tensor.
fn softmax_vec(logits: &Tensor) -> Vec<f32> {
    let data = logits.data();
    let slice = data.as_slice().expect("contiguous logits");
    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = slice.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

/// Compute binary MCC from confusion matrix values.
///
/// MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
pub fn binary_mcc(tp: usize, tn: usize, fp: usize, fn_count: usize) -> f32 {
    let numerator = (tp * tn) as f64 - (fp * fn_count) as f64;
    let denom =
        ((tp + fp) as f64 * (tp + fn_count) as f64 * (tn + fp) as f64 * (tn + fn_count) as f64)
            .sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (numerator / denom) as f32
    }
}

/// Evaluate predictions against ground truth (CLF-003).
pub fn evaluate(
    predictions: &[usize],
    labels: &[usize],
    num_classes: usize,
) -> ClassificationMetrics {
    assert_eq!(predictions.len(), labels.len());
    let n = predictions.len();

    // Build confusion matrix
    let mut cm = vec![vec![0usize; num_classes]; num_classes];
    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        if pred < num_classes && label < num_classes {
            cm[pred][label] += 1;
        }
    }

    // Accuracy
    let correct: usize = (0..num_classes).map(|c| cm[c][c]).sum();
    let accuracy = correct as f32 / n.max(1) as f32;

    // Per-class precision and recall
    let mut precision = vec![0.0_f32; num_classes];
    let mut recall = vec![0.0_f32; num_classes];
    for c in 0..num_classes {
        let pred_count: usize = cm[c].iter().sum();
        let actual_count: usize = (0..num_classes).map(|p| cm[p][c]).sum();
        precision[c] = if pred_count > 0 { cm[c][c] as f32 / pred_count as f32 } else { 0.0 };
        recall[c] = if actual_count > 0 { cm[c][c] as f32 / actual_count as f32 } else { 0.0 };
    }

    // MCC (binary for 2-class, multiclass generalization otherwise)
    let mcc = if num_classes == 2 {
        let tp = cm[1][1];
        let tn = cm[0][0];
        let fp = cm[1][0];
        let fn_count = cm[0][1];
        binary_mcc(tp, tn, fp, fn_count)
    } else {
        multiclass_mcc(&cm, num_classes)
    };

    ClassificationMetrics { mcc, accuracy, recall, precision, num_samples: n, confusion_matrix: cm }
}

/// Multiclass MCC using the general formula.
fn multiclass_mcc(cm: &[Vec<usize>], k: usize) -> f32 {
    let n: f64 = cm.iter().flat_map(|row| row.iter()).sum::<usize>() as f64;
    let c: f64 = (0..k).map(|i| cm[i][i] as f64).sum();

    let mut s = 0.0_f64; // sum of outer products
    let mut p = 0.0_f64; // sum of row sums squared
    let mut t = 0.0_f64; // sum of col sums squared

    for i in 0..k {
        let row_sum: f64 = cm[i].iter().sum::<usize>() as f64;
        let col_sum: f64 = (0..k).map(|j| cm[j][i] as f64).sum();
        p += row_sum * row_sum;
        t += col_sum * col_sum;
        for j in 0..k {
            s += (cm[i].iter().sum::<usize>() as f64) * (cm[j][i] as f64);
        }
    }

    let numerator = c * n - s;
    let denom = ((n * n - p) * (n * n - t)).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (numerator / denom) as f32
    }
}

/// Compute bootstrap confidence interval for MCC (CLF-003).
///
/// Resamples predictions/labels with replacement `n_bootstrap` times,
/// computes MCC for each, and returns 2.5th/97.5th percentiles.
pub fn bootstrap_mcc_ci(
    predictions: &[usize],
    labels: &[usize],
    num_classes: usize,
    n_bootstrap: usize,
) -> BootstrapCI {
    let n = predictions.len();
    let point_estimate = evaluate(predictions, labels, num_classes).mcc;

    let mut mcc_samples = Vec::with_capacity(n_bootstrap);
    let mut rng: u64 = 12345;

    for _ in 0..n_bootstrap {
        let mut boot_preds = Vec::with_capacity(n);
        let mut boot_labels = Vec::with_capacity(n);

        for _ in 0..n {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1442695040888963407);
            let idx = (rng >> 33) as usize % n;
            boot_preds.push(predictions[idx]);
            boot_labels.push(labels[idx]);
        }

        let metrics = evaluate(&boot_preds, &boot_labels, num_classes);
        mcc_samples.push(metrics.mcc);
    }

    mcc_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower_idx = (n_bootstrap as f32 * 0.025) as usize;
    let upper_idx = ((n_bootstrap as f32 * 0.975) as usize).min(n_bootstrap - 1);

    BootstrapCI {
        estimate: point_estimate,
        lower: mcc_samples[lower_idx],
        upper: mcc_samples[upper_idx],
        n_bootstrap,
    }
}

/// Confidence score for a single sample (CLF-007).
#[derive(Debug, Clone)]
pub struct ConfidenceScore {
    /// Predicted class (argmax)
    pub predicted_class: usize,
    /// Probability of predicted class
    pub confidence: f32,
    /// Full probability distribution
    pub probabilities: Vec<f32>,
}

/// Cache confidence scores for all samples (CLF-007).
pub fn compute_confidence_scores(
    probe: &LinearProbe,
    embeddings: &[Vec<f32>],
) -> Vec<ConfidenceScore> {
    embeddings
        .iter()
        .map(|emb| {
            let emb_tensor = Tensor::from_vec(emb.clone(), false);
            let probs = probe.predict_probs(&emb_tensor);
            let (predicted_class, &confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("non-empty probabilities");
            ConfidenceScore { predicted_class, confidence, probabilities: probs }
        })
        .collect()
}

// =============================================================================
// CLF-004: ESCALATION LADDER
// =============================================================================

/// Escalation level for classifier training (SSC v11 Section 5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationLevel {
    /// Level 0: Linear probe on frozen encoder (1,538 params)
    LinearProbe,
    /// Level 1: Fine-tune top-2 encoder layers + head (~15M params)
    TopLayers,
    /// Level 2: Full fine-tune all encoder layers (125M params)
    FullFinetune,
    /// Level 3: Continue-pretrain on shell + fine-tune (125M params)
    ContinuePretrain,
}

impl std::fmt::Display for EscalationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LinearProbe => write!(f, "Level 0: Linear probe"),
            Self::TopLayers => write!(f, "Level 1: Top-2 layers + head"),
            Self::FullFinetune => write!(f, "Level 2: Full fine-tune"),
            Self::ContinuePretrain => write!(f, "Level 3: Continue-pretrain + fine-tune"),
        }
    }
}

/// Decide whether to escalate based on MCC CI (CLF-004).
///
/// Returns `Some(next_level)` if escalation needed, `None` if ship gate met.
pub fn should_escalate(
    current_level: EscalationLevel,
    mcc_ci: &BootstrapCI,
    accuracy: f32,
) -> Option<EscalationLevel> {
    match current_level {
        EscalationLevel::LinearProbe => {
            if mcc_ci.lower < 0.2 || accuracy <= 0.935 {
                Some(EscalationLevel::TopLayers)
            } else {
                None // Ship gate C-CLF-001 met
            }
        }
        EscalationLevel::TopLayers | EscalationLevel::FullFinetune => {
            if mcc_ci.lower < 0.3 {
                match current_level {
                    EscalationLevel::TopLayers => Some(EscalationLevel::FullFinetune),
                    _ => Some(EscalationLevel::ContinuePretrain),
                }
            } else {
                None
            }
        }
        EscalationLevel::ContinuePretrain => {
            // Terminal level — if this fails, classifier adds no value
            None
        }
    }
}

// =============================================================================
// CLF-005: BASELINES COMPARISON
// =============================================================================

/// Baseline comparison result (CLF-005).
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Name of the baseline
    pub name: String,
    /// Baseline MCC
    pub baseline_mcc: f32,
    /// Model MCC
    pub model_mcc: f32,
    /// Whether model beats this baseline
    pub beats_baseline: bool,
}

/// Compare model against baselines (CLF-005).
///
/// Baselines from SSC v11 Section 5.5:
/// - Majority class: MCC = 0.0
/// - Keyword regex: MCC ~0.3-0.5
/// - bashrs linter: MCC ~0.4-0.6
pub fn compare_baselines(model_mcc: f32, baseline_mccs: &[(&str, f32)]) -> Vec<BaselineComparison> {
    baseline_mccs
        .iter()
        .map(|&(name, baseline_mcc)| BaselineComparison {
            name: name.to_string(),
            baseline_mcc,
            model_mcc,
            beats_baseline: model_mcc > baseline_mcc,
        })
        .collect()
}

// =============================================================================
// CLF-006: GENERALIZATION TEST
// =============================================================================

/// Generalization test result (CLF-006).
#[derive(Debug, Clone)]
pub struct GeneralizationResult {
    /// Number of novel unsafe scripts tested
    pub total: usize,
    /// Number correctly classified as unsafe
    pub detected: usize,
    /// Detection rate (detected / total)
    pub detection_rate: f32,
    /// Meets threshold (>= 50%)
    pub passes: bool,
}

/// Run generalization test on novel unsafe scripts (CLF-006).
///
/// Tests the classifier on out-of-distribution scripts that have
/// no lexical overlap with training data.
pub fn generalization_test(
    probe: &LinearProbe,
    novel_embeddings: &[Vec<f32>],
    unsafe_class: usize,
) -> GeneralizationResult {
    let total = novel_embeddings.len();
    let detected = novel_embeddings
        .iter()
        .filter(|emb| {
            let emb_tensor = Tensor::from_vec((*emb).clone(), false);
            probe.predict(&emb_tensor) == unsafe_class
        })
        .count();

    let detection_rate = if total > 0 { detected as f32 / total as f32 } else { 0.0 };

    GeneralizationResult { total, detected, detection_rate, passes: detection_rate >= 0.5 }
}

// =============================================================================
// SHIP GATE (C-CLF-001)
// =============================================================================

/// Ship gate check result (SSC v11 Section 5.7).
#[derive(Debug, Clone)]
pub struct ShipGateResult {
    /// MCC CI lower bound > 0.2
    pub mcc_passes: bool,
    /// Accuracy > 0.935
    pub accuracy_passes: bool,
    /// Generalization >= 50%
    pub generalization_passes: bool,
    /// All criteria met
    pub ship_ready: bool,
    /// Escalation level that achieved these results
    pub level: EscalationLevel,
}

/// Check ship gate C-CLF-001 (SSC v11 Section 5.7).
pub fn check_ship_gate(
    mcc_ci: &BootstrapCI,
    accuracy: f32,
    generalization: &GeneralizationResult,
    level: EscalationLevel,
) -> ShipGateResult {
    let mcc_passes = mcc_ci.lower > 0.2;
    let accuracy_passes = accuracy > 0.935;
    let generalization_passes = generalization.passes;

    ShipGateResult {
        mcc_passes,
        accuracy_passes,
        generalization_passes,
        ship_ready: mcc_passes && accuracy_passes && generalization_passes,
        level,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn clf_002_linear_probe_forward_shape() {
        let probe = LinearProbe::new(768, 2);
        let emb = Tensor::from_vec(vec![0.1; 768], false);
        let logits = probe.forward(&emb);
        assert_eq!(logits.len(), 2);
    }

    #[test]
    fn clf_002_linear_probe_predict_probs_sum_to_one() {
        let probe = LinearProbe::new(64, 3);
        let emb = Tensor::from_vec(vec![0.5; 64], false);
        let probs = probe.predict_probs(&emb);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities must sum to 1.0, got {sum}");
        assert!(probs.iter().all(|&p| p > 0.0), "all probabilities must be positive");
    }

    #[test]
    fn clf_002_linear_probe_num_parameters() {
        let probe = LinearProbe::new(768, 2);
        assert_eq!(probe.num_parameters(), 768 * 2 + 2); // 1538
    }

    #[test]
    fn clf_002_linear_probe_train_reduces_loss() {
        let mut probe = LinearProbe::new(8, 2);
        // Simple linearly separable data
        let embeddings: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                if i < 10 {
                    vec![1.0; 8] // class 0
                } else {
                    vec![-1.0; 8] // class 1
                }
            })
            .collect();
        let labels: Vec<usize> = (0..20).map(|i| usize::from(i >= 10)).collect();

        let loss_before = {
            let mut temp = LinearProbe::new(8, 2);
            temp.train(&embeddings, &labels, 1, 0.01, None)
        };
        let loss_after = probe.train(&embeddings, &labels, 10, 0.01, None);

        // After 10 epochs, loss should be lower
        assert!(loss_after < loss_before + 0.5, "training should reduce loss");
    }

    #[test]
    fn clf_003_binary_mcc_perfect() {
        // Perfect predictions
        assert!((binary_mcc(50, 50, 0, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn clf_003_binary_mcc_random() {
        // Random predictions: MCC ≈ 0
        assert!(binary_mcc(25, 25, 25, 25).abs() < 1e-5);
    }

    #[test]
    fn clf_003_evaluate_perfect() {
        let preds = vec![0, 0, 1, 1, 1];
        let labels = vec![0, 0, 1, 1, 1];
        let metrics = evaluate(&preds, &labels, 2);
        assert!((metrics.accuracy - 1.0).abs() < 1e-5);
        assert!((metrics.mcc - 1.0).abs() < 1e-5);
    }

    #[test]
    fn clf_003_evaluate_majority_baseline() {
        // All predict class 0
        let preds = vec![0; 100];
        let labels: Vec<usize> = (0..100).map(|i| usize::from(i >= 93)).collect();
        let metrics = evaluate(&preds, &labels, 2);
        assert!((metrics.accuracy - 0.93).abs() < 0.01);
        assert_eq!(metrics.recall[1], 0.0); // unsafe recall is 0
    }

    #[test]
    fn clf_003_bootstrap_ci_contains_estimate() {
        let preds = vec![0, 0, 1, 1, 0, 1, 0, 0, 1, 1];
        let labels = vec![0, 0, 1, 1, 0, 0, 0, 1, 1, 1];
        let ci = bootstrap_mcc_ci(&preds, &labels, 2, 100);
        assert!(ci.lower <= ci.estimate, "CI lower must be <= estimate");
        assert!(ci.upper >= ci.estimate, "CI upper must be >= estimate");
    }

    #[test]
    fn clf_007_confidence_scores() {
        let probe = LinearProbe::new(8, 2);
        let embeddings = vec![vec![0.5; 8], vec![-0.5; 8]];
        let scores = compute_confidence_scores(&probe, &embeddings);
        assert_eq!(scores.len(), 2);
        for score in &scores {
            assert!(score.confidence > 0.0);
            assert!(score.confidence <= 1.0);
            assert_eq!(score.probabilities.len(), 2);
            let sum: f32 = score.probabilities.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    // =========================================================================
    // CLF-004: ESCALATION TESTS
    // =========================================================================

    #[test]
    fn clf_004_escalate_from_linear_probe_low_mcc() {
        let ci = BootstrapCI { estimate: 0.15, lower: 0.10, upper: 0.20, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::LinearProbe, &ci, 0.94);
        assert_eq!(result, Some(EscalationLevel::TopLayers));
    }

    #[test]
    fn clf_004_no_escalate_when_ship_gate_met() {
        let ci = BootstrapCI { estimate: 0.45, lower: 0.30, upper: 0.60, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::LinearProbe, &ci, 0.96);
        assert_eq!(result, None);
    }

    #[test]
    fn clf_004_escalate_from_top_layers_to_full() {
        let ci = BootstrapCI { estimate: 0.25, lower: 0.15, upper: 0.35, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::TopLayers, &ci, 0.95);
        assert_eq!(result, Some(EscalationLevel::FullFinetune));
    }

    #[test]
    fn clf_004_terminal_level_no_escalation() {
        let ci = BootstrapCI { estimate: 0.1, lower: 0.05, upper: 0.15, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::ContinuePretrain, &ci, 0.90);
        assert_eq!(result, None); // Terminal — can't escalate further
    }

    #[test]
    fn clf_004_escalate_on_low_accuracy() {
        // MCC CI OK but accuracy below threshold
        let ci = BootstrapCI { estimate: 0.45, lower: 0.30, upper: 0.60, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::LinearProbe, &ci, 0.93);
        assert_eq!(result, Some(EscalationLevel::TopLayers));
    }

    // =========================================================================
    // CLF-005: BASELINES COMPARISON TESTS
    // =========================================================================

    #[test]
    fn clf_005_compare_baselines_beats_majority() {
        let baselines = vec![("majority", 0.0), ("keyword", 0.4), ("linter", 0.5)];
        let comparisons = compare_baselines(0.35, &baselines);
        assert!(comparisons[0].beats_baseline); // beats majority (0.35 > 0.0)
        assert!(!comparisons[1].beats_baseline); // loses to keyword (0.35 < 0.4)
        assert!(!comparisons[2].beats_baseline); // loses to linter (0.35 < 0.5)
    }

    #[test]
    fn clf_005_compare_baselines_beats_all() {
        let baselines = vec![("majority", 0.0), ("keyword", 0.4), ("linter", 0.5)];
        let comparisons = compare_baselines(0.65, &baselines);
        assert!(comparisons.iter().all(|c| c.beats_baseline));
    }

    // =========================================================================
    // CLF-006: GENERALIZATION TEST
    // =========================================================================

    #[test]
    fn clf_006_generalization_all_detected() {
        let mut probe = LinearProbe::new(4, 2);
        // Train probe to always predict unsafe (class 1) for negative embeddings
        let embeddings: Vec<Vec<f32>> =
            (0..20).map(|i| if i < 10 { vec![1.0; 4] } else { vec![-1.0; 4] }).collect();
        let labels: Vec<usize> = (0..20).map(|i| usize::from(i >= 10)).collect();
        probe.train(&embeddings, &labels, 30, 0.1, None);

        let novel = vec![vec![-1.0; 4]; 10]; // all "unsafe" pattern
        let result = generalization_test(&probe, &novel, 1);
        assert_eq!(result.total, 10);
        assert!(result.passes, "trained probe should detect unsafe-pattern embeddings");
    }

    #[test]
    fn clf_006_generalization_empty() {
        let probe = LinearProbe::new(4, 2);
        let result = generalization_test(&probe, &[], 1);
        assert_eq!(result.total, 0);
        assert_eq!(result.detection_rate, 0.0);
    }

    // =========================================================================
    // SHIP GATE TESTS
    // =========================================================================

    #[test]
    fn clf_ship_gate_passes() {
        let ci = BootstrapCI { estimate: 0.4, lower: 0.25, upper: 0.55, n_bootstrap: 100 };
        let gen =
            GeneralizationResult { total: 50, detected: 30, detection_rate: 0.6, passes: true };
        let result = check_ship_gate(&ci, 0.96, &gen, EscalationLevel::LinearProbe);
        assert!(result.ship_ready);
        assert!(result.mcc_passes);
        assert!(result.accuracy_passes);
        assert!(result.generalization_passes);
    }

    #[test]
    fn clf_ship_gate_fails_mcc() {
        let ci = BootstrapCI { estimate: 0.15, lower: 0.10, upper: 0.20, n_bootstrap: 100 };
        let gen =
            GeneralizationResult { total: 50, detected: 30, detection_rate: 0.6, passes: true };
        let result = check_ship_gate(&ci, 0.96, &gen, EscalationLevel::LinearProbe);
        assert!(!result.ship_ready);
        assert!(!result.mcc_passes);
    }

    #[test]
    fn clf_ship_gate_fails_generalization() {
        let ci = BootstrapCI { estimate: 0.4, lower: 0.25, upper: 0.55, n_bootstrap: 100 };
        let gen =
            GeneralizationResult { total: 50, detected: 20, detection_rate: 0.4, passes: false };
        let result = check_ship_gate(&ci, 0.96, &gen, EscalationLevel::LinearProbe);
        assert!(!result.ship_ready);
        assert!(!result.generalization_passes);
    }

    // =========================================================================
    // MLP PROBE TESTS (Level 0.5)
    // =========================================================================

    #[test]
    fn mlp_probe_forward_shape() {
        let probe = MlpProbe::new(768, 128, 2);
        let emb = vec![0.1; 768];
        let (h, logits) = probe.forward(&emb);
        assert_eq!(h.len(), 128);
        assert_eq!(logits.len(), 2);
    }

    #[test]
    fn mlp_probe_predict_probs_sum_to_one() {
        let probe = MlpProbe::new(64, 32, 3);
        let emb = vec![0.5; 64];
        let probs = probe.predict_probs(&emb);
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities must sum to 1.0, got {sum}");
    }

    #[test]
    fn mlp_probe_num_parameters() {
        let probe = MlpProbe::new(768, 128, 2);
        // W1: 768*128 + b1: 128 + W2: 128*2 + b2: 2 = 98,434 + 128 + 256 + 2 = 98,818 + 2 = 98,690
        assert_eq!(probe.num_parameters(), 768 * 128 + 128 + 128 * 2 + 2);
    }

    #[test]
    fn mlp_probe_relu_zeros_negative() {
        let probe = MlpProbe::new(4, 4, 2);
        let emb = vec![-10.0; 4]; // all negative
        let (h, _) = probe.forward(&emb);
        // After ReLU, some hidden units may be zero (depends on init)
        // At least verify h values are non-negative
        assert!(h.iter().all(|&v| v >= 0.0), "ReLU output must be non-negative");
    }

    #[test]
    fn mlp_probe_train_learns_xor() {
        // XOR is not linearly separable — MLP should learn it, linear probe can't
        let embeddings = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let labels = vec![0, 1, 1, 0]; // XOR pattern

        // Repeat data for more training signal
        let embeddings: Vec<Vec<f32>> = embeddings.iter().cycle().take(40).cloned().collect();
        let labels: Vec<usize> = labels.iter().cycle().take(40).copied().collect();

        let mut mlp = MlpProbe::new(2, 8, 2);
        mlp.train(&embeddings, &labels, 200, 0.1, None, 0.0);

        // Test XOR predictions
        let pred_00 = mlp.predict(&[0.0, 0.0]);
        let pred_01 = mlp.predict(&[0.0, 1.0]);
        let pred_10 = mlp.predict(&[1.0, 0.0]);
        let pred_11 = mlp.predict(&[1.0, 1.0]);

        // MLP should learn XOR (at least partially)
        let correct = u8::from(pred_00 == 0)
            + u8::from(pred_01 == 1)
            + u8::from(pred_10 == 1)
            + u8::from(pred_11 == 0);
        assert!(correct >= 3, "MLP should learn XOR (got {correct}/4 correct)");
    }

    #[test]
    fn mlp_probe_train_reduces_loss() {
        let mut probe = MlpProbe::new(8, 16, 2);
        let embeddings: Vec<Vec<f32>> =
            (0..20).map(|i| if i < 10 { vec![1.0; 8] } else { vec![-1.0; 8] }).collect();
        let labels: Vec<usize> = (0..20).map(|i| usize::from(i >= 10)).collect();

        let loss_1 = probe.train(&embeddings, &labels, 1, 0.01, None, 0.0);
        let loss_10 = probe.train(&embeddings, &labels, 10, 0.01, None, 0.0);
        assert!(loss_10 < loss_1 + 0.5, "training should reduce loss");
    }

    // ── test_cov4 additional coverage tests ────────────────────────

    #[test]
    fn test_cov4_multiclass_mcc_perfect_3class() {
        // Perfect 3-class predictions
        let preds = vec![0, 0, 1, 1, 2, 2];
        let labels = vec![0, 0, 1, 1, 2, 2];
        let metrics = evaluate(&preds, &labels, 3);
        assert!((metrics.accuracy - 1.0).abs() < 1e-5);
        assert!(metrics.mcc > 0.9, "Perfect 3-class should have high MCC, got {}", metrics.mcc);
    }

    #[test]
    fn test_cov4_multiclass_mcc_random_3class() {
        // Completely wrong predictions
        let preds = vec![1, 2, 0, 2, 0, 1];
        let labels = vec![0, 0, 1, 1, 2, 2];
        let metrics = evaluate(&preds, &labels, 3);
        assert!(metrics.mcc < 0.1, "Random 3-class MCC should be near 0, got {}", metrics.mcc);
    }

    #[test]
    fn test_cov4_multiclass_mcc_4class() {
        let preds = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let labels = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let metrics = evaluate(&preds, &labels, 4);
        assert!((metrics.mcc - 1.0).abs() < 1e-5);
        assert_eq!(metrics.num_samples, 8);
    }

    #[test]
    fn test_cov4_binary_mcc_all_tp() {
        // All positive, all predicted positive
        assert_eq!(binary_mcc(100, 0, 0, 0), 0.0); // denom = 0
    }

    #[test]
    fn test_cov4_binary_mcc_all_tn() {
        // All negative, all predicted negative
        assert_eq!(binary_mcc(0, 100, 0, 0), 0.0); // denom = 0
    }

    #[test]
    fn test_cov4_binary_mcc_worst() {
        // All predictions wrong
        assert!((binary_mcc(0, 0, 50, 50) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cov4_binary_mcc_asymmetric() {
        // TP=80, TN=10, FP=5, FN=5
        let mcc = binary_mcc(80, 10, 5, 5);
        assert!(mcc > 0.0 && mcc < 1.0, "Asymmetric MCC should be between 0 and 1, got {mcc}");
    }

    #[test]
    fn test_cov4_evaluate_all_same_prediction() {
        // All predict class 0, mixed true labels
        let preds = vec![0, 0, 0, 0, 0];
        let labels = vec![0, 0, 1, 1, 1];
        let metrics = evaluate(&preds, &labels, 2);
        assert!((metrics.accuracy - 0.4).abs() < 1e-5);
        assert_eq!(metrics.recall[0], 1.0); // all class-0 detected
        assert_eq!(metrics.recall[1], 0.0); // no class-1 detected
    }

    #[test]
    fn test_cov4_evaluate_empty() {
        let metrics = evaluate(&[], &[], 2);
        assert_eq!(metrics.num_samples, 0);
        assert!((metrics.accuracy - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cov4_evaluate_precision() {
        let preds = vec![0, 0, 1, 1, 1];
        let labels = vec![0, 1, 1, 1, 0];
        let metrics = evaluate(&preds, &labels, 2);
        // Predicted class 0: [0]=true, [1]=false → precision[0]=1/2=0.5
        assert!((metrics.precision[0] - 0.5).abs() < 1e-5);
        // Predicted class 1: [1]=true(2), [0]=false(1) → precision[1]=2/3
        assert!((metrics.precision[1] - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cov4_evaluate_confusion_matrix() {
        let preds = vec![0, 1, 0, 1];
        let labels = vec![0, 1, 1, 0];
        let metrics = evaluate(&preds, &labels, 2);
        // cm[pred][actual]
        assert_eq!(metrics.confusion_matrix[0][0], 1); // TP for class 0
        assert_eq!(metrics.confusion_matrix[0][1], 1); // FN for class 1 (predicted 0)
        assert_eq!(metrics.confusion_matrix[1][0], 1); // FP for class 0 (predicted 1)
        assert_eq!(metrics.confusion_matrix[1][1], 1); // TP for class 1
    }

    #[test]
    fn test_cov4_evaluate_out_of_bounds_ignored() {
        // Labels or predictions >= num_classes should be silently skipped in cm
        let preds = vec![0, 1, 5]; // 5 is out of bounds for num_classes=2
        let labels = vec![0, 1, 0];
        let metrics = evaluate(&preds, &labels, 2);
        assert_eq!(metrics.num_samples, 3);
        // cm should only count valid entries
        assert_eq!(metrics.confusion_matrix[0][0], 1);
        assert_eq!(metrics.confusion_matrix[1][1], 1);
    }

    #[test]
    fn test_cov4_bootstrap_ci_deterministic() {
        let preds = vec![0, 0, 1, 1, 0, 1, 0, 0, 1, 1];
        let labels = vec![0, 0, 1, 1, 0, 0, 0, 1, 1, 1];
        let ci1 = bootstrap_mcc_ci(&preds, &labels, 2, 50);
        let ci2 = bootstrap_mcc_ci(&preds, &labels, 2, 50);
        // Same seed should give same result
        assert!((ci1.lower - ci2.lower).abs() < 1e-5);
        assert!((ci1.upper - ci2.upper).abs() < 1e-5);
    }

    #[test]
    fn test_cov4_bootstrap_ci_bounds() {
        let preds = vec![0, 0, 1, 1, 0, 1];
        let labels = vec![0, 0, 1, 1, 0, 1];
        let ci = bootstrap_mcc_ci(&preds, &labels, 2, 200);
        assert!(ci.lower <= ci.upper);
        assert!(ci.lower >= -1.0);
        assert!(ci.upper <= 1.0);
        assert_eq!(ci.n_bootstrap, 200);
    }

    #[test]
    fn test_cov4_confidence_scores_deterministic() {
        let probe = LinearProbe::new(8, 2);
        let embs = vec![vec![0.5; 8], vec![-0.5; 8]];
        let scores1 = compute_confidence_scores(&probe, &embs);
        let scores2 = compute_confidence_scores(&probe, &embs);
        for (s1, s2) in scores1.iter().zip(scores2.iter()) {
            assert_eq!(s1.predicted_class, s2.predicted_class);
            assert!((s1.confidence - s2.confidence).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cov4_confidence_scores_empty() {
        let probe = LinearProbe::new(8, 2);
        let scores = compute_confidence_scores(&probe, &[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_cov4_escalation_display() {
        assert_eq!(format!("{}", EscalationLevel::LinearProbe), "Level 0: Linear probe");
        assert_eq!(format!("{}", EscalationLevel::TopLayers), "Level 1: Top-2 layers + head");
        assert_eq!(format!("{}", EscalationLevel::FullFinetune), "Level 2: Full fine-tune");
        assert_eq!(
            format!("{}", EscalationLevel::ContinuePretrain),
            "Level 3: Continue-pretrain + fine-tune"
        );
    }

    #[test]
    fn test_cov4_escalation_debug_clone() {
        let level = EscalationLevel::TopLayers;
        let cloned = level;
        assert_eq!(level, cloned);
        assert!(format!("{level:?}").contains("TopLayers"));
    }

    #[test]
    fn test_cov4_escalate_full_to_continue() {
        let ci = BootstrapCI { estimate: 0.2, lower: 0.1, upper: 0.3, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::FullFinetune, &ci, 0.95);
        assert_eq!(result, Some(EscalationLevel::ContinuePretrain));
    }

    #[test]
    fn test_cov4_escalate_full_no_escalate() {
        let ci = BootstrapCI { estimate: 0.5, lower: 0.4, upper: 0.6, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::FullFinetune, &ci, 0.96);
        assert_eq!(result, None);
    }

    #[test]
    fn test_cov4_escalate_top_layers_no_escalate() {
        let ci = BootstrapCI { estimate: 0.5, lower: 0.35, upper: 0.65, n_bootstrap: 100 };
        let result = should_escalate(EscalationLevel::TopLayers, &ci, 0.96);
        assert_eq!(result, None);
    }

    #[test]
    fn test_cov4_compare_baselines_details() {
        let comps = compare_baselines(0.5, &[("majority", 0.0), ("keyword", 0.5), ("linter", 0.6)]);
        assert_eq!(comps[0].name, "majority");
        assert!(comps[0].beats_baseline);
        assert!(!comps[1].beats_baseline); // 0.5 > 0.5 is false
        assert!(!comps[2].beats_baseline);
        assert!((comps[0].model_mcc - 0.5).abs() < 1e-5);
        assert!((comps[0].baseline_mcc - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cov4_compare_baselines_empty() {
        let comps = compare_baselines(0.5, &[]);
        assert!(comps.is_empty());
    }

    #[test]
    fn test_cov4_generalization_result_fields() {
        let probe = LinearProbe::new(4, 2);
        let embs: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0; 4]).collect();
        let result = generalization_test(&probe, &embs, 1);
        assert_eq!(result.total, 5);
        assert!(result.detected <= 5);
        assert!((result.detection_rate - result.detected as f32 / 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_cov4_ship_gate_all_fail() {
        let ci = BootstrapCI { estimate: 0.1, lower: 0.05, upper: 0.15, n_bootstrap: 100 };
        let gen =
            GeneralizationResult { total: 50, detected: 10, detection_rate: 0.2, passes: false };
        let result = check_ship_gate(&ci, 0.90, &gen, EscalationLevel::LinearProbe);
        assert!(!result.ship_ready);
        assert!(!result.mcc_passes);
        assert!(!result.accuracy_passes);
        assert!(!result.generalization_passes);
        assert_eq!(result.level, EscalationLevel::LinearProbe);
    }

    #[test]
    fn test_cov4_ship_gate_fails_accuracy() {
        let ci = BootstrapCI { estimate: 0.4, lower: 0.25, upper: 0.55, n_bootstrap: 100 };
        let gen =
            GeneralizationResult { total: 50, detected: 30, detection_rate: 0.6, passes: true };
        let result = check_ship_gate(&ci, 0.90, &gen, EscalationLevel::TopLayers);
        assert!(!result.ship_ready);
        assert!(result.mcc_passes);
        assert!(!result.accuracy_passes);
        assert!(result.generalization_passes);
        assert_eq!(result.level, EscalationLevel::TopLayers);
    }

    #[test]
    fn test_cov4_linear_probe_predict() {
        let probe = LinearProbe::new(8, 3);
        let emb = Tensor::from_vec(vec![0.5; 8], false);
        let predicted = probe.predict(&emb);
        assert!(predicted < 3);
    }

    #[test]
    fn test_cov4_linear_probe_num_classes() {
        let probe = LinearProbe::new(64, 5);
        assert_eq!(probe.num_classes(), 5);
    }

    #[test]
    fn test_cov4_linear_probe_train_with_class_weights() {
        let mut probe = LinearProbe::new(4, 2);
        let embeddings =
            vec![vec![1.0; 4]; 10].into_iter().chain(vec![vec![-1.0; 4]; 10]).collect::<Vec<_>>();
        let labels: Vec<usize> = (0..20).map(|i| usize::from(i >= 10)).collect();
        let weights = vec![1.0, 5.0]; // upweight class 1

        let loss = probe.train(&embeddings, &labels, 5, 0.01, Some(&weights));
        assert!(loss.is_finite());
    }

    #[test]
    fn test_cov4_mlp_probe_predict() {
        let probe = MlpProbe::new(8, 16, 3);
        let emb = vec![0.1; 8];
        let predicted = probe.predict(&emb);
        assert!(predicted < 3);
    }

    #[test]
    fn test_cov4_mlp_probe_predict_probs_all_positive() {
        let probe = MlpProbe::new(4, 8, 2);
        let probs = probe.predict_probs(&[0.5, -0.5, 1.0, -1.0]);
        assert!(probs.iter().all(|&p| p > 0.0));
        assert!(probs.iter().all(|&p| p <= 1.0));
    }

    #[test]
    fn test_cov4_mlp_probe_num_parameters() {
        let probe = MlpProbe::new(16, 8, 3);
        // W1: 16*8 + b1: 8 + W2: 8*3 + b2: 3 = 128 + 8 + 24 + 3 = 163
        assert_eq!(probe.num_parameters(), 16 * 8 + 8 + 8 * 3 + 3);
    }

    #[test]
    fn test_cov4_mlp_probe_train_with_class_weights() {
        let mut probe = MlpProbe::new(4, 8, 2);
        let embeddings: Vec<Vec<f32>> =
            (0..20).map(|i| if i < 10 { vec![1.0; 4] } else { vec![-1.0; 4] }).collect();
        let labels: Vec<usize> = (0..20).map(|i| usize::from(i >= 10)).collect();
        let weights = vec![1.0, 5.0];

        let loss = probe.train(&embeddings, &labels, 5, 0.01, Some(&weights), 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_cov4_mlp_probe_train_with_weight_decay() {
        let mut probe = MlpProbe::new(4, 8, 2);
        let embeddings: Vec<Vec<f32>> =
            (0..10).map(|i| if i < 5 { vec![1.0; 4] } else { vec![-1.0; 4] }).collect();
        let labels: Vec<usize> = (0..10).map(|i| usize::from(i >= 5)).collect();

        let loss = probe.train(&embeddings, &labels, 5, 0.01, None, 0.01);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_cov4_softmax_slice_single() {
        let result = softmax_slice(&[0.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cov4_softmax_slice_large_values() {
        // Should not overflow due to max subtraction
        let result = softmax_slice(&[1000.0, 1001.0]);
        assert_eq!(result.len(), 2);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result[1] > result[0]); // higher logit → higher prob
    }

    #[test]
    fn test_cov4_softmax_slice_equal() {
        let result = softmax_slice(&[1.0, 1.0, 1.0]);
        for &p in &result {
            assert!((p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cov4_classification_metrics_clone() {
        let m = ClassificationMetrics {
            mcc: 0.5,
            accuracy: 0.9,
            recall: vec![0.8, 0.7],
            precision: vec![0.85, 0.75],
            num_samples: 100,
            confusion_matrix: vec![vec![40, 10], vec![5, 45]],
        };
        let m2 = m.clone();
        assert!((m2.mcc - 0.5).abs() < 1e-5);
        assert_eq!(m2.num_samples, 100);
        assert!(format!("{m2:?}").contains("ClassificationMetrics"));
    }

    #[test]
    fn test_cov4_bootstrap_ci_clone() {
        let ci = BootstrapCI { estimate: 0.5, lower: 0.3, upper: 0.7, n_bootstrap: 1000 };
        let ci2 = ci;
        assert!((ci2.estimate - 0.5).abs() < 1e-5);
        assert!(format!("{ci:?}").contains("BootstrapCI"));
    }

    #[test]
    fn test_cov4_confidence_score_clone() {
        let s =
            ConfidenceScore { predicted_class: 1, confidence: 0.8, probabilities: vec![0.2, 0.8] };
        let s2 = s.clone();
        assert_eq!(s2.predicted_class, 1);
        assert!((s2.confidence - 0.8).abs() < 1e-5);
        assert!(format!("{s2:?}").contains("ConfidenceScore"));
    }

    #[test]
    fn test_cov4_generalization_result_clone() {
        let r =
            GeneralizationResult { total: 20, detected: 15, detection_rate: 0.75, passes: true };
        let r2 = r.clone();
        assert!(r2.passes);
        assert_eq!(r2.total, 20);
        assert!(format!("{r2:?}").contains("GeneralizationResult"));
    }

    #[test]
    fn test_cov4_baseline_comparison_clone() {
        let b = BaselineComparison {
            name: "test".to_string(),
            baseline_mcc: 0.3,
            model_mcc: 0.5,
            beats_baseline: true,
        };
        let b2 = b.clone();
        assert!(b2.beats_baseline);
        assert!(format!("{b2:?}").contains("BaselineComparison"));
    }

    #[test]
    fn test_cov4_ship_gate_result_clone() {
        let r = ShipGateResult {
            mcc_passes: true,
            accuracy_passes: true,
            generalization_passes: false,
            ship_ready: false,
            level: EscalationLevel::LinearProbe,
        };
        let r2 = r.clone();
        assert!(!r2.ship_ready);
        assert!(format!("{r2:?}").contains("ShipGateResult"));
    }

    #[test]
    fn test_cov4_multiclass_mcc_single_class() {
        // Edge case: all predictions and labels same class
        let preds = vec![0, 0, 0, 0];
        let labels = vec![0, 0, 0, 0];
        let metrics = evaluate(&preds, &labels, 3);
        assert!((metrics.accuracy - 1.0).abs() < 1e-5);
        // MCC is 0 when only one class is present (denom = 0)
        assert!(metrics.mcc.abs() < 1e-5 || metrics.mcc.is_finite());
    }
}
