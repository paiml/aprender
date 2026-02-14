//! Weak Supervision and Label Model.
//!
//! Combines noisy labels from multiple labeling functions to produce
//! high-quality training labels without manual annotation.
//!
//! # Approach
//!
//! Based on the Snorkel paradigm:
//! 1. Define labeling functions (LFs) that provide noisy labels
//! 2. Learn a label model that weighs LF outputs
//! 3. Generate probabilistic labels for training
//!
//! # Reference
//!
//! - Ratner, A., et al. (2017). Snorkel: Rapid Training Data Creation with
//!   Weak Supervision. VLDB.
//! - Ratner, A., et al. (2019). Training Complex Models with Multi-Task
//!   Weak Supervision. AAAI.

use crate::primitives::Vector;

/// Abstain value for labeling functions (no label provided).
pub const ABSTAIN: i32 = -1;

/// Labeling function output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LFOutput {
    /// No label for this sample
    Abstain,
    /// Assigned class label
    Label(usize),
}

impl LFOutput {
    /// Convert to i32 representation (-1 for abstain).
    #[must_use]
    pub fn to_i32(&self) -> i32 {
        match self {
            LFOutput::Abstain => ABSTAIN,
            LFOutput::Label(c) => *c as i32,
        }
    }

    /// Create from i32 (-1 means abstain).
    #[must_use]
    pub fn from_i32(v: i32) -> Self {
        if v < 0 {
            LFOutput::Abstain
        } else {
            LFOutput::Label(v as usize)
        }
    }
}

/// Label Model for combining weak supervision sources.
///
/// Learns the accuracy and coverage of labeling functions to produce
/// optimal probabilistic labels.
///
/// # Example
///
/// ```ignore
/// use aprender::weak_supervision::{LabelModel, LFOutput};
///
/// // Matrix of LF outputs: [n_samples x n_labeling_functions]
/// let lf_matrix = vec![
///     vec![LFOutput::Label(1), LFOutput::Abstain, LFOutput::Label(1)],
///     vec![LFOutput::Label(0), LFOutput::Label(0), LFOutput::Abstain],
///     // ...
/// ];
///
/// let mut model = LabelModel::new(2, 3); // 2 classes, 3 LFs
/// model.fit(&lf_matrix, 100, 0.01);
///
/// let probs = model.predict_proba(&lf_matrix);
/// ```
#[derive(Debug, Clone)]
pub struct LabelModel {
    /// Number of classes
    n_classes: usize,
    /// Number of labeling functions
    n_lfs: usize,
    /// Estimated accuracy per LF per class [`n_lfs` x `n_classes`]
    accuracies: Vec<Vec<f32>>,
    /// Prior class probabilities [`n_classes`]
    class_priors: Vec<f32>,
}

impl LabelModel {
    /// Create a new label model.
    ///
    /// # Arguments
    ///
    /// * `n_classes` - Number of output classes
    /// * `n_lfs` - Number of labeling functions
    #[must_use]
    pub fn new(n_classes: usize, n_lfs: usize) -> Self {
        Self {
            n_classes,
            n_lfs,
            accuracies: vec![vec![0.7; n_classes]; n_lfs], // Initial estimate
            class_priors: vec![1.0 / n_classes as f32; n_classes],
        }
    }

    /// Fit the label model using EM algorithm.
    ///
    /// # Arguments
    ///
    /// * `lf_matrix` - Matrix of LF outputs [`n_samples` x `n_lfs`]
    /// * `n_epochs` - Number of EM iterations
    /// * `lr` - Learning rate for parameter updates
    pub fn fit(&mut self, lf_matrix: &[Vec<LFOutput>], n_epochs: usize, lr: f32) {
        let n_samples = lf_matrix.len();
        if n_samples == 0 {
            return;
        }

        // EM Algorithm
        for _ in 0..n_epochs {
            // E-step: Estimate class probabilities for each sample
            let mut class_posteriors = vec![vec![0.0; self.n_classes]; n_samples];

            for (i, lf_outputs) in lf_matrix.iter().enumerate() {
                // Compute posterior for each class
                let mut log_probs = vec![0.0_f32; self.n_classes];

                for (c, log_prob) in log_probs.iter_mut().enumerate() {
                    // Prior
                    *log_prob = self.class_priors[c].ln();

                    // Likelihood from each LF
                    for (j, &output) in lf_outputs.iter().enumerate() {
                        if let LFOutput::Label(lf_class) = output {
                            // Accuracy-based likelihood
                            if lf_class == c {
                                *log_prob += self.accuracies[j][c].ln();
                            } else {
                                // Probability of wrong label
                                let wrong_prob =
                                    (1.0 - self.accuracies[j][c]) / (self.n_classes - 1) as f32;
                                *log_prob += wrong_prob.max(1e-10).ln();
                            }
                        }
                        // Abstain doesn't contribute
                    }
                }

                // Convert to probabilities
                let max_log = log_probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_probs: Vec<f32> = log_probs.iter().map(|&l| (l - max_log).exp()).collect();
                let sum: f32 = exp_probs.iter().sum();

                for (c, exp_prob) in exp_probs.into_iter().enumerate() {
                    class_posteriors[i][c] = exp_prob / sum;
                }
            }

            // M-step: Update parameters
            // Update class priors
            let mut new_priors = vec![0.0; self.n_classes];
            for posterior in &class_posteriors {
                for (c, &p) in posterior.iter().enumerate() {
                    new_priors[c] += p;
                }
            }
            for prior in &mut new_priors {
                *prior /= n_samples as f32;
            }

            for (c, &new_prior) in new_priors.iter().enumerate() {
                self.class_priors[c] = (1.0 - lr) * self.class_priors[c] + lr * new_prior;
            }

            // Update LF accuracies
            for j in 0..self.n_lfs {
                #[allow(clippy::needless_range_loop)] // c used as both index and comparison value
                for c in 0..self.n_classes {
                    let mut correct_count = 0.0;
                    let mut total_count = 0.0;

                    for (i, lf_outputs) in lf_matrix.iter().enumerate() {
                        if let LFOutput::Label(lf_class) = lf_outputs[j] {
                            let posterior = class_posteriors[i][c];
                            total_count += posterior;
                            if lf_class == c {
                                correct_count += posterior;
                            }
                        }
                    }

                    if total_count > 1e-6 {
                        let new_acc = (correct_count / total_count).clamp(0.1, 0.99);
                        self.accuracies[j][c] = (1.0 - lr) * self.accuracies[j][c] + lr * new_acc;
                    }
                }
            }
        }
    }

    /// Get probabilistic labels for samples.
    #[must_use]
    pub fn predict_proba(&self, lf_matrix: &[Vec<LFOutput>]) -> Vec<Vector<f32>> {
        lf_matrix
            .iter()
            .map(|lf_outputs| {
                let mut log_probs = vec![0.0_f32; self.n_classes];

                for (c, log_prob) in log_probs.iter_mut().enumerate() {
                    *log_prob = self.class_priors[c].ln();

                    for (j, &output) in lf_outputs.iter().enumerate() {
                        if let LFOutput::Label(lf_class) = output {
                            if lf_class == c {
                                *log_prob += self.accuracies[j][c].ln();
                            } else {
                                let wrong_prob =
                                    (1.0 - self.accuracies[j][c]) / (self.n_classes - 1) as f32;
                                *log_prob += wrong_prob.max(1e-10).ln();
                            }
                        }
                    }
                }

                // Softmax
                let max_log = log_probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_probs: Vec<f32> = log_probs.iter().map(|&l| (l - max_log).exp()).collect();
                let sum: f32 = exp_probs.iter().sum();
                let probs: Vec<f32> = exp_probs.iter().map(|&e| e / sum).collect();

                Vector::from_slice(&probs)
            })
            .collect()
    }

    /// Get hard label predictions (argmax of probabilities).
    #[must_use]
    pub fn predict(&self, lf_matrix: &[Vec<LFOutput>]) -> Vec<usize> {
        self.predict_proba(lf_matrix)
            .into_iter()
            .map(|probs| {
                probs
                    .as_slice()
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx)
            })
            .collect()
    }

    /// Get estimated LF accuracies.
    #[must_use]
    pub fn get_accuracies(&self) -> &[Vec<f32>] {
        &self.accuracies
    }

    /// Get estimated class priors.
    #[must_use]
    pub fn get_class_priors(&self) -> &[f32] {
        &self.class_priors
    }

    /// Get coverage of each LF (fraction of samples with non-abstain votes).
    #[must_use]
    pub fn get_lf_coverage(&self, lf_matrix: &[Vec<LFOutput>]) -> Vec<f32> {
        let n_samples = lf_matrix.len() as f32;
        if n_samples == 0.0 {
            return vec![0.0; self.n_lfs];
        }

        let mut coverage = vec![0.0; self.n_lfs];
        for outputs in lf_matrix {
            for (j, &output) in outputs.iter().enumerate() {
                if output != LFOutput::Abstain {
                    coverage[j] += 1.0;
                }
            }
        }
        for c in &mut coverage {
            *c /= n_samples;
        }
        coverage
    }
}

/// Confident Learning for label noise detection.
///
/// Identifies mislabeled samples by finding confident joint distribution
/// discrepancies between noisy labels and model predictions.
///
/// # Reference
///
/// - Northcutt, C., et al. (2021). Confident Learning: Estimating Uncertainty
///   in Dataset Labels. JAIR.
#[derive(Debug, Clone)]
pub struct ConfidentLearning {
    /// Number of classes
    n_classes: usize,
    /// Confidence threshold for filtering
    threshold: f32,
}

impl ConfidentLearning {
    /// Create new confident learning instance.
    #[must_use]
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            threshold: 0.5,
        }
    }

    /// Create with custom threshold.
    #[must_use]
    pub fn with_threshold(n_classes: usize, threshold: f32) -> Self {
        Self {
            n_classes,
            threshold,
        }
    }

    /// Find potentially mislabeled samples.
    ///
    /// # Arguments
    ///
    /// * `labels` - Noisy labels
    /// * `pred_probs` - Model-predicted class probabilities
    ///
    /// # Returns
    ///
    /// Indices of samples likely to be mislabeled.
    #[must_use]
    pub fn find_label_issues(&self, labels: &[usize], pred_probs: &[Vector<f32>]) -> Vec<usize> {
        let mut issues = Vec::new();

        for (i, (&label, probs)) in labels.iter().zip(pred_probs.iter()).enumerate() {
            let predicted_class = probs
                .as_slice()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);

            let label_prob = probs[label];

            // Issue if predicted class differs and label confidence is low
            if predicted_class != label && label_prob < self.threshold {
                issues.push(i);
            }
        }

        issues
    }

    /// Compute confident joint matrix.
    ///
    /// Element (i,j) represents count of samples with noisy label i
    /// and confident predicted label j.
    #[must_use]
    pub fn compute_confident_joint(
        &self,
        labels: &[usize],
        pred_probs: &[Vector<f32>],
    ) -> Vec<Vec<f32>> {
        let mut joint = vec![vec![0.0; self.n_classes]; self.n_classes];

        // Per-class thresholds (average predicted probability for each class)
        let mut class_thresholds = vec![0.0_f32; self.n_classes];
        let mut class_counts = vec![0usize; self.n_classes];

        for probs in pred_probs {
            let predicted = probs
                .as_slice()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);

            class_thresholds[predicted] += probs[predicted];
            class_counts[predicted] += 1;
        }

        for (c, thresh) in class_thresholds.iter_mut().enumerate() {
            if class_counts[c] > 0 {
                *thresh /= class_counts[c] as f32;
            } else {
                *thresh = 0.5;
            }
        }

        // Build joint
        for (&noisy_label, probs) in labels.iter().zip(pred_probs.iter()) {
            let predicted = probs
                .as_slice()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);

            // Only count if confident
            if probs[predicted] >= class_thresholds[predicted] {
                joint[noisy_label][predicted] += 1.0;
            }
        }

        // Normalize to get probabilities
        let total: f32 = joint.iter().flat_map(|row| row.iter()).sum();
        if total > 0.0 {
            for row in &mut joint {
                for cell in row {
                    *cell /= total;
                }
            }
        }

        joint
    }

    /// Estimate noise transition matrix P(noisy=i | true=j).
    #[must_use]
    pub fn estimate_noise_matrix(
        &self,
        labels: &[usize],
        pred_probs: &[Vector<f32>],
    ) -> Vec<Vec<f32>> {
        let joint = self.compute_confident_joint(labels, pred_probs);

        // Normalize each column to get conditional probabilities
        let mut noise_matrix = vec![vec![0.0; self.n_classes]; self.n_classes];

        for j in 0..self.n_classes {
            let col_sum: f32 = (0..self.n_classes).map(|i| joint[i][j]).sum();
            if col_sum > 0.0 {
                for i in 0..self.n_classes {
                    noise_matrix[i][j] = joint[i][j] / col_sum;
                }
            } else {
                // Default to identity
                noise_matrix[j][j] = 1.0;
            }
        }

        noise_matrix
    }
}

#[cfg(test)]
#[path = "weak_supervision_tests.rs"]
mod tests;
