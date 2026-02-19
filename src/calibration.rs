//! Model calibration for confidence estimation.
//!
//! Calibration ensures model confidence matches actual accuracy.
//!
//! # Methods
//! - Temperature Scaling: Single parameter post-hoc calibration
//! - Platt Scaling: Logistic regression on logits
//! - Expected Calibration Error (ECE): Calibration metric

use crate::primitives::Vector;

/// Temperature scaling calibrator.
/// Divides logits by learned temperature T before softmax.
#[derive(Debug, Clone)]
pub struct TemperatureScaling {
    temperature: f32,
}

impl Default for TemperatureScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl TemperatureScaling {
    #[must_use]
    pub fn new() -> Self {
        Self { temperature: 1.0 }
    }

    /// Fit temperature on validation set using grid search.
    pub fn fit(&mut self, logits: &[Vector<f32>], labels: &[usize]) {
        let mut best_temp = 1.0;
        let mut best_nll = f32::INFINITY;

        for t in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] {
            let nll = Self::compute_nll(logits, labels, t);
            if nll < best_nll {
                best_nll = nll;
                best_temp = t;
            }
        }
        self.temperature = best_temp;
    }

    fn compute_nll(logits: &[Vector<f32>], labels: &[usize], temp: f32) -> f32 {
        let mut total_nll = 0.0;
        for (logit, &label) in logits.iter().zip(labels.iter()) {
            let scaled: Vec<f32> = logit.as_slice().iter().map(|&x| x / temp).collect();
            let probs = softmax(&scaled);
            total_nll -= probs[label].max(1e-10).ln();
        }
        total_nll / logits.len() as f32
    }

    /// Calibrate logits by dividing by temperature.
    #[must_use]
    pub fn calibrate(&self, logits: &Vector<f32>) -> Vector<f32> {
        let scaled: Vec<f32> = logits
            .as_slice()
            .iter()
            .map(|&x| x / self.temperature)
            .collect();
        Vector::from_slice(&scaled)
    }

    /// Get calibrated probabilities.
    #[must_use]
    pub fn predict_proba(&self, logits: &Vector<f32>) -> Vector<f32> {
        let calibrated = self.calibrate(logits);
        Vector::from_slice(&softmax(calibrated.as_slice()))
    }

    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

/// Platt scaling: fits logistic regression on logits.
#[derive(Debug, Clone)]
pub struct PlattScaling {
    a: f32,
    b: f32,
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl PlattScaling {
    #[must_use]
    pub fn new() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// Fit Platt scaling parameters using gradient descent.
    pub fn fit(&mut self, logits: &[f32], labels: &[bool]) {
        let mut a = 1.0_f32;
        let mut b = 0.0_f32;
        let lr = 0.01;

        for _ in 0..1000 {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;

            for (&logit, &label) in logits.iter().zip(labels.iter()) {
                let p = sigmoid(a * logit + b);
                let y = if label { 1.0 } else { 0.0 };
                let diff = p - y;
                grad_a += diff * logit;
                grad_b += diff;
            }

            a -= lr * grad_a / logits.len() as f32;
            b -= lr * grad_b / logits.len() as f32;
        }

        self.a = a;
        self.b = b;
    }

    /// Get calibrated probability.
    #[must_use]
    pub fn predict_proba(&self, logit: f32) -> f32 {
        sigmoid(self.a * logit + self.b)
    }

    #[must_use]
    pub fn params(&self) -> (f32, f32) {
        (self.a, self.b)
    }
}

/// Expected Calibration Error (ECE) - measures calibration quality.
#[provable_contracts_macros::contract("calibration-v1", equation = "expected_calibration_error")]
#[must_use]
pub fn expected_calibration_error(predictions: &[f32], labels: &[bool], n_bins: usize) -> f32 {
    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_correct = vec![0.0; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        let bin = ((pred * n_bins as f32) as usize).min(n_bins - 1);
        bin_sums[bin] += pred;
        bin_correct[bin] += if label { 1.0 } else { 0.0 };
        bin_counts[bin] += 1;
    }

    let n = predictions.len() as f32;
    let mut ece = 0.0;

    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            let avg_conf = bin_sums[i] / bin_counts[i] as f32;
            let avg_acc = bin_correct[i] / bin_counts[i] as f32;
            ece += (bin_counts[i] as f32 / n) * (avg_conf - avg_acc).abs();
        }
    }
    ece
}

/// Maximum Calibration Error (MCE).
#[provable_contracts_macros::contract("calibration-v1", equation = "maximum_calibration_error")]
#[must_use]
pub fn maximum_calibration_error(predictions: &[f32], labels: &[bool], n_bins: usize) -> f32 {
    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_correct = vec![0.0; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        let bin = ((pred * n_bins as f32) as usize).min(n_bins - 1);
        bin_sums[bin] += pred;
        bin_correct[bin] += if label { 1.0 } else { 0.0 };
        bin_counts[bin] += 1;
    }

    let mut mce = 0.0_f32;
    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            let avg_conf = bin_sums[i] / bin_counts[i] as f32;
            let avg_acc = bin_correct[i] / bin_counts[i] as f32;
            mce = mce.max((avg_conf - avg_acc).abs());
        }
    }
    mce
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// ONE PATH: Delegates to `nn::functional::sigmoid_scalar` (UCBD §4).
fn sigmoid(x: f32) -> f32 {
    crate::nn::functional::sigmoid_scalar(x)
}

/// Isotonic Regression calibrator.
///
/// Non-parametric calibration method that fits a monotonically increasing
/// step function to the data. Guaranteed to be monotonic.
///
/// # Reference
///
/// - Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into
///   accurate multiclass probability estimates.
#[derive(Debug, Clone)]
pub struct IsotonicRegression {
    /// Threshold values (sorted)
    pub thresholds: Vec<f32>,
    /// Calibrated values at each threshold
    pub values: Vec<f32>,
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl IsotonicRegression {
    /// Create a new isotonic regression calibrator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            thresholds: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Fit isotonic regression on predictions and labels.
    ///
    /// Uses Pool Adjacent Violators (PAV) algorithm.
    pub fn fit(&mut self, predictions: &[f32], labels: &[bool]) {
        assert_eq!(predictions.len(), labels.len());

        // Sort by predictions
        let mut pairs: Vec<(f32, f32)> = predictions
            .iter()
            .zip(labels.iter())
            .map(|(&p, &l)| (p, if l { 1.0 } else { 0.0 }))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if pairs.is_empty() {
            return;
        }

        // PAV algorithm: merge adjacent violators
        let mut blocks: Vec<(f32, f32, usize)> = Vec::new(); // (x_min, sum, count)

        for (x, y) in pairs {
            blocks.push((x, y, 1));

            // While last two blocks violate monotonicity, merge them
            while blocks.len() >= 2 {
                let len = blocks.len();
                let avg_last = blocks[len - 1].1 / blocks[len - 1].2 as f32;
                let avg_prev = blocks[len - 2].1 / blocks[len - 2].2 as f32;

                if avg_last < avg_prev {
                    // Merge: keep x_min from prev, sum totals
                    // Safe: we check blocks.len() >= 2 in the while condition
                    let last = blocks
                        .pop()
                        .expect("blocks guaranteed non-empty by while condition");
                    let prev = blocks
                        .pop()
                        .expect("blocks guaranteed to have 2+ elements by while condition");
                    blocks.push((prev.0, prev.1 + last.1, prev.2 + last.2));
                } else {
                    break;
                }
            }
        }

        // Extract thresholds and values
        self.thresholds.clear();
        self.values.clear();

        for (x_min, sum, count) in blocks {
            self.thresholds.push(x_min);
            self.values.push(sum / count as f32);
        }
    }

    /// Get calibrated probability for a prediction.
    #[must_use]
    pub fn predict(&self, prediction: f32) -> f32 {
        if self.thresholds.is_empty() {
            return prediction;
        }

        // Binary search to find appropriate block
        let idx = self.thresholds.partition_point(|&t| t <= prediction);

        if idx == 0 {
            self.values[0]
        } else if idx >= self.values.len() {
            self.values[self.values.len() - 1]
        } else {
            // Linear interpolation between blocks
            let t0 = self.thresholds[idx - 1];
            let t1 = self.thresholds[idx];
            let v0 = self.values[idx - 1];
            let v1 = self.values[idx];

            if (t1 - t0).abs() < 1e-10 {
                v0
            } else {
                let alpha = (prediction - t0) / (t1 - t0);
                v0 + alpha * (v1 - v0)
            }
        }
    }

    /// Get calibrated probabilities for multiple predictions.
    #[must_use]
    pub fn predict_batch(&self, predictions: &[f32]) -> Vec<f32> {
        predictions.iter().map(|&p| self.predict(p)).collect()
    }
}

/// Generate reliability diagram data.
///
/// Returns (`mean_predicted_prob`, `fraction_positive`) for each bin.
#[must_use]
pub fn reliability_diagram(predictions: &[f32], labels: &[bool], n_bins: usize) -> Vec<(f32, f32)> {
    let mut bins: Vec<(f32, f32, usize)> = vec![(0.0, 0.0, 0); n_bins]; // (sum_pred, sum_pos, count)

    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        let bin = ((pred * n_bins as f32) as usize).min(n_bins - 1);
        bins[bin].0 += pred;
        bins[bin].1 += if label { 1.0 } else { 0.0 };
        bins[bin].2 += 1;
    }

    bins.iter()
        .enumerate()
        .map(|(i, &(sum_pred, sum_pos, count))| {
            if count > 0 {
                (sum_pred / count as f32, sum_pos / count as f32)
            } else {
                // Empty bin: use bin center, 0.0 accuracy
                let center = (i as f32 + 0.5) / n_bins as f32;
                (center, 0.0)
            }
        })
        .collect()
}

/// Brier score - measures calibration quality.
///
/// ```text
/// Brier = (1/n) * Σ(p_i - y_i)²
/// ```
///
/// Lower is better. Perfect calibration = 0.
#[must_use]
pub fn brier_score(predictions: &[f32], labels: &[bool]) -> f32 {
    let n = predictions.len() as f32;
    predictions
        .iter()
        .zip(labels.iter())
        .map(|(&p, &l)| {
            let y = if l { 1.0 } else { 0.0 };
            (p - y).powi(2)
        })
        .sum::<f32>()
        / n
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;
