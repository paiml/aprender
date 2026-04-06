//! Classification metrics: Accuracy, Precision, Recall, F1
//!
//! Thresholding (continuous predictions → discrete labels) is entrenar's concern.
//! Metric computation on discrete labels delegates to `aprender::metrics::classification`.

use crate::Tensor;

use super::Metric;

/// Convert continuous predictions and targets to discrete binary labels.
///
/// Entrenar's training concern: model outputs are continuous (logits/probabilities),
/// so thresholding is part of evaluation. After thresholding, the discrete labels
/// can be passed to aprender for metric computation.
fn threshold_to_labels(
    predictions: &Tensor,
    targets: &Tensor,
    threshold: f32,
) -> (Vec<usize>, Vec<usize>) {
    let y_pred: Vec<usize> =
        predictions.data().iter().map(|&p| usize::from(p >= threshold)).collect();
    let y_true: Vec<usize> = targets.data().iter().map(|&t| usize::from(t >= 0.5)).collect();
    (y_pred, y_true)
}

/// Accuracy metric for classification
///
/// For binary classification: fraction of correct predictions
/// For multi-class: fraction where argmax(pred) == argmax(target)
///
/// # Example
///
/// ```
/// use entrenar::train::{Accuracy, Metric};
/// use entrenar::Tensor;
///
/// let metric = Accuracy::new(0.5);  // threshold for binary
/// let pred = Tensor::from_vec(vec![0.9, 0.2, 0.8], false);
/// let target = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);
///
/// let acc = metric.compute(&pred, &target);
/// assert_eq!(acc, 1.0);  // All correct
/// ```
#[derive(Debug, Clone)]
pub struct Accuracy {
    /// Threshold for binary classification
    pub(crate) threshold: f32,
}

impl Accuracy {
    /// Create new accuracy metric with given threshold for binary classification
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Create accuracy metric with default threshold of 0.5
    pub fn default_threshold() -> Self {
        Self::new(0.5)
    }
}

impl Default for Accuracy {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for Accuracy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        if predictions.is_empty() {
            return 0.0;
        }

        // Threshold to discrete labels (entrenar's concern), then delegate to aprender
        let (y_pred, y_true) = threshold_to_labels(predictions, targets, self.threshold);
        aprender::metrics::classification::accuracy(&y_pred, &y_true)
    }

    fn name(&self) -> &'static str {
        "Accuracy"
    }
}

/// Precision metric (true positives / predicted positives)
///
/// # Example
///
/// ```
/// use entrenar::train::{Precision, Metric};
/// use entrenar::Tensor;
///
/// let metric = Precision::new(0.5);
/// let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2], false);
/// let target = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);
///
/// let prec = metric.compute(&pred, &target);
/// assert_eq!(prec, 0.5);  // 1 TP / 2 predicted positives
/// ```
#[derive(Debug, Clone)]
pub struct Precision {
    pub(crate) threshold: f32,
}

impl Precision {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for Precision {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        // Threshold to discrete labels (entrenar), count from labels
        // Note: aprender's precision() uses macro/micro/weighted averaging which
        // differs from binary positive-class precision. We threshold via entrenar,
        // then compute TP/FP from discrete labels.
        let (y_pred, y_true) = threshold_to_labels(predictions, targets, self.threshold);

        let mut true_positives = 0usize;
        let mut predicted_positives = 0usize;

        for (&p, &t) in y_pred.iter().zip(y_true.iter()) {
            if p == 1 {
                predicted_positives += 1;
                if t == 1 {
                    true_positives += 1;
                }
            }
        }

        if predicted_positives == 0 {
            return 0.0;
        }

        true_positives as f32 / predicted_positives as f32
    }

    fn name(&self) -> &'static str {
        "Precision"
    }
}

/// Recall metric (true positives / actual positives)
///
/// # Example
///
/// ```
/// use entrenar::train::{Recall, Metric};
/// use entrenar::Tensor;
///
/// let metric = Recall::new(0.5);
/// let pred = Tensor::from_vec(vec![0.9, 0.2, 0.8], false);
/// let target = Tensor::from_vec(vec![1.0, 1.0, 0.0], false);
///
/// let rec = metric.compute(&pred, &target);
/// assert_eq!(rec, 0.5);  // 1 TP / 2 actual positives
/// ```
#[derive(Debug, Clone)]
pub struct Recall {
    pub(crate) threshold: f32,
}

impl Recall {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Default for Recall {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for Recall {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        // Threshold to discrete labels (entrenar), count from labels
        let (y_pred, y_true) = threshold_to_labels(predictions, targets, self.threshold);

        let mut true_positives = 0usize;
        let mut actual_positives = 0usize;

        for (&p, &t) in y_pred.iter().zip(y_true.iter()) {
            if t == 1 {
                actual_positives += 1;
                if p == 1 {
                    true_positives += 1;
                }
            }
        }

        if actual_positives == 0 {
            return 0.0;
        }

        true_positives as f32 / actual_positives as f32
    }

    fn name(&self) -> &'static str {
        "Recall"
    }
}

/// F1 Score (harmonic mean of precision and recall)
///
/// F1 = 2 * (precision * recall) / (precision + recall)
///
/// # Example
///
/// ```
/// use entrenar::train::{F1Score, Metric};
/// use entrenar::Tensor;
///
/// let metric = F1Score::new(0.5);
/// let pred = Tensor::from_vec(vec![0.9, 0.8, 0.2, 0.1], false);
/// let target = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], false);
///
/// let f1 = metric.compute(&pred, &target);
/// assert!(f1 > 0.0 && f1 <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct F1Score {
    precision: Precision,
    recall: Recall,
}

impl F1Score {
    pub fn new(threshold: f32) -> Self {
        Self { precision: Precision::new(threshold), recall: Recall::new(threshold) }
    }
}

impl Default for F1Score {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Metric for F1Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        let precision = self.precision.compute(predictions, targets);
        let recall = self.recall.compute(predictions, targets);

        if precision + recall == 0.0 {
            return 0.0;
        }

        2.0 * (precision * recall) / (precision + recall)
    }

    fn name(&self) -> &'static str {
        "F1"
    }
}
