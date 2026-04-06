//! Multi-class classification metrics

use super::average::Average;
use super::confusion::ConfusionMatrix;

/// Multi-class classification metrics
#[derive(Clone, Debug)]
pub struct MultiClassMetrics {
    /// Per-class precision
    pub precision: Vec<f64>,
    /// Per-class recall
    pub recall: Vec<f64>,
    /// Per-class F1 score
    pub f1: Vec<f64>,
    /// Per-class support (count)
    pub support: Vec<usize>,
    /// Number of classes
    pub n_classes: usize,
}

impl MultiClassMetrics {
    /// Compute metrics from confusion matrix
    pub fn from_confusion_matrix(cm: &ConfusionMatrix) -> Self {
        let n_classes = cm.n_classes();
        let mut precision = Vec::with_capacity(n_classes);
        let mut recall = Vec::with_capacity(n_classes);
        let mut f1 = Vec::with_capacity(n_classes);
        let mut support = Vec::with_capacity(n_classes);

        for class in 0..n_classes {
            let tp = cm.true_positives(class) as f64;
            let fp = cm.false_positives(class) as f64;
            let fn_ = cm.false_negatives(class) as f64;

            let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };

            precision.push(p);
            recall.push(r);
            f1.push(f);
            support.push(cm.support(class));
        }

        Self { precision, recall, f1, support, n_classes }
    }

    /// Compute from predictions and ground truth
    pub fn from_predictions(y_pred: &[usize], y_true: &[usize]) -> Self {
        let cm = ConfusionMatrix::from_predictions(y_pred, y_true);
        Self::from_confusion_matrix(&cm)
    }

    /// Get averaged precision
    pub fn precision_avg(&self, average: Average) -> f64 {
        self.average_metric(&self.precision, average)
    }

    /// Get averaged recall
    pub fn recall_avg(&self, average: Average) -> f64 {
        self.average_metric(&self.recall, average)
    }

    /// Get averaged F1
    pub fn f1_avg(&self, average: Average) -> f64 {
        self.average_metric(&self.f1, average)
    }

    fn average_metric(&self, values: &[f64], average: Average) -> f64 {
        match average {
            Average::Macro => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            Average::Micro => {
                // For micro-averaging, we need to recalculate from totals
                // Currently uses macro-average as fallback (FUTURE: full micro-avg)
                self.average_metric(values, Average::Macro)
            }
            Average::Weighted => {
                let total_support: usize = self.support.iter().sum();
                if total_support == 0 {
                    return 0.0;
                }
                values.iter().zip(self.support.iter()).map(|(&v, &s)| v * s as f64).sum::<f64>()
                    / total_support as f64
            }
            Average::None => {
                // Return macro as default for single value
                self.average_metric(values, Average::Macro)
            }
        }
    }
}
