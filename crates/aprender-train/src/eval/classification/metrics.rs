//! Multi-class classification metrics

use super::average::Average;
use super::confusion::ConfusionMatrix;

/// Average precomputed per-class F1 values over explicit class indices.
///
/// Returns `None` for an empty selection or an out-of-range class index.
#[must_use]
pub fn f1_average_for_classes(f1: &[f64], classes: &[usize]) -> Option<f64> {
    if classes.is_empty() || classes.iter().any(|&class| class >= f1.len()) {
        return None;
    }
    let total = classes.iter().map(|&class| f1[class]).sum::<f64>();
    Some(total / classes.len() as f64)
}

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
    ///
    /// The class count is INFERRED from the observed indices (`max + 1`). A class that appears
    /// in neither vector is therefore absent from the result — see
    /// [`Self::from_predictions_with_min_classes`] when the declared label map is known and a
    /// zero-support class must still be represented.
    pub fn from_predictions(y_pred: &[usize], y_true: &[usize]) -> Self {
        let cm = ConfusionMatrix::from_predictions(y_pred, y_true);
        Self::from_confusion_matrix(&cm)
    }

    /// Compute from predictions and ground truth over AT LEAST `min_classes` classes.
    ///
    /// # Why the declared size has to be passed in
    ///
    /// [`Self::from_predictions`] infers the class count from the data, which is right when
    /// nothing else knows it. A benchmark row does know it: the head's ordered label map is the
    /// authority, and a split in which the last class happens to have zero support would
    /// otherwise silently produce a SHORTER metric vector — so `f1[2]` would mean `favor` on one
    /// row of a results table and be out of range on the next, and an official score selecting
    /// class 2 would report "not computable" for a class that merely did not occur.
    ///
    /// With the declared size supplied, an absent class is present and scores the shipped
    /// zero-division value (`0.0`), which is the convention [`Self::from_confusion_matrix`]
    /// already applies to every degenerate precision, recall and F1.
    pub fn from_predictions_with_min_classes(
        y_pred: &[usize],
        y_true: &[usize],
        min_classes: usize,
    ) -> Self {
        let cm = ConfusionMatrix::from_predictions_with_min_classes(y_pred, y_true, min_classes);
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

    /// Average F1 over an explicit subset of class indices.
    ///
    /// This supports benchmarks whose official score excludes a neutral or
    /// background class. For example, TweetEval stance reports
    /// `(F1_against + F1_favor) / 2`, corresponding to class indices `[1, 2]`
    /// in the canonical abortion stance label mapping.
    ///
    /// Returns `None` when `classes` is empty or any requested class is not
    /// present, preventing a silently mislabelled benchmark score.
    #[must_use]
    pub fn f1_avg_for_classes(&self, classes: &[usize]) -> Option<f64> {
        f1_average_for_classes(&self.f1, classes)
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
