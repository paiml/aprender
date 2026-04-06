//! Classification report functions

use super::average::Average;
use super::confusion::ConfusionMatrix;
use super::metrics::MultiClassMetrics;

/// Compute confusion matrix from predictions and ground truth
///
/// # Arguments
/// * `y_pred` - Predicted class labels
/// * `y_true` - Ground truth class labels
///
/// # Returns
/// A ConfusionMatrix where element [i][j] is count of true label i predicted as j
///
/// # Example
/// ```ignore
/// use entrenar::eval::confusion_matrix;
///
/// let y_pred = vec![0, 1, 1, 2, 0];
/// let y_true = vec![0, 1, 0, 2, 1];
/// let cm = confusion_matrix(&y_pred, &y_true);
///
/// assert_eq!(cm.get(0, 0), 1);  // True 0, predicted 0
/// assert_eq!(cm.get(0, 1), 1);  // True 0, predicted 1
/// ```
pub fn confusion_matrix(y_pred: &[usize], y_true: &[usize]) -> ConfusionMatrix {
    contract_pre_confusion_matrix!();
    ConfusionMatrix::from_predictions(y_pred, y_true)
}

/// Generate sklearn-style classification report
///
/// # Arguments
/// * `y_pred` - Predicted class labels
/// * `y_true` - Ground truth class labels
///
/// # Returns
/// A formatted string containing per-class and overall metrics
///
/// # Example
/// ```ignore
/// use entrenar::eval::classification_report;
///
/// let report = classification_report(&y_pred, &y_true);
/// println!("{}", report);
/// ```
pub fn classification_report(y_pred: &[usize], y_true: &[usize]) -> String {
    let cm = ConfusionMatrix::from_predictions(y_pred, y_true);
    let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

    let mut report = String::new();

    // Header
    report.push_str(&format!(
        "{:>12} {:>10} {:>10} {:>10} {:>10}\n",
        "", "precision", "recall", "f1-score", "support"
    ));
    report.push_str(&"-".repeat(54));
    report.push('\n');

    // Per-class metrics
    for class in 0..metrics.n_classes {
        report.push_str(&format!(
            "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}\n",
            format!("Class {}", class),
            metrics.precision[class],
            metrics.recall[class],
            metrics.f1[class],
            metrics.support[class]
        ));
    }

    report.push_str(&"-".repeat(54));
    report.push('\n');

    // Averages
    let total_support: usize = metrics.support.iter().sum();

    report.push_str(&format!(
        "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}\n",
        "macro avg",
        metrics.precision_avg(Average::Macro),
        metrics.recall_avg(Average::Macro),
        metrics.f1_avg(Average::Macro),
        total_support
    ));

    report.push_str(&format!(
        "{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}\n",
        "weighted avg",
        metrics.precision_avg(Average::Weighted),
        metrics.recall_avg(Average::Weighted),
        metrics.f1_avg(Average::Weighted),
        total_support
    ));

    report.push_str(&format!("\nAccuracy: {:.4}\n", cm.accuracy()));

    report
}
