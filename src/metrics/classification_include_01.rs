
/// Compute per-class F1 scores.
///
/// Returns a vector of F1 scores, one per class (ordered by class index).
/// For binary classification, index 1 is the positive-class F1.
///
/// F1_i = 2 * precision_i * recall_i / (precision_i + recall_i)
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
///
/// # Returns
///
/// Vector of per-class F1 scores (each in 0.0..=1.0)
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::f1_per_class;
///
/// let y_true = vec![1, 0, 1, 0];
/// let y_pred = vec![1, 1, 0, 0];
/// let per_class = f1_per_class(&y_pred, &y_true);
/// assert_eq!(per_class.len(), 2);
/// // Both classes: precision=0.5, recall=0.5 → F1=0.5
/// assert!((per_class[0] - 0.5).abs() < 1e-5);
/// assert!((per_class[1] - 0.5).abs() < 1e-5);
/// ```
#[must_use]
pub fn f1_per_class(y_pred: &[usize], y_true: &[usize]) -> Vec<f32> {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    let (tp, fp, fn_counts, _) = compute_tp_fp_fn(y_pred, y_true, n_classes);
    (0..n_classes)
        .map(|i| class_f1(tp[i], fp[i], fn_counts[i]))
        .collect()
}

/// Compute confusion matrix.
///
/// Returns a matrix where element `[i,j]` is the count of samples
/// with true label i and predicted label j.
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - True class labels
///
/// # Returns
///
/// Confusion matrix as `Matrix<usize>`
///
/// # Panics
///
/// Panics if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::metrics::classification::confusion_matrix;
///
/// let y_true = vec![0, 0, 1, 1, 2, 2];
/// let y_pred = vec![0, 1, 1, 1, 2, 0];
/// let cm = confusion_matrix(&y_pred, &y_true);
/// assert_eq!(cm.n_rows(), 3);
/// assert_eq!(cm.n_cols(), 3);
/// ```
#[must_use]
#[provable_contracts_macros::contract("metrics-classification-v1", equation = "confusion_matrix")]
pub fn confusion_matrix(y_pred: &[usize], y_true: &[usize]) -> Matrix<usize> {
    assert_eq!(y_pred.len(), y_true.len(), "Vectors must have same length");
    assert!(!y_true.is_empty(), "Vectors cannot be empty");

    let n_classes = y_true
        .iter()
        .chain(y_pred.iter())
        .max()
        .map_or(0, |&m| m + 1);

    let mut data = vec![0usize; n_classes * n_classes];

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        data[true_label * n_classes + pred_label] += 1;
    }

    Matrix::from_vec(n_classes, n_classes, data)
        .expect("Confusion matrix dimensions match data length")
}

/// Helper function to compute TP, FP, FN for each class.
fn compute_tp_fp_fn(
    y_pred: &[usize],
    y_true: &[usize],
    n_classes: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut tp = vec![0usize; n_classes];
    let mut fp = vec![0usize; n_classes];
    let mut fn_counts = vec![0usize; n_classes];
    let mut support = vec![0usize; n_classes];

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        support[true_label] += 1;

        if true_label == pred_label {
            tp[true_label] += 1;
        } else {
            fp[pred_label] += 1;
            fn_counts[true_label] += 1;
        }
    }

    (tp, fp, fn_counts, support)
}

#[path = "classification_part_02.rs"]
mod classification_part_02;
pub use classification_part_02::classification_report;
