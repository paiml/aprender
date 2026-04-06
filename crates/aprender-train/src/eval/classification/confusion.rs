//! Confusion matrix for multi-class classification

use std::fmt;

/// Confusion matrix for multi-class classification
///
/// Element [i][j] represents count of samples with true label i predicted as j
#[derive(Clone, Debug)]
pub struct ConfusionMatrix {
    /// The matrix data: matrix[true_label][predicted_label] = count
    matrix: Vec<Vec<usize>>,
    /// Number of classes
    n_classes: usize,
    /// Class labels (indices)
    labels: Vec<usize>,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix with given number of classes
    pub fn new(n_classes: usize) -> Self {
        Self {
            matrix: vec![vec![0; n_classes]; n_classes],
            n_classes,
            labels: (0..n_classes).collect(),
        }
    }

    /// Create from predictions and ground truth
    pub fn from_predictions(y_pred: &[usize], y_true: &[usize]) -> Self {
        Self::from_predictions_with_min_classes(y_pred, y_true, 0)
    }

    /// Create from predictions and ground truth, ensuring at least `min_classes` classes
    pub fn from_predictions_with_min_classes(
        y_pred: &[usize],
        y_true: &[usize],
        min_classes: usize,
    ) -> Self {
        assert_eq!(y_pred.len(), y_true.len(), "Predictions and targets must have same length");

        // Determine number of classes (at least min_classes)
        let observed = y_pred.iter().chain(y_true.iter()).max().map_or(0, |&m| m + 1);
        let n_classes = observed.max(min_classes);

        let mut cm = Self::new(n_classes);

        for (&pred, &true_label) in y_pred.iter().zip(y_true.iter()) {
            if pred < n_classes && true_label < n_classes {
                cm.matrix[true_label][pred] += 1;
            }
        }

        cm
    }

    /// Get the raw matrix
    pub fn matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }

    /// Get the class labels
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Get number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Get element at [true_label][predicted_label]
    pub fn get(&self, true_label: usize, predicted_label: usize) -> usize {
        self.matrix[true_label][predicted_label]
    }

    /// Calculate true positives for a class
    pub fn true_positives(&self, class: usize) -> usize {
        self.matrix[class][class]
    }

    /// Calculate false positives for a class (predicted as class but wasn't)
    pub fn false_positives(&self, class: usize) -> usize {
        (0..self.n_classes).filter(|&i| i != class).map(|i| self.matrix[i][class]).sum()
    }

    /// Calculate false negatives for a class (was class but predicted differently)
    pub fn false_negatives(&self, class: usize) -> usize {
        (0..self.n_classes).filter(|&j| j != class).map(|j| self.matrix[class][j]).sum()
    }

    /// Calculate true negatives for a class
    pub fn true_negatives(&self, class: usize) -> usize {
        let total: usize = self.matrix.iter().flatten().sum();
        total
            - self.true_positives(class)
            - self.false_positives(class)
            - self.false_negatives(class)
    }

    /// Calculate support (total true instances) for a class
    pub fn support(&self, class: usize) -> usize {
        self.matrix[class].iter().sum()
    }

    /// Total number of samples
    pub fn total(&self) -> usize {
        self.matrix.iter().flatten().sum()
    }

    /// Calculate accuracy
    pub fn accuracy(&self) -> f64 {
        contract_pre_accuracy!();
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let correct: usize = (0..self.n_classes).map(|i| self.matrix[i][i]).sum();
        correct as f64 / total as f64
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Confusion Matrix:")?;

        // Header
        write!(f, "      ")?;
        for j in 0..self.n_classes {
            write!(f, "Pred {j} ")?;
        }
        writeln!(f)?;

        // Rows
        for i in 0..self.n_classes {
            write!(f, "True {i}")?;
            for j in 0..self.n_classes {
                write!(f, "{:>6} ", self.matrix[i][j])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
