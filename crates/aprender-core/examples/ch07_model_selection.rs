#![allow(clippy::disallowed_methods)]
//! Chapter 7: Model Selection and Evaluation
//!
//! Demonstrates KFold cross-validation with a real model + metrics.
//! Citation: Kohavi, "Cross-Validation and Bootstrap," IJCAI 1995
//! Contract: contracts/apr-book-ch07-v1.yaml (v2 — api_calls enforced)

use aprender::prelude::*;
use aprender::metrics::classification::{accuracy, precision, recall, Average};
use aprender::model_selection::KFold;

fn main() {
    // Dataset: two linearly separable classes
    let x = Matrix::from_vec(12, 2, vec![
        1.0, 1.5, 2.0, 2.5, 1.5, 2.0, 2.5, 1.0, 3.0, 2.0, 2.0, 1.5,
        7.0, 7.5, 8.0, 8.5, 7.5, 8.0, 8.5, 7.0, 9.0, 8.0, 7.0, 7.5,
    ]).expect("valid 12x2 matrix");
    let y: Vec<usize> = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

    // KFold cross-validation with 3 folds
    let kfold = KFold::new(3).with_shuffle(true).with_random_state(42);
    let splits = kfold.split(x.n_rows());

    println!("KFold cross-validation: k=3, {} samples", x.n_rows());
    assert_eq!(splits.len(), 3, "3-fold must produce 3 splits");

    let mut fold_accuracies = Vec::new();
    for (fold_idx, (train_idx, test_idx)) in splits.iter().enumerate() {
        // Extract train/test data by index
        let mut x_train_data = Vec::new();
        let mut y_train = Vec::new();
        for &i in train_idx {
            for col in 0..2 { x_train_data.push(x.get(i, col)); }
            y_train.push(y[i]);
        }
        let mut x_test_data = Vec::new();
        let mut y_test = Vec::new();
        for &i in test_idx {
            for col in 0..2 { x_test_data.push(x.get(i, col)); }
            y_test.push(y[i]);
        }

        let x_train = Matrix::from_vec(train_idx.len(), 2, x_train_data).unwrap();
        let x_test = Matrix::from_vec(test_idx.len(), 2, x_test_data).unwrap();

        // Train a decision tree on this fold
        let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
        tree.fit(&x_train, &y_train).expect("fit on fold");
        let preds = tree.predict(&x_test);

        let acc = accuracy(&preds, &y_test);
        fold_accuracies.push(acc);
        println!("  Fold {}: train={} test={} accuracy={acc:.2}",
            fold_idx + 1, train_idx.len(), test_idx.len());
    }

    // Cross-validation mean accuracy
    let mean_acc: f32 = fold_accuracies.iter().sum::<f32>() / fold_accuracies.len() as f32;
    println!("\nMean accuracy: {mean_acc:.2}");
    assert!(mean_acc > 0.5, "Mean CV accuracy must exceed chance");

    // Full-data metrics
    let mut full_tree = DecisionTreeClassifier::new().with_max_depth(3);
    full_tree.fit(&x, &y).unwrap();
    let full_preds = full_tree.predict(&x);
    let prec = precision(&full_preds, &y, Average::Macro);
    let rec = recall(&full_preds, &y, Average::Macro);
    println!("Full-data precision={prec:.2}, recall={rec:.2}");
    assert!(prec > 0.0 && prec <= 1.0, "Precision in (0,1]");
    assert!(rec > 0.0 && rec <= 1.0, "Recall in (0,1]");

    println!("Chapter 7 contracts: PASSED");
}
