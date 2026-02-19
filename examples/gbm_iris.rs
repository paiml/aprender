#![allow(clippy::disallowed_methods)]
//! Gradient Boosting Machine on Iris Dataset
//!
//! This example demonstrates Gradient Boosting (GBM) for binary classification
//! on Iris data, comparing with other classifiers and showing the power of
//! sequential ensemble learning.

use aprender::classification::{GaussianNB, LinearSVM};
use aprender::primitives::Matrix;
use aprender::tree::GradientBoostingClassifier;

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gradient Boosting: Iris Binary Classification ===\n");

    // Binary classification: Setosa vs Versicolor
    let (x_train, y_train, x_test, y_test) = load_binary_iris_data()?;

    println!(
        "Dataset: {} training samples, {} test samples",
        x_train.n_rows(),
        x_test.n_rows()
    );
    println!("Classes: 0=Setosa, 1=Versicolor\n");

    // Part 1: Basic Gradient Boosting
    println!("=== Part 1: Basic Gradient Boosting ===\n");

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(50)
        .with_learning_rate(0.1)
        .with_max_depth(3);

    gbm.fit(&x_train, &y_train)?;
    let predictions = gbm.predict(&x_test)?;
    let accuracy = compute_accuracy(&predictions, &y_test);

    println!("Gradient Boosting Accuracy: {:.1}%", accuracy * 100.0);
    println!("Number of trees trained: {}\n", gbm.n_estimators());

    // Part 2: Hyperparameter Effects
    println!("=== Part 2: Effect of Number of Estimators ===\n");

    for &n_est in &[10, 30, 50, 100] {
        let mut gbm_n = GradientBoostingClassifier::new()
            .with_n_estimators(n_est)
            .with_learning_rate(0.1)
            .with_max_depth(3);

        gbm_n.fit(&x_train, &y_train)?;
        let preds = gbm_n.predict(&x_test)?;
        let acc = compute_accuracy(&preds, &y_test);

        println!("n_estimators={:3}: Accuracy = {:.1}%", n_est, acc * 100.0);
    }
    println!();

    println!("=== Part 3: Effect of Learning Rate ===\n");

    for &lr in &[0.01, 0.05, 0.1, 0.5] {
        let mut gbm_lr = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(lr)
            .with_max_depth(3);

        gbm_lr.fit(&x_train, &y_train)?;
        let preds = gbm_lr.predict(&x_test)?;
        let acc = compute_accuracy(&preds, &y_test);

        println!("learning_rate={:.2}: Accuracy = {:.1}%", lr, acc * 100.0);
    }
    println!();

    println!("=== Part 4: Effect of Tree Depth ===\n");

    for &depth in &[1, 2, 3, 5] {
        let mut gbm_depth = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_max_depth(depth);

        gbm_depth.fit(&x_train, &y_train)?;
        let preds = gbm_depth.predict(&x_test)?;
        let acc = compute_accuracy(&preds, &y_test);

        println!("max_depth={}: Accuracy = {:.1}%", depth, acc * 100.0);
    }
    println!();

    // Part 5: Comparison with Other Classifiers
    println!("=== Part 5: Comparison with Other Classifiers ===\n");

    // Naive Bayes
    let mut nb = GaussianNB::new();
    nb.fit(&x_train, &y_train)?;
    let nb_predictions = nb.predict(&x_test)?;
    let nb_accuracy = compute_accuracy(&nb_predictions, &y_test);

    // Linear SVM
    let mut svm = LinearSVM::new()
        .with_c(1.0)
        .with_max_iter(1000)
        .with_learning_rate(0.1);
    svm.fit(&x_train, &y_train)?;
    let svm_predictions = svm.predict(&x_test)?;
    let svm_accuracy = compute_accuracy(&svm_predictions, &y_test);

    println!("Classifier               Accuracy");
    println!("──────────────────────────────────");
    println!("Gradient Boosting        {:.1}%", accuracy * 100.0);
    println!("Naive Bayes              {:.1}%", nb_accuracy * 100.0);
    println!("Linear SVM               {:.1}%\n", svm_accuracy * 100.0);

    // Part 6: Probabilistic Predictions
    println!("=== Part 6: Probabilistic Predictions ===\n");

    let probas = gbm.predict_proba(&x_test)?;

    println!("Sample  Predicted  P(Setosa)  P(Versicolor)");
    println!("────────────────────────────────────────────");

    for i in 0..5.min(x_test.n_rows()) {
        let pred = predictions[i];
        let p0 = probas[i][0];
        let p1 = probas[i][1];

        let species = if pred == 0 {
            "Setosa    "
        } else {
            "Versicolor"
        };

        println!("   {i}     {species}   {p0:.3}      {p1:.3}");
    }
    println!();

    // Part 7: Summary
    println!("=== Summary ===\n");
    println!("Gradient Boosting Characteristics:");
    println!("✓ Sequential ensemble of weak learners");
    println!("✓ Learns from residual errors");
    println!("✓ Lower learning rate + more trees = better generalization");
    println!("✓ Shallow trees prevent overfitting");
    println!("✓ State-of-the-art for tabular data\n");

    println!("Hyperparameter Guidelines:");
    println!("- n_estimators: 50-500 (more trees = better fit, slower)");
    println!("- learning_rate: 0.01-0.3 (lower = better gen., needs more trees)");
    println!("- max_depth: 3-8 (shallow trees prevent overfitting)\n");

    println!("GBM vs Naive Bayes vs SVM:");
    println!(
        "- Accuracy: GBM {:.1}% vs NB {:.1}% vs SVM {:.1}%",
        accuracy * 100.0,
        nb_accuracy * 100.0,
        svm_accuracy * 100.0
    );
    println!("- Training: GBM iterative vs NB instant vs SVM iterative");
    println!("- Prediction: All O(trees·depth) or O(features)");
    println!("- Strength: GBM complex patterns vs NB probabilistic vs SVM margin");

    Ok(())
}

/// Load binary Iris dataset (Setosa vs Versicolor only)
#[allow(clippy::type_complexity)]
fn load_binary_iris_data(
) -> Result<(Matrix<f32>, Vec<usize>, Matrix<f32>, Vec<usize>), &'static str> {
    // Binary classification: Setosa (0) vs Versicolor (1)
    // Training set: 14 samples (7 Setosa, 7 Versicolor)
    let x_train = Matrix::from_vec(
        14,
        4,
        vec![
            // Setosa
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, // Versicolor
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6,
        ],
    )?;

    let y_train = vec![
        0, 0, 0, 0, 0, 0, 0, // Setosa
        1, 1, 1, 1, 1, 1, 1, // Versicolor
    ];

    // Test set: 6 samples (3 Setosa, 3 Versicolor)
    let x_test = Matrix::from_vec(
        6,
        4,
        vec![
            // Setosa
            5.0, 3.3, 1.4, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5, 0.1, // Versicolor
            4.9, 2.4, 3.3, 1.0, 6.6, 2.9, 4.6, 1.3, 5.2, 2.7, 3.9, 1.4,
        ],
    )?;

    let y_test = vec![
        0, 0, 0, // Setosa
        1, 1, 1, // Versicolor
    ];

    Ok((x_train, y_train, x_test, y_test))
}

/// Compute classification accuracy
fn compute_accuracy(predictions: &[usize], true_labels: &[usize]) -> f32 {
    let correct = predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();

    correct as f32 / true_labels.len() as f32
}
