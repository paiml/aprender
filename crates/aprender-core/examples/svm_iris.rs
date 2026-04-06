#![allow(clippy::disallowed_methods)]
//! Linear Support Vector Machine Classification on Iris Dataset
//!
//! This example demonstrates Linear SVM classification on the Iris flower dataset.
//! We'll explore:
//! - Binary classification with maximum-margin decision boundary
//! - Effect of regularization parameter C
//! - Comparison with Naive Bayes and kNN
//! - Decision function values and margins
//! - Robustness to outliers

use aprender::classification::{GaussianNB, KNearestNeighbors, LinearSVM};
use aprender::primitives::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Linear SVM: Iris Binary Classification ===\n");

    // For SVM demo, we'll use binary classification (Setosa vs Versicolor)
    // This makes visualization and understanding easier
    let (x_train, y_train, x_test, y_test) = load_binary_iris_data()?;

    println!(
        "Binary Dataset: {} training samples, {} test samples",
        x_train.n_rows(),
        x_test.n_rows()
    );
    println!("Classes: 0=Setosa, 1=Versicolor\n");

    // Part 1: Basic Linear SVM
    println!("=== Part 1: Basic Linear SVM ===\n");
    let mut svm = LinearSVM::new()
        .with_c(1.0)
        .with_max_iter(1000)
        .with_learning_rate(0.1);
    svm.fit(&x_train, &y_train)?;

    let predictions = svm.predict(&x_test)?;
    let accuracy = compute_accuracy(&predictions, &y_test);
    println!("Test Accuracy: {:.1}%\n", accuracy * 100.0);

    // Part 2: Decision Function Values
    println!("=== Part 2: Decision Function & Margins ===\n");

    let decisions = svm.decision_function(&x_test)?;

    println!("Sample predictions with decision values:");
    println!("Sample  True  Predicted  Decision  Margin");
    println!("───────────────────────────────────────────");

    for i in 0..5.min(x_test.n_rows()) {
        let true_label = y_test[i];
        let pred = predictions[i];
        let decision = decisions[i];
        let margin = if true_label == 1 { decision } else { -decision };

        println!("   {i}      {true_label}      {pred}       {decision:.3}    {margin:.3}");
    }
    println!();

    // Part 3: Effect of Regularization Parameter C
    println!("=== Part 3: Effect of Regularization (C) ===\n");

    for &c_value in &[0.01, 0.1, 1.0, 10.0, 100.0] {
        let mut svm_c = LinearSVM::new()
            .with_c(c_value)
            .with_max_iter(1000)
            .with_learning_rate(0.1);
        svm_c.fit(&x_train, &y_train)?;
        let preds = svm_c.predict(&x_test)?;
        let acc = compute_accuracy(&preds, &y_test);

        println!("C={:6.2}: Accuracy = {:.1}%", c_value, acc * 100.0);
    }
    println!();

    // Part 4: Comparison with Other Classifiers
    println!("=== Part 4: Comparison with Other Classifiers ===\n");

    // Naive Bayes
    let mut nb = GaussianNB::new();
    nb.fit(&x_train, &y_train)?;
    let nb_predictions = nb.predict(&x_test)?;
    let nb_accuracy = compute_accuracy(&nb_predictions, &y_test);

    // kNN
    let mut knn = KNearestNeighbors::new(5).with_weights(true);
    knn.fit(&x_train, &y_train)?;
    let knn_predictions = knn.predict(&x_test)?;
    let knn_accuracy = compute_accuracy(&knn_predictions, &y_test);

    println!("Classifier         Accuracy");
    println!("─────────────────────────────");
    println!("Linear SVM         {:.1}%", accuracy * 100.0);
    println!("Naive Bayes        {:.1}%", nb_accuracy * 100.0);
    println!("k-NN (k=5)         {:.1}%\n", knn_accuracy * 100.0);

    // Part 5: Understanding SVM Decision Boundary
    println!("=== Part 5: Understanding the Model ===\n");

    println!("Linear SVM Characteristics:");
    println!("✓ Maximizes margin between classes");
    println!("✓ Robust to outliers (with appropriate C)");
    println!("✓ Fast prediction (linear decision function)");
    println!("✓ Convex optimization (guaranteed convergence)");
    println!("✓ Effective in high-dimensional spaces\n");

    println!("Regularization Parameter C:");
    println!("- Small C (0.01-0.1): Large margin, simpler model, more regularization");
    println!("- Large C (10-100): Small margin, complex model, less regularization");
    println!("- Default C=1.0: Balanced trade-off\n");

    // Part 6: Per-Class Analysis
    println!("=== Part 6: Per-Class Performance ===\n");

    let mut class_correct = [0; 2];
    let mut class_total = [0; 2];

    for (&pred, &true_label) in predictions.iter().zip(y_test.iter()) {
        class_total[true_label] += 1;
        if pred == true_label {
            class_correct[true_label] += 1;
        }
    }

    let species = ["Setosa", "Versicolor"];
    println!("Species      Correct  Total  Accuracy");
    println!("──────────────────────────────────────");
    for i in 0..2 {
        let acc = class_correct[i] as f32 / class_total[i] as f32 * 100.0;
        println!(
            "{:12}  {}/{}     {:2}     {:.1}%",
            species[i], class_correct[i], class_total[i], class_total[i], acc
        );
    }
    println!();

    // Part 7: Summary
    println!("=== Summary ===\n");
    println!("Linear SVM vs Naive Bayes vs k-NN:");
    println!("- Training: SVM iterative vs NB instant vs kNN instant (lazy)");
    println!("- Prediction: SVM O(p) vs NB O(p·c) vs kNN O(n·p)");
    println!("- Decision: SVM margin-based vs NB probabilistic vs kNN similarity");
    println!("- Regularization: SVM C parameter vs NB variance smoothing vs kNN k");
    println!(
        "- Accuracy: SVM {:.1}% vs NB {:.1}% vs kNN {:.1}%",
        accuracy * 100.0,
        nb_accuracy * 100.0,
        knn_accuracy * 100.0
    );

    Ok(())
}

/// Load binary Iris dataset (Setosa vs Versicolor only)
#[allow(clippy::type_complexity)]
fn load_binary_iris_data(
) -> Result<(Matrix<f32>, Vec<usize>, Matrix<f32>, Vec<usize>), &'static str> {
    // Binary classification: Setosa (0) vs Versicolor (1)
    // Features: [sepal_length, sepal_width, petal_length, petal_width]

    // Training set: 14 samples (7 Setosa, 7 Versicolor)
    let x_train = Matrix::from_vec(
        14,
        4,
        vec![
            // Setosa (small petals, large sepals)
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4,
            0.3, // Versicolor (medium petals
            // and sepals)
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
