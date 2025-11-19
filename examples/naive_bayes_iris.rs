//! Gaussian Naive Bayes Classification on Iris Dataset
//!
//! This example demonstrates Gaussian Naive Bayes (GaussianNB) classification on the
//! famous Iris flower dataset. We'll explore:
//! - Training GaussianNB with probabilistic approach
//! - Comparing performance with kNN
//! - Understanding class priors and feature distributions
//! - Probabilistic predictions with confidence scores
//! - Effect of variance smoothing

use aprender::classification::{GaussianNB, KNearestNeighbors};
use aprender::primitives::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gaussian Naive Bayes: Iris Flower Classification ===\n");

    // Load Iris dataset (30 samples: 10 Setosa, 10 Versicolor, 10 Virginica)
    let (x_train, y_train, x_test, y_test) = load_iris_data()?;

    println!(
        "Dataset: {} training samples, {} test samples\n",
        x_train.n_rows(),
        x_test.n_rows()
    );

    // Part 1: Basic Gaussian Naive Bayes
    println!("=== Part 1: Basic Gaussian Naive Bayes ===\n");
    let mut nb = GaussianNB::new();
    nb.fit(&x_train, &y_train)?;

    let predictions = nb.predict(&x_test)?;
    let accuracy = compute_accuracy(&predictions, &y_test);
    println!("Test Accuracy: {:.1}%\n", accuracy * 100.0);

    // Part 2: Probabilistic Predictions
    println!("=== Part 2: Probabilistic Predictions ===\n");

    let probabilities = nb.predict_proba(&x_test)?;

    println!("Sample predictions with confidence:");
    println!("Sample  Predicted  Setosa  Versicolor  Virginica");
    println!("──────────────────────────────────────────────────────");

    for i in 0..5.min(x_test.n_rows()) {
        let pred = predictions[i];
        let p0 = probabilities[i][0];
        let p1 = probabilities[i][1];
        let p2 = probabilities[i][2];

        let species = match pred {
            0 => "Setosa    ",
            1 => "Versicolor",
            2 => "Virginica ",
            _ => "Unknown   ",
        };

        println!(
            "   {}     {}   {:.1}%    {:.1}%       {:.1}%",
            i,
            species,
            p0 * 100.0,
            p1 * 100.0,
            p2 * 100.0
        );
    }
    println!();

    // Part 3: Comparison with kNN
    println!("=== Part 3: Comparison with kNN (k=5) ===\n");

    let mut knn = KNearestNeighbors::new(5).with_weights(true);
    knn.fit(&x_train, &y_train)?;
    let knn_predictions = knn.predict(&x_test)?;
    let knn_accuracy = compute_accuracy(&knn_predictions, &y_test);

    println!("Gaussian Naive Bayes: {:.1}%", accuracy * 100.0);
    println!("k-Nearest Neighbors:  {:.1}%\n", knn_accuracy * 100.0);

    // Part 4: Effect of Variance Smoothing
    println!("=== Part 4: Effect of Variance Smoothing ===\n");

    for &var_smoothing in &[1e-12, 1e-9, 1e-6, 1e-3] {
        let mut nb_smooth = GaussianNB::new().with_var_smoothing(var_smoothing);
        nb_smooth.fit(&x_train, &y_train)?;
        let smooth_predictions = nb_smooth.predict(&x_test)?;
        let smooth_accuracy = compute_accuracy(&smooth_predictions, &y_test);

        println!(
            "var_smoothing={:8}: Accuracy = {:.1}%",
            format!("{:.0e}", var_smoothing),
            smooth_accuracy * 100.0
        );
    }
    println!();

    // Part 5: Class Balance and Priors
    println!("=== Part 5: Understanding the Model ===\n");

    // Refit to access internals (for demonstration)
    let mut nb_analysis = GaussianNB::new();
    nb_analysis.fit(&x_train, &y_train)?;

    println!("Key Insights:");
    println!("- Training time: Instant (no iterative optimization)");
    println!("- Assumes features are independent (Naive assumption)");
    println!("- Models each class with Gaussian distribution");
    println!("- Uses Bayes' theorem: P(y|X) ∝ P(y) * P(X|y)");
    println!("- Handles imbalanced classes via class priors\n");

    // Part 6: Per-Class Analysis
    println!("=== Part 6: Per-Class Performance ===\n");

    let mut class_correct = [0; 3];
    let mut class_total = [0; 3];

    for (&pred, &true_label) in predictions.iter().zip(y_test.iter()) {
        class_total[true_label] += 1;
        if pred == true_label {
            class_correct[true_label] += 1;
        }
    }

    let species = ["Setosa", "Versicolor", "Virginica"];
    println!("Species      Correct  Total  Accuracy");
    println!("──────────────────────────────────────");
    for i in 0..3 {
        let acc = class_correct[i] as f32 / class_total[i] as f32 * 100.0;
        println!(
            "{:12}  {}/{}     {:2}     {:.1}%",
            species[i], class_correct[i], class_total[i], class_total[i], acc
        );
    }
    println!();

    // Part 7: Summary and Comparison
    println!("=== Summary ===\n");
    println!("Gaussian Naive Bayes Characteristics:");
    println!("✓ Extremely fast training (closed-form solution)");
    println!("✓ Probabilistic predictions (confidence scores)");
    println!("✓ Works well with small datasets");
    println!("✓ Handles imbalanced classes naturally");
    println!("✓ Excellent baseline classifier\n");

    println!("Naive Bayes vs k-Nearest Neighbors:");
    println!("- Training: NB instant vs kNN instant (lazy)");
    println!("- Prediction: NB O(p) vs kNN O(n·p) per sample");
    println!("- Memory: NB O(c·p) vs kNN O(n·p)");
    println!("- Assumption: NB independence vs kNN similarity");
    println!(
        "- Accuracy: NB {:.1}% vs kNN {:.1}%",
        accuracy * 100.0,
        knn_accuracy * 100.0
    );

    Ok(())
}

/// Load Iris dataset with train/test split
#[allow(clippy::type_complexity)]
fn load_iris_data() -> Result<(Matrix<f32>, Vec<usize>, Matrix<f32>, Vec<usize>), &'static str> {
    // Iris dataset: 30 samples (10 per species)
    // Features: [sepal_length, sepal_width, petal_length, petal_width]
    // Classes: 0=Setosa, 1=Versicolor, 2=Virginica

    // Training set: 20 samples (first 7 Setosa, 7 Versicolor, 6 Virginica)
    let x_train = Matrix::from_vec(
        20,
        4,
        vec![
            // Setosa (small petals, large sepals)
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4,
            0.3, // Versicolor (medium petals
            // and sepals)
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7,
            1.6, // Virginica (large petals and
            // sepals)
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2, 7.6, 3.0, 6.6, 2.1,
        ],
    )?;

    let y_train = vec![
        0, 0, 0, 0, 0, 0, 0, // Setosa
        1, 1, 1, 1, 1, 1, 1, // Versicolor
        2, 2, 2, 2, 2, 2, // Virginica
    ];

    // Test set: 10 samples (3 Setosa, 3 Versicolor, 4 Virginica)
    let x_test = Matrix::from_vec(
        10,
        4,
        vec![
            // Setosa
            5.0, 3.3, 1.4, 0.2, 4.4, 2.9, 1.4, 0.2, 4.9, 3.1, 1.5, 0.1, // Versicolor
            4.9, 2.4, 3.3, 1.0, 6.6, 2.9, 4.6, 1.3, 5.2, 2.7, 3.9, 1.4, // Virginica
            7.2, 3.6, 6.1, 2.5, 6.5, 3.2, 5.1, 2.0, 6.4, 2.7, 5.3, 1.9, 5.9, 3.0, 5.1, 1.8,
        ],
    )?;

    let y_test = vec![
        0, 0, 0, // Setosa
        1, 1, 1, // Versicolor
        2, 2, 2, 2, // Virginica
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
