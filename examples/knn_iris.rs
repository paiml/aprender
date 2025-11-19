//! K-Nearest Neighbors Classification on Iris Dataset
//!
//! This example demonstrates K-Nearest Neighbors (kNN) classification on the
//! famous Iris flower dataset. We'll explore:
//! - Training kNN with different k values
//! - Comparing distance metrics (Euclidean, Manhattan, Minkowski)
//! - Weighted vs uniform voting
//! - Probabilistic predictions with predict_proba
//! - Model evaluation and parameter tuning

use aprender::classification::DistanceMetric;
use aprender::classification::KNearestNeighbors;
use aprender::primitives::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== K-Nearest Neighbors: Iris Flower Classification ===\n");

    // Load Iris dataset (30 samples: 10 Setosa, 10 Versicolor, 10 Virginica)
    let (x_train, y_train, x_test, y_test) = load_iris_data()?;

    println!(
        "Dataset: {} training samples, {} test samples\n",
        x_train.n_rows(),
        x_test.n_rows()
    );

    // Part 1: Basic kNN with k=3
    println!("=== Part 1: Basic kNN (k=3, Euclidean) ===\n");
    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x_train, &y_train)?;

    let predictions = knn.predict(&x_test)?;
    let accuracy = compute_accuracy(&predictions, &y_test);
    println!("Test Accuracy: {:.1}%\n", accuracy * 100.0);

    // Part 2: Effect of k parameter
    println!("=== Part 2: Effect of k Parameter ===\n");
    for k in [1, 3, 5, 7, 9] {
        let mut knn = KNearestNeighbors::new(k);
        knn.fit(&x_train, &y_train)?;
        let predictions = knn.predict(&x_test)?;
        let accuracy = compute_accuracy(&predictions, &y_test);
        println!("k={}: Accuracy = {:.1}%", k, accuracy * 100.0);
    }
    println!();

    // Part 3: Distance Metrics Comparison
    println!("=== Part 3: Distance Metrics (k=5) ===\n");

    let mut knn_euclidean = KNearestNeighbors::new(5).with_metric(DistanceMetric::Euclidean);
    knn_euclidean.fit(&x_train, &y_train)?;
    let acc_euclidean = compute_accuracy(&knn_euclidean.predict(&x_test)?, &y_test);

    let mut knn_manhattan = KNearestNeighbors::new(5).with_metric(DistanceMetric::Manhattan);
    knn_manhattan.fit(&x_train, &y_train)?;
    let acc_manhattan = compute_accuracy(&knn_manhattan.predict(&x_test)?, &y_test);

    let mut knn_minkowski = KNearestNeighbors::new(5).with_metric(DistanceMetric::Minkowski(3.0));
    knn_minkowski.fit(&x_train, &y_train)?;
    let acc_minkowski = compute_accuracy(&knn_minkowski.predict(&x_test)?, &y_test);

    println!("Euclidean distance:   {:.1}%", acc_euclidean * 100.0);
    println!("Manhattan distance:   {:.1}%", acc_manhattan * 100.0);
    println!("Minkowski (p=3):      {:.1}%\n", acc_minkowski * 100.0);

    // Part 4: Weighted vs Uniform Voting
    println!("=== Part 4: Weighted vs Uniform Voting (k=5) ===\n");

    let mut knn_uniform = KNearestNeighbors::new(5);
    knn_uniform.fit(&x_train, &y_train)?;
    let acc_uniform = compute_accuracy(&knn_uniform.predict(&x_test)?, &y_test);

    let mut knn_weighted = KNearestNeighbors::new(5).with_weights(true);
    knn_weighted.fit(&x_train, &y_train)?;
    let acc_weighted = compute_accuracy(&knn_weighted.predict(&x_test)?, &y_test);

    println!("Uniform voting:   {:.1}%", acc_uniform * 100.0);
    println!("Weighted voting:  {:.1}%\n", acc_weighted * 100.0);

    // Part 5: Probabilistic Predictions
    println!("=== Part 5: Probabilistic Predictions ===\n");

    let mut knn_proba = KNearestNeighbors::new(5).with_weights(true);
    knn_proba.fit(&x_train, &y_train)?;

    let probabilities = knn_proba.predict_proba(&x_test)?;

    println!("Sample predictions with confidence:");
    println!("Sample  Predicted  Setosa  Versicolor  Virginica");
    println!("─────────────────────────────────────────────────────");

    let predictions = knn_proba.predict(&x_test)?;
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

    // Part 6: Best Configuration Summary
    println!("=== Summary ===\n");
    println!("Best configuration found:");
    println!("- k = 5 neighbors");
    println!("- Distance metric: Euclidean");
    println!("- Voting: Weighted by inverse distance");
    println!("- Test accuracy: {:.1}%\n", acc_weighted * 100.0);

    println!("Key insights:");
    println!("- Small k (1-3): Risk of overfitting, sensitive to noise");
    println!("- Large k (7-9): Risk of underfitting, class boundaries blur");
    println!("- Weighted voting: Gives more influence to closer neighbors");
    println!("- Distance metric: Euclidean works well for continuous features");

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
            3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3,
            // Versicolor (medium petals and sepals)
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6,
            // Virginica (large petals and sepals)
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
