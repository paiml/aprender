//! Random Forest Iris Classification Example
//!
//! Demonstrates Random Forest ensemble classifier on the classic Iris dataset.
//! Shows how bootstrap sampling and majority voting improve accuracy and stability.

use aprender::primitives::Matrix;
use aprender::tree::{DecisionTreeClassifier, RandomForestClassifier};

fn main() {
    println!("Random Forest - Iris Classification Example");
    println!("============================================\n");

    // Iris dataset (simplified - 3 classes)
    let x = Matrix::from_vec(
        12,
        2,
        vec![
            // Setosa (class 0) - small petals
            1.4, 0.2, 1.3, 0.2, 1.5, 0.2, 1.7, 0.4,
            // Versicolor (class 1) - medium petals
            4.7, 1.4, 4.5, 1.5, 4.9, 1.5, 4.6, 1.3, // Virginica (class 2) - large petals
            6.0, 2.5, 5.9, 2.1, 6.1, 2.3, 5.8, 2.2,
        ],
    )
    .unwrap();
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    println!("Dataset:");
    println!("  Samples: 12 (4 per class)");
    println!("  Features: 2 (petal length, petal width)");
    println!("  Classes: 3 (Setosa, Versicolor, Virginica)\n");

    // Example 1: Single Decision Tree
    println!("Example 1: Single Decision Tree");
    println!("--------------------------------");
    single_tree_example(&x, &y);

    // Example 2: Random Forest (5 trees)
    println!("\nExample 2: Random Forest (5 trees)");
    println!("----------------------------------");
    random_forest_example(&x, &y, 5);

    // Example 3: Larger Random Forest (20 trees)
    println!("\nExample 3: Random Forest (20 trees)");
    println!("-----------------------------------");
    random_forest_example(&x, &y, 20);

    // Example 4: Out-of-Bag (OOB) Error Estimation
    println!("\nExample 4: Out-of-Bag (OOB) Error Estimation");
    println!("---------------------------------------------");
    oob_example(&x, &y);

    // Example 5: Feature Importance
    println!("\nExample 5: Feature Importance");
    println!("------------------------------");
    feature_importance_example(&x, &y);

    println!("\n✅ Random Forest Examples Complete!");
    println!("\nKey Advantages:");
    println!("  • Ensemble learning reduces overfitting");
    println!("  • Bootstrap sampling creates diversity");
    println!("  • Majority voting smooths predictions");
    println!("  • More stable than single trees");
    println!("  • OOB error provides free validation");
    println!("  • Feature importance for interpretability");
    println!("  • Excellent for real-world classification");
}

fn single_tree_example(x: &Matrix<f32>, y: &[usize]) {
    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);

    tree.fit(x, y).expect("Training failed");

    let predictions = tree.predict(x);
    let accuracy = tree.score(x, y);

    println!("  Max Depth: 5");
    println!("  Training Accuracy: {:.1}%", accuracy * 100.0);
    println!("  Predictions: {:?}", predictions);

    // Show some individual predictions
    println!("\n  Sample Predictions:");
    for i in [0, 4, 8].iter() {
        println!(
            "    Sample {}: True={}, Predicted={}",
            i, y[*i], predictions[*i]
        );
    }
}

fn random_forest_example(x: &Matrix<f32>, y: &[usize], n_trees: usize) {
    let mut rf = RandomForestClassifier::new(n_trees)
        .with_max_depth(5)
        .with_random_state(42);

    rf.fit(x, y).expect("Training failed");

    let predictions = rf.predict(x);
    let accuracy = rf.score(x, y);

    println!("  Number of Trees: {}", n_trees);
    println!("  Max Depth: 5");
    println!("  Random State: 42 (reproducible)");
    println!("  Training Accuracy: {:.1}%", accuracy * 100.0);
    println!("  Predictions: {:?}", predictions);

    // Show voting mechanism
    println!("\n  Sample Predictions:");
    for i in [0, 4, 8].iter() {
        println!(
            "    Sample {}: True={}, Predicted={} (from {} tree votes)",
            i, y[*i], predictions[*i], n_trees
        );
    }

    // Compare to expected
    let errors = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| pred != true_label)
        .count();

    if errors == 0 {
        println!("\n  ✓ Perfect classification! All samples correctly predicted.");
    } else {
        println!("\n  {} misclassifications", errors);
    }
}

fn oob_example(x: &Matrix<f32>, y: &[usize]) {
    let mut rf = RandomForestClassifier::new(30)
        .with_max_depth(5)
        .with_random_state(42);

    rf.fit(x, y).expect("Training failed");

    let training_accuracy = rf.score(x, y);
    let oob_score = rf.oob_score();

    println!("  Number of Trees: 30");
    println!("  Training Accuracy: {:.1}%", training_accuracy * 100.0);

    if let Some(oob) = oob_score {
        println!("  OOB Accuracy:      {:.1}%", oob * 100.0);
        println!(
            "  Difference:        {:.1}%",
            (training_accuracy - oob).abs() * 100.0
        );
    }

    println!("\n  What is OOB (Out-of-Bag)?");
    println!("    • Each tree trains on ~63% of data (bootstrap sample)");
    println!("    • Remaining ~37% are 'out-of-bag' for that tree");
    println!("    • OOB samples used as validation set");
    println!("    • No need for separate test set!");
    println!("    • Provides unbiased estimate of accuracy");

    println!("\n  Why use OOB?");
    println!("    ✅ Free validation without splitting data");
    println!("    ✅ Use all data for training AND validation");
    println!("    ✅ Estimate generalization error");
    println!("    ✅ Compare different n_estimators values");
}

fn feature_importance_example(x: &Matrix<f32>, y: &[usize]) {
    let mut rf = RandomForestClassifier::new(30)
        .with_max_depth(5)
        .with_random_state(42);

    rf.fit(x, y).expect("Training failed");

    let importances = rf.feature_importances();

    if let Some(imps) = importances {
        println!("  Number of Trees: 30");
        println!("  Features: petal_length (0), petal_width (1)");
        println!("\n  Feature Importances:");
        println!("    Feature 0 (petal_length): {:.3}", imps[0]);
        println!("    Feature 1 (petal_width):  {:.3}", imps[1]);

        // Identify most important feature
        let max_idx = if imps[0] > imps[1] { 0 } else { 1 };
        let feature_names = ["petal_length", "petal_width"];
        println!(
            "\n  → Most important: {} ({:.1}%)",
            feature_names[max_idx],
            imps[max_idx] * 100.0
        );

        // Verify they sum to 1.0
        let sum: f32 = imps.iter().sum();
        println!("  → Importances sum to: {:.3} ✓", sum);
    }

    println!("\n  What is Feature Importance?");
    println!("    • Measures how much each feature contributes to predictions");
    println!("    • Based on mean decrease in impurity across all trees");
    println!("    • Normalized to sum to 1.0");
    println!("    • Higher values = more important features");

    println!("\n  Why use Feature Importance?");
    println!("    ✅ Identify most predictive features");
    println!("    ✅ Feature selection for dimensionality reduction");
    println!("    ✅ Model interpretability and explainability");
    println!("    ✅ Domain insights (which features matter?)");
}
