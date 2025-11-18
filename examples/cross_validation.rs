//! Cross-Validation Example
//!
//! Demonstrates train/test splitting and K-Fold cross-validation for model evaluation.
//! Essential for assessing model performance and preventing overfitting.

use aprender::linear_model::LinearRegression;
use aprender::model_selection::{cross_validate, train_test_split, KFold};
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;

fn main() {
    println!("Cross-Validation - Model Selection Example");
    println!("===========================================\n");

    // Example 1: Train/Test Split
    println!("Example 1: Train/Test Split");
    println!("---------------------------");
    train_test_split_example();

    // Example 2: K-Fold Cross-Validation (Manual)
    println!("\nExample 2: K-Fold Cross-Validation (Manual)");
    println!("-------------------------------------------");
    kfold_example();

    // Example 3: Automated cross_validate()
    println!("\nExample 3: Automated cross_validate()");
    println!("-------------------------------------");
    cross_validate_example();

    println!("\n✅ Cross-validation complete!");
    println!("\nKey Benefits:");
    println!("  • Unbiased performance estimates");
    println!("  • Detect overfitting early");
    println!("  • Maximize use of limited data");
    println!("  • Industry best practice for ML validation");
}

fn train_test_split_example() {
    // Create synthetic dataset: y = 3x + 2 + noise
    let x_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 3.0 * x + 2.0).collect();

    let x = Matrix::from_vec(100, 1, x_data).unwrap();
    let y = Vector::from_vec(y_data);

    // Split 80/20
    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, 0.2, Some(42)).expect("Split failed");

    println!("  Dataset: 100 samples");
    println!("  Split: 80% train, 20% test");
    println!("  Training samples: {}", x_train.shape().0);
    println!("  Test samples: {}", x_test.shape().0);

    // Train model on training set
    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train).expect("Training failed");

    println!(
        "\n  Fitted model: y = {:.2}x + {:.2}",
        model.coefficients()[0],
        model.intercept()
    );

    // Evaluate on both sets
    let train_score = model.score(&x_train, &y_train);
    let test_score = model.score(&x_test, &y_test);

    println!("\n  Training R²: {:.4}", train_score);
    println!("  Test R²:     {:.4}", test_score);

    let generalization_gap = (train_score - test_score).abs();
    println!("  Generalization gap: {:.4}", generalization_gap);

    if generalization_gap < 0.05 {
        println!("  ✓ Model generalizes well!");
    } else {
        println!("  ⚠ Possible overfitting detected");
    }
}

fn kfold_example() {
    // Create synthetic dataset: y = 2x + 1
    let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let x = Matrix::from_vec(50, 1, x_data).unwrap();
    let y = Vector::from_vec(y_data);

    // 5-Fold cross-validation
    let kfold = KFold::new(5).with_random_state(42);
    let splits = kfold.split(50);

    println!("  Dataset: 50 samples");
    println!("  K-Fold: 5 folds");
    println!("  Reproducible: Yes (random_state=42)\n");

    let mut fold_scores = Vec::new();

    for (fold_num, (train_idx, test_idx)) in splits.iter().enumerate() {
        // Extract fold data
        let (x_train_fold, y_train_fold) = extract_samples(&x, &y, train_idx);
        let (x_test_fold, y_test_fold) = extract_samples(&x, &y, test_idx);

        // Train model on this fold
        let mut model = LinearRegression::new();
        model
            .fit(&x_train_fold, &y_train_fold)
            .expect("Training failed");

        // Evaluate
        let score = model.score(&x_test_fold, &y_test_fold);
        fold_scores.push(score);

        println!(
            "  Fold {}: train_size={:2}, test_size={:2}, R² = {:.4}",
            fold_num + 1,
            train_idx.len(),
            test_idx.len(),
            score
        );
    }

    // Compute statistics
    let mean_score = fold_scores.iter().sum::<f32>() / fold_scores.len() as f32;
    let variance = fold_scores
        .iter()
        .map(|&score| (score - mean_score).powi(2))
        .sum::<f32>()
        / fold_scores.len() as f32;
    let std_dev = variance.sqrt();

    println!("\n  Cross-Validation Results:");
    println!("  -------------------------");
    println!("  Mean R²: {:.4} ± {:.4}", mean_score, std_dev);
    println!(
        "  Min R²:  {:.4}",
        fold_scores.iter().cloned().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max R²:  {:.4}",
        fold_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    if std_dev < 0.05 {
        println!("  ✓ Stable model across folds!");
    }
}

fn cross_validate_example() {
    // Create synthetic dataset: y = 4x - 3
    let x_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 4.0 * x - 3.0).collect();

    let x = Matrix::from_vec(100, 1, x_data).unwrap();
    let y = Vector::from_vec(y_data);

    // Create model and cross-validation splitter
    let model = LinearRegression::new();
    let kfold = KFold::new(10).with_random_state(42);

    println!("  Dataset: 100 samples");
    println!("  Model: LinearRegression");
    println!("  CV Strategy: 10-Fold (reproducible)\n");

    // Run automated cross-validation
    let results = cross_validate(&model, &x, &y, &kfold).expect("Cross-validation failed");

    println!("  Results:");
    println!("  --------");
    println!("  Mean R²: {:.4}", results.mean());
    println!("  Std Dev: {:.4}", results.std());
    println!("  Min R²:  {:.4}", results.min());
    println!("  Max R²:  {:.4}", results.max());
    println!("  Folds:   {}", results.scores.len());

    // Show all fold scores
    println!("\n  Fold Scores:");
    for (i, &score) in results.scores.iter().enumerate() {
        println!("    Fold {:2}: {:.4}", i + 1, score);
    }

    // Interpret results
    println!("\n  Interpretation:");
    if results.mean() > 0.95 {
        println!("  ✓ Excellent model performance (R² > 0.95)");
    }
    if results.std() < 0.05 {
        println!("  ✓ Very stable across folds (std < 0.05)");
    }

    println!("\n  Advantages:");
    println!("  • Single function call - no manual loop");
    println!("  • Automatic fold management");
    println!("  • Built-in statistics (mean, std, min, max)");
    println!("  • Reproducible with random_state");
}

/// Helper function to extract samples by indices
fn extract_samples(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    indices: &[usize],
) -> (Matrix<f32>, Vector<f32>) {
    let n_features = x.shape().1;
    let mut x_data = Vec::with_capacity(indices.len() * n_features);
    let mut y_data = Vec::with_capacity(indices.len());

    for &idx in indices {
        for j in 0..n_features {
            x_data.push(x.get(idx, j));
        }
        y_data.push(y.as_slice()[idx]);
    }

    let x_subset =
        Matrix::from_vec(indices.len(), n_features, x_data).expect("Failed to create matrix");
    let y_subset = Vector::from_vec(y_data);

    (x_subset, y_subset)
}
