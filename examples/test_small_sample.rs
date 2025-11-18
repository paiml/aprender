//! LinearRegression Sample Size Requirements Example
//!
//! Demonstrates error handling for underdetermined systems and shows
//! the minimum sample requirements for linear regression.

use aprender::prelude::*;

fn main() {
    println!("LinearRegression: Sample Size Requirements");
    println!("==========================================\n");

    // TEST 1: Underdetermined system (TOO FEW SAMPLES)
    println!("Test 1: Underdetermined System (n_samples < n_features + 1)");
    println!("-----------------------------------------------------------");
    let x_small = Matrix::from_vec(
        3,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ],
    )
    .unwrap();
    let y_small = Vector::from_vec(vec![10.0, 20.0, 30.0]);

    println!("  n_samples: 3");
    println!("  n_features: 5");
    println!("  fit_intercept: true (requires 5 + 1 = 6 samples)\n");

    let mut model = LinearRegression::new();
    match model.fit(&x_small, &y_small) {
        Ok(_) => println!("  ✅ Success"),
        Err(e) => println!("  ❌ Error: {}\n", e),
    }

    // TEST 2: Exactly determined system (MINIMUM SAMPLES)
    println!("Test 2: Exactly Determined System (n_samples == n_features + 1)");
    println!("---------------------------------------------------------------");
    // Use orthogonal features for numerical stability
    let x_exact = Matrix::from_vec(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let y_exact = Vector::from_vec(vec![1.0, 2.0, 3.0, 6.0]);

    println!("  n_samples: 4");
    println!("  n_features: 3");
    println!("  fit_intercept: true (requires 3 + 1 = 4 samples)\n");

    let mut model2 = LinearRegression::new();
    match model2.fit(&x_exact, &y_exact) {
        Ok(_) => {
            println!("  ✅ Success! Model fitted.");
            let r2 = model2.score(&x_exact, &y_exact);
            println!("  Training R²: {:.3}", r2);
            println!("  Coefficients: {:?}\n", model2.coefficients().as_slice());
        }
        Err(e) => println!("  ❌ Error: {}\n", e),
    }

    // TEST 3: Overdetermined system (RECOMMENDED)
    println!("Test 3: Overdetermined System (n_samples >> n_features)");
    println!("--------------------------------------------------------");
    // Simple linear data: y = 2x + 1
    let x_over = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y_over = Vector::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

    println!("  n_samples: 10");
    println!("  n_features: 1");
    println!("  fit_intercept: true (requires 1 + 1 = 2 samples)\n");

    let mut model3 = LinearRegression::new();
    match model3.fit(&x_over, &y_over) {
        Ok(_) => {
            println!("  ✅ Success! Model fitted.");
            let r2 = model3.score(&x_over, &y_over);
            println!("  Training R²: {:.3}", r2);
            println!("  Coefficients: {:?}", model3.coefficients().as_slice());
            println!("  Intercept: {:.3}\n", model3.intercept());
        }
        Err(e) => println!("  ❌ Error: {}\n", e),
    }

    // SUMMARY
    println!("Summary");
    println!("-------");
    println!("• With intercept: need n_samples >= n_features + 1");
    println!("• Without intercept: need n_samples >= n_features");
    println!("• Recommended: n_samples >> n_features (at least 10x)");
    println!("• For small samples: consider Ridge regression (future feature)");
}
