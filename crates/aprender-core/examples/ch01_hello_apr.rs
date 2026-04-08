#![allow(clippy::disallowed_methods)]
//! Chapter 1: Why Rust for Machine Learning
//!
//! Demonstrates the Estimator trait with LinearRegression.
//! Citation: Rajbhandari et al., "ZeRO," arXiv:1910.02054
//! Contract: contracts/apr-book-ch01-v1.yaml

use aprender::prelude::*;

fn main() {
    // Create training data: y = 2x + 1
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    // Train linear regression
    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("linear regression fit");

    // Predict
    let x_test = Matrix::from_vec(1, 1, vec![5.0]).expect("valid matrix");
    let pred = model.predict(&x_test);
    println!("Prediction for x=5: {:.2}", pred[0]);
    assert!(
        (pred[0] - 11.0).abs() < 1.0,
        "Linear fit sanity: expected ~11, got {}",
        pred[0]
    );

    // Score
    let r2 = model.score(&x, &y);
    println!("R-squared: {r2:.4}");
    assert!(r2 > 0.99, "R-squared must be >0.99 for perfect linear data");

    println!("Chapter 1 contracts: PASSED");
}
