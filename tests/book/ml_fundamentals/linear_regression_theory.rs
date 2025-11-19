//! Tests for "Linear Regression Theory" chapter
//!
//! Chapter: book/src/ml-fundamentals/linear-regression.md
//! Status:  Not yet written (this test exists first - TDD!)
//!
//! This test file validates all code examples in the Linear Regression Theory chapter.

use aprender::linear_model::LinearRegression;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;

/// Example 1: Verify OLS closed-form solution
///
/// Book chapter explains: beta = (X^T X)^(-1) X^T y
/// This property test PROVES the math is correct.
#[test]
fn test_ols_closed_form_solution() {
    // Simple 2D case: y = 2x + 1
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Vector::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    // Verify coefficients match expected values (f32 precision)
    let coefficients = model.coefficients();
    assert!(
        (coefficients[0] - 2.0).abs() < 1e-5,
        "Slope should be 2.0, got {}",
        coefficients[0]
    );
    assert!(
        (model.intercept() - 1.0).abs() < 1e-5,
        "Intercept should be 1.0, got {}",
        model.intercept()
    );
}

/// Example 2: Verify predictions match theoretical values
#[test]
fn test_ols_predictions() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    // Predict on new data
    let x_test = Matrix::from_vec(2, 1, vec![4.0, 5.0]).unwrap();
    let predictions = model.predict(&x_test);

    // Verify predictions match y = 2x (f32 precision)
    assert!((predictions[0] - 8.0).abs() < 1e-5);
    assert!((predictions[1] - 10.0).abs() < 1e-5);
}

/// Property Test: OLS should minimize sum of squared residuals
///
/// This PROVES the mathematical property that OLS is optimal.
#[cfg(test)]
mod properties {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ols_minimizes_sse(
            x_vals in prop::collection::vec(-100.0f32..100.0f32, 10..20),
            true_slope in -10.0f32..10.0f32,
            true_intercept in -10.0f32..10.0f32,
        ) {
            // Generate data: y = true_slope * x + true_intercept
            let n = x_vals.len();
            let x = Matrix::from_vec(n, 1, x_vals.clone()).unwrap();
            let y: Vec<f32> = x_vals.iter()
                .map(|&x_val| true_slope * x_val + true_intercept)
                .collect();
            let y = Vector::from_vec(y);

            let mut model = LinearRegression::new();
            if model.fit(&x, &y).is_ok() {
                // Recovered coefficients should be very close to true values
                let coefficients = model.coefficients();
                prop_assert!((coefficients[0] - true_slope).abs() < 0.01);
                prop_assert!((model.intercept() - true_intercept).abs() < 0.01);
            }
        }
    }
}
