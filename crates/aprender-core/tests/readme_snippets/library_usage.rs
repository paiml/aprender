// Compiled witness for README.md "## Library usage" (F-FR-U00-010).
// readme_contract.rs asserts this file contains the README block verbatim, so
// the snippet the README promises is compiled by this test target.
#[allow(dead_code, unused_variables)]
fn readme_library_snippet() {
use aprender::linear_model::LinearRegression;
use aprender::prelude::{Matrix, Vector};
use aprender::traits::Estimator;

// Train on y = 2*x + 1
let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix dimensions");
let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

let mut model = LinearRegression::new();
model.fit(&x, &y).expect("linear regression fit should succeed");
let predictions = model.predict(&x);
let r2 = model.score(&x, &y);
assert!(r2 > 0.99);
}
