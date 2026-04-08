#![allow(clippy::disallowed_methods)]
//! Chapter 6: Ensemble Methods
//!
//! Demonstrates RandomForestClassifier on separable data.
//! Citation: Chen & Guestrin, "XGBoost," arXiv:1603.02754
//! Contract: contracts/apr-book-ch06-v1.yaml

use aprender::primitives::Matrix;
use aprender::tree::RandomForestClassifier;
use aprender::metrics::classification::accuracy;

fn main() {
    // Simple classification dataset
    let x = Matrix::from_vec(6, 2, vec![
        1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 6.0, 5.0, 7.0, 8.0, 8.0, 6.0,
    ])
    .expect("valid 6x2 matrix");
    let y: Vec<usize> = vec![0, 0, 0, 1, 1, 1];

    // Random forest classifier
    let mut rf = RandomForestClassifier::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("random forest fit");
    let preds = rf.predict(&x);

    let acc = accuracy(&preds, &y);
    println!("RandomForest accuracy: {acc:.2}");
    assert!(acc >= 0.8, "RF training accuracy contract");

    println!("Ensemble: 10 trees (Breiman, 2001)");
    println!("Chapter 6 contracts: PASSED");
}
