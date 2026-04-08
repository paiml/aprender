#![allow(clippy::disallowed_methods)]
//! Chapter 4: Supervised Learning
//!
//! Demonstrates DecisionTree and LogisticRegression on separable data.
//! Citation: Ruder, "Gradient Descent Optimization," arXiv:1609.04747
//! Contract: contracts/apr-book-ch04-v1.yaml

use aprender::prelude::*;
use aprender::metrics::classification::accuracy;

fn main() {
    // Training data: two separable classes
    let x = Matrix::from_vec(6, 2, vec![
        1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 6.0, 5.0, 7.0, 8.0, 8.0, 6.0,
    ])
    .expect("valid 6x2 matrix");
    let y: Vec<usize> = vec![0, 0, 0, 1, 1, 1];

    // Decision tree classifier
    let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
    tree.fit(&x, &y).expect("decision tree fit");
    let preds = tree.predict(&x);
    let acc = accuracy(&preds, &y);
    println!("DecisionTree accuracy: {acc:.2}");
    assert!(acc >= 0.8, "Decision tree training accuracy contract");

    // Logistic regression
    let mut lr = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);
    lr.fit(&x, &y).expect("logistic regression fit");
    let lr_preds = lr.predict(&x);
    let lr_acc = accuracy(&lr_preds, &y);
    println!("LogisticRegression accuracy: {lr_acc:.2}");

    println!("Chapter 4 contracts: PASSED");
}
