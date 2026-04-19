#![allow(clippy::disallowed_methods)]
//! Chapter 26: Switch From ndarray/nalgebra/linfa
//!
//! API equivalence: ndarray/linfa → aprender
//! Contract: contracts/apr-book-ch26-v1.yaml

use aprender::metrics::classification::accuracy;
use aprender::prelude::*;

fn main() {
    println!("=== Switch From ndarray/nalgebra/linfa ===");
    println!();

    // API equivalence table
    println!("| ndarray/linfa                  | aprender                           |");
    println!("|--------------------------------|------------------------------------|");
    println!("| Array2::from_shape_vec((r,c))  | Matrix::from_vec(r, c, vec)        |");
    println!("| Array1::from_vec(vec)          | Vector::from_slice(&vec)           |");
    println!("| a.dot(&b)                      | a.dot(&b)                          |");
    println!("| linfa::KMeans::params(k)       | KMeans::new(k)                     |");
    println!("| linfa_linear::LinearRegression | LinearRegression::new()            |");
    println!("| dataset.fit(&model)            | model.fit(&x, &y)                  |");
    println!("| model.predict(&x)              | model.predict(&x)                  |");
    println!();

    // Matrix operations — same patterns
    let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid 3x2 matrix");
    let v = Vector::from_slice(&[1.0, 1.0]);
    let result = m.matvec(&v).expect("matvec");
    println!(
        "Matrix (3x2) @ Vector (2) = [{:.0}, {:.0}, {:.0}]",
        result[0], result[1], result[2]
    );
    assert!((result[0] - 3.0).abs() < 1e-4, "matvec contract");

    // LinearRegression — same API pattern as linfa
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);
    let mut lr = LinearRegression::new();
    lr.fit(&x, &y).expect("fit");
    let pred = lr.predict(&x);
    let r2 = lr.score(&x, &y);
    println!();
    println!("LinearRegression: R2 = {r2:.4}, first prediction = {:.4}", pred.as_slice()[0]);
    assert!(r2 > 0.99, "Perfect linear fit");

    // KMeans — same pattern as linfa
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 5.0, 5.0, 5.1, 4.9, 4.9, 5.1],
    )
    .expect("valid");
    let mut km = KMeans::new(2).with_random_state(42);
    km.fit(&data).expect("fit");
    let labels = km.predict(&data);
    println!("KMeans labels: {:?}", labels);
    assert_eq!(labels[0], labels[1], "Cluster coherence");
    assert_ne!(labels[0], labels[3], "Cluster separation");

    // DecisionTree for classification
    let x_cls = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 6.0, 5.0, 7.0, 8.0, 8.0, 6.0],
    )
    .expect("valid");
    let y_cls: Vec<usize> = vec![0, 0, 0, 1, 1, 1];
    let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
    tree.fit(&x_cls, &y_cls).expect("fit");
    let preds = tree.predict(&x_cls);
    let acc = accuracy(&preds, &y_cls);
    println!("DecisionTree accuracy: {acc:.2}");
    assert!(acc > 0.8, "Accuracy contract");

    println!();
    println!("Key differences from ndarray/linfa:");
    println!("  1. aprender uses f32 by default (ML-optimized, SIMD-friendly)");
    println!("  2. No Dataset wrapper needed — fit(&Matrix, &[label]) directly");
    println!("  3. 70 crates: compute, serve, train, contracts (not just ML)");
    println!("  4. cargo install aprender gives you `apr` CLI (57 commands)");

    println!();
    println!("Chapter 26 contracts: PASSED");
}
