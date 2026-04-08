#![allow(clippy::disallowed_methods)]
//! Chapter 5: Unsupervised Learning
//!
//! Demonstrates KMeans clustering on well-separated data.
//! Citation: Halko et al., "Finding Structure with Randomness," arXiv:0909.4061
//! Contract: contracts/apr-book-ch05-v1.yaml

use aprender::prelude::*;

fn main() {
    // Two obvious clusters
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 5.0, 5.0, 5.1, 4.9, 4.9, 5.1],
    )
    .expect("valid 6x2 matrix");

    // KMeans with k=2
    let mut kmeans = KMeans::new(2).with_max_iter(100).with_random_state(42);
    kmeans.fit(&data).expect("kmeans fit");

    // Predict cluster labels
    let labels = kmeans.predict(&data);
    println!("KMeans labels: {:?}", labels);

    // Cluster coherence: points 0-2 same cluster, 3-5 same cluster
    assert_eq!(labels[0], labels[1], "Cluster coherence: 0 and 1");
    assert_eq!(labels[1], labels[2], "Cluster coherence: 1 and 2");
    assert_eq!(labels[3], labels[4], "Cluster coherence: 3 and 4");
    assert_eq!(labels[4], labels[5], "Cluster coherence: 4 and 5");
    // Cluster separation
    assert_ne!(labels[0], labels[3], "Cluster separation: 0 vs 3");

    println!("Chapter 5 contracts: PASSED");
}
