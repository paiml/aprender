#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Machine Learning: Vector Similarity and Normalization
//!
//! This example demonstrates common ML operations using Trueno:
//! - Cosine similarity (used in recommendation systems, semantic search)
//! - L2 normalization (feature scaling)
//! - Euclidean distance (clustering, nearest neighbors)
//!
//! Run with:
//! ```
//! cargo run --release --example ml_similarity
//! ```

use trueno::Vector;

fn main() {
    println!("🤖 Machine Learning Vector Operations with Trueno\n");
    println!("{}", "=".repeat(80));

    // Example: Document similarity in a recommendation system
    println!("\n📚 Use Case 1: Document Similarity (Recommendation System)\n");

    // TF-IDF vectors for three documents
    let doc1 = Vector::from_slice(&[0.5, 0.8, 0.0, 0.3, 0.1]); // "machine learning tutorial"
    let doc2 = Vector::from_slice(&[0.6, 0.7, 0.1, 0.2, 0.0]); // "deep learning guide"
    let doc3 = Vector::from_slice(&[0.1, 0.0, 0.9, 0.8, 0.7]); // "cooking recipes"

    println!("Document 1: {:?}", doc1.as_slice());
    println!("Document 2: {:?}", doc2.as_slice());
    println!("Document 3: {:?}", doc3.as_slice());

    // Compute cosine similarities
    let sim_1_2 = cosine_similarity(&doc1, &doc2);
    let sim_1_3 = cosine_similarity(&doc1, &doc3);
    let sim_2_3 = cosine_similarity(&doc2, &doc3);

    println!("\nCosine Similarities:");
    println!("  Doc1 vs Doc2 (ML vs DL): {:.4} - Similar topics!", sim_1_2);
    println!("  Doc1 vs Doc3 (ML vs Cooking): {:.4} - Different topics", sim_1_3);
    println!("  Doc2 vs Doc3 (DL vs Cooking): {:.4} - Different topics", sim_2_3);

    // Example: Feature normalization for neural network input
    println!("\n{}", "-".repeat(80));
    println!("\n🧠 Use Case 2: Feature Normalization (Neural Network Preprocessing)\n");

    // Raw features: [age, salary, years_experience]
    let features = Vector::from_slice(&[35.0, 75000.0, 8.0]);
    println!("Raw features: {:?}", features.as_slice());
    println!("  (age=35, salary=$75k, experience=8 years)");

    // L2 normalization (unit vector)
    let normalized = l2_normalize(&features);
    println!("\nL2 Normalized: {:?}", normalized.as_slice());

    // Verify it's a unit vector (magnitude = 1.0)
    let magnitude = (normalized.dot(&normalized).expect("Example should not fail")).sqrt();
    println!("Magnitude: {:.6} (should be ~1.0)", magnitude);

    // Example: k-Nearest Neighbors distance calculation
    println!("\n{}", "-".repeat(80));
    println!("\n🎯 Use Case 3: k-Nearest Neighbors (Classification)\n");

    // Training samples in 2D feature space
    let sample1 = Vector::from_slice(&[1.0, 2.0]); // Class A
    let sample2 = Vector::from_slice(&[1.5, 1.8]); // Class A
    let sample3 = Vector::from_slice(&[8.0, 8.0]); // Class B
    let sample4 = Vector::from_slice(&[9.0, 7.5]); // Class B

    // New point to classify
    let new_point = Vector::from_slice(&[2.0, 2.5]);

    println!("Training samples:");
    println!("  Sample 1 (Class A): {:?}", sample1.as_slice());
    println!("  Sample 2 (Class A): {:?}", sample2.as_slice());
    println!("  Sample 3 (Class B): {:?}", sample3.as_slice());
    println!("  Sample 4 (Class B): {:?}", sample4.as_slice());
    println!("\nNew point: {:?}", new_point.as_slice());

    // Calculate Euclidean distances
    let dist1 = euclidean_distance(&new_point, &sample1);
    let dist2 = euclidean_distance(&new_point, &sample2);
    let dist3 = euclidean_distance(&new_point, &sample3);
    let dist4 = euclidean_distance(&new_point, &sample4);

    println!("\nDistances to training samples:");
    println!("  Distance to Sample 1: {:.4}", dist1);
    println!("  Distance to Sample 2: {:.4}", dist2);
    println!("  Distance to Sample 3: {:.4}", dist3);
    println!("  Distance to Sample 4: {:.4}", dist4);

    // Find nearest neighbor
    let min_dist = dist1.min(dist2).min(dist3).min(dist4);
    let class = if min_dist == dist1 || min_dist == dist2 { "A" } else { "B" };
    println!("\nNearest neighbor distance: {:.4}", min_dist);
    println!("Predicted class: {} ✓", class);

    // Performance note
    println!("\n{}", "=".repeat(80));
    println!("\n⚡ Performance Note:\n");
    println!("All vector operations use SIMD (SSE2/AVX2) automatically:");
    println!("  • Dot product: ~340% faster than scalar");
    println!("  • Sum (for normalization): ~315% faster");
    println!("  • Element-wise operations: Modest speedup\n");
    println!("See examples/performance_demo.rs for detailed benchmarks");
    println!("{}", "=".repeat(80));
}

/// Compute cosine similarity between two vectors
///
/// Cosine similarity = (A · B) / (||A|| × ||B||)
/// Range: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite
fn cosine_similarity(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
    let dot = a.dot(b).expect("Example should not fail");
    let mag_a = a.dot(a).expect("Example should not fail").sqrt();
    let mag_b = b.dot(b).expect("Example should not fail").sqrt();
    dot / (mag_a * mag_b)
}

/// Normalize vector to unit length (L2 normalization)
///
/// Returns a new vector where ||v|| = 1
fn l2_normalize(v: &Vector<f32>) -> Vector<f32> {
    let magnitude = v.dot(v).expect("Example should not fail").sqrt();
    let data: Vec<f32> = v.as_slice().iter().map(|x| x / magnitude).collect();
    Vector::from_slice(&data)
}

/// Compute Euclidean distance between two vectors
///
/// Distance = sqrt(sum((a_i - b_i)^2))
fn euclidean_distance(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
    let diff = a.as_slice().iter().zip(b.as_slice()).map(|(x, y)| (x - y).powi(2)).sum::<f32>();
    diff.sqrt()
}
