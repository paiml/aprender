use super::*;

#[test]
fn test_empty_index() {
    let index = HNSWIndex::new(16, 200, 0.0);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_add_single_item() {
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("item1", Vector::from_slice(&[1.0, 2.0, 3.0]));
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn test_search_single_item() {
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("item1", Vector::from_slice(&[1.0, 2.0, 3.0]));

    let query = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let results = index.search(&query, 1);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "item1");
    assert!(results[0].1 < 1e-6); // Distance ~0
}

#[test]
fn test_search_multiple_items() {
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("a", Vector::from_slice(&[1.0, 0.0, 0.0]));
    index.add("b", Vector::from_slice(&[0.0, 1.0, 0.0]));
    index.add("c", Vector::from_slice(&[0.0, 0.0, 1.0]));

    let query = Vector::from_slice(&[0.9, 0.1, 0.0]);
    let results = index.search(&query, 2);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "a"); // Closest to [1,0,0]
}

#[test]
fn test_search_k_larger_than_index() {
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("a", Vector::from_slice(&[1.0]));
    index.add("b", Vector::from_slice(&[2.0]));

    let query = Vector::from_slice(&[1.5]);
    let results = index.search(&query, 10);

    assert_eq!(results.len(), 2); // Only 2 items available
}

#[test]
fn test_cosine_distance() {
    // Identical vectors
    let a = Vector::from_slice(&[1.0, 0.0]);
    let b = Vector::from_slice(&[1.0, 0.0]);
    assert!(HNSWIndex::distance(&a, &b) < 1e-10);

    // Orthogonal vectors
    let a = Vector::from_slice(&[1.0, 0.0]);
    let b = Vector::from_slice(&[0.0, 1.0]);
    assert!((HNSWIndex::distance(&a, &b) - 1.0).abs() < 1e-10);

    // Opposite vectors
    let a = Vector::from_slice(&[1.0, 0.0]);
    let b = Vector::from_slice(&[-1.0, 0.0]);
    assert!((HNSWIndex::distance(&a, &b) - 2.0).abs() < 1e-10);
}

#[test]
fn test_search_order_by_similarity() {
    let mut index = HNSWIndex::new(16, 200, 0.0);

    // Add items at different angles (cosine distance measures angle, not magnitude)
    index.add("closest", Vector::from_slice(&[1.0, 0.1])); // ~6 degrees from x-axis
    index.add("medium", Vector::from_slice(&[1.0, 1.0])); // 45 degrees from x-axis
    index.add("farthest", Vector::from_slice(&[0.1, 1.0])); // ~84 degrees from x-axis

    let query = Vector::from_slice(&[1.0, 0.0]); // 0 degrees from x-axis
    let results = index.search(&query, 3);

    assert_eq!(results.len(), 3);
    // Results should be ordered by cosine distance (ascending)
    assert!(
        results[0].1 <= results[1].1,
        "First result should be closest"
    );
    assert!(
        results[1].1 <= results[2].1,
        "Second result should be closer than or equal to third"
    );

    // Verify that "farthest" has the largest distance
    let farthest_dist = results
        .iter()
        .find(|(id, _)| id == "farthest")
        .expect("farthest should be in results")
        .1;
    let closest_dist = results
        .iter()
        .find(|(id, _)| id == "closest")
        .expect("closest should be in results")
        .1;
    assert!(
        farthest_dist > closest_dist,
        "Farthest should have larger distance than closest"
    );
}

#[test]
fn test_empty_search() {
    let index = HNSWIndex::new(16, 200, 0.0);
    let query = Vector::from_slice(&[1.0, 2.0]);
    let results = index.search(&query, 5);
    assert!(results.is_empty());
}

#[test]
fn test_search_debug() {
    let mut index = HNSWIndex::new(16, 200, 0.0);

    index.add("a", Vector::from_slice(&[1.0, 1.0]));
    index.add("b", Vector::from_slice(&[2.0, 2.0]));
    index.add("c", Vector::from_slice(&[10.0, 10.0]));

    let query = Vector::from_slice(&[0.9, 0.9]);
    let results = index.search(&query, 3);

    // Print results for debugging
    for (i, (id, dist)) in results.iter().enumerate() {
        eprintln!("[{i}] id={id}, dist={dist:.6}");
    }

    // Just check that we get 3 results
    assert_eq!(results.len(), 3);
}

// ================================================================
// Additional coverage tests for missed branches
// ================================================================

#[test]
fn test_distance_mismatched_lengths() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[1.0, 2.0]);
    let dist = HNSWIndex::distance(&a, &b);
    assert!(dist.is_infinite());
}

#[test]
fn test_distance_zero_vector_a() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let dist = HNSWIndex::distance(&a, &b);
    assert!(dist.is_infinite());
}

#[test]
fn test_distance_zero_vector_b() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let dist = HNSWIndex::distance(&a, &b);
    assert!(dist.is_infinite());
}

#[test]
fn test_distance_both_zero_vectors() {
    let a = Vector::from_slice(&[0.0, 0.0]);
    let b = Vector::from_slice(&[0.0, 0.0]);
    let dist = HNSWIndex::distance(&a, &b);
    assert!(dist.is_infinite());
}

#[test]
fn test_m_accessor() {
    let index = HNSWIndex::new(32, 100, 0.0);
    assert_eq!(index.m(), 32);
}

#[test]
fn test_ef_construction_accessor() {
    let index = HNSWIndex::new(16, 300, 0.0);
    assert_eq!(index.ef_construction(), 300);
}

#[test]
fn test_search_empty_query_on_nonempty_index() {
    // Test with a zero-length query vector against items with different length
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("item1", Vector::from_slice(&[1.0, 2.0, 3.0]));

    // Query with different dimension
    let query = Vector::from_slice(&[1.0, 2.0]);
    let results = index.search(&query, 1);

    // Should still return results (distance will be infinity)
    assert_eq!(results.len(), 1);
}

#[test]
fn test_add_many_items_triggers_pruning() {
    // Adding many items with small M to trigger connection pruning
    let mut index = HNSWIndex::new(2, 10, 0.0);

    for i in 0..20 {
        let val = (i as f64) * 0.1;
        index.add(format!("item{i}"), Vector::from_slice(&[val, 1.0 - val]));
    }

    assert_eq!(index.len(), 20);

    // Search should still work after pruning
    let query = Vector::from_slice(&[0.5, 0.5]);
    let results = index.search(&query, 5);
    assert!(!results.is_empty());
    assert!(results.len() <= 5);
}

#[test]
fn test_search_k_zero() {
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("item1", Vector::from_slice(&[1.0, 0.0]));

    let query = Vector::from_slice(&[1.0, 0.0]);
    let results = index.search(&query, 0);

    // k=0 means take(0) which yields empty
    assert!(results.is_empty());
}

#[test]
fn test_add_duplicate_id() {
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("same", Vector::from_slice(&[1.0, 0.0]));
    index.add("same", Vector::from_slice(&[0.0, 1.0]));

    // Both nodes exist (len=2) but item_to_node maps to the latest
    assert_eq!(index.len(), 2);
}

#[test]
fn test_high_dimensional_vectors() {
    let mut index = HNSWIndex::new(16, 200, 0.0);

    // 100-dimensional vectors (orthogonal-ish to avoid cosine distance edge cases)
    let v1: Vec<f64> = (0..100).map(|i| if i < 33 { 1.0 } else { 0.0 }).collect();
    let v2: Vec<f64> = (0..100)
        .map(|i| if (33..66).contains(&i) { 1.0 } else { 0.0 })
        .collect();
    let v3: Vec<f64> = (0..100).map(|i| if i >= 66 { 1.0 } else { 0.0 }).collect();

    index.add("a", Vector::from_slice(&v1));
    index.add("b", Vector::from_slice(&v2));
    index.add("c", Vector::from_slice(&v3));

    let query = Vector::from_slice(&v1);
    let results = index.search(&query, 2);

    assert_eq!(results.len(), 2);
    // "a" should be closest to itself
    assert_eq!(results[0].0, "a");
    assert!(results[0].1 < 1e-10);
}

#[test]
fn test_collinear_vectors() {
    // Vectors pointing in same direction but different magnitudes
    // Cosine distance should be 0 (identical direction)
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[2.0, 4.0]);
    let dist = HNSWIndex::distance(&a, &b);
    assert!(dist < 1e-10, "Collinear vectors should have distance ~0");
}

#[test]
fn test_search_single_item_k_greater_than_one() {
    let mut index = HNSWIndex::new(16, 200, 0.0);
    index.add("only", Vector::from_slice(&[1.0, 0.0]));

    let query = Vector::from_slice(&[0.9, 0.1]);
    let results = index.search(&query, 5);

    // Only 1 item in index
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "only");
}
