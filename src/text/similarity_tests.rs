pub(crate) use super::*;

#[test]
fn test_cosine_similarity_identical() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let sim = cosine_similarity(&v, &v).expect("should succeed");
    assert!((sim - 1.0).abs() < 1e-10);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let v1 = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let v2 = Vector::from_slice(&[0.0, 1.0, 0.0]);
    let sim = cosine_similarity(&v1, &v2).expect("should succeed");
    assert!(sim.abs() < 1e-10);
}

#[test]
fn test_jaccard_similarity() {
    let a = vec!["the", "cat", "sat"];
    let b = vec!["the", "dog", "sat"];
    let sim = jaccard_similarity(&a, &b).expect("should succeed");
    assert!((sim - 0.5).abs() < 1e-10); // 2 common / 4 total
}

#[test]
fn test_edit_distance() {
    let dist = edit_distance("kitten", "sitting").expect("should succeed");
    assert_eq!(dist, 3);

    let dist = edit_distance("", "abc").expect("should succeed");
    assert_eq!(dist, 3);

    let dist = edit_distance("same", "same").expect("should succeed");
    assert_eq!(dist, 0);
}

#[test]
fn test_edit_distance_similarity() {
    let sim = edit_distance_similarity("hello", "hello").expect("should succeed");
    assert!((sim - 1.0).abs() < 1e-10);

    let sim = edit_distance_similarity("abc", "xyz").expect("should succeed");
    assert!(sim < 0.5);
}

#[test]
fn test_top_k_similar() {
    let query = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let docs = vec![
        Vector::from_slice(&[2.0, 3.0, 4.0]),
        Vector::from_slice(&[0.0, 0.0, 1.0]),
        Vector::from_slice(&[1.0, 2.0, 2.9]),
    ];

    let top = top_k_similar(&query, &docs, 2).expect("should succeed");
    assert_eq!(top.len(), 2);
    assert!(top[0].1 > top[1].1); // Sorted by similarity
}

#[test]
fn test_cosine_similarity_different_lengths() {
    let v1 = Vector::from_slice(&[1.0, 2.0]);
    let v2 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert!(cosine_similarity(&v1, &v2).is_err());
}

#[test]
fn test_cosine_similarity_empty() {
    let v1 = Vector::from_slice(&[]);
    let v2 = Vector::from_slice(&[]);
    assert!(cosine_similarity(&v1, &v2).is_err());
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let v1 = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let v2 = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let sim = cosine_similarity(&v1, &v2).expect("should succeed");
    assert_eq!(sim, 0.0); // Zero vector is orthogonal
}

#[test]
fn test_jaccard_similarity_both_empty() {
    let a: Vec<&str> = vec![];
    let b: Vec<&str> = vec![];
    let sim = jaccard_similarity(&a, &b).expect("should succeed");
    assert_eq!(sim, 1.0); // Empty sets are identical
}

#[test]
fn test_jaccard_similarity_one_empty() {
    let a = vec!["word"];
    let b: Vec<&str> = vec![];
    let sim = jaccard_similarity(&a, &b).expect("should succeed");
    assert_eq!(sim, 0.0); // No overlap possible
}

#[test]
fn test_edit_distance_empty_second() {
    let dist = edit_distance("abc", "").expect("should succeed");
    assert_eq!(dist, 3);
}

#[test]
fn test_edit_distance_similarity_empty() {
    let sim = edit_distance_similarity("", "").expect("should succeed");
    assert_eq!(sim, 1.0); // Empty strings are identical
}

#[test]
fn test_pairwise_cosine_similarity() {
    let docs = vec![
        Vector::from_slice(&[1.0, 0.0]),
        Vector::from_slice(&[0.0, 1.0]),
        Vector::from_slice(&[1.0, 1.0]),
    ];

    let sim_matrix = pairwise_cosine_similarity(&docs).expect("should succeed");
    assert_eq!(sim_matrix.len(), 3);
    assert_eq!(sim_matrix[0].len(), 3);

    // Diagonal should be 1.0 (self-similarity)
    assert!((sim_matrix[0][0] - 1.0).abs() < 1e-10);
    assert!((sim_matrix[1][1] - 1.0).abs() < 1e-10);

    // Symmetric
    assert!((sim_matrix[0][1] - sim_matrix[1][0]).abs() < 1e-10);
}

#[test]
fn test_pairwise_cosine_similarity_empty() {
    let docs: Vec<Vector<f64>> = vec![];
    let sim_matrix = pairwise_cosine_similarity(&docs).expect("should succeed");
    assert!(sim_matrix.is_empty());
}

#[test]
fn test_top_k_similar_empty_docs() {
    let query = Vector::from_slice(&[1.0, 2.0]);
    let docs: Vec<Vector<f64>> = vec![];
    let top = top_k_similar(&query, &docs, 5).expect("should succeed");
    assert!(top.is_empty());
}
